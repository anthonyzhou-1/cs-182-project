# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utilities for reading open sourced Learning Complex Physics data."""

import functools
import numpy as np
import tensorflow.compat.v1 as tf
import os
import torch
import functools
import json

# Create a description of the features.
_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}


def convert_to_tensor(x, encoded_dtype):
    if len(x) == 1:
        out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
    else:
        out = []
        for el in x:
            out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
    out = tf.convert_to_tensor(np.array(out))
    return out


def parse_serialized_simulation_example(example_proto, metadata):
    """Parses a serialized simulation tf.SequenceExample.
    Args:
      example_proto: A string encoding of the tf.SequenceExample proto.
      metadata: A dict of metadata for the dataset.
    Returns:
      context: A dict, with features that do not vary over the trajectory.
      parsed_features: A dict of tf.Tensors representing the parsed examples
        across time, where axis zero is the time axis.
    """
    if 'context_mean' in metadata:
        feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
    else:
        feature_description = _FEATURE_DESCRIPTION
    context, parsed_features = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=_CONTEXT_FEATURES,
        sequence_features=feature_description)
    for feature_key, item in parsed_features.items():
        convert_fn = functools.partial(
            convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
        parsed_features[feature_key] = tf.py_function(
            convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out'])

    # There is an extra frame at the beginning so we can calculate pos change
    # for all frames used in the paper.
    position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]

    # Reshape positions to correct dim:
    parsed_features['position'] = tf.reshape(parsed_features['position'],
                                             position_shape)
    # Set correct shapes of the remaining tensors.
    sequence_length = metadata['sequence_length'] + 1
    if 'context_mean' in metadata:
        context_feat_len = len(metadata['context_mean'])
        parsed_features['step_context'] = tf.reshape(
            parsed_features['step_context'],
            [sequence_length, context_feat_len])
    # Decode particle type explicitly
    context['particle_type'] = tf.py_function(
        functools.partial(convert_fn, encoded_dtype=np.int64),
        inp=[context['particle_type'].values],
        Tout=[tf.int64])
    context['particle_type'] = tf.reshape(context['particle_type'], [-1])
    return context, parsed_features


def read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())

def prepare_rollout_inputs(context, features):
    """Prepares an inputs trajectory for rollout."""
    out_dict = {**context}
    # Position is encoded as [sequence_length, num_particles, dim] but the model
    # expects [num_particles, sequence_length, dim].
    pos = tf.transpose(features['position'], [1, 0, 2])
    out_dict['position'] = pos
    # Compute the number of nodes
    out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
    if 'step_context' in features:
        out_dict['step_context'] = features['step_context']
    out_dict['is_trajectory'] = tf.constant([True], tf.bool)
    return out_dict

def tf2torch(data):
    for key in data:
        tensor = data[key]
        numpy_arr = tensor.numpy()
        if(isinstance(numpy_arr, np.ndarray)):
            data[key] = torch.from_numpy(numpy_arr)
    return data

def parse_data(path, metadata):
    ds = tf.data.TFRecordDataset(path)
    ds = ds.map(functools.partial(parse_serialized_simulation_example, metadata=metadata))
    ds = ds.map(prepare_rollout_inputs)
    parsed_data = []
    for element in ds:
        parsed_data.append(tf2torch(element))
    return parsed_data

def get_positions(parsed_data):
    positions = [parsed_data[i]['position'] for i in np.arange(len(parsed_data))]
    return positions

def combine_positions(positions, all_pos=None):
    for i in np.arange(len(positions)):
        if all_pos is None:
            all_pos = positions[i]
        else: 
            all_pos = torch.cat([all_pos, positions[i]])
    return all_pos

def update_dictionary_position(parsed_data, positions):
    for i in np.arange(len(parsed_data)):
        parsed_data[i]['position'] = positions[i]  
    return 

def half_trajectories(data_path, n=2):
    metadata = read_metadata(data_path)
    parsed_data_train = parse_data(data_path + "train.tfrecord", metadata)
    parsed_data_test = parse_data(data_path + "test.tfrecord", metadata)
    parsed_data_valid = parse_data(data_path + "valid.tfrecord", metadata)

    positions_train = get_positions(parsed_data_train)
    positions_test = get_positions(parsed_data_test)
    positions_valid = get_positions(parsed_data_valid)

    # take every-other position
    positions_train = [position[:, ::n, :] for position in positions_train]
    positions_test = [position[:, ::n, :] for position in positions_test]
    positions_valid = [position[:, ::n, :] for position in positions_valid]

    # update metadata statistics
    all_pos = combine_positions(positions_train)
    all_pos = combine_positions(positions_test, all_pos=all_pos)
    all_pos = combine_positions(positions_valid, all_pos=all_pos)

    velocity = torch.diff(all_pos, dim=1)
    vel_mean = torch.mean(velocity, dim=[0,1])
    vel_std = torch.std(velocity, dim=[0,1])

    acceleration = torch.diff(velocity, dim=1)
    acc_mean = torch.mean(acceleration, dim=[0,1])
    acc_std = torch.std(acceleration, dim=[0,1])

    # update dictionaries
    metadata['dt'] = metadata['dt'] * n
    metadata['vel_mean'] = vel_mean.tolist()
    metadata['vel_std'] = vel_std.tolist()
    metadata['acc_mean'] = acc_mean.tolist()
    metadata['acc_std'] = acc_std.tolist()

    metadata['sequence_length'] = positions_test[0].shape[1] - 1

    update_dictionary_position(parsed_data_test, positions_test)
    update_dictionary_position(parsed_data_train, positions_train)
    update_dictionary_position(parsed_data_valid, positions_valid)

    return metadata, parsed_data_train, parsed_data_test, parsed_data_valid