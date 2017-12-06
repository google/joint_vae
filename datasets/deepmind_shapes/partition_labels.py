#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert continous deepmind-concepts latents to discrete labels."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from itertools import product
import json
import math
import random

import numpy as np
import tensorflow as tf

gfile = tf.gfile


class PartitionLabels(object):
  """Define a discretization scheme from continous latents to labels."""

  def __init__(self, label_split_file, label_map_file, random_seed=42):
    """Initialization.

    Args:
      label_split_file: string, A JSON file path where the label
        combinations from training, val and test will be stored.
      label_map_file: Stores a mapping from label id to label name.
      random_seed: Random seed for partitioning labels.
    """
    # Random seed for repeatable train, val, test set creation.
    self._random_seed = random_seed
    random.seed(self._random_seed)

    self.label_split_file = label_split_file
    self.label_map_file = label_map_file

    self._setup_labels()
    self._setup_labelmap()

  def _setup_labels(self):
    """Define the labels to induce on the dataset, and rules to identify."""
    # Unary labels only depend on one other dimension in latent space.
    self.unary_labels = ('shape', 'scale', 'orientation')
    # Multi labels depend on multple dimensions of latent space.
    self.multi_labels = tuple('location')

    self.labels = self.unary_labels + self.multi_labels

    self.state_space = OrderedDict()
    self.state_space['shape'] = ('square', 'ellipse', 'heart')
    self.state_space['scale'] = ('big', 'small')
    self.state_space['orientation'] = ('pi/4', 'pi/2', '3pi/4', 'pi', '5pi/4',
                                       '3pi/2', '7pi/4', '2pi')
    self.state_space['location'] = ('top-left', 'top-right', 'bottom-left',
                                    'bottom-right')
    # Setup rules to map latents to labels.
    self._setup_membership_functions()

  def _setup_labelmap(self):
    """Generate mapping from item in a state space to its label id."""
    self.label_map = OrderedDict()

    for label_type in self.state_space:
      self.label_map[label_type] = {
          index: v
          for index, v in enumerate(self.state_space[label_type])
      }

  def _setup_membership_functions(self):
    """Setup membership functions to decide label for latent."""
    self.membership_functions = {}
    # Shape.
    self.membership_functions['square'] = lambda x: x == 1.0
    self.membership_functions['ellipse'] = lambda x: x == 2.0
    self.membership_functions['heart'] = lambda x: x == 3.0

    # Scale.
    self.membership_functions['big'] = lambda x: x > 0.7
    self.membership_functions['small'] = lambda x: x <= 0.7

    # Orientation.
    self.membership_functions['pi/4'] = lambda x: x >= 0.0 and x <= np.pi / 4
    self.membership_functions['pi/2'] = (
        lambda x: x > np.pi / 4 and x <= np.pi / 2)
    self.membership_functions['3pi/4'] = (
        lambda x: x > np.pi / 2 and x <= 3 * np.pi / 4)
    self.membership_functions['pi'] = (
        lambda x: x > 3 * np.pi / 4 and x <= np.pi)
    self.membership_functions['5pi/4'] = (
        lambda x: x > np.pi and x <= 5 * np.pi / 4)
    self.membership_functions['3pi/2'] = (
        lambda x: x > 5 * np.pi / 4 and x <= 3 * np.pi / 2)
    self.membership_functions['7pi/4'] = (
        lambda x: x > 3 * np.pi / 2 and x <= 7 * np.pi / 4)
    self.membership_functions['2pi'] = (
        lambda x: x > 7 * np.pi / 4 and x <= 2 * np.pi + 1e-3)

    # Location.
    self.membership_functions['top-left'] = lambda x, y: x <= 0.5 and y <= 0.5
    self.membership_functions['top-right'] = lambda x, y: x > 0.5 and y <= 0.5
    self.membership_functions['bottom-right'] = lambda x, y: x > 0.5 and y > 0.5
    self.membership_functions['bottom-left'] = lambda x, y: x <= 0.5 and y > 0.5

  def get_unary_labels(self, latent, label_type='shape'):
    """Get class labels which only depend on one latent dimension.

    Other classes with more complicated dependencies are handled in separate
    functions for each label.

    Args:
      latent: list of float with 1 element
      label_type: one among 'shape', 'scale', 'orientation'
    Returns:
      index: class label assigned based on the data point.
    Raises:
      ValueError: if invalid label type (say non-unary) is provided.
      RuntimeError: if the continous latent value could not be categorized
        using the defined rules.
    """
    if label_type not in self.unary_labels:
      raise ValueError('Invalid label_type. Only shape, scale,'
                       'orientation supported.')

    for index, item in enumerate(self.state_space[label_type]):
      if all(map(self.membership_functions[item], [latent])) is True:
        return index
    raise RuntimeError('Could not assign a class to the data point %f' % latent)

  def get_location_labels(self, posx, posy):
    """Define a mapping from latents to location variables.

    For non-unary labels like location, define a separate function implementing
    the rule for classifying multiple continous latents into a single label.

    Args:
      posx: float, x-coordinate of a shape
      posy: float, y-coordinate of a shape
    Returns:
      index: Assinged class label based on x, y coordinates
    Raises:
      RuntimeError: if the continous latent value could not be categorized
        using the defined rules.
    """
    for index, item in enumerate(self.state_space['location']):
      if all(map(self.membership_functions[item], [posx], [posy])) is True:
        return index
    raise RuntimeError('Could not assign a class to the data point.')

  def get_all_label_combs(self):
    """Compute the exhaustive state space across all labels.

    Computes the set of all label combinations in the dataset, and shuffles
    them.

    Returns:
      all_label_combinations: list of tuples, The set of all label combinations
        in the dataset.
    """
    all_label_combinations = []
    for labels in product(self.state_space['shape'], self.state_space['scale'],
                          self.state_space['orientation'],
                          self.state_space['location']):
      all_label_combinations.append(labels)

    random.shuffle(all_label_combinations)

    return all_label_combinations

  def generate_partitions(self, mylist, train_val_test_split):
    """Retrieve data after splitting the dataset into train, val, test.

    Args:
      mylist: list, Input list with items to split.
      train_val_test_split: Tuple, with a split of train, val and test. Sum
        of entries in the tuple must be 1.

    Returns:
      split_name_to_attributes: Dict, with key as the split name (train/val/
        test) and value a list of tuples containing label combinations in
        the corresponding split.
    """
    assert sum(train_val_test_split) == 1, 'Invalid train val test split.'
    rounded_train_val_indices = [0]

    train_val_test_split = np.cumsum(np.array(train_val_test_split))
    split_names = ['train', 'val', 'test']

    for item in train_val_test_split:
      rounded_train_val_indices.append(int(math.floor(item * len(mylist))))

    split_name_to_attributes = {}
    for index, item in enumerate(rounded_train_val_indices):
      if index == len(rounded_train_val_indices) - 1:
        break
      split_name_to_attributes[split_names[index]] = mylist[
          item:rounded_train_val_indices[index + 1]]
    return split_name_to_attributes

  def split_train_val_test(self, train_val_test_split=(.85, .05, .10)):
    """Split label combinations into train/ val and test.

    Args:
      train_val_test_split: Tuple, with a split of train, val and test. Sum
        of entries in the tuple must be 1.
    Returns:
      label_splits: Dict, with key as the split of the dataset (train/val/test)
        and values a list of tuples containing the labels in that split. For
        example {'train':[(square,small,pi/2,bottom-right)], 'val':[(heart,big,
        pi/2,bottom-left)], 'test':[(square, big, pi/2, bottom-left)]}.
    """
    all_label_combinations = self.get_all_label_combs()

    label_splits = self.generate_partitions(all_label_combinations,
                                            train_val_test_split)
    self.serialize_label_splits(label_splits)

    return label_splits

  def serialize_label_splits(self, label_splits):
    """Utilities for dumping label map and split info to JSON."""
    serialized = json.dumps(label_splits)
    with tf.gfile.Open(self.label_split_file, 'w') as f:
      f.write(serialized)

    # Write label map as a tuple of key value pairs to maintain order.
    label_map_list = [(k, v) for k, v in self.label_map.iteritems()]

    serialized = json.dumps(label_map_list)
    with tf.gfile.Open(self.label_map_file, 'w') as f:
      f.write(serialized)
