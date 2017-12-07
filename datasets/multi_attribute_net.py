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

"""Contains the definition of a multi-attribute classification network.

The model in this file is a simple convolutional network with two
convolutional layers, two pooling layers, followed by two fully connected
layers. A single dropout layer is used between the two fully connected layers.
The network has "n" heads after the convolutional part, where "n" is the number
of attributes that we want to predict. Each attribute prediction problem is
treated as a supervised multi-class classification problem.

Author: vrama@
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim



def get_tower_name(layer, tower_id):
  return layer + '_' + tower_id


def conv_multi_attribute_net(images,
                             num_classes_per_attribute=None,
                             attribute_names=None,
                             hidden_units=1024,
                             is_training=False):
  """Creates the convolutional Deepmind shapes to labels model.

  Args:
    images: The images, a tensor of size [batch_size, image_size, image_size,
        num_channels]
    num_classes_per_attribute: list, indicates the number of classes for each
      attribute in the dataset.
    attribute_names: list of str, provides the name of each attribute for
      naming layers in each tower.
    hidden_units: number of units in the first FC layer.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.

  Returns:
    the output logits_list, a list of `Tensor` where each `Tensor` entry 'i' is
    [batch_size, NUM_CLASSES[i]]
  """
  if attribute_names is None:
    attribute_names = [str(x) for x in xrange(len(num_classes_per_attribute))]
  else:
    # Avoid passing invalid attribute names as scope
    attribute_names = [
        re.sub('[^A-Za-z0-9]', '_', str(x)) for x in attribute_names
    ]

  # Adds a convolutional layer with 32 filters of size [5x5], followed by
  # the default (implicit) Relu activation.
  net = slim.conv2d(images, 32, [5, 5], padding='SAME', scope='conv1')

  # Adds a [2x2] pooling layer with a stride of 2.
  net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')

  # Adds a convolutional layer with 64 filters of size [5x5], followed by
  # the default (implicit) Relu activation.
  net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='conv2')

  # Adds a [2x2] pooling layer with a stride of 2.
  net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')

  # Reshapes the hidden units such that instead of 2D maps, they are 1D vectors:
  base_net = slim.flatten(net)

  # Add a for loop over the number of classes, to build four towers for
  # predicting each modality.
  logits = []
  for output_tower_id, num_classes in enumerate(num_classes_per_attribute):
    # Adds a fully-connected layer with 1024 hidden units,
    # followed by the default Relu activation.
    net = slim.fully_connected(
        base_net,
        hidden_units,
        scope=get_tower_name('fc3', attribute_names[output_tower_id]))

    # Adds a dropout layer during training.
    net = slim.dropout(
        net,
        0.5,
        is_training=is_training,
        scope=get_tower_name('dropout3', attribute_names[output_tower_id]))

    # Adds a fully connected layer with 'num_classes' outputs. Note
    # that the default Relu activation has been overridden to use no activation.
    net = slim.fully_connected(
        net,
        num_classes,
        activation_fn=None,
        scope=get_tower_name('fc4', attribute_names[output_tower_id]))
    logits.append(net)

  print(np.sum([np.prod(v.get_shape())
             for v in tf.trainable_variables()]))
  print('Total number of parameters')

  return logits


def mlp_multi_attribute_net(images,
                            num_classes_per_attribute=None,
                            attribute_names=None,
                            hidden_units=1024,
                            is_training=False):
  """Creates the convolutional CUB IRv2 features to labels model.

  Args:
    images: The images, a tensor of size [batch_size, image_size, image_size,
        num_channels]
    num_classes_per_attribute: list, indicates the number of classes for each
      attribute in the dataset.
    attribute_names: list of str, provides the name of each attribute for
      naming layers in each tower.
    hidden_units: number of units in the first FC layer.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.

  Returns:
    the output logits_list, a list of `Tensor` where each `Tensor` entry 'i' is
    [batch_size, NUM_CLASSES[i]]
  """
  if attribute_names is None:
    attribute_names = [str(x) for x in xrange(len(num_classes_per_attribute))]
  else:
    # Avoid passing invalid attribute names as scope
    attribute_names = [
        re.sub('[^A-Za-z0-9]', '_', str(x)) for x in attribute_names
    ]

  # Make sure the inputs are flat, just in case.
  net = slim.flatten(images)

  # Process with some shared parameters.
  net = slim.fully_connected(net, hidden_units, scope='fc1')

  net = slim.fully_connected(net, hidden_units, scope='fc2')

  # Add a for loop over the number of classes, to build a tower for predicting
  # each modality.
  logits = []
  for output_tower_id, num_classes in enumerate(num_classes_per_attribute):
    net = slim.fully_connected(
        net,
        hidden_units // 2,
        scope=get_tower_name('fc3', attribute_names[output_tower_id]))

    # Adds a dropout layer during training.
    net = slim.dropout(
        net,
        0.5,
        is_training=is_training,
        scope=get_tower_name('dropout3', attribute_names[output_tower_id]))

    # Adds a fully connected layer with 'num_classes' outputs. Note
    # that the default Relu activation has been overridden to use no activation.
    net = slim.fully_connected(
        net,
        num_classes,
        activation_fn=None,
        scope=get_tower_name('fc4', attribute_names[output_tower_id]))
    logits.append(net)

  print(np.sum([np.prod(v.get_shape())
             for v in tf.trainable_variables()]),
     'Total number of parameters')

  return logits
