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

r"""Provides data for the deepmind shapes with labels dataset.

Provide data loading utilities for an augmented version of the shapes
datasets used in the following paper:

 Beta-VAE: Learning Basic Visual Concepts with a Constrained
 Variational Framework
   Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess,
   Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner

  ICLR 2017

This data loader loads an augmented version of the above dataset where
additional label field specifies a discretization of the continous
latent space. The Train/Val/Test splits are done to study compositional
generalization. That is, there is no intersection between
train val test in terms of the label-sets present in them.

Author: vrama@
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import dataset
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

_FILE_PATTERN = 'deepmind_shapes_labels_gray_sq_el_ht_tf_example_%s-*'

_DATASET_DIR = ('/placer/prod/home/vale-project-placer/datasets/deepmind_shapes_with_labels')

_SPLIT_TYPE = 'comp'

_SPLITS_TO_SIZES = {
    'train': 625920,
    'val': 34560,
    'test': 76800,
}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [64 x 64 x 1] grayscale image.',
    'labels': 'Labels induced on latent states used to generate the image.',
    'latents': 'Latents used to generate the image.'
}

# There are four labels in the dataset corresponding to
# shape, size, orientation, and location of the object.
# 3 classes for shape (square, ellipse, heart),
# 2 classes for size (big, small)
# 8 classes for orientation(\pi/4, \pi/2, 3\pi/4, \pi, 5\pi/4, 3\pi/2, 7\pi/4,
# 2\pi)
# 4 classes for location (top-left, top-right, bottom-left, bottom-right)
_NUM_CLASSES_PER_ATTRIBUTE = (3, 2, 8, 4)

_NUM_LATENTS = 6

def get_split(split_name='train',
              dataset_dir=None,
              num_classes_per_attribute=None):
  """Gets a dataset tuple with instructions for reading 2D shapes data.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    num_classes_per_attribute: The number of labels for the classfication
      problem corresponding to each attribute. For example, if the first
      attribute is "shape" and there are three possible shapes, then
      then provide a value 3 in the first index, and so on.

  Returns:
    A `Dataset` namedtuple.
    metadata: A dictionary with some metadata about the dataset we just
      constructed.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if num_classes_per_attribute is None:
    num_classes_per_attribute = _NUM_CLASSES_PER_ATTRIBUTE

  if dataset_dir is None:
    dataset_dir = _DATASET_DIR

  file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % split_name)

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'labels': tf.FixedLenFeature([len(num_classes_per_attribute)], tf.int64),
      'latents': tf.FixedLenFeature([_NUM_LATENTS], tf.float32),
  }

  items_to_handlers = {
      'image': tfexample_decoder.Image(shape=[64, 64, 3]),
      'labels': tfexample_decoder.Tensor('labels'),
      'latents': tfexample_decoder.Tensor('latents'),
  }

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  metadata = {
      'num_classes_per_attribute': num_classes_per_attribute,
      'split_type': _SPLIT_TYPE
  }

  return dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS), metadata
