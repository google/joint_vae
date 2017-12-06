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

r"""Provides data for the mnist with attributes dataset.

Provide data loading utilities for an augmented version of the
MNIST dataset which contains the following attributes:
  1. Location (digits are translated on a canvas and placed around
    one of four locations/regions in the canvas). Each location
    is a gaussian placed at four quadrants of the canvas.
  2. Scale (We vary scale from 0.4 to 1.0), with two gaussians
    placed at 0.5 +- 0.1 and 0.9 +- 0.1 repsectively.
  3. Orientation: we vary orientation from -90 to +90 degrees,
    sampling actual values from gaussians at +30 +- 10 and
    -30 +-10. On a third of the occasions we dont orient the
    digit at all which means a rotation of 0 degrees.

The original data after transformations is binarized as per the
procedure described in the following paper:

  Salakhutdinov, Ruslan, and Iain Murray. 2008. ``On the Quantitative Analysis of
  Deep Belief Networks.'' In Proceedings of the 25th International Conference on
Author: vrama@
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.contrib.slim.python.slim.data import dataset
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

# Only provides option to load the binarized version of the dataset.
_FILE_PATTERN = 'binarized_True_replication_1_retrieval_at_1.tf_record'

# TODO(vrama): Update
_DATASET_DIR = (
    '/nethome/rvedantam3/data/mnista/'
)

_SPLIT_TYPE = 'retrieval'

_SPLITS_TO_SIZES = {'retrieval': 70000}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [64 x 64 x 1] grayscale image.',
    'labels': 'Labels induced on latent states used to generate the image.',
    'latents': 'Latents used to generate the image.'
}

# There are four labels in the dataset corresponding to
# shape, size, orientation, and location of the object.
# 10 classes for digits (0-9),
# 2 classes for size (big, small)
# 3 classes for orientation (clockwise, center, counterclockwise)
# 4 classes for location (top-left, top-right, bottom-left, bottom-right)
_NUM_CLASSES_PER_ATTRIBUTE = (10, 2, 3, 4)

_NUM_LATENTS = 5


def get_split(split_name='retrieval',
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

  file_pattern = os.path.join(dataset_dir, _FILE_PATTERN)

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

  decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                               items_to_handlers)

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
