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
  Machine Learning, 872-79.

Author: vrama@
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.contrib.slim.python.slim.data import dataset
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder
from datasets.celeba.image_decoder import ImageDecodeProcess

# Only provides option to load the binarized version of the dataset.
_FILE_PATTERN = '%s-*'

_SPLIT_TYPE = 'iid'

_DATASET_DIR = '/srv/share/datasets/celeba_for_tf_ig'

_SPLITS_TO_SIZES = {'train': 162770, 'val': 19867, 'test': 19962}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [218 x 178 x 3] RGB image.',
    'labels': 'Attributes corresponding to the image.',
}

_NUM_CLASSES_PER_ATTRIBUTE = tuple([2]*18)


def get_split(split_name='train',
              split_type="iid",
              dataset_dir=None,
              image_length=64,
              num_classes_per_attribute=None):
  """Gets a dataset tuple with instructions for reading 2D shapes data.

  Args:
    split_name: A train/test split name.
    split_type: str, type of split being loaded "iid" or "comp"
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

  if split_type is not "iid":
    raise ValueError("Only IID split available for CelebA.")

  if num_classes_per_attribute is None:
    num_classes_per_attribute = _NUM_CLASSES_PER_ATTRIBUTE

  if dataset_dir is None or dataset_dir == '':
    dataset_dir = _DATASET_DIR

  # Load attribute label map file.
  label_map_json = os.path.join(dataset_dir,
                                          'attribute_label_map.json')

  file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % split_name)
  tf.logging.info('Loading from %s file.' % (file_pattern))

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'image/labels': tf.FixedLenFeature([len(num_classes_per_attribute)], tf.int64),
  }
  # TODO(vrama): See
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/data/tfexample_decoder.py#L270
  # For where changes would need to be made to preprocess the images which
  # get loaded.

  items_to_handlers = {
      'image': ImageDecodeProcess(shape=[218, 178, 3], image_length=64),
      'labels': tfexample_decoder.Tensor('image/labels'),
  }

  decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                               items_to_handlers)

  metadata = {
      'num_classes_per_attribute': num_classes_per_attribute,
      'split_type': _SPLIT_TYPE,
      'label_map_json': label_map_json,
  }

  return dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS), metadata
