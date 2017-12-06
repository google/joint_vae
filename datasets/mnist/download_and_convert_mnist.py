# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts MNIST data to TFRecords of TF-Example protos.

This module downloads the MNIST data, uncompresses it, reads the files
that make up the MNIST data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys

import numpy as np
from six.moves import urllib
import tensorflow as tf

from datasets.mnist import dataset_utils

tf.app.flags.DEFINE_string('dataset_dir', '/tmp/mnist', 'Location where we want'
                           'to store the MNIST dataset.')

tf.app.flags.DEFINE_boolean('create_validation', False, 'Create a validation'
                            ' split of the dataset from train images.')


FLAGS = tf.app.flags.FLAGS

# The URLs where the MNIST data can be downloaded.
_DATA_URL = 'http://yann.lecun.com/exdb/mnist/'
_TRAIN_DATA_FILENAME = 'train-images-idx3-ubyte.gz'
_TRAIN_LABELS_FILENAME = 'train-labels-idx1-ubyte.gz'
_TEST_DATA_FILENAME = 't10k-images-idx3-ubyte.gz'
_TEST_LABELS_FILENAME = 't10k-labels-idx1-ubyte.gz'

_IMAGE_SIZE = 28
_NUM_CHANNELS = 1

_VALID_OUTPUT_SPLITS = {
    'train_small': (0, 50000),
    'valid': (50000, 60000),
    'test': (60000, 70000),
}

_OUTPUT_SPLITS = {
    'train': (0, 60000),
    'test': (60000, 70000),
}

# The names of the classes.
_CLASS_NAMES = [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'size',
    'seven',
    'eight',
    'nine',
]


def _extract_images(filename, num_images):
  """Extract the images into a numpy array.

  Args:
    filename: The path to an MNIST images file.
    num_images: The number of images in the file.

  Returns:
    A numpy array of shape [number_of_images, height, width, channels].
  """
  print('Extracting images from: ', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(
        _IMAGE_SIZE * _IMAGE_SIZE * num_images * _NUM_CHANNELS)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
  return data


def _extract_labels(filename, num_labels):
  """Extract the labels into a vector of int64 label IDs.

  Args:
    filename: The path to an MNIST labels file.
    num_labels: The number of labels in the file.

  Returns:
    A numpy array of shape [number_of_labels]
  """
  print('Extracting labels from: ', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_labels)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels


def _extract_images_and_labels(data_filename, labels_filename, num_datapoints):
  """Loads data from binary MNIST files and returns values.

  Args:
    data_filename: The filename of the MNIST images.
    labels_filename: The filename of the MNIST labels.
  Returns:
    images: np.array of images [N, 28, 28, 1]
    labels: np.array of labels [N]
  """
  images = _extract_images(data_filename, num_datapoints)
  labels = _extract_labels(labels_filename, num_datapoints)

  return images, labels


def _add_to_tfrecord(images, labels, tfrecord_writer, end_idx, start_idx=0):
  """Loads data from the binary MNIST files and writes files to a TFRecord.

  Args:

    images: np.array of images [N, 28, 28, 1]
    labels: np.array of labels [N]
    tfrecord_writer: The TFRecord writer to use for writing.
    end_idx: The last index to read from images array. (0-indexed)
    start_idx: The start index to read from images array. (0-indexed)
  """
  shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
  with tf.Graph().as_default():
    image = tf.placeholder(dtype=tf.uint8, shape=shape)
    encoded_png = tf.image.encode_png(image)

    with tf.Session('') as sess:
      for j in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, end_idx))
        sys.stdout.flush()

        png_string = sess.run(encoded_png, feed_dict={image: images[j]})

        example = dataset_utils.image_to_tfexample(
            png_string, 'png'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, labels[j])
        tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(dataset_dir, split_name):
  """Creates the output filename.

  Args:
    dataset_dir: The directory where the temporary files are stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  return '%s/mnist_%s.tfrecord' % (dataset_dir, split_name)


def _download_dataset(dataset_dir):
  """Downloads MNIST locally.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  for filename in [_TRAIN_DATA_FILENAME,
                   _TRAIN_LABELS_FILENAME,
                   _TEST_DATA_FILENAME,
                   _TEST_LABELS_FILENAME]:
    filepath = os.path.join(dataset_dir, filename)

    if not os.path.exists(filepath):
      print('Downloading file %s...' % filename)
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %.1f%%' % (
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(_DATA_URL + filename,
                                               filepath,
                                               _progress)
      print()
      with tf.gfile.GFile(filepath) as f:
        size = f.size()
      print('Successfully downloaded', filename, size, 'bytes.')


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  for filename in [_TRAIN_DATA_FILENAME,
                   _TRAIN_LABELS_FILENAME,
                   _TEST_DATA_FILENAME,
                   _TEST_LABELS_FILENAME]:
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)


def main(unused_argv):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(FLAGS.dataset_dir):
    tf.gfile.MakeDirs(FLAGS.dataset_dir)

  if FLAGS.create_validation:
    output_splits = _VALID_OUTPUT_SPLITS
  else:
    output_splits = _OUTPUT_SPLITS

  _download_dataset(FLAGS.dataset_dir)

  # First, load the training data:
  data_filename = os.path.join(FLAGS.dataset_dir, _TRAIN_DATA_FILENAME)
  labels_filename = os.path.join(FLAGS.dataset_dir, _TRAIN_LABELS_FILENAME)
  train_images, train_labels = _extract_images_and_labels(
      data_filename, labels_filename, 60000)

  # Next, laod the testing data:
  data_filename = os.path.join(FLAGS.dataset_dir, _TEST_DATA_FILENAME)
  labels_filename = os.path.join(FLAGS.dataset_dir, _TEST_LABELS_FILENAME)
  test_images, test_labels = _extract_images_and_labels(
      data_filename, labels_filename, 10000)

  all_images = np.concatenate((train_images, test_images), axis=0)
  all_labels = np.concatenate((train_labels, test_labels), axis=0)

  for split_name, indices in output_splits.iteritems():
    write_filename = _get_output_filename(FLAGS.dataset_dir, split_name)

    assert len(indices) == 2, "Indices has the wrong shape"

    if tf.gfile.Exists(write_filename):
      print('Dataset files already exist. Exiting without re-creating them.')
      continue

    with tf.python_io.TFRecordWriter(write_filename) as tfrecord_writer:
      _add_to_tfrecord(all_images,
                       all_labels,
                       tfrecord_writer,
                       indices[-1],
                       indices[0])

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
  dataset_utils.write_label_file(labels_to_class_names, FLAGS.dataset_dir)

  _clean_up_temporary_files(FLAGS.dataset_dir)
  print('\nFinished converting the MNIST dataset!')

if __name__ == "__main__":
    tf.app.run()
