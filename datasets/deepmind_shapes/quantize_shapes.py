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

r"""Add labels to the Deepmind shapes dataset by quantizing the latent space.

The dataset has grayscale shapes which are all smoothly transformed
as per some latent state.

This code adds labels to the deepmind dataset by quantizing the continous
valued latent space into discrete bins.

Author: vrama@
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

from sstable import SSTableWriter

from partition_labels import PartitionLabels

app = tf.app
Example = tf.Example
flags = tf.flags
gfile = tf.gfile

_TRAIN_VAL_TEST_SPLIT = (0.85, 0.05, 0.10)

FLAGS = tf.app.flags.FLAGS

# Input/ Output Files.
flags.DEFINE_string('input_tfexample', '', 'Path to deepmind-concepts 2D shapes'
                    'tf example sstables.')
flags.DEFINE_string('output_train_tfexample', '', 'Path to deepmind-concepts'
                    '-labels 2D shapes train tf example sstables.')
flags.DEFINE_string('output_val_tfexample', '', 'Path to deepmind-concepts'
                    '-labels 2D shapes val tf example sstables.')
flags.DEFINE_string('output_test_tfexample', '', 'Path to deepmind-concepts'
                    '-labels 2D shapes test tf example sstables.')

# Label maps.
flags.DEFINE_string('label_split_json', '',
                    'Json file where label splits will be stored.')
flags.DEFINE_string('label_map_json', '', 'Text file with label maps')

# Other options.
flags.DEFINE_boolean('semi_supervised', False, 'Semi-supervised data SSTables.')
flags.DEFINE_integer('random_seed', 42, 'Random seed for dataset creation.')
flags.DEFINE_boolean(
    'compositional_split', True, 'True: Split dataset into train val test'
    'ensuring no overlap in labels, False: Split dataset into train val'
    ' test IID.')


def write_tuples_to_sstable(tuple_list, sstable_path, name='train'):
  """Write out a list of key, value pairs to an SSTable."""
  logging.info('Writing %s sstable.', (name))
  builder = SSTableWriter(sstable_path)
  for key, value in tuple_list:
    builder.Add(key, value)
  builder.FinishTable()


def induce_labels_and_split(latents, partition_labels):
  """Given latent states from deepmind-concepts dataset, infer labels.

  Args:
    latents: [6] list, Continous latent space for a given tf Example.
    partition_labels: A list of tuples containing the labels/attributes. For
      example [(square,small,pi/2,bottom-right)].
  Returns:
    labels: [4] length list of int, Labels assigned based on the current data
      point. For example, [0, 0, 1, 2]
    label_names: [4] length list of str,
      Example: ["square", "big", "pi/2", "bottom-left"]
  """
  labels = []
  label_names = []
  assert len(latents) == 6, 'Check latents, wrong dimensions.'

  # Labels[1] corresponds to shape - 'square'(0), 'ellipse'(1), 'heart' (2).
  label_id = partition_labels.get_unary_labels(latents[1], 'shape')
  labels.append(label_id)
  label_name = partition_labels.label_map['shape'][label_id]
  label_names.append(label_name)

  # Labels[2] corresponds to scale - 'big' (0) 'small' (1).
  label_id = partition_labels.get_unary_labels(latents[2], 'scale')
  labels.append(label_id)
  label_name = partition_labels.label_map['scale'][label_id]
  label_names.append(label_name)

  # Labels[3] corresponds to orientation -
  # pi/4 (0), pi/2 (1), 3pi/4 (2), pi (3), 5pi/4 (4), 3pi/2 (5), 7pi/4 (6)
  # 2p (7).
  label_id = partition_labels.get_unary_labels(latents[3], 'orientation')
  labels.append(label_id)
  label_name = partition_labels.label_map['orientation'][label_id]
  label_names.append(label_name)

  # Lables[4] corresponds to position- top-left (0), top-right(1),
  # bottom-left(2), bottom-right(3).
  label_id = partition_labels.get_location_labels(latents[4], latents[5])
  labels.append(label_id)
  label_name = partition_labels.label_map['location'][label_id]
  label_names.append(label_name)

  return labels, label_names


def read_and_process_sstable(input_sstable, partition_labels, label_splits):
  """Reads a TF Example SSTable, induces labels, writes output TF example.

  Given as input an sstable in the TF example format, along with some latents
  which describe the data generation process, this code takes as input some
  options to discretize the latent space, and based on that labels items from
  input TF example. Then, it separates the data into train, val, and test
  splits based on specifications.

  Args:
    input_sstable: string, input sstable with deepmind 2D shapes data.
    partition_labels: An instance of PartitionLabels class.
    label_splits: Dict, with key as the split of the dataset (train/val/test)
      and values a list of tuples containing the labels in that split. For
      example {'train':[(square,small,pi/2,bottom-right)], 'val':[(heart,big,
      pi/2,bottom-left)], 'test':[(square, big, pi/2, bottom-left)]}
  Raises:
    RuntimeError: If the label in the dataset is not in train, val or test.
  """
  sstable_filelist = tf.gfile.GenerateShardedFilenames(input_sstable)
  sst = sstable.MergedSSTable(sstable_filelist)
  logging.info('Reading from SStable...')

  iter_index = 0
  train_items = []
  val_items = []
  test_items = []

  if not FLAGS.compositional_split:
    cumulative_train_val_test_split = np.array(_TRAIN_VAL_TEST_SPLIT)
    cumulative_train_val_test_split = np.cumsum(cumulative_train_val_test_split)

  for key, value in sst.iteritems():
    value_example = Example.FromString(value)
    latent = value_example.features.feature['latents'].float_list.value
    labels, label_names = induce_labels_and_split(latent, partition_labels)
    new_feature = value_example.features.feature
    new_feature['labels'].int64_list.value.extend(labels)

    label_names = tuple(label_names)
    # Check which split of data the label_names belong to. For example, if
    # label_names = ["square", "big", "pi/2", "bottom-left"], then we look
    # it up in a list of all label names for each split to find the data split
    # that the data point should belong to, and then we store it to the
    # corresponding file.
    if FLAGS.compositional_split is True:
      if label_names in label_splits['train']:
        train_items.append((key, value_example.SerializeToString()))
      elif label_names in label_splits['val']:
        val_items.append((key, value_example.SerializeToString()))
      elif label_names in label_splits['test']:
        test_items.append((key, value_example.SerializeToString()))
      else:
        raise RuntimeError
    else:
      # This is the case where train/val/test sets are all built IID.
      choose_dp = np.random.uniform(0, 1)
      if choose_dp <= cumulative_train_val_test_split[0]:
        train_items.append((key, value_example.SerializeToString()))
      if choose_dp > cumulative_train_val_test_split[
          0] and choose_dp <= cumulative_train_val_test_split[1]:
        val_items.append((key, value_example.SerializeToString()))
      if choose_dp > cumulative_train_val_test_split[
          1] and choose_dp <= cumulative_train_val_test_split[2]:
        test_items.append((key, value_example.SerializeToString()))

    iter_index += 1
    if iter_index % 1000 == 0:
      logging.info('Finished %d examples', (iter_index))

  logging.info('There are %d train, %d val, %d test instances.',
               len(train_items), len(val_items), len(test_items))

  write_tuples_to_sstable(
      train_items, FLAGS.output_train_tfexample, name='train')
  write_tuples_to_sstable(val_items, FLAGS.output_val_tfexample, name='val')
  write_tuples_to_sstable(test_items, FLAGS.output_test_tfexample, name='test')


def main(unused_argv):
  assert FLAGS.input_tfexample is not None, 'Missing input_tfexample path.'
  assert FLAGS.output_train_tfexample is not None, ('Missing '
                                                    'output_train_tfexample '
                                                    'path.')
  assert FLAGS.output_val_tfexample is not None, ('Missing output_val_tfexample'
                                                  ' path.')
  assert FLAGS.output_test_tfexample is not None, (
      'Missing '
      'output_test_tfexample path.')
  assert FLAGS.label_split_json is not None, 'Missing label_split_json path.'
  assert FLAGS.label_map_json is not None, 'Missing label_map_json path.'

  if not FLAGS.compositional_split:
    np.random.seed(FLAGS.random_seed)

  partition_labels = PartitionLabels(
      FLAGS.label_split_json,
      FLAGS.label_map_json,
      random_seed=FLAGS.random_seed)
  label_splits = partition_labels.split_train_val_test(_TRAIN_VAL_TEST_SPLIT)

  read_and_process_sstable(FLAGS.input_tfexample, partition_labels,
                           label_splits)


if __name__ == '__main__':
  tf.app.run()
