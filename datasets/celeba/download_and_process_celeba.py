#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright  2017 rvedantam3 <vrama91@vt.edu>
#
# Distributed under terms of the MIT license.
"""
Script to download and process the celebrities with attributes (CelebA) dataset.
The celebrities with attributes dataset is from the following paper.

Liu, Z., P. Luo, X. Wang, and X. Tang. 2015.
Deep Learning Face Attributes in the Wild.
Proceedings of the IEEE. cv-foundation.org.
http://www.cv-foundation.org/openaccess/content_iccv_2015/html/\
    Liu_Deep_Learning_Face_ICCV_2015_paper.html

Once downloaded we would like to put the dataset into TFRecords so that the data
can be consumed by tensorflow code.

A lot of the choices for how to preprocess the Celeb-A dataset are taken from
the following codebase:
    https://github.com/andersbll/autoencoding_beyond_pixels

Specifically we crop out a 148x148 patch from the image and then resize it to
64x64 patch.
"""
from collections import namedtuple
from datetime import datetime
import json
import numpy as np
import os
import sys
from six.moves import urllib
import tensorflow as tf
import threading
import zipfile

from scipy.io import savemat
from utils.data_utils import ImageCoder
from utils.data_utils import _process_image
from utils.data_utils import _int64_feature
from utils.data_utils import _bytes_feature

tf.app.flags.DEFINE_string('dataset_dir', '/tmp/celeba', 'Destination for the'
                           ' Celeb A dataset.')

tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory.')

tf.app.flags.DEFINE_string(
    'attribute_label_map', '/tmp/celeba/attribute_label_map.json',
    'Where to store the attribute label map for the file.')

tf.app.flags.DEFINE_string('attribute_subset_list', 'attribute_subset_icgan.txt',
                           'Subset of attributes to pick for modeling.')

tf.app.flags.DEFINE_string('split_name', 'val', 'Name of the split to process.')

tf.app.flags.DEFINE_integer('shards', 100,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 10,
                            'Number of threads to preprocess the images.')


FLAGS = tf.app.flags.FLAGS

attributeAndLabelMap = namedtuple('attribute_and_label_map', ['attribute', 'label_map'])

_ALIGNED_IMGS_URL = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1'

_PARTITIONS_URL = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADxLE5t6HqyD8sQCmzWJRcHa/Eval/list_eval_partition.txt?dl=1'

_ATTRIBUTES_URL = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAC7-uCaJkmPmvLX2_P5qy0ga/Anno/list_attr_celeba.txt?dl=1'

_SPLIT_TO_ID = {
    "train": 0,
    "val": 1,
    "test": 2,
}


def download_and_unzip_celeba():
  """Downloads the celeba dataset and unzips it, and returns paths to files.

  Returns:
    data_to_path: A dict with keys "images", "partitions", "attributes" and
      values the corresponding paths.
  """
  file_list = ("images", "partitions", "attributes")
  data_to_path = {}

  for url, file_item in zip(
      [_ALIGNED_IMGS_URL, _PARTITIONS_URL, _ATTRIBUTES_URL], file_list):
    filename = url.split('?')[0].split('/')[-1]
    filepath = os.path.join(FLAGS.dataset_dir, filename)

    print('Downloading file %s' % filename)
    print(filepath)

    if not tf.gfile.Exists(filepath):

      def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %.1f%%' %
            (float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
      if '.zip' in filename:
        print('Extracting..')
        with zipfile.ZipFile(filepath, 'r') as f:
          f.extractall(FLAGS.dataset_dir)

    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded and extracted %s, size %s bytes.' %
          (filename, size))

    data_to_path[file_item] = filepath

  return data_to_path


def _original_attribute_names_to_labelmap(attributes):
  """Converts binary attribute names into a general label map format.

  The label map format basically lists out each attribute along with possible
  states. For CelebA since there is only presence or absence of attributes
  we keep things as attribute name followed by "absent" or "present" as labels.

  Args:
    attributes: a list of str

  Returns:
    label_map: a list of tuples of str, and dict.
  """
  label_map = list()
  for index, attribute_name in enumerate(attributes):
    label_map.append(attributeAndLabelMap(attribute_name, {0: "absent", 1: "present"}))
  return label_map


def _convert_to_example(filename, image_buffer, label, height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height':
      _int64_feature(height),
      'image/width':
      _int64_feature(width),
      'image/colorspace':
      _bytes_feature(colorspace.encode()),
      'image/channels':
      _int64_feature(channels),
      'image/labels':
      _int64_feature(label),
      'image/format':
      _bytes_feature(image_format.encode()),
      'image/filename':
      _bytes_feature(os.path.basename(filename).encode()),
      'image/encoded':
      _bytes_feature(image_buffer)
  }))
  return example


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               labels, num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).

  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]

      image_buffer, height, width = _process_image(filename, coder)
      example = _convert_to_example(filename, image_buffer, label, height,
                                    width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_shards))


def _process_image_files(name, filenames, labels, num_shards):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames, labels, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' % (datetime.now(),
                                                             len(filenames)))
  sys.stdout.flush()


def _select_images_and_attributes_in_split(
    split_name, split_file, attribute_file, attribute_label_map_file):
  """Given a split file figure out the subset of images in each split.

  More specifically, the split file contains the keys in the following format:
      image0 0
      image1 0
      image2 1
      image3 2
  where 0, 1, 2 etc. refers to the split of the dataset which we are loading.

  Also load the attribute vectors corresponding to each file. The attribute
  file has the following format:
      #num_items
      attribute_name_1 attribute_name_2 attribute_name_3
      image0 -1 1 -1
      image1 1 -1 -1
      .
      .

  Args:
    split_name: string, "train", "val" or "test"
    split_file: string, name of the file containing the split of the dataset
      into train val test.
  Returns:

  """
  with tf.gfile.Open(split_file, 'r') as f:
    filelist = [x.rstrip() for x in f.readlines()]
    filelist = [(x.split(' ')[0], int(x.split(' ')[-1])) for x in filelist]

  images_in_split = [
      x[0] for x in filelist if x[-1] == _SPLIT_TO_ID[split_name]
  ]

  # Also load the subset of attributes for items in the split.
  with tf.gfile.Open(attribute_file, 'r') as f:
    file_lines = [x.strip() for x in f.readlines()]

  # First line is the number of items in the file
  num_items = int(file_lines[0])
  original_attribute_names = file_lines[1].split(' ')
  attribute_label_map = _original_attribute_names_to_labelmap(original_attribute_names)


  # Optionally load a subset of attributes which actually get used.
  if FLAGS.attribute_subset_list != '':
    with tf.gfile.Open(FLAGS.attribute_subset_list, 'r') as f:
      all_lines  = f.readlines()
    num_attributes_subset = int(all_lines[0])
    attribute_names_in_subset = [x.rstrip() for x in all_lines[1:]]
    attribute_subset_indices = [index for index, name in enumerate(original_attribute_names) if name in attribute_names_in_subset]
    # Filter/ modify the attribute_label_map based on the subset.
    attribute_label_map = [item
                                  for item in attribute_label_map if item.attribute in attribute_names_in_subset]
  else:
    attribute_subset_indices = range(len(original_attribute_names))
    attribute_names_in_subset = original_attribute_names

  # Write the attribute label map file.
  with tf.gfile.Open(attribute_label_map_file, 'w') as f:
    json.dump(attribute_label_map, f)

  filename_to_attributes = {}
  for index in xrange(2, len(file_lines)):
    items = [x for x in file_lines[index].split(' ') if x != '']
    assert len(items) == len(original_attribute_names) + 1, ("Reading from incorrect"
                                                    "attribute file.")
    # Filter based on the attributes we think should be in the subset.
    attributes_for_item = [int(x) for index, x in enumerate(items[1:]) if index in attribute_subset_indices]
    # CelebA attributes are -1, 1, convert it into 0, 1.
    assert any([x==-1 for x in attributes_for_item]), (
        "Expect attributes in celeba ground truth to be -1 or 1")
    filename_to_attributes[items[0]] = [int(x == 1) for x in attributes_for_item]

  attributes_in_split = []
  for image in images_in_split:
    attributes_in_split.append(filename_to_attributes[image])

  # Dump a .mat file with the labels for datapoints in the dataset. That is
  # we dump a file with N x D datapoints in it, where N is the number of
  # images in the split and D is the number of classes.
  write_attributes_in_split = np.array(attributes_in_split)
  savemat(os.path.join(FLAGS.output_directory, split_name+'_dataset_labels.mat'),
          {
              'data_labels': write_attributes_in_split,
              'label_names': [x.rstrip() for x in attribute_names_in_subset],
           })

  # Get paths to the images in the split.
  paths_to_images = [
      os.path.join(FLAGS.dataset_dir, 'img_align_celeba', x)
      for x in images_in_split
  ]

  return paths_to_images, attributes_in_split


def _process_dataset(split_name, data_path, output_directory, num_shards):
  # Select the subset of images in the particular split.
  images_in_split, attributes_in_split = _select_images_and_attributes_in_split(
      split_name, data_path["partitions"], data_path["attributes"],
      FLAGS.attribute_label_map)

  # Process those images.
  _process_image_files(split_name, images_in_split, attributes_in_split,
                       num_shards)


def main(unused_argv):
  if not tf.gfile.Exists(FLAGS.dataset_dir):
    tf.gfile.MakeDirs(FLAGS.dataset_dir)

  if not tf.gfile.Exists(FLAGS.output_directory):
    tf.gfile.MakeDirs(FLAGS.output_directory)

  data_path = download_and_unzip_celeba()

  _process_dataset(FLAGS.split_name, data_path, FLAGS.dataset_dir, FLAGS.shards)


if __name__ == "__main__":
  tf.app.run()
