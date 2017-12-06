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

"""Creates a retrieval set for MNIST with attributes.

For each image in the MNIST datset, we create "K" images where for each of
the K images we randomly sample a location, scale and rotation parameter.
Based on the parameters we sample we also assign class labels for images.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import json
import logging

import numpy as np
import tensorflow as tf

from sstable import SSTableWriter
import original_mnist_data_provider
import partition_labels
import utils

app = tf.app
flags = tf.flags
gfile = tf.gfile

FLAGS = tf.app.flags.FLAGS

# I/O
flags.DEFINE_string("output_retrieval_tfexample", "", "Path to the SSTable with"
                    " retrieval images.")

# Label maps.
flags.DEFINE_string("label_split_json", "/tmp/label_split.json",
                    "Json file where label splits will be stored.")
flags.DEFINE_string("label_map_json", "/tmp/label_map.json",
                    "Text file with label maps")

# Legacy arguments, mostly here for no good reason than convenience.
flags.DEFINE_string(
    "image_split_json", "/tmp/num_images_split.json",
    "Tells us the number of images in each split of the dataset.")
flags.DEFINE_string("split_type", "iid", "iid or comp split.")

flags.DEFINE_integer("replication", 1, "Sample these many mnist with "
                     "attribute images for each ground truth mnist image.")
flags.DEFINE_integer("canvas_size", 64, "Size of the output canvas for MNIST"
                     "with attributes")
flags.DEFINE_boolean("binarize", True, "Performs binarization of the "
                     "dataset as described in Salakhudtinov and Murray, 08.")
flags.DEFINE_integer("random_seed", 42, "Random seed for dataset creation.")

# Names are defined at:
# //third_party/tensorflow_models/slim/datasets/mnist.py
split_names_to_mnist_names = {
    "train": "train_small",
    "val": "valid",
    "test": "test",
}


_DATA_SPLITS = tuple(["train", "val", "test"])

def write_sstables(data, sstable_path):

  colorspace = "RGB"
  channels = 1
  image_format = "JPEG"
  height = FLAGS.canvas_size
  width = FLAGS.canvas_size

  image_coder = utils.ImageCoder(FLAGS.canvas_size)

  builder = SSTableWriter(sstable_path)
  logging.info("Writing %s SSTable.", sstable_path)

  for index, datum in enumerate(data):
    # Convert the contents of the datum into a tf example.

    image_encoded = image_coder.encode_image(datum.images)
    latents = [float(x) for x in datum.true_latents]
    labels = datum.attribute_labels

    example = tf.train.Example(features=tf.train.Features(
        feature={
            "image/height": utils.int64_feature(height),
            "image/width": utils.int64_feature(width),
            "image/colorspace": utils.string_feature(colorspace.encode()),
            "image/channels": utils.int64_feature(channels),
            "latents": utils.float_feature(latents),
            "labels": utils.int64_feature(labels),
            "image/format": utils.string_feature(image_format.encode()),
            "image/encoded": utils.bytes_feature(image_encoded)
        }))
    builder.Add("%.16d" % (index), example.SerializeToString())

    if index % 1000 == 0:
      logging.info("Wrote %d datapoints to the sstable.", index)

  builder.FinishTable()


def process_images_and_create_labels(dataset, create_affine_mnist):
  attribute_dataset = []

  for split, data in dataset.iteritems():
    logging.info("Processing split %s.", split)

    dataset_images, dataset_labels = data

    if dataset_images.shape[0] != dataset_labels.shape[0]:
      raise ValueError("Images and labels must have the same shape.")

    for index in xrange(dataset_images.shape[0]):
      for _ in xrange(FLAGS.replication):
        transformed_image, attribute_names, attribute_labels, true_latents = (
            create_affine_mnist.assign_labels_to_datapoint_and_process(
                dataset_images[index], dataset_labels[index]))
        attribute_datum = utils.AttributeData(
            transformed_image, attribute_names, attribute_labels, true_latents)
        attribute_dataset.append(attribute_datum)

      if index % 1000 == 0:
        logging.info("Finished adding labels for %d images.", index)

  return attribute_dataset


def load_minst_dataset(split_name, use_batch_size=200):
  """Load the given split of the MNIST dataset.

  If nothing is specified then we load all the splits of the dataset.

  Args:
    split_name: tuple, specifies the names of the splits of the MNIST
      dataset that we wish to load.
  Returns:
    TODO(Vrama):
  """
  split_to_dataset = {}

  with tf.Graph().as_default():
    for name in split_name:
      logging.info("Using %s split.", name)
      images, labels, num_samples = original_mnist_data_provider.provide_data(
          split_names_to_mnist_names[name],
          batch_size=use_batch_size,)
      if num_samples % use_batch_size != 0:
        raise ValueError("Invalid batch size %d", use_batch_size)

      num_iters_loading = int(num_samples / use_batch_size)

      sess = tf.Session()
      coordinator = tf.train.Coordinator()
      tf.train.start_queue_runners(sess, coord=coordinator)

      images_in_split = []
      class_in_split = []
      for load_iteration in xrange(num_iters_loading):
        loaded_image, loaded_label = sess.run([images, labels])
        images_in_split.append(loaded_image)
        class_in_split.append(loaded_label)
        if load_iteration % 10 == 0:
          logging.info("Loaded %d of %d", load_iteration * use_batch_size,
                       num_samples)

      images_in_split = np.vstack(images_in_split)
      classes_in_split = np.vstack(class_in_split)

      _, categorical_classes = np.nonzero(classes_in_split)

      split_to_dataset[name] = (images_in_split, categorical_classes)

  return split_to_dataset


def main(argv=()):
  del argv  # Unused.

  assert FLAGS.output_retrieval_tfexample is not None, (
      "Missing "
      "output_retrieval_tfexample "
      "path.")
  np.random.seed(FLAGS.random_seed)

  data_splits = _DATA_SPLITS

  # Load the MNIST dataset from existing SSTables.
  split_to_dataset = load_minst_dataset(data_splits)

  # Class to add attributes to MNIST.
  create_affine_mnist = partition_labels.CreateMnistWithAttributes(
      FLAGS.label_split_json,
      FLAGS.label_map_json,
      split_type=FLAGS.split_type,
      canvas_size=FLAGS.canvas_size,
      random_seed=FLAGS.random_seed,
  )

  # Process each record and add the images and attriutes to MNIST.
  affine_mnist = process_images_and_create_labels(
      split_to_dataset, create_affine_mnist)

  logging.info("Retrieval set has %d images", len(affine_mnist))

  with tf.gfile.Open(FLAGS.image_split_json, "w") as f:
    f.write(json.dumps([len(affine_mnist)]))

  # Can binarize images here, after making sure that the np random seed is
  # set again.
  if FLAGS.binarize:
    logging.info("Binarizing the images.")
    affine_mnist = utils.binarize(affine_mnist,
                                           FLAGS.random_seed)
  # Write output SSTables.
  write_sstables(affine_mnist, FLAGS.output_retrieval_tfexample)

if __name__ == "__main__":
  tf.app.run()
