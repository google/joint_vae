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

"""Creates a version of the MNIST dataset with attributes.

For each image in the MNIST datset, we create "K" images where for each of
the K images we randomly sample a location, scale and rotation parameter.
Based on the parameters we sample we also assign class labels for images.

TODO(vrama): Provide details of the usage.
Usage:
  python create_dataset.py --output_train_tfexample /tmp/train\
      --output_val_tfexample /tmp/val\
      --output_test_tfexample /tmp/test\


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from datasets.mnist import mnist_valid as mnist_dataset
import json

import numpy as np
import tensorflow as tf

import partition_labels
import utils

app = tf.app
flags = tf.flags
gfile = tf.gfile

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

# I/O
flags.DEFINE_string("output_train_tfexample", "", "Path to deepmind-concepts"
                    "-labels 2D shapes train tf example sstables.")
flags.DEFINE_string("output_val_tfexample", "", "Path to deepmind-concepts"
                    "-labels 2D shapes val tf example sstables.")
flags.DEFINE_string("output_test_tfexample", "", "Path to deepmind-concepts"
                    "-labels 2D shapes test tf example sstables.")

flags.DEFINE_string("path_to_original_mnist", "/tmp/mnist", "Path to the "
                    "original MNIST dataset")

# Label maps.
flags.DEFINE_string("label_split_json", "/tmp/label_split.json",
                    "Json file where label splits will be stored.")
flags.DEFINE_string("label_map_json", "/tmp/label_map.json",
                    "Text file with label maps")
flags.DEFINE_string(
    "image_split_json", "/tmp/num_images_split.json",
    "Tells us the number of images in each split of the dataset.")

flags.DEFINE_integer("replication", 5, "Sample these many mnist with "
                     "attribute images for each ground truth mnist image.")
flags.DEFINE_integer("canvas_size", 64, "Size of the output canvas for MNIST"
                     "with attributes")
flags.DEFINE_boolean("binarize", True, "Performs binarization of the "
                     "dataset as described in Salakhudtinov and Murray, 08.")
flags.DEFINE_string("split_type", "iid", "iid or comp split.")
flags.DEFINE_integer("random_seed", 42, "Random seed for dataset creation.")

# Names are defined at:
# https://github.com/tensorflow/models/blob/master/slim/datasets/mnist.py
split_names_to_mnist_names = {
    "train": "train_small",
    "val": "valid",
    "test": "test",
}

_DATA_SPLITS = tuple(["train", "val", "test"])

_TRAIN_VAL_TEST_SPLIT = (0.85, 0.05, 0.1)


def binarize_images(train_images, val_images, test_images, random_seed):

  train_images = utils.binarize(train_images, random_seed)
  val_images = utils.binarize(val_images, random_seed)
  test_images = utils.binarize(test_images, random_seed)

  return train_images, val_images, test_images

def write_tfexamples(data, tfexample_path):

  colorspace = "RGB"
  channels = 1
  image_format = "JPEG"
  height = FLAGS.canvas_size
  width = FLAGS.canvas_size

  image_coder = utils.ImageCoder(FLAGS.canvas_size)

  builder = tf.python_io.TFRecordWriter(tfexample_path)
  tf.logging.info("Writing %s TF Example.", tfexample_path)

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
    builder.write(example.SerializeToString())

    if index % 1000 == 0:
      tf.logging.info("Wrote %d datapoints to the tfexample.", index)

  builder.close()


def split_train_val_test(attribute_data, label_splits):
  train_items = []
  val_items = []
  test_items = []

  if FLAGS.split_type != "comp":
    cumulative_train_val_test_split = np.array(_TRAIN_VAL_TEST_SPLIT)
    cumulative_train_val_test_split = np.cumsum(cumulative_train_val_test_split)

  for datum in attribute_data:
    if FLAGS.split_type == "comp":
      if datum.attribute_names in label_splits["train"]:
        train_items.append(datum)
      elif datum.attribute_names in label_splits["val"]:
        val_items.append(datum)
      elif datum.attribute_names in label_splits["test"]:
        test_items.append(datum)
      else:
        raise RuntimeError
    elif FLAGS.split_type == "iid":
      # This is the case where train/val/test sets are all built IID.
      choose_dp = np.random.uniform(0, 1)
      if choose_dp <= cumulative_train_val_test_split[0]:
        train_items.append(datum)
      if (choose_dp > cumulative_train_val_test_split[0] and
          choose_dp <= cumulative_train_val_test_split[1]):
        val_items.append(datum)
      if (choose_dp > cumulative_train_val_test_split[1] and
          choose_dp <= cumulative_train_val_test_split[2]):
        test_items.append(datum)

  return (train_items, val_items, test_items)


def process_images_and_create_labels(dataset,
                                     create_affine_mnist,
                                     data_splits=_DATA_SPLITS):
  attribute_dataset = []

  for split in data_splits:
    tf.logging.info("Processing split %s.", split)

    dataset_images, dataset_labels = dataset[split]

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
        tf.logging.info("Finished adding labels for %d images.", index)

  return attribute_dataset


def load_mnist_dataset(split_name, use_batch_size=200):
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
      tf.logging.info("Using %s split.", name)
      dataset_mnist = mnist_dataset.get_split(
          split_names_to_mnist_names[name],
          dataset_dir=FLAGS.path_to_original_mnist,
      )
      num_samples = dataset_mnist.num_samples
      mnist_provider = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
          dataset_mnist,
          shuffle=False,
      )
      [image, label] = mnist_provider.get(['image', 'label'])

      sess = tf.Session()
      coordinator = tf.train.Coordinator()
      tf.train.start_queue_runners(sess, coord=coordinator)

      images_in_split = []
      classes_in_split = []
      for load_iteration in xrange(num_samples):
        loaded_image, loaded_label = sess.run([image, label])
        images_in_split.append(loaded_image[np.newaxis, :])
        classes_in_split.append(loaded_label)

        if load_iteration % 5000 == 0:
          tf.logging.info("Loaded %d of %d", load_iteration,
                       num_samples)

      images_in_split = np.vstack(images_in_split)
      classes_in_split = np.array(classes_in_split)

      split_to_dataset[name] = (images_in_split, classes_in_split)

  return split_to_dataset


def main(argv=()):
  del argv  # Unused.

  assert FLAGS.output_train_tfexample is not None, ("Missing "
                                                    "output_train_tfexample "
                                                    "path.")
  assert FLAGS.output_val_tfexample is not None, ("Missing output_val_tfexample"
                                                  " path.")
  assert FLAGS.output_test_tfexample is not None, (
      "Missing "
      "output_test_tfexample path.")

  np.random.seed(FLAGS.random_seed)

  data_splits = _DATA_SPLITS

  # Load the MNIST dataset from existing SSTables.
  split_to_dataset = load_mnist_dataset(data_splits)

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

  # At this point we will either be splitting based on the data point in an
  # IID manner or we will be splitting into unique attribute combinations.
  label_splits = create_affine_mnist.split_train_val_test(
      _TRAIN_VAL_TEST_SPLIT)

  # Split into Train Val Test.
  train_split, val_split, test_split = split_train_val_test(
      affine_mnist, label_splits)

  num_images_in_each_split = {}
  num_images_in_each_split["train"] = len(train_split)
  num_images_in_each_split["val"] = len(val_split)
  num_images_in_each_split["test"] = len(test_split)

  for key, value in num_images_in_each_split.iteritems():
    tf.logging.info("Split %s has %d images", key, value)

  with tf.gfile.Open(FLAGS.image_split_json, "w") as f:
    f.write(json.dumps(num_images_in_each_split))

  # Can binarize images here, after making sure that the np random seed is
  # set again.
  if FLAGS.binarize:
    tf.logging.info("Binarizing the images.")
    train_split, val_split, test_split = binarize_images(
        train_split, val_split, test_split, FLAGS.random_seed)

  # Write output TFExamples.
  write_tfexamples(train_split, FLAGS.output_train_tfexample)
  write_tfexamples(val_split, FLAGS.output_val_tfexample)
  write_tfexamples(test_split, FLAGS.output_test_tfexample)


if __name__ == "__main__":
  tf.app.run()
