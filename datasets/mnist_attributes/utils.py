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

"""Some utilities for creating the MNIST with attributes dataset.
"""
from collections import namedtuple
import copy

import numpy as np
import tensorflow as tf

XY = namedtuple("XY", ["x", "y"])

AttributeData = namedtuple("AttributeData", [
    "images", "attribute_names", "attribute_labels", "true_latents"
])


def binarize(split_images, random_seed):
  np.random.seed(random_seed)

  if np.max(split_images[0].images) == 1.0:
    raise ValueError("Images are expected in the range (0, 255).")

  binarized_split_images = []
  for datum in split_images:
    image = datum.images
    # Scale Image to lie between 0 and 1
    image_prob = image / 255.0
    # Use the values from a single channel to predict the binarized
    # version.
    binarized_image = np.random.binomial(
        1, np.squeeze(image_prob[:, :, 1])).astype(np.uint8)
    binarized_image = np.repeat(binarized_image[:, :, np.newaxis], 3, axis=-1)
    binarized_image *= 255

    binarized_datum = AttributeData(binarized_image, datum.attribute_names,
                                    datum.attribute_labels, datum.true_latents)

    binarized_split_images.append(binarized_datum)

  return binarized_split_images


class ImageCoder(object):
  """Intitializes a class that handles encoding/ decoding operations for images."""

  def __init__(self, image_size):
    self.raw_image = tf.placeholder(
        shape=[image_size, image_size, None], dtype=np.uint8)
    self.encoded_image = tf.image.encode_jpeg(self.raw_image)

    self.session = tf.Session()

  def encode_image(self, image_to_encode):
    encoded_image = self.session.run([self.encoded_image],
                                     {self.raw_image: image_to_encode})
    return encoded_image


def int64_feature(ints):
  if not isinstance(ints, list):
    ints = list([ints])
  return tf.train.Feature(int64_list=tf.train.Int64List(value=ints))


def float_feature(floats):
  if not isinstance(floats, list):
    floats = list([floats])
  return tf.train.Feature(float_list=tf.train.FloatList(value=floats))


def bytes_feature(strings):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=strings))

def string_feature(string):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[string]))


def pil_image_to_np_array(pil_image):
  image_size_x, image_size_y = pil_image.size
  im_arr = np.fromstring(pil_image.tobytes(), dtype=np.uint8)
  # Note x, y changes to y, x when we go from PIL to numpy.
  np_array_image = np.reshape(im_arr, (image_size_y, image_size_x, -1))
  return np_array_image
