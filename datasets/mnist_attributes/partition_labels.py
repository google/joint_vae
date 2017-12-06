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

"""Create discrete-valued labels for mnist with attributes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from itertools import product
import json
import math
import logging
import random

import numpy as np
from PIL import Image

import tensorflow as tf
import perturb_images
import utils


gfile = tf.gfile

def check_in_image(paste_image_location, paste_image_size, canvas_image_size):
  """Checks whether the location for the pasted image is within the canvas.

  Args:
    paste_image_location: a namedtuple of utils.XY, with 'x' and 'y' coordinates
      of
      the center of the image we want to paste.
    paste_image_size: a namedtuple of utils.XY, with 'x' and 'y' coordinates
      corresponding to the size of the image we are pasting.
    canvas_image_size: the size of the canvas that we are pasting the image to.
  Returns:
    True if the pasted image would lie within the canvas, False otherwise.
  """
  offset_x = int(paste_image_size.x / 2) + 1
  offset_y = int(paste_image_size.y / 2) + 1

  if (paste_image_location.x + offset_x > canvas_image_size or
      paste_image_location.x - offset_x < 1 or
      paste_image_location.y + offset_y > canvas_image_size or
      paste_image_location.y - offset_y < 1):
    return False
  return True


class CreateMnistWithAttributes(object):
  """Add attribute-label data to mnist dataset images."""

  def __init__(self,
               label_split_file,
               label_map_file,
               split_type,
               canvas_size=64,
               random_seed=42):
    """Initialization.

    Args:
      label_split_file: string, A JSON file path where the label
        combinations from training, val and test will be stored.
      label_map_file: Stores a mapping from label id to label name.
      canvas_size: The size of the canvas that we paste MNIST digits on.
      random_seed: Random seed for partitioning labels.
    """
    # Random seed for repeatable train, val, test set creation.
    self._random_seed = random_seed
    random.seed(self._random_seed)

    self.label_split_file = label_split_file
    self.label_map_file = label_map_file
    self.split_type = split_type
    self.canvas_size = canvas_size

    self._setup_labels()
    self._setup_labelmap()

  def _setup_labels(self):
    """Define the labels to induce on the dataset."""
    self._attribute_names = ("digit", "scale", "orientation", "location")

    self.state_space = OrderedDict()
    self.state_space["digit"] = ("0", "1", "2", "3", "4", "5", "6", "7", "8",
                                 "9")

    self.state_space["scale"] = ("big", "small")

    self.state_space["orientation"] = ("clockwise", "upright",
                                       "counterclockwise")

    self.state_space["location"] = ("top-left", "top-right", "bottom-left",
                                    "bottom-right")

  def _setup_labelmap(self):
    """Generate mapping from item in a state space to its label id."""
    self.label_map = OrderedDict()

    for label_type in self.state_space:
      self.label_map[label_type] = {
          index: v
          for index, v in enumerate(self.state_space[label_type])
      }

  def _attribute_dict_to_tuple(self, attribute_dict):
    """Given a dict convert to a tuple in order of attributes."""
    # state space is an ordereddict, read attribute dict in the order implied
    # by state space.
    if set(attribute_dict.keys()) != set(self.state_space.keys()):
      raise ValueError("Invalid attribute dict.")

    values = []
    for attribute in self.state_space.keys():
      if isinstance(attribute_dict[attribute], utils.XY):
        values.extend(list(attribute_dict[attribute]))
      elif isinstance(attribute_dict[attribute], np.ndarray):
        # TODO(vrama): Think through if this covers all cases, this is a bandaid
        # for now.
        values.extend(list(attribute_dict[attribute]))
      else:
        values.append(attribute_dict[attribute])
    return values

  def get_images_and_continuous_latents_for_labels(self, gt_attributes, image):
    """Get continuous valued latent values from attributes.

    Args:
      gt_attributes: dict, with key as the attribute name and value as the
        label assigned in the ground truth for that attribute.
      image: A PIL Image.
    Returns:
      final_transformed_image: A PIL Image.
      continuous_latent_values: dict, with key as the semantic name of the
        latent state and value as the value assigned to the latent state.
    """
    continuous_latent_values = {}

    def infer_latents_for_scale(scale_label):
      if self.state_space["scale"][scale_label] == "small":
        return np.random.randn(1) * 0.1 + 0.6
      if self.state_space["scale"][scale_label] == "big":
        return np.random.randn(1) * 0.1 + 0.9

    def infer_latents_for_orientation(orientation_label):
      if self.state_space["orientation"][orientation_label] == "clockwise":
        return np.random.randn(1) * 10 - 45
      if self.state_space["orientation"][
          orientation_label] == "counterclockwise":
        return np.random.randn(1) * 10 + 45
      if self.state_space["orientation"][orientation_label] == "upright":
        return 0.0

    def infer_latents_for_location(location_label):
      # Center offset shifts the center towards the corners by a specified
      # amount.
      center_offset = self.canvas_size / 16
      # center is in format x, y
      if self.state_space["location"][location_label] == "top-left":
        center = utils.XY(self.canvas_size / 4 - center_offset,
                          self.canvas_size / 4 - center_offset)
      elif self.state_space["location"][location_label] == "top-right":
        center = utils.XY(3 * self.canvas_size / 4 + center_offset,
                          self.canvas_size / 4 - center_offset)
      elif self.state_space["location"][location_label] == "bottom-left":
        center = utils.XY(1 * self.canvas_size / 4 - center_offset,
                          3 * self.canvas_size / 4 + center_offset)
      elif self.state_space["location"][location_label] == "bottom-right":
        center = utils.XY(3 * self.canvas_size / 4 + center_offset,
                          3 * self.canvas_size / 4 + center_offset)

      selected_location_x = int(
          np.random.randn(1) * self.canvas_size / 16 + center.x)
      selected_location_y = int(
          np.random.randn(1) * self.canvas_size / 16 + center.y)

      return utils.XY(selected_location_x, selected_location_y)

    continuous_latent_values["digit"] = gt_attributes["digit"]

    continuous_latent_values["orientation"] = None
    while (continuous_latent_values["orientation"] is None or
           continuous_latent_values["orientation"] < -90 or
           continuous_latent_values["orientation"] > 90):
      continuous_latent_values["orientation"] = infer_latents_for_orientation(
          gt_attributes["orientation"])

    continuous_latent_values["scale"] = None
    while (continuous_latent_values["scale"] is None or
           continuous_latent_values["scale"] < 0.4 or
           continuous_latent_values["scale"] > 1.0):
      continuous_latent_values["scale"] = infer_latents_for_scale(
          gt_attributes["scale"])

    rotated_image, rotated_canvas_size = perturb_images.rotate_image(
        image, continuous_latent_values["orientation"])
    scaled_rotated_image, scaled_rotated_canvas_size = (
        perturb_images.scale_image(rotated_image, rotated_canvas_size,
                                   continuous_latent_values["scale"]))

    latent_location = utils.XY(-1, -1)

    # Resample from the candidate location gaussians until we find something
    # that is within the image boundaries.
    while not check_in_image(latent_location, scaled_rotated_canvas_size,
                             self.canvas_size):
      latent_location = infer_latents_for_location(gt_attributes["location"])

    continuous_latent_values["location"] = latent_location
    # To paste the image on the canvas we need to pass the top-left location
    # of the box we computed. So we compute that below.
    paste_location = utils.XY(continuous_latent_values["location"].x - int(
        scaled_rotated_canvas_size.x / 2) - 1,
                              continuous_latent_values["location"].y - int(
                                  scaled_rotated_canvas_size.y / 2) - 1)

    final_transformed_image = perturb_images.paste_image_on_black_canvas(
        scaled_rotated_image, paste_location, self.canvas_size)

    final_transformed_image = utils.pil_image_to_np_array(
        final_transformed_image)

    return final_transformed_image, continuous_latent_values

  @staticmethod
  def format_image(image):
    """Converts original MNIST np.array image into a PIL image.
    """
    if image.shape[-1] == 1:
      pil_image = np.repeat(image, 3, axis=-1)
    pil_image = Image.fromarray(pil_image)
    return pil_image

  def assign_labels_to_datapoint_and_process(self, image, labels):
    """Assigns labels to datapoint and process output images.

    Args:
      image: an np.uint8 array of [image_size, image_size, nchannels]
      labels: int, an integer categorical label value.
    Returns:
      transformed_image: a PIL Image.
      gt_attributes: list, with a list of labels for each attribute.
      continuous_latent_values: list, with a list of labels for each latent
      state.
    Raises:
      ValueError: if labels is not an integer.
      ValueError: if image is not an np.array.
    """

    if not isinstance(labels, int):
      raise ValueError("Labels must be integer.")

    if not image.dtype == np.uint8:
      raise ValueError("Image must be np.uint8")

    # Process images.
    image = self.format_image(image)

    # Convert to categorical values from one-hot labels.
    gt_attributes = {}

    gt_attributes["digit"] = labels

    # First perform uniform sampling within each bin.
    gt_attributes["scale"] = np.random.choice(
        range(len(self.state_space["scale"])))

    gt_attributes["orientation"] = np.random.choice(
        range(len(self.state_space["orientation"])))

    gt_attributes["location"] = np.random.choice(
        range(len(self.state_space["location"])))

    # Given the values that we sampled, figure out what the continuous valued
    # latent world states for the classes should be, and generate an image with
    # those attributes.
    transformed_image, continuous_latent_values = (
        self.get_images_and_continuous_latents_for_labels(gt_attributes, image))

    # Get the names of the attributes that we sampled.
    attribute_names = []
    for key in self.state_space.keys():
      attribute_names.append(self.state_space[key][gt_attributes[key]])
    attribute_names = tuple(attribute_names)

    return transformed_image, attribute_names, self._attribute_dict_to_tuple(
        gt_attributes), self._attribute_dict_to_tuple(continuous_latent_values)

  def get_all_label_combs(self):
    """Compute the exhaustive state space across all labels.

    Computes the set of all label combinations in the dataset, and shuffles
    them.

    Returns:
      all_label_combinations: list of tuples, The set of all label combinations
        in the dataset.
    """
    all_label_combinations = []
    for labels in product(self.state_space["digit"], self.state_space["scale"],
                          self.state_space["orientation"],
                          self.state_space["location"]):
      all_label_combinations.append(labels)

    random.shuffle(all_label_combinations)

    return all_label_combinations

  def generate_partitions(self, mylist, train_val_test_split):
    """Retrieve data after splitting the dataset into train, val, test.

    Args:
      mylist: list, Input list with items to split.
      train_val_test_split: Tuple, with a split of train, val and test. Sum
        of entries in the tuple must be 1.

    Returns:
      split_name_to_attributes: Dict, with key as the split name (train/val/
        test) and value a list of tuples containing label combinations in
        the corresponding split.
    """
    assert sum(train_val_test_split) == 1, "Invalid train val test split."
    rounded_train_val_indices = [0]

    train_val_test_split = np.cumsum(np.array(train_val_test_split))
    split_names = ["train", "val", "test"]

    for item in train_val_test_split:
      rounded_train_val_indices.append(int(math.floor(item * len(mylist))))

    split_name_to_attributes = {}
    if self.split_type == 'iid':
      for sp_name in split_names:
        split_name_to_attributes[sp_name] = mylist
    elif self.split_type == 'comp':
      for index, item in enumerate(rounded_train_val_indices):
        if index == len(rounded_train_val_indices) - 1:
          break
        split_name_to_attributes[split_names[index]] = mylist[
            item:rounded_train_val_indices[index + 1]]
    return split_name_to_attributes

  def serialize_label_splits(self, label_splits):
    """Utilities for dumping label map and split info to JSON."""
    serialized = json.dumps(label_splits)
    with tf.gfile.Open(self.label_split_file, "w") as f:
      f.write(serialized)

    # Write label map as a tuple of key value pairs to maintain order.
    label_map_list = [(k, v) for k, v in self.label_map.iteritems()]

    serialized = json.dumps(label_map_list)
    with tf.gfile.Open(self.label_map_file, "w") as f:
      f.write(serialized)

  def split_train_val_test(self, train_val_test_split=(.80, .10, .10)):
    """Split label combinations into train/ val and test.

    Args:
      train_val_test_split: Tuple, with a split of train, val and test. Sum
        of entries in the tuple must be 1.
    Returns:
      label_splits: Dict, with key as the split of the dataset (train/val/test)
        and values a list of tuples containing the labels in that split. For
        example {"train":[(9,small,clockwise,bottom-right)], "val":[(8,big,
        center,bottom-left)], "test":[(7, big, center, bottom-left)]}.
    """
    all_label_combinations = self.get_all_label_combs()

    label_splits = self.generate_partitions(all_label_combinations,
                                            train_val_test_split)
    self.serialize_label_splits(label_splits)

    return label_splits
