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

"""Load a label map file and provide utilities to use it.

The label map attribute file is a List of tuples with the first element as
the name of the attribute and second as a Dict.
The value dict has the key label_id as a string, and value name of the label.

For example:
  [(
    "shape",
    {
     "0": "square",
     "1": "ellipse",
     "2": "heart"
    }
   ),
   (
    "scale",
    {
     "0": "big",
     "1": "small"
    }
   ),
  ]

Where "shape" and "scale" are the attributes, each associated with a label
map.

Author: vrama@
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import json

import tensorflow as tf

gfile = tf.gfile


class LabelMap(object):
  """Provides utilities to load from a label map attribute file."""

  def __init__(self, label_map_json_file):
    """Intitalize.

    Args:
      label_map_json_file: JSON file with the label map.
        Format:
          [(
            "shape",
            {
             "0": "square",
             "1": "ellipse",
             "2": "heart"
            }
           ),
           (
            "scale",
            {
             "0": "big",
             "1": "small"
            }
           ),
          ]
    """
    self._label_map_json_file = label_map_json_file

    with tf.gfile.Open(self._label_map_json_file, 'r') as f:
      json_dump = f.read()
      attribute_label_map = OrderedDict(json.loads(json_dump))
      copy_label_map = OrderedDict()

      for attribute, label_map in attribute_label_map.iteritems():
        out_label_map = {}
        for label_id, label_name in label_map.iteritems():
          out_label_map[int(label_id)] = label_name
        copy_label_map[attribute] = out_label_map

      self._label_map = copy_label_map

  def name_for_label(self, attribute_id, label_id):
    """Get the name corresponding to a label.

    Args:
      attribute_id: Index of the attribute ("shape")
      label_id: Index of the label within the set of labels for
        an attribute ("square")
    Returns:
      str, label for a given attribute, label id.
    """
    return self._label_map[self.attributes[attribute_id]][label_id]

  def label_for_name(self, label_name_vector):
    """Get the labels corresponding to a name.

    Args:
      label_name_vector: list of str, of length num_attributes, specifying the
        name for each label.
    Returns:
      label_vector: List of int.
    """
    label_vector = []
    for attribute_index, attribute_name in enumerate(label_name_vector):
      label_vector.extend([
          k
          for k, v in self._label_map[self.attributes[attribute_index]]
          .iteritems() if v == attribute_name
      ])
    return label_vector

  def attribute_for_name(self, name):
    """Get the attribute id for a given named attribute ('shape')."""
    return sorted(self._label_map.keys()).index(name)

  @property
  def attributes(self):
    """Return a list of all attributes."""
    return self._label_map.keys()

  @property
  def count_labels(self):
    """Return the number of labels for each attribute, in order."""
    return tuple([len(v) for _, v in self._label_map.iteritems()])

  @property
  def count_for_label(self):
    return {k: len(v) for k, v in self._label_map.items()}
