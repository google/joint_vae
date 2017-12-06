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

"""Evaluates a multi-attribute classification model on deepmind shapes.

See the README.md file for compilation and running instructions.
"""

import math

import dataset_provider

import tensorflow as tf
import tensorflow.contrib.slim as slim

from .. import label_map
from .. import multi_attribute_net

app = tf.app
flags = tf.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_integer('batch_size', 128, 'The number of images in each batch.')

flags.DEFINE_string('split_type', 'compositional',
                    'compositional: Train/val/test on a compositional split.'
                    'iid: Train/val/test on an IID split.')

# Loading same JSON file for both 'iid'/'compositional'.
flags.DEFINE_string(
    'label_map_json',
    'research/vale/imagination/classify_deepmind_labels/deepmind_2dshapes_labelmap.json',
    'A json file storing a list tuple of (attribute keys (str), label maps for'
    ' each attribute (dict)).')

flags.DEFINE_boolean('grayscale', True,
                     'Indicates whether the input image is grayscale or color')

flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')

flags.DEFINE_string('checkpoint_dir', '/tmp/deepmind_shapes_with_labels/',
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', '/tmp/deepmind_shapes_with_labels/',
                    'Directory where the results are saved to.')

flags.DEFINE_integer('eval_interval_secs', 60,
                     'The frequency, in seconds, with which evaluation is run.')

flags.DEFINE_string('split_name', 'val',
                    """Either 'train' or 'val' or 'test'.""")


def main(_):
  g = tf.Graph()
  with g.as_default():
    images, gt_labels_list, _, num_samples, num_classes_per_attribute = (
        dataset_provider.provide_data(
            FLAGS.split_name,
            FLAGS.batch_size,
            split_type=FLAGS.split_type,
            grayscale=FLAGS.grayscale))

    # Load a label map with attributes and corresponding labels.
    attribute_label_map = label_map.LabelMap(FLAGS.label_map_json)

    assert attribute_label_map.count_labels == num_classes_per_attribute, (
        'Please check your label_map_file, to make sure it corresponds to the'
        ' dataset being loaded.')

    # Define the model:
    logits_list = multi_attribute_net.multi_attribute_net(
        images,
        num_classes_per_attribute=num_classes_per_attribute,
        attribute_names=attribute_label_map.attributes,
        is_training=False)

    predictions_list = [tf.argmax(logits, 1) for logits in logits_list]

    # Define the metrics:
    metrics_to_computation = {}
    for tower_id, sparse_gt_labels_list in enumerate(gt_labels_list):
      metrics_to_computation[
          'Accuracy_'
          + attribute_label_map.attributes[tower_id]] = slim.metrics.accuracy(
              predictions_list[tower_id], sparse_gt_labels_list)
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
        metrics_to_computation)

    # Also compute the average accuracy across all towers, each tower
    # corresponds to an attribute.
    mean_accuracy = tf.zeros((1))
    for name, value in names_to_values.iteritems():
      slim.summaries.add_scalar_summary(
          value, name, prefix='eval', print_summary=True)
      mean_accuracy += value

    mean_accuracy /= len(names_to_values.keys())
    slim.summaries.add_scalar_summary(
        mean_accuracy[0], 'Accuracy', prefix='eval', print_summary=True)

    # This ensures that we make a single pass over all of the data.
    num_batches = math.ceil(num_samples / float(FLAGS.batch_size))

    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
  tf.app.run()
