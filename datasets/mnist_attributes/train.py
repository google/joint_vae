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

"""Trains a supervised multi-label classifier on MNIST-A.

The model learns a mapping from images to different attributes / labels for
the image. For example, the attributes can be shape, size, orientation, and
location for the objects. For each attribute we solve a supervised
classification problem.

NOTE(vrama): Everytime we change the train/val/test split on the original
dataset, need to retrain evaluation models with this script.

See the README.md file for compilation and running instructions.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

import dataset_provider
from experiments import configuration

from .. import label_map
from .. import multi_attribute_net

app = tf.app
flags = tf.flags
gfile = tf.gfile

FLAGS = tf.app.flags.FLAGS

flags.DEFINE_integer('batch_size', 64, 'The number of images in each batch.')

flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')

# TODO(vrama): Binarize is a bit of an abuse of terminology here, since
# the option we really want to enable is to say that we would not like to do
# mean subtraction.
flags.DEFINE_string('train_log_dir', '/tmp/deepmind_shapes_with_labels/',
                    'Directory where to write event logs.')

flags.DEFINE_integer(
    'save_summaries_secs', 120,
    'The frequency with which summaries are saved, in seconds.')

flags.DEFINE_integer('save_interval_secs', 120,
                     'The frequency with which the model is saved, in seconds.')

flags.DEFINE_integer('max_number_of_steps', 50000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  config = configuration.Configuration()

  g = tf.Graph()
  with g.as_default():
    # If ps_tasks is zero, the local device is used. When using multiple
    # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
    # across the different devices.
    with tf.device(tf.ReplicaDeviceSetter(FLAGS.ps_tasks)):
      if config.dataset == 'cub':
        images, gt_labels_list, _ = (
            cub_provider.provide_data(
                'train',
                FLAGS.batch_size,
                split_type=config.split_type,
                preprocess_options=config.preprocess_options,
                image_resize=config.image_size,
                shuffle_data=True,
                use_inception_resnet_v2=config.cub_irv2_features,
                box_cox_lambda=config.cub_box_cox_lambda,
                categorical=config.cub_categorical,
                classes_only=config.cub_classes_only,
                skip_classes=config.cub_skip_classes,
            ))
        num_classes_per_attribute = config.num_classes_per_attribute
      else:
        images, gt_labels_list, _, _, num_classes_per_attribute = (
            dataset_provider.provide_data(
                'train',
                FLAGS.batch_size,
                split_type=config.split_type,
                preprocess_options=config.preprocess_options,
                grayscale=config.grayscale))

      attribute_label_map = label_map.LabelMap(config.label_map_json)

      assert attribute_label_map.count_labels == num_classes_per_attribute, (
          'Please check your label_map_file, to make sure it corresponds to '
          'the dataset being loaded.')

      # Define the model:
      if config.dataset == 'cub' and config.cub_irv2_features:
        logits_list = multi_attribute_net.mlp_multi_attribute_net(
            images,
            num_classes_per_attribute=num_classes_per_attribute,
            attribute_names=attribute_label_map.attributes,
            hidden_units=config.comprehensibility_hidden_units,
            is_training=True)
      else:
        logits_list = multi_attribute_net.conv_multi_attribute_net(
            images,
            num_classes_per_attribute=num_classes_per_attribute,
            attribute_names=attribute_label_map.attributes,
            hidden_units=config.comprehensibility_hidden_units,
            is_training=True)

      # Specify the loss function:
      for logits, sparse_gt_labels in zip(logits_list, gt_labels_list):
        tf.contrib.losses.add_loss(
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=sparse_gt_labels, logits=logits)))
      total_loss = tf.contrib.losses.get_total_loss()
      tf.contrib.deprecated.scalar_summary('Total Loss', total_loss)

      # Specify the optimization scheme:
      optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)

      # Set up training.
      train_op = slim.learning.create_train_op(total_loss, optimizer)

      # Run training.
      slim.learning.train(
          train_op=train_op,
          logdir=FLAGS.train_log_dir,
          master=FLAGS.master,
          is_chief=FLAGS.task == 0,
          number_of_steps=FLAGS.max_number_of_steps,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
  tf.app.run()
