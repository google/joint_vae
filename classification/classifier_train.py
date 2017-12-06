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
"""Trains a unimodal/multimodal variational autoencoder.

Provides options to specify the following models for training:
  > multi: a multimodal variational autoencoder.
  > single: a unimodal variational autoencoder.
  > kronecker: a multimodal variational autoencoder with a low rank inductive
  bias.

See the README.md file for compilation and running instructions.

Author: vrama@
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from classification import classification_model

FLAGS = tf.app.flags.FLAGS

# Training configuration.
tf.app.flags.DEFINE_string('dataset_dir', '', 'Path to the dataset we want to'
                           ' train on.')
tf.app.flags.DEFINE_integer('batch_size', 32,
                            'The number of images in each batch.')

# Data augmentation.
tf.app.flags.DEFINE_boolean('blur_image', False, 'Whether to blur input images.')

tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                          'Learning rate to train the model.')

tf.app.flags.DEFINE_boolean('finetune', False,
                            'If True, finetune all the layers of the model.')

# Logistics.
tf.app.flags.DEFINE_string('path_to_irv2_checkpoint', '',
                           'Path to the inception resnet'
                           ' checkpoint file.')
tf.app.flags.DEFINE_string('master', 'local',
                           'BNS name of the TensorFlow master to use.')

tf.app.flags.DEFINE_string('train_log_dir', '/tmp/inception_resnet_train/',
                           'Directory where to write event logs.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 1800,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 1800,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer('max_number_of_steps', 500000,
                            'The maximum number of gradient steps.')

tf.app.flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  if not tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  if not FLAGS.path_to_irv2_checkpoint:
    raise ValueError('Please provide path to inception resnet checkpoint.')

  g = tf.Graph()
  with g.as_default():
    # If ps_tasks is zero, the local device is used. When using multiple
    # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
    # across the different devices.
    classifier = classification_model.ClassifyFaces(
        mode='train', finetune=FLAGS.finetune)
    classifier.build_model()

    init_fn = classifier.get_init_fn(FLAGS.path_to_irv2_checkpoint)
    variables_to_train = classifier.get_trainable_vars()

    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

    # Set up training.
    train_op = slim.learning.create_train_op(
        classifier.loss,
        optimizer,
        check_numerics=False,
        variables_to_train=variables_to_train)

    saver = classifier.setup_saver()

    # Run training.
    slim.learning.train(
        train_op=train_op,
        logdir=FLAGS.train_log_dir,
        init_fn=init_fn,
        graph=g,
        number_of_steps=FLAGS.max_number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        saver=saver)


if __name__ == '__main__':
  tf.app.run()
