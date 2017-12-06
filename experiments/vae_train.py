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


import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import convolutional_multi_vae
import configuration


_MODEL_LIST = ('multi', 'single', 'kronecker')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_type', 'multi',
                    'Kind of VAE to train, multi or single.')

tf.app.flags.DEFINE_integer('batch_size', 64, 'The number of images in each batch.')

tf.app.flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')

tf.app.flags.DEFINE_string('train_log_dir', '/tmp/unimodal_vae/',
                    'Directory where to write event logs.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer('save_interval_secs', 600,
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
  assert FLAGS.model_type in _MODEL_LIST, 'Invalid model specified.'

  if not tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  config = configuration.get_configuration()
  config.batch_size = FLAGS.batch_size

  training_config = configuration.TrainingConfig()

  g = tf.Graph()
  with g.as_default():
    # If ps_tasks is zero, the local device is used. When using multiple
    # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
    # across the different devices.
    if FLAGS.model_type == 'multi':
      vae = convolutional_multi_vae.ConvolutionalMultiVae(
          config, mode='train', split_name='train')
    elif FLAGS.model_type == 'single' or FLAGS.model_type == 'kronecker':
      raise NotImplementedError("%s not implemented" % (FLAGS.model_type))
    vae.build_model()

    optimizer = training_config.optimizer(training_config.learning_rate)

    tf.losses.add_loss(vae.loss)
    total_loss = tf.losses.get_total_loss()

    # Set up training.
    train_op = slim.learning.create_train_op(
      total_loss, optimizer, check_numerics=False)
    saver = vae.setup_saver()

    if config.loss_type == 'fwkl':
      init_fn = vae.get_forward_kl_init_fn(FLAGS.fwkl_init_dir)
    else:
      init_fn = None

    # Run training.
    slim.learning.train(
        train_op=train_op,
        init_fn=init_fn,
        logdir=FLAGS.train_log_dir,
        graph=g,
        number_of_steps=FLAGS.max_number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        saver=saver)


if __name__ == '__main__':
  tf.app.run()
