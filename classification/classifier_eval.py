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
"""Evaluates inception resnet.

Evaluates the generated images on the comp split, stores a webpage with
results, and computes metrics based on supervised learning.
NOTE: Currently this eval is specific to deepmind 2d shapes and labels.

Author: vrama@
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import logging
import math
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from classification import classification_model
from joint_vae import utils

app = tf.app
flags = tf.flags
gfile = tf.gfile

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 64,
                            'The number of images in each batch.')

tf.app.flags.DEFINE_string('dataset_dir', '', 'Path to the dataset we want to'
                           ' train on.')
# Data augmentation.
tf.app.flags.DEFINE_boolean('blur_image', False, 'Whether to blur input images.')

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/inception_resnet_train/',
                           'Directory where the model was written to.')

tf.app.flags.DEFINE_string('eval_dir', '/tmp/inception_resnet_train/',
                           'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'eval_interval_secs', 120,
    'The frequency, in seconds, with which evaluation is run.')

tf.app.flags.DEFINE_string('split_name', 'val',
                           """Either 'train' or 'val' or 'test'.""")

tf.logging.set_verbosity(tf.logging.INFO)


def create_restore_fn(checkpoint_path, saver):
  """Return a function to restore variables.

  Args:
    checkpoint_path: Path to the checkpoint to load.
    saver: tf.train.Saver object
  Returns:
    restore_fn: An op to which we can pass a session to restore
      variables.
    global_step_ckpt: Int, the global checkpoint number.
  Raises:
    ValueError: If invalid checkpoint name is found or there are no checkpoints
      in the directory specified.
  """
  tf.logging.info('Checkpoint path: %s', (checkpoint_path))
  global_step_ckpt = checkpoint_path.split('-')[-1]

  # Checks for fraudulent checkpoints without a global_step.
  if global_step_ckpt == checkpoint_path:
    raise ValueError('Invalid checkpoint name %s.' % (checkpoint_path))

  def restore_fn(sess):
    """Restore model and feature extractor."""
    tf.logging.info('Restoring the model from %s', (checkpoint_path))
    saver.restore(sess, checkpoint_path)

  return restore_fn, global_step_ckpt


def evaluate_loop():
  """Run evaluation in a loop."""
  batch_size = FLAGS.batch_size

  g = tf.Graph()
  with g.as_default():
    tf.set_random_seed(123)
    classifier = classification_model.ClassifyFaces(
        mode='eval', split=FLAGS.split_name)
    classifier.build_model()
    saver = classifier.setup_saver()
    num_iter = int(math.ceil(classifier.num_samples / float(FLAGS.batch_size)))
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    for checkpoint_path in slim.evaluation.checkpoints_iterator(
        FLAGS.checkpoint_dir, FLAGS.eval_interval_secs):
      init_fn, global_step = create_restore_fn(checkpoint_path, saver)
      with tf.Session() as sess:
        init_fn(sess)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        total_correct = []
        total_data = []
        for this_iter in xrange(num_iter):
          logits, labels = sess.run([classifier.logits, classifier.labels])
          predictions = np.argmax(logits, axis=-1)

          total_correct.append(predictions == labels)
          total_data.append(predictions.shape[0])
          if this_iter % 10 == 0:
            tf.logging.info('Done with iteration %d/%d' % (this_iter, num_iter))

        if len(total_correct) > 1:
          total_correct = np.vstack(total_correct)
        else:
          total_correct = total_correct[0]

        total_correct = np.sum(total_correct, axis=0)
        accuracy_per_class = total_correct / float(np.sum(total_data))
        overall_accuracy = np.mean(accuracy_per_class)

        tf.logging.info('Accuracy: %f' % (overall_accuracy))
        utils.add_simple_summary(
            summary_writer,
            overall_accuracy,
            'Accuracy',
            global_step,)

        # Add summaries for each class.
        for attribute_name, accuracy in zip(classifier.attribute_names,
                                            accuracy_per_class):
          utils.add_simple_summary(summary_writer, accuracy,
                                   'Accuracy_' + attribute_name, global_step)
          tf.logging.info('Accuracy for class %s is %f' % (attribute_name,
                                                           accuracy))

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)


def main(_):
  np.random.seed(42)
  assert FLAGS.checkpoint_dir is not None, ('Please specify a checkpoint '
                                            'directory.')
  assert FLAGS.eval_dir is not None, 'Please specify an evaluation directory.'

  evaluate_loop()


if __name__ == '__main__':
  tf.app.run()
