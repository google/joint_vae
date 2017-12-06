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

"""Evaluates unimodal variational autoencoder on deepmind 2d shapes with labels.

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

from datasets import label_map
from joint_vae import utils
from convolutional_multi_vae import ConvolutionalMultiVae
from experiments import image_utils
from experiments import configuration

app = tf.app
flags = tf.flags
gfile = tf.gfile


_MODEL_LIST = ('multi', 'single', 'kronecker')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_type', 'multi',
                    'Kind of VAE to train, multi or single.')

tf.app.flags.DEFINE_integer('batch_size', 128, 'The number of images in each batch.')

tf.app.flags.DEFINE_integer('num_samples_predictive_dist', 10, 'Number of samples to'
                     'average over when estimating p(y| x).')

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/unimodal_vae/',
                    'Directory where the model was written to.')

tf.app.flags.DEFINE_string('eval_dir', '/tmp/unimodal_vae/',
                    'Directory where the results are saved to.')

tf.app.flags.DEFINE_string('results_tag', '',
                    'Additional tag to prepend to output file names.')

tf.app.flags.DEFINE_integer('eval_interval_secs', 1200,
                     'The frequency, in seconds, with which evaluation is run.')

tf.app.flags.DEFINE_integer('num_result_datapoints', 100,
                     'Subset of the results to write our to a pickle file.')

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


def evaluate_once_and_log(supervisor, ops_to_run,
                          result_keys, num_iter, subset,
                          global_step, label_mapping):
  """Evaluates the model and serializes qualitative results.

  Run evaluation once on the specified split of the data and compute some
  metrics indicating how well we are doing. Also stores some qualitative
  results and images to a pickle file.

  Args:
    supervisor: tf.train.Supervisor object.
    ops_to_run: dictionary mapping names to ops to run in the session.
    result_keys: dict with key 'list' or `Tensor` and values list of
      outputs we get from the session, based on whether they are of
      type list of `Tensor` or `Tensor`.
    num_iter: Number of iterations to run evaluation for.
    subset: list, subset of num_iter * batch_size values which are to be
      stored to disk.
    global_step: Int, global step for evaluation.
    label_mapping: LabelMap object.
  """

  all_results = {
      key: []
      for type_result in result_keys for key in result_keys[type_result]
  }

  all_results_subset = {}

  with supervisor.managed_session(start_standard_services=True) as sess:
    for eval_iter in range(num_iter):

      results = sess.run(ops_to_run)
      results_from_session = dict(**results)

      # TODO(vrama)
      # Evaluation: Run the supervised model for inference.
      # TODO(vrama): Make results_from_session keys and result_keys monolithic.
      all_results['gen_images_cond_label'].append(
          results_from_session['predicted_image_from_label'])

      all_results['gen_images_cond_image'].append(
          results_from_session['predicted_image_from_image'])

      all_results['gen_images_cond_image_label'].append(
          results_from_session['predicted_image_from_image_label'])

      all_results['gt_images'].append(results_from_session['input_images'])

      if not all_results['gt_labels']:
        for gt_label, predicted_label in zip(
            results_from_session['input_labels'],
            results_from_session['predicted_labels_from_image']):
          all_results['gt_labels'].append([gt_label])
          all_results['predicted_labels'].append([predicted_label])
      else:
        for index, _ in enumerate(all_results['gt_labels']):
          all_results['gt_labels'][index].append(
              results_from_session['input_labels'][index])
        for index, _ in enumerate(all_results['predicted_labels']):
          all_results['predicted_labels'][index].append(
              results_from_session['predicted_labels_from_image'][index])

      # TODO(vrama): Make a class to manage accuracy ops, and then the naming
      # can automatically become consistent in Accuracy_"${attribute}" below.
      tf.logging.info('Finished %d of %d. Accuracy %f', eval_iter + 1, num_iter,
                   results_from_session['Accuracy_overall'])

  for key, result in all_results.iteritems():
    if key in result_keys['list']:
      all_results_subset[key] = []
      for index, attribute in enumerate(all_results[key]):
        all_results[key][index] = np.hstack(attribute)
        all_results_subset[key].append(all_results[key][index][subset])
    else:
      all_results[key] = np.vstack(result)
      all_results_subset[key] = all_results[key][subset]

  guessing_rates = []
  top_value_rates = []
  for field in sorted(results_from_session):
    if field.startswith('Accuracy_'):
      accuracy = results_from_session[field]
      attribute = field.replace('Accuracy_', '')
      if attribute == 'overall':
        guessing_rate = np.mean(guessing_rates)
        top_value_rate = np.mean(top_value_rates)
      else:
        attribute_id = label_mapping.attribute_for_name(attribute)
        gt_labels = all_results['gt_labels'][attribute_id]
        guessing_rate = 1.0 / label_mapping.count_for_label[attribute]
        top_value_rate = np.max(np.bincount(gt_labels)) / float(
            gt_labels.shape[-1])
        guessing_rates.append(guessing_rate)
        top_value_rates.append(top_value_rate)
      tf.logging.info('Attribute %s has accuracy %.4f\n'
                   '\t\t\t\t\t\t(chance: %.4f, top: %.4f)',
                   attribute, accuracy, guessing_rate, top_value_rate)

  #qj(all_results['gt_images'], 'gt_images', n=10)
  #qj(all_results['gen_images_cond_image'], 'gen_images_cond_image', n=10)
  #qj(all_results['gen_images_cond_image_label'],
  #   'gen_images_cond_image_label', n=10)
  #qj(all_results['gen_images_cond_label'], 'gen_images_cond_label', n=10)

  result_pickle_file = os.path.join(FLAGS.eval_dir, 'results'
                                    '-' + global_step + '.p')

  with tf.gfile.Open(result_pickle_file, 'w') as f:
    pickle.dump(all_results_subset, f)

  gt_labels = zip(*all_results_subset['gt_labels'])
  pred_labels = zip(*all_results_subset['predicted_labels'])

  n_attrs = 6  # Maximum number of attributes we expect to be able to show.
  annotations = [
      [
          {
              'label': (':'.join([str(gtl) for gtl in gt_labels[i][:n_attrs]])),
              'color': '#000000',
          },
          {
              'label': (':'.join([str(pl) for pl in pred_labels[i][:n_attrs]])),
              'color': image_utils.get_color(pred_labels[i], gt_labels[i]),
          },
      ]
      for i in range(len(subset))
  ]

  for key in all_results_subset:
    if 'images' in key:
      raw_images = all_results_subset[key]
      if raw_images.shape[-1] in [1, 3, 1536]:
        images = [raw_images]
        tags = ['']
      else:
        channels = raw_images.shape[-1] // 2
        assert raw_images.shape[-1] == channels * 2, 'Unexpected image shape'
        images = [raw_images[..., :channels], raw_images[..., channels:]]
        tags = ['_means', '_samples']

      for image, tag in zip(images, tags):
        images_file = os.path.join(FLAGS.eval_dir, '%s__%s%s_%s' %
                                   (FLAGS.results_tag, key, tag, global_step))
        if image.shape[-1] == 1536:
          image = np.reshape(image, [-1, 16, 32, 3])
          image = np.pad(image, [[0, 0], [8, 8], [0, 0], [0, 0]], 'constant')
        image_utils.plot_images(
            image,
            n=int(np.ceil(np.sqrt(len(subset)))),
            annotations=annotations,
            filename=images_file)


def evaluate_loop():
  """Run evaluation in a loop."""
  config = configuration.get_configuration()
  config.batch_size = FLAGS.batch_size

  label_mapping = label_map.LabelMap(config.label_map_json)

  g = tf.Graph()
  with g.as_default():
    tf.set_random_seed(123)
    if FLAGS.model_type == 'multi':
      vae = ConvolutionalMultiVae(config,
                                  mode=FLAGS.split_name,
                                  split_name=FLAGS.split_name)
    elif FLAGS.model_type == 'single' or 'kronecker':
      raise NotImplementedError

    vae.build_model()

    # Build some utility operations for various things we would like to do with
    # the joint model.
    image_generation_op = vae.generate_images_conditioned_label(
        mean_or_sample='both')

    image_generation_op_conditioned_both = (
        vae.generate_image_conditioned_image_label(mean_or_sample='both'))

    unsup_image_generation_op = vae.generate_image_conditioned_image(
        mean_or_sample='both')

    image_classification_op = vae.generate_label_conditioned_image(
        num_samples_predictive_dist=FLAGS.num_samples_predictive_dist)

    accuracy_names, update_op_list = utils.add_accuracy_ops(
        image_classification_op, vae.labels, label_mapping.attributes,
        mode=FLAGS.split_name)

    saver = tf.train.Saver(var_list=tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES))

    tensor_result_keys = [
        'gen_images_cond_label', 'gen_images_cond_image',
        'gen_images_cond_image_label', 'gt_images', 'gt_labels'
    ]
    # Some of the results are not just tensors, but lists of tensors, like
    # attribute labels, specified separately below.
    list_result_keys = ['predicted_labels', 'gt_labels']

    result_keys = {'list': list_result_keys, 'tensor': tensor_result_keys}

    ops_to_run = [
        image_classification_op, image_generation_op, unsup_image_generation_op,
        image_generation_op_conditioned_both, vae.images, vae.labels
    ]

    ops_to_run += update_op_list

    result_names = ([
        'predicted_labels_from_image',
        'predicted_image_from_label',
        'predicted_image_from_image',
        'predicted_image_from_image_label',
        'input_images',
        'input_labels'
    ] + accuracy_names)

    ops_to_run = {name: op for name, op in zip(result_names, ops_to_run)}

    num_iter = int(math.ceil(vae.num_samples / FLAGS.batch_size))

    # Save a subset of results to disk.
    subset = np.random.choice(
        num_iter * FLAGS.batch_size,
        size=FLAGS.num_result_datapoints,
        replace=False)

    for checkpoint_path in slim.evaluation.checkpoints_iterator(
        FLAGS.checkpoint_dir, FLAGS.eval_interval_secs):
      restore_fn, global_step = create_restore_fn(checkpoint_path, saver)
      sv = tf.train.Supervisor(
          graph=g, init_fn=restore_fn, logdir=FLAGS.eval_dir, saver=None)
      evaluate_once_and_log(sv,
                            ops_to_run,
                            result_keys,
                            num_iter,
                            subset,
                            global_step,
                            label_mapping)


def main(_):
  np.random.seed(42)
  assert FLAGS.model_type in _MODEL_LIST, 'Invalid model specified.'
  if FLAGS.model_type == 'single':
    raise ValueError('Cannot run this eval for single model type.')
  assert FLAGS.checkpoint_dir is not None, ('Please specify a checkpoint '
                                            'directory.')
  assert FLAGS.eval_dir is not None, 'Please specify an evaluation directory.'

  evaluate_loop()


if __name__ == '__main__':
  tf.app.run()
