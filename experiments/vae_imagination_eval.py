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

"""Evaluate imagination function in a multimodal VAE.

A good imagination function has the following desiderata:
  1. Correctness: q(z| y) should lead to a point in 'z' which corresponds
    to the concept "y".
  2. Coverage: q(z| y) should cover the variation of the concept 'y'
    based on how specific the concept is.
  3. Compositionality: We should be able to generalize to unseen/novel
    label combinations.
  4. Comprehnsability: We should be able to generate an image that
    captures the enssence of what we conditioned on.

NOTE: Currently this eval is specific to deepmind 2d shapes and labels.

Author: vrama@
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import namedtuple
import cPickle as pickle
import math
import os

import numpy as np
from scipy import stats
import tensorflow as tf

from datasets import label_map

from experiments import image_utils
from experiments import configuration
from experiments import vae_eval  # pylint: disable=unused-import
from experiments.convolutional_multi_vae import ConvolutionalMultiVae
from experiments import comprehensibility

from joint_vae import utils
from third_party.interpolate import interpolate


flags = tf.flags
gfile = tf.gfile
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('iclr_results_path', '/tmp/results',
                    'Directory where all the NIPS\'17 Imagination results will '
                    'be stored.')

tf.app.flags.DEFINE_integer('max_global_step', 500000, 'Maximum global step')

tf.app.flags.DEFINE_string('distance_metric', 'kl', 'Options: \'kl\' or \'l2\'.'
                    'Metric to use for retrieval.')

tf.app.flags.DEFINE_string('input_queries', '',
                    'A set of queries and associated masks for eval. If not '
                    'specified, the labels from validation are used to query.')

tf.app.flags.DEFINE_boolean(
    'visualize_means', True,
    'If true, visualize the mean images, if false visualize all images.')

tf.app.flags.DEFINE_boolean(
    'run_interpolation', True,
    'If true, run interpolation between pairs of queries.')

tf.app.flags.DEFINE_boolean(
    'evaluate_once', False, 'If true just evaluate once and break.')

tf.app.flags.DEFINE_integer('num_images_comprehension_eval', 10,
                     'Number of samples to draw '
                     'from the model for performing comprehension evaluation.')

tf.app.flags.DEFINE_integer('interp_samples', 100,
                     'Number of pairs of queries to interpolate between.')

tf.app.flags.DEFINE_integer('interp_steps', 9,
                     'Number of interpolation steps between queries.')

Query = namedtuple('Query', ['label', 'mask'])
RetrievalDatapoint = namedtuple('RetrievalDatapoint',
                                ['gaussian', 'label', 'latent'])
QueryDatapoint = namedtuple('QueryDatapoint', ['gaussian', 'label'])

FeatureExtractionOps = namedtuple('FeatureExtractionOps', [
    'latent_conditioned_image', 'labels', 'true_latents', 'saver'
])

InferenceOps = namedtuple('InferenceOps', [
    'inference_label_ph', 'ignore_label_mask_ph', 'latent_conditioned_label',
    'images_generation_op', 'z_ph', 'p_x_z_mean', 'p_x_z_sample', 'saver'
])


def construct_feature_extraction_graph(config):
  """Construct the graph for the model in feature extration mode.

  We set the model in validation mode, which means that we turn off random
  shuffling. We then read from a FIFO queue, extracting representations for each
  image in say the validation set.

  Args:
    config: An object of class configuration.get_configuration()
  Returns:
    g_features: A tf.Graph() object.
    vae: A ConvolutionalMultiVae/ConvolutionalVae/KroneckerMultiVae object.
    temp_saver: A tf.train.Saver() object
    num_iter: Int, number of iterations to run the graph for.
  """

  g_features = tf.Graph()
  with g_features.as_default():
    tf.set_random_seed(123)
    if FLAGS.model_type == 'multi':
      vae = ConvolutionalMultiVae(
          config,
          mode=FLAGS.split_name,
          split_name=FLAGS.split_name,
          add_summary=False)
    elif FLAGS.model_type == 'single' or 'kronecker':
      raise NotImplementedError

    vae.build_model()

    latent_conditioned_image = vae.latent_conditioned_image
    gt_label = vae.labels
    true_latents = vae.true_latents
    temp_saver = tf.train.Saver(
        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    feature_extraction_ops = FeatureExtractionOps(
        latent_conditioned_image, gt_label, true_latents, temp_saver)
    print(vae.num_samples)

    num_iter = int(math.ceil(vae.num_samples / FLAGS.batch_size))
  return g_features, feature_extraction_ops, temp_saver, num_iter


def construct_inference_graph(config):
  """Construct the graph for the model in inference mode.

  Inference mode provides utilities to pass in inputs to the model as
  placeholders, enabling us to interact with the model freely.

  Args:
    config: An object of class configuration.get_configuration()
  Returns:
    g_inference: A tf.Graph() object.
    vae: A ConvolutionalMultiVae/ConvolutionalVae/KroneckerMultiVae object.
  """
  g_inference = tf.Graph()
  with g_inference.as_default():
    tf.set_random_seed(123)

    if FLAGS.model_type == 'multi':
      vae = ConvolutionalMultiVae(
          config, mode='inference', split_name=FLAGS.split_name)
    elif FLAGS.model_type == 'single' or 'kronecker':
      raise NotImplementedError
    vae.build_model()

    inference_label_ph = vae.inference_label_ph
    ignore_label_mask_ph = vae.ignore_label_mask_ph
    latent_conditioned_label = vae.latent_conditioned_label

    mean_or_sample = 'sample'
    if FLAGS.visualize_means:
      mean_or_sample = 'both'

    images_generation_op = vae.generate_images_conditioned_label(
        FLAGS.num_images_comprehension_eval, mean_or_sample=mean_or_sample)

    z_ph = tf.placeholder(dtype=tf.float32, shape=[None, config.num_latent])
    p_x_z, _ = vae.model.predict(z_ph)

    saver = tf.train.Saver()

    inference_ops = InferenceOps(inference_label_ph, ignore_label_mask_ph,
                                 latent_conditioned_label, images_generation_op,
                                 z_ph, p_x_z.mean(), p_x_z.sample(), saver)

  return g_inference, inference_ops


def extract_features(feature_ops, inference_ops, g_features, g_inference, saver,
                     num_iter, checkpoint_path):
  """Extract attribute and image features for retrieval.

  The function first constructs a graph to process the validation or test set
  as specified and extracts the features for each image. Then it processes
  either a set of queries provided externally (if available) or the labels from
  validation as the queries and extracts the representations for them and
  finally retruns the image representations and the label/query representations.

  Args:
    feature_ops: An object of FeatureExtractionOps which contains some ops to
      run during feature extraction.
    inference_ops: An object of InferenceOps which contains some ops to
      run during feature extraction.
    g_features: A tf.Graph() object in which the feature extraction has been
      instantiated.
    g_inference: A tf.Graph() object in which the vae_inference model has been
      instantiated.
    saver: a tf.train.Saver() object used for saving back the checkpoint we loaded
      from after extracting features, since checkpoints get deleted
      periodically.
    num_iter: int, number of iterations to go through model evaluation when
      extracting features for the whole dataset.
    checkpoint_path: The path to the checkpoint to load the model parameters
      from.
  Returns:
    image_latents_and_labels: A list of objects of class RetrievalDatapoint
    query_latents_and_labels: A list of objects of class RetrievalDatapoint
  """
  with g_features.as_default() as g:
    latent_conditioned_image = feature_ops.latent_conditioned_image
    gt_label = feature_ops.labels
    true_latents = feature_ops.true_latents
    init_fn, global_step = utils.create_restore_fn(checkpoint_path,
                                                   feature_ops.saver)
    sv = tf.train.Supervisor(
        graph=g,
        init_fn=init_fn,
        logdir=FLAGS.eval_dir,
        saver=None,
        summary_op=None,
        summary_writer=None)
    with sv.managed_session(start_standard_services=True) as sess:
      image_latents_and_labels = []

      if not FLAGS.input_queries:

        for batch_iter in range(num_iter):
          tf.logging.info('Extracting latent representations for %d of %d',
                       batch_iter + 1, num_iter)
          gaussian_means, gaussian_stds, labels, latents = sess.run([
              latent_conditioned_image.density.loc,
              latent_conditioned_image.density.scale, gt_label, true_latents
          ])
          gaussian_means = np.split(gaussian_means, len(gaussian_means), axis=0)
          gaussian_stds = np.split(gaussian_stds, len(gaussian_stds), axis=0)
          labels = utils.unbatchify_list(labels)
          labels = [tuple(x) for x in labels]
          latents = np.split(latents, len(latents), axis=0)
          for gauss_mean, gauss_std, label, latent in zip(
              gaussian_means, gaussian_stds, labels, latents):
            gaussian = utils.Gaussian(gauss_mean, gauss_std)
            image_latents_and_labels.append(
                RetrievalDatapoint(gaussian, label, latent))

      # Save back the checkpoint we loaded because it might have vanished, as
      # TF deletes checkpoints which are too old.
      temp_checkpoint_path = os.path.join(FLAGS.eval_dir,
                                          'eval_model.ckpt-%s' % (global_step))
      saver.save(
          sess,
          save_path=temp_checkpoint_path,
          latest_filename='checkpoint_temp')

    # The query generator produces output in the following format:
    # {
    #   'queries': list of queries, each an np.array of [1, num_attributes]
    #       with each entry in the array setting the label value for the
    #       attribute.
    #   'masks': list of masks, each an np.array of [1, num_attributes]
    #       with 0/1 values specifying attributes to select and which to
    #       ignore.
    # }
    # Each entry in masks[i] correponds to the entry in queries[i]
    if FLAGS.input_queries:
      tf.logging.info('****Using input query file %s.****', FLAGS.input_queries)

      with tf.gfile.Open(FLAGS.input_queries, 'r') as f:
        queries_and_masks = pickle.load(f)
        queries = queries_and_masks['queries']
        masks = queries_and_masks['masks']
    # If no query file is specified, use the labels from the validation set as
    # the queries.
    else:
      tf.logging.info('Defaulting to using queries from SSTables.')
      queries = set([x.label for x in image_latents_and_labels])
      queries = [np.expand_dims(np.array(x), axis=0) for x in queries]
      masks = np.ones((len(queries), queries[0].shape[1]))
      masks = np.split(masks, len(masks), axis=0)

    with g_inference.as_default() as g:
      inference_label_ph = inference_ops.inference_label_ph
      ignore_label_mask_ph = inference_ops.ignore_label_mask_ph
      latent_conditioned_label = inference_ops.latent_conditioned_label
      images_generation_op = inference_ops.images_generation_op
      z_ph = inference_ops.z_ph
      pxz_mu = inference_ops.p_x_z_mean
      pxz_sample = inference_ops.p_x_z_sample

      init_fn, global_step = utils.create_restore_fn(temp_checkpoint_path,
                                                     inference_ops.saver)
      sv = tf.train.Supervisor(
          graph=g,
          init_fn=init_fn,
          logdir=FLAGS.eval_dir,
          saver=None,
          summary_op=None,
          summary_writer=None,)

      with sv.managed_session(start_standard_services=True) as sess:
        query_latents_and_labels = []
        tf.logging.info('Extracting representations for queries and computing '
                        'posterior entropies.')

        queries_and_generated_images = []
        entropies_for_queries = []

        for query, mask in zip(queries, masks):
          gaussian_mean, gaussian_std, generated_images_for_query = sess.run(
              [
                  latent_conditioned_label.density.loc,
                  latent_conditioned_label.density.scale,
                  images_generation_op,
              ],
              feed_dict={inference_label_ph: query,
                         ignore_label_mask_ph: mask})
          gaussian_query = utils.Gaussian(gaussian_mean, gaussian_std)

          entropy_for_query = 0.5 * np.log(np.prod(2*np.pi*np.e*np.square(gaussian_std)))
          # Need to compute the entropy of this gaussian
          current_query = Query(query, mask)
          entropies_for_queries.append({'Query': current_query,
                                      'Entropy': entropy_for_query})


          # FIX Decoder Size Error: If we misspecified the decoder size, then
          # we subsample the channels for further evaluation.
          if not isinstance(generated_images_for_query, list):
            generated_images_for_query = [generated_images_for_query]

          if FLAGS.dataset=='affine_mnist' and generated_images_for_query[0].shape[-1] == 6:
            use_gen_images_for_query = []
            for gen_image in generated_images_for_query:
              use_gen_images_for_query.append(gen_image[:, :, :, [0, 3]])
            generated_images_for_query = use_gen_images_for_query

          # We typically draw multiple samples for each image to get robustness.
          queries_and_generated_images.append((current_query,
                                                 generated_images_for_query))

          query_latents_and_labels.append(
              QueryDatapoint(gaussian_query, current_query))

        # Store / serialize the entropy results

        pickle_name = (FLAGS.results_tag + '_' + FLAGS.split_name + '_' +
                   '_'.join(FLAGS.eval_dir.split('/')[-2:]) +
                   FLAGS.distance_metric +'_entropy_' + '_%s.p' % global_step)
        output_file = os.path.join(FLAGS.iclr_results_path, pickle_name)

        tf.logging.info('Writing the entropy file to %s.' % (output_file))
        with tf.gfile.Open(output_file, 'w') as f:
          pickle.dump(entropies_for_queries, f)

        if FLAGS.run_interpolation:
          queries_and_interpolated_images = interpolate_queries(
            sess, query_latents_and_labels, z_ph, pxz_mu, pxz_sample)
        else:
          queries_and_interpolated_images = None

        tf.logging.info('Extracted representations for all %d queries',
                     len(queries))
        image_latents_and_labels = None

  return (image_latents_and_labels, query_latents_and_labels,
          queries_and_generated_images, queries_and_interpolated_images,
          global_step)


def interpolate_queries(sess,
                        query_latents_and_labels,
                        z_ph,
                        mean_images_op,
                        sample_images_op,
                        also_sample=True,
                        interp_samples=FLAGS.interp_samples,
                        interp_steps=FLAGS.interp_steps):
  """Interpolate between all pairs of queries and generate images."""
  qs_and_imgs = []
  interps = []
  rng = np.random.RandomState(1)
  if len(query_latents_and_labels) > 24:
    queries = rng.choice(
        len(query_latents_and_labels),
        size=min(len(query_latents_and_labels), interp_samples * 2),
        replace=False)
    start_queries = queries[:len(queries) // 2]
    end_queries = queries[len(queries) // 2:]
    for start_q_i, end_q_i in zip(start_queries, end_queries):
      start_q = query_latents_and_labels[start_q_i]
      end_q = query_latents_and_labels[end_q_i]
      interps.append(((start_q, end_q), interpolate.do_interpolation(
          interpolate.slerp, start_q.gaussian.mean[0], end_q.gaussian.mean[0],
          interp_steps)))
      if also_sample:
        for _ in range(3):
          start_v = rng.normal(start_q.gaussian.mean[0],
                               start_q.gaussian.std[0])
          end_v = rng.normal(end_q.gaussian.mean[0], end_q.gaussian.std[0])
          interps.append(((start_q, end_q), interpolate.do_interpolation(
              interpolate.slerp, start_v, end_v, interp_steps)))
  else:
    start_queries = []
    end_queries = []
    for i, start_q in enumerate(query_latents_and_labels):
      for j, end_q in enumerate(query_latents_and_labels[i + 1:]):
        start_queries.append(i)
        end_queries.append(i + j + 1)
        interps.append(((start_q, end_q), interpolate.do_interpolation(
            interpolate.slerp, start_q.gaussian.mean[0], end_q.gaussian.mean[0],
            interp_steps)))
        if also_sample:
          for _ in range(3):
            start_v = rng.normal(start_q.gaussian.mean[0],
                                 start_q.gaussian.std[0])
            end_v = rng.normal(end_q.gaussian.mean[0], end_q.gaussian.std[0])
            interps.append(((start_q, end_q), interpolate.do_interpolation(
                interpolate.slerp, start_v, end_v, interp_steps)))

  for i, (queries, interpolated_means) in enumerate(interps):
    if i % 100 == 0:
      print(i, 'Running interpolation')

    z_means, z_samples = sess.run(
        [mean_images_op, sample_images_op],
        feed_dict={z_ph: interpolated_means})

    # Add a grey line between each image.
    z_means[:-1, :, -1, :] = 0.5
    z_samples[:-1, :, -1, :] = 0.5

    query_index = i // 4 if also_sample else i
    start_label = query_latents_and_labels[start_queries[query_index]].label
    start_label = ':'.join(
        str(l) if start_label.mask[0][j] == 1 else '_'
        for j, l in enumerate(start_label.label[0]))
    end_label = query_latents_and_labels[end_queries[query_index]].label
    end_label = ':'.join(
        str(l) if end_label.mask[0][j] == 1 else '_'
        for j, l in enumerate(end_label.label[0]))

    labels = [[dict(label='', color='#000000')]
              for _ in range(interp_steps + 1)]
    labels[0][0]['label'] = start_label
    labels[-1][0]['label'] = end_label

    filename_suffix = ('%04d_sl%s_el%s' %
                       (i, start_label.replace(':', '_').replace('?', 'q'),
                        end_label.replace(':', '_').replace('?', 'q')))

    image_utils.plot_images(
        z_means,
        n=len(z_means),
        annotations=labels,
        filename=os.path.join(FLAGS.iclr_results_path, '%s_interp_mean_%s' %
                              (FLAGS.results_tag, filename_suffix)))

    # image_utils.plot_images(
    #     z_samples,
    #     n=len(z_samples),
    #     annotations=labels,
    #     filename=os.path.join(FLAGS.iclr_results_path, '%s_interp_samp_%s' %
    #                           (FLAGS.results_tag, filename_suffix)))

    qs_and_imgs.append((queries, z_means, z_samples))

  return qs_and_imgs


def evaluate_comprehensibility(queries_and_gen_images,
                               comprehensibility_eval,
                               global_step,
                               num_classes_per_attribute,
                               metric_results,
                               summary_writer=None):
  """Evaluate comprehensibility of generated images.

  Comprehensibility is evaluated by passing generated images through a
  a pretrained classifier which tries to check if the images that were generated
  yeild the same predictions as what we conditioned on.

  Args:
    queries_and_gen_images: Set of queries and generated images, a tuple of
      namedtuple query and list of np.array images.
    comprehensibility_eval: an instance of class Comprehensibility, which
      is a helper for evaluating comprehensibility.
    global_step: Global step at which we are doing the evaluation.
    num_classes_per_attribute: List of ints.
    metric_results: Stores the results of all the metrics indexed by
      query_label, and metric name. For example (4, cluster_recall_100) means
      we are asking for results at the leaf node (for MNISTa) for cluster recall
      @100 metric.
    summary_writer: An object of tf.summary.FileWriter to write summaries.
  Returns:
    metric_results: metric results with the comprehensability numbers.
  """
  tf.logging.info('Starting comprehensibility eval.')

  all_query_data = []

  for query, gen_image_list in queries_and_gen_images:
    comprehensibility_scores = []
    visualization_images = []
    # Each query can have multiple images drawn for it, which is stored in
    # a list.
    all_predicted_labels = []
    # Compute the reference uniform distributions.
    empirical_histograms = []
    for attribute in num_classes_per_attribute:
      empirical_histograms.append(np.zeros(attribute))

    for gen_image in gen_image_list:
      compre_eval, predicted_label, vis_image = comprehensibility_eval.evaluate(
          gen_image, query.label, query.mask)
      visualization_images.append(vis_image)
      all_predicted_labels.append(predicted_label)

      for attribute in xrange(predicted_label.shape[-1]):
        empirical_histograms[attribute][predicted_label[attribute]] += 1

      comprehensibility_scores.append(compre_eval.sum())
    # Normalize the emprical histogram into a distribution, and compute the KL
    # divergence against a reference uniform distribution.
    all_kld = []
    all_jsd_sim = []

    # Consolidated JSD is jenson shannon computed across all attributes --
    # observed as well as unobserved.
    all_consolidated_jsd_sim = []
    for index, _ in enumerate(empirical_histograms):
      attribute_histogram = empirical_histograms[index] / len(gen_image_list)

      # Unspecified dimensions should be ignored.
      if query.mask[0, index] == 0:
        # Compute the jenson shannon divergence.
        kld = -1 * stats.entropy(attribute_histogram) + np.log(
            num_classes_per_attribute[index])
        reference_distribution = np.ones(
            num_classes_per_attribute[index]) / num_classes_per_attribute[index]
      elif query.mask[0, index] == 1:
        # For observed queries the reference distribution is a one hot vector
        # with the observed bit on.
        reference_distribution = np.zeros(num_classes_per_attribute[index])
        reference_distribution[query.label[0, index]] = 1
      jenson_shannon_m = (
          0.5 * attribute_histogram + 0.5 * reference_distribution)
      # Confusingly, stats.entropoy computes KLD when given two inputs.
      jsd = 0.5 * (stats.entropy(attribute_histogram, jenson_shannon_m) +
                   stats.entropy(reference_distribution, jenson_shannon_m))
      jsd_sim = 1 - jsd

      assert jsd >= 0 and jsd <= 1, 'Invalid value for JSD.'
      if query.mask[0, index] == 0:
        assert kld >= 0, 'Invalid value for KL divergence.'
        all_kld.append(kld)
        all_jsd_sim.append(jsd_sim)
      all_consolidated_jsd_sim.append(jsd_sim)

    assert len(all_kld) == query.mask.shape[-1] - np.sum(
        query.mask), 'Check the KLD array, wrong dimensions.'
    assert len(all_jsd_sim) == query.mask.shape[-1] - np.sum(
        query.mask), 'Check the JSD array, wrong dimensions.'

    if not all_kld:
      overall_kld = 0
    else:
      overall_kld = np.mean(all_kld)

    if not all_jsd_sim:
      overall_jsd_sim = 1
    else:
      overall_jsd_sim = np.mean(all_jsd_sim)

    overall_consolidated_jsd_sim = np.mean(all_consolidated_jsd_sim)

    # Compute the KL(p, q) we see in the predictions, relative to the unifrom
    # distribution. q in this case is the predictions we make. p is the uniform
    # distribution.

    metric_results[(
        query.mask.sum(),
        'comprehensibility')].append(np.mean(comprehensibility_scores))
    metric_results[(query.mask.sum(),
                    'parametric_coverage')].append(overall_kld)
    metric_results[(query.mask.sum(),
                    'parametric_jsd_sim')].append(overall_jsd_sim)
    metric_results[(query.mask.sum(),
                    'parametric_consolidated_jsd_sim')].append(
                        overall_consolidated_jsd_sim)

    # Only store the first 10 images to disk.
    subset_predicted_labels = all_predicted_labels[:10]
    visualization_images = visualization_images[:10]
    comprehensibility_scores = comprehensibility_scores[:10]

    # Serialize information about the query to disk.
    all_query_data.append(
        (query.label, query.mask, subset_predicted_labels, visualization_images,
         comprehensibility_scores, overall_kld, overall_jsd_sim, overall_consolidated_jsd_sim))

  for k, v in metric_results.iteritems():
    if (k[1] == 'comprehensibility' or k[1] == 'parametric_coverage' or
        k[1] == 'parametric_jsd_sim' or k[1] == 'parametric_consolidated_jsd_sim'):
      tf.logging.info('%s error: %f', str(k[0]) + '_' + k[1], np.mean(v))
      if summary_writer:
        utils.add_simple_summary(summary_writer,
                                 np.mean(v), str(k[0]) + '_' + k[1],
                                 global_step)

  if len(all_query_data) > 1000:
    # Only store results for a maximum of 1000 queries.
    subset = np.random.choice(range(len(all_query_data)), 1000)
    all_query_data = [all_query_data[idx] for idx in subset]
  # Serialize the results to disk.
  tf.logging.info('Writing evaluation metadata to disk.')

  pickle_name = (FLAGS.results_tag + '_' + FLAGS.split_name + '_' +
                 '_'.join(FLAGS.eval_dir.split('/')[-2:]) +
                 'parametric_eval_metadata_%s.p' % global_step)
  full_result_filepath = os.path.join(FLAGS.iclr_results_path, pickle_name)

  with tf.gfile.Open(full_result_filepath, 'w') as f:
    pickle.dump(all_query_data, f)

  return metric_results


def extract_features_and_eval_imagination(summary_writer):
  """Extract features for retrieval and perform imagination evals.

  Args:
    summary_writer: an instance of tf.summary.FileWriter
  """
  config = configuration.get_configuration()
  config.batch_size = FLAGS.batch_size

  label_mapping = label_map.LabelMap(config.label_map_json)

  g_features, feature_ops, temp_saver, num_iter = (
      construct_feature_extraction_graph(config))

  g_inference, inference_ops = construct_inference_graph(config)

  comprehensibility_eval = comprehensibility.Comprehensibility(
      config.comprehensibility_ckpt,
      config,
      config.num_classes_per_attribute,
      config.image_size,
      FLAGS.visualize_means,
      attribute_names=label_mapping.attributes,
      hidden_units=config.comprehensibility_hidden_units)

  for checkpoint_path in slim.evaluation.checkpoints_iterator(
      FLAGS.checkpoint_dir, FLAGS.eval_interval_secs):

    (image_latents_and_labels, query_latents_and_labels, queries_and_gen_images,
     queries_and_interp_images, global_step) = (extract_features(
         feature_ops, inference_ops, g_features, g_inference, temp_saver,
         num_iter, checkpoint_path))

    # Stores the results of all the metrics indexed by query_label, and
    # metric name. For example (4, cluster_recall_100) means that we are
    # asking for results at the leaf node (for MNISTa) for cluster recall
    # @100 metric.
    all_metric_results = defaultdict(list)

    #tf.logging.info('Performing Retrieval.')
    #all_metric_results = retrieval_utils.text_to_image_retrieval(
    #    image_latents_and_labels,
    #    query_latents_and_labels,
    #    global_step,
    #    FLAGS.eval_dir,
    #    all_metric_results,
    #    distance_metric=FLAGS.distance_metric,
    #    summary_writer=summary_writer)

    tf.logging.info('Evaluating Comprehensibility.')
    all_metric_results = evaluate_comprehensibility(
        queries_and_gen_images,
        comprehensibility_eval,
        global_step,
        config.num_classes_per_attribute,
        all_metric_results,
        summary_writer=summary_writer)

    if all_metric_results is None:  # Never True
      all_metric_results['queries_and_interpolated_images'] = (
          queries_and_interp_images)

    # Dump the results into a Pickle.
    pickle_name = (FLAGS.results_tag + '_' + FLAGS.split_name + '_' +
                   '_'.join(FLAGS.eval_dir.split('/')[-2:]) +
                   FLAGS.distance_metric + '_%s.p' % global_step)
    full_result_filepath = os.path.join(FLAGS.iclr_results_path, pickle_name)
    tf.logging.info('Storing results at %s.', full_result_filepath)
    with tf.gfile.Open(full_result_filepath, 'w') as f:
      pickle.dump(all_metric_results, f)

    if global_step >= FLAGS.max_global_step or FLAGS.evaluate_once:
      print(global_step, 'No longer waiting for a new checkpoint.')
      break


def main(_):
  if not FLAGS.iclr_results_path:
    raise ValueError('iclr_results_path must be specified!')

  iclr_results_path = FLAGS.iclr_results_path
  split_names = ['val', 'test']
  for i, split_name in enumerate(split_names):
    np.random.seed(42)

    FLAGS.split_name = split_name
    print(FLAGS.split_name)

    if FLAGS.input_queries:
      FLAGS.input_queries = FLAGS.input_queries.replace(
          '_' + split_names[1 - i] + '_', '_' + split_name + '_')
    print(FLAGS.input_queries)

    FLAGS.iclr_results_path = iclr_results_path + '_' + FLAGS.split_name
    print(FLAGS.iclr_results_path)
    if not tf.gfile.Exists(FLAGS.iclr_results_path):
      tf.logging.info('Creating the ICLR results directory %s',
                   FLAGS.iclr_results_path)
      tf.gfile.MakeDirs(FLAGS.iclr_results_path)

    assert FLAGS.checkpoint_dir is not None, ('Please specify a checkpoint '
                                              'directory.')
    assert FLAGS.eval_dir is not None, 'Please specify an evaluation directory.'

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    extract_features_and_eval_imagination(summary_writer)


if __name__ == '__main__':
  tf.app.run()
