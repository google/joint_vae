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

"""Run experiments on vae geometry.

Tests models related to NIPS 17 paper submission.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import traceback

from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import datasets  # Nope

from joint_vae import bivcca_loss
from joint_vae import encoder_decoder
from joint_vae import jmvae_loss
from joint_vae import joint_vae
from joint_vae import multimodal_elbo_loss

from joint_vae import utils

import image_utils

flags = tf.flags
tf.app.flags.DEFINE_string('output_path', '', 'Directory where to write the images.')
tf.app.flags.DEFINE_bool('sparse_image_writes', False,
                  'Set to True to write fewer images')

tf.app.flags.DEFINE_integer('task', 0, 'Task id')
tf.app.flags.DEFINE_integer('num_tasks', 1, 'Number of tasks launched')
tf.app.flags.DEFINE_integer('num_steps', 11000, 'Number of steps to train for.')

FLAGS = tf.app.flags.FLAGS

gfile = tf.gfile


def main(_):
  # pylint: disable=unused-variable
  # Experiment parameters
  num_steps = FLAGS.num_steps
  batch_size = 64

  # Visualization parameters
  sample_range = 3
  x_sample_steps = 32
  y_sample_steps = 256
  base_dir = FLAGS.output_path
  sparse_image_writes = FLAGS.sparse_image_writes

  if not os.path.isdir(FLAGS.output_path):
    raise ValueError('"%s" is not a directory.' % FLAGS.output_path)

  # Model parameters
  mlp_dims = 512
  latent_dims = 2

  num_ys = [
      2,  # pylint: disable=bad-continuation
      #       3,  # pylint: disable=bad-continuation
  ]

  architectures = [
      #       'multi_vae',  # pylint: disable=bad-continuation
      #       'multi_qxy',  # pylint: disable=bad-continuation
      #       'multi_qx_qy',  # pylint: disable=bad-continuation
      #       'multi_qx_qxy',  # pylint: disable=bad-continuation
      #       'multi_qy_qxy',  # pylint: disable=bad-continuation
      'multi_convo_qx_qy_qxy',
      'multi_qx_qy_qxy',
      'bivcca_qx_qy_qxy',
      'jmvae_qx_qy_qxy',
      #       'jmvae',  # pylint: disable=bad-continuation
      #       'bivcca',  # pylint: disable=bad-continuation
  ]

  is_product_of_experts = [
      #       0,  # pylint: disable=bad-continuation
      1,  # pylint: disable=bad-continuation
  ]

  universal_experts = [
      #       0,  # pylint: disable=bad-continuation
      #       0,  # pylint: disable=bad-continuation
      #       0,  # pylint: disable=bad-continuation
      #       0,  # pylint: disable=bad-continuation
      #       0,  # pylint: disable=bad-continuation
      #       0,  # pylint: disable=bad-continuation
      #       0,  # pylint: disable=bad-continuation
      #       0,  # pylint: disable=bad-continuation
      #       0,  # pylint: disable=bad-continuation
      #       0,  # pylint: disable=bad-continuation
      #       1,  # pylint: disable=bad-continuation
      #       1,  # pylint: disable=bad-continuation
      #       1,  # pylint: disable=bad-continuation
      #       1,  # pylint: disable=bad-continuation
      #       1,  # pylint: disable=bad-continuation
      #       1,  # pylint: disable=bad-continuation
      #       1,  # pylint: disable=bad-continuation
      #       1,  # pylint: disable=bad-continuation
      #       1,  # pylint: disable=bad-continuation
      1,  # pylint: disable=bad-continuation
  ]

  alpha_xs = [
      #       0.01,  # pylint: disable=bad-continuation
      #       0.1,   # pylint: disable=bad-continuation
      1.0,  # pylint: disable=bad-continuation
  ]

  alpha_ys = [
      #       1.0,  # pylint: disable=bad-continuation
      #       5.0,  # pylint: disable=bad-continuation
      #       10.0,  # pylint: disable=bad-continuation
      #       15.0,  # pylint: disable=bad-continuation
      #       20.0,  # pylint: disable=bad-continuation
      #       25.0,  # pylint: disable=bad-continuation
      #       50.0,  # pylint: disable=bad-continuation
      100.0,  # pylint: disable=bad-continuation
      #       400.0,  # pylint: disable=bad-continuation
      #       1000.0,  # pylint: disable=bad-continuation
      #       10000.0,  # pylint: disable=bad-continuation
  ]

  l1_pyzs = [
      #       0.0,  # pylint: disable=bad-continuation
      1e-5,  # pylint: disable=bad-continuation
      #       1e-4,  # pylint: disable=bad-continuation
      #       1e-3,  # pylint: disable=bad-continuation
      #       1e-2,  # pylint: disable=bad-continuation
      #       1e-1,  # pylint: disable=bad-continuation
  ]

  alpha_y_elbo_ys = [
      #       0.0,  # pylint: disable=bad-continuation
      #       0.01,  # pylint: disable=bad-continuation
      #       0.1,  # pylint: disable=bad-continuation
      1.0,  # pylint: disable=bad-continuation
      #       5.0,  # pylint: disable=bad-continuation
      #       10.0,  # pylint: disable=bad-continuation
      #       25.0,  # pylint: disable=bad-continuation
      #       50.0,  # pylint: disable=bad-continuation
      #       100.0,  # pylint: disable=bad-continuation
      #       400.0,  # pylint: disable=bad-continuation
      #       1000.0,  # pylint: disable=bad-continuation
  ]

  stop_gradients = [
      #       0,  # pylint: disable=bad-continuation
      1,  # pylint: disable=bad-continuation
  ]

  # jmvae kl scale param
  jmvae_alphas = [
      #       0.01,  # pylint: disable=bad-continuation
      #       0.1,  # pylint: disable=bad-continuation
      #       1.0,  # pylint: disable=bad-continuation
      10.0,  # pylint: disable=bad-continuation
  ]

  # bivcca tradeoff param, 0 <= mu <= 1
  bivcca_mus = [
      #       0.1,  # pylint: disable=bad-continuation
      #       0.3,  # pylint: disable=bad-continuation
      #       0.5,  # pylint: disable=bad-continuation
      #       0.7,  # pylint: disable=bad-continuation
      0.9,  # pylint: disable=bad-continuation
  ]

  # Will set dropout=True whenever keep_prob > 0.0.
  keep_probs = [
      0.0,
      0.5,  # pylint: disable=bad-continuation
  ]

  models_to_train = [
      model
      for model in itertools.product(num_ys, architectures,
                                     is_product_of_experts, universal_experts,
                                     alpha_xs, alpha_ys, l1_pyzs, keep_probs)
  ]

  num_extra_models_to_train = np.sum([
      len(alpha_y_elbo_ys) * len(stop_gradients) - 1 for m in models_to_train
      if 'multi' in m[1]
  ] + [len(jmvae_alphas) - 1 for m in models_to_train if 'jmvae' in m[1]
      ] + [len(bivcca_mus) - 1 for m in models_to_train if 'bivcca' in m[1]])

  num_models = len(models_to_train) + num_extra_models_to_train
  print('About to train %d models!' % num_models)

  i = 0
  for model in models_to_train:
    (num_y, architecture, product_of_experts, universal_expert, alpha_x,
     alpha_y, l1_pyz, keep_prob) = model

    if 'multi' in architecture:
      final_alpha_y_elbo_ys = alpha_y_elbo_ys
      final_stop_gradients = stop_gradients
    else:
      final_alpha_y_elbo_ys = alpha_y_elbo_ys[:1]
      final_stop_gradients = stop_gradients[:1]

    if 'jmvae' in architecture:
      final_jmvc_alphas = jmvae_alphas
    else:
      final_jmvc_alphas = jmvae_alphas[:1]

    if 'bivcca' in architecture:
      final_bivcca_mus = bivcca_mus
    else:
      final_bivcca_mus = bivcca_mus[:1]

    for extra_params in itertools.product(final_alpha_y_elbo_ys,
                                          final_stop_gradients,
                                          final_jmvc_alphas, final_bivcca_mus):
      i += 1

      (alpha_y_elbo_y, stop_gradient, jmvae_alpha, bivcca_mu) = extra_params

      outname = ('m{i:04d}_{architecture}_'
                 'poe{product_of_experts}_ue{universal_expert}_'
                 'ax{alpha_x}_ay{alpha_y}_l1{l1_pyz}_'
                 'ayey{alpha_y_elbo_y}_sg{stop_gradient}_'
                 'ja{jmvae_alpha}_bm{bivcca_mu}'
                 '_ny{num_y}_kp{keep_prob}'.format(**locals()))
      outname = outname.replace('_', '-').replace('.', '_')

      if i % FLAGS.num_tasks == FLAGS.task:
        print(outname,
           'Training model %d of %d...' % (i, num_models))

        if FLAGS.num_tasks == 1:
          # Running only 1 task means probably not running on borg, so don't
          # catch crashes.
          build_graph_and_run(**locals())
          continue

        try:
          build_graph_and_run(**locals())
        except:  # pylint: disable=bare-except
          print(outname, 'Retrying after a crash...')
          try:
            build_graph_and_run(**locals())
          except:  # pylint: disable=bare-except
            print(outname,
               'Failed to train, skipping.')
      else:
        print(outname, 'Skipping model %d of %d...' % (i, num_models))
  # pylint: enable=unused-variable


def build_graph_and_run(
    num_steps,
    batch_size,
    num_y,
    sample_range,
    x_sample_steps,
    y_sample_steps,
    mlp_dims,
    latent_dims,
    architecture,
    product_of_experts,
    universal_expert,
    alpha_x,
    alpha_y,
    l1_pyz,
    alpha_y_elbo_y,
    stop_gradient,
    bivcca_mu,  # pylint: disable=unused-argument
    jmvae_alpha,  # pylint: disable=unused-argument
    keep_prob,
    base_dir,
    outname,
    sparse_image_writes,
    **_):
  """Build the graph and run the experiment."""
  # Set up sample grid.
  sample_zxs = np.array(
      [[x, y]
       for y in np.linspace(-sample_range, sample_range, x_sample_steps)
       for x in np.linspace(-sample_range, sample_range, x_sample_steps)])
  sample_zys = np.array(
      [[x, y]
       for y in np.linspace(-sample_range, sample_range, y_sample_steps)
       for x in np.linspace(-sample_range, sample_range, y_sample_steps)])

  base_colors = [
      '#e0e0e0',  # Gray.
      '#5ba857',  # Green.
      '#ba151b',  # Red.
      '#be9d41',  # Yellow.
  ]

  if product_of_experts:
    if num_y == 2:
      attribute_vecs = [
          [0, 0],
          [0, 1],
          [1, 0],
          [1, 1],
          # For masked queries.
          [0, 0],
          [1, 0],
          [0, 0],
          [0, 1],
      ]
      attribute_colors = base_colors + [
          # Masked query colors.
          '#ee935d',  # I care about small things: black + green.
          '#7957ab',  # I care about big things: red + yellow.
          '#2bebe3',  # I care about even things: black + red.
          '#ee4f00',  # I care about odd things: green + yellow.
          '#ff85be',  # Prior color: pink.
      ]
      mask_vecs = [
          [1, 1],
          [1, 1],
          [1, 1],
          [1, 1],
          # Masked queries
          [1, 0],  # I care about small things.
          [1, 0],  # I care about big things.
          [0, 1],  # I care about even things.
          [0, 1],  # I care about odd things.
      ]
    elif num_y == 3:
      attribute_vecs = [
          [0, 0, 0],
          [1, 0, 0],
          [0, 0, 0],
          [0, 1, 0],
          [0, 0, 0],
          [0, 0, 1],
      ]
      attribute_colors = base_colors + [
          '#e0e0e0',  # Gray.
          '#697fcd',  # Blue.
          '#ff85be',  # Prior color: pink.
      ]
      mask_vecs = [
          [1, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [0, 1, 0],
          [0, 0, 1],
          [0, 0, 1],
      ]
  else:
    if num_y == 2:
      attribute_vecs = [
          [0, 0],
          [0, 1],
          [1, 0],
          [1, 1],
      ]
      attribute_colors = base_colors + [
          '#ff85be',  # Prior color: pink.
      ]
    elif num_y == 3:
      attribute_vecs = [
          [0, 0, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 1, 0],
          [0, 0, 1],
          [0, 1, 1],
          [1, 0, 1],
          [1, 1, 1],
      ]
      attribute_colors = base_colors + [
          '#697fcd',  # Blue.
          '#54acdb',  # Cyan.
          '#c762b8',  # Magenta.
          '#101010',  # Black.
          '#ff85be',  # Prior color: pink.
      ]

  tf.reset_default_graph()

  # Set up dataset.
  data_dim_x = [28, 28, 1]
  data_dim_y = [num_y]

  def get_data(batch_size=batch_size, **kwargs):
    """Get images and labels."""
    dataset = datasets.Mnist(batch_size=batch_size, **kwargs)

    print((dataset.description.image.min, dataset.description.image.max))
    print(dataset.num_examples)

    data = dataset()
    image_p = tf.contrib.distributions.Bernoulli(probs=data.image)
    image = tf.cast(image_p.sample(), tf.float32)

    label_big_small = tf.expand_dims(
        tf.cast(
            tf.greater_equal(data.label, tf.constant(5, dtype=tf.int64)),
            tf.float32), 1)

    label_odd_even = tf.expand_dims(
        tf.minimum(
            tf.reduce_sum(
                tf.cast(
                    tf.equal(
                        tf.expand_dims(data.label, 1),
                        tf.expand_dims(
                            tf.constant([1, 3, 5, 7, 9], dtype=tf.int64), 0)),
                    tf.float32),
                axis=-1), 1), 1)

    label_prime = tf.expand_dims(
        tf.minimum(
            tf.reduce_sum(
                tf.cast(
                    tf.equal(
                        tf.expand_dims(data.label, 1),
                        tf.expand_dims(
                            tf.constant([2, 3, 5, 7], dtype=tf.int64), 0)),
                    tf.float32),
                axis=-1), 1), 1)

    all_labels = [label_big_small, label_odd_even, label_prime]
    label = tf.concat(all_labels[:num_y], axis=-1)

    return image, label

  # Get the full dataset for evaluation use after training.
  all_batch_size = 100  # Need an evenly-dividing batch size to get everything.
  all_image, all_label = get_data(
      batch_size=all_batch_size, mode='all', sample='sequential')

  # Get the regular dataset.
  image, label = get_data(mode='train')

  # TODO(iansf): Something with labels to test SSL?

  # Build the graph
  mask = None
  dropout = keep_prob > 0.0
  if 'convo' in architecture:
    # This is only used for testing the new convolutional multi VAE
    # architectures.
    # TODO(hmoraldo): get rid of this once implementation is finished.
    encoders, decoders, prior = (
        encoder_decoder.get_convolutional_multi_vae_networks(
            latent_dims,
            product_of_experts_y=product_of_experts,
            decoder_output_channels=[10, 10, 10, 10, 1],
            decoder_output_shapes=[2, 4, 7, 14, 28],
            decoder_kernel_shapes=[4, 4, 4, 4, 4],
            decoder_strides=[2, 2, 2, 2, 2],
            decoder_paddings=['SAME', 'SAME', 'SAME', 'SAME', 'SAME'],
            encoder_output_channels=[1],
            encoder_kernel_shapes=[3],
            encoder_strides=[1],
            encoder_paddings=['VALID'],
            activation_fn=tf.nn.relu,
            use_batch_norm=False,
            mlp_layers=[5],
            output_distribution='Bernoulli',
            dropout=dropout,
            keep_prob=keep_prob,
            is_training=True,
            mlp_dim=mlp_dims,
            data_dim_y=data_dim_y,
            batch_size=-1,
            l1_pyz=l1_pyz,
            universal_expert=universal_expert))
  else:
    encoders, decoders, prior = encoder_decoder.get_jmvae_networks(
        mlp_dims,
        latent_dims,
        data_dim_x=data_dim_x,
        data_dim_y=data_dim_y,
        batch_size=-1,
        product_of_experts_y=product_of_experts,
        universal_expert=universal_expert,
        l1_pyz=l1_pyz,
        use_sparse_label_decoder=False,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=True)

  if 'multi' in architecture:
    loss_object = multimodal_elbo_loss.MultimodalElboLoss(
        encoders,
        decoders,
        prior,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        alpha_y_elbo_y=alpha_y_elbo_y,
        rescale_amodal_x=True,
        stop_gradient=stop_gradient,
        predict_both=False)

    x_mask = 'qx_' in architecture or 'vae' in architecture
    y_mask = 'qy' in architecture
    xy_mask = 'qxy' in architecture
    print((x_mask, y_mask, xy_mask), 'mask')
    mask = tf.concat(
        [
            tf.ones([batch_size, 1]) * x_mask,
            tf.ones([batch_size, 1]) * y_mask,
            tf.ones([batch_size, 1]) * xy_mask,
        ],
        axis=-1)
  elif 'jmvae' in architecture:
    loss_object = jmvae_loss.JmvaeLoss(
        encoders,
        decoders,
        prior,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        jmvae_alpha=jmvae_alpha,
        joint_distribution_index=2)
  elif 'bivcca' in architecture:
    loss_object = bivcca_loss.BiVccaLoss(
        encoders,
        decoders,
        prior,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        mu_tradeoff=bivcca_mu,
        joint_distribution_index=2)
  else:
    raise ValueError('Architecture %s not yet implemented.' % architecture)

  model = joint_vae.JointVae(encoders, decoders, prior, loss_object)
  loss = model.build_nelbo_loss([image, label, [image, label]], mask)

  # Set up placeholders and grab the core component tensors.
  image_ph = tf.placeholder(dtype=tf.float32, shape=[None] + data_dim_x)
  label_ph = tf.placeholder(dtype=tf.float32, shape=[None] + data_dim_y)
  mask_y_ph = tf.placeholder(dtype=tf.float32, shape=[None] + data_dim_y)
  z_ph = tf.placeholder(dtype=tf.float32, shape=[None, latent_dims])

  # All variables named *_dropout are constructed from model instead of
  # model_eval, which means they use is_training=True.
  # TODO(hmoraldo): remove *_dropout tensors once we stop using them.
  q_z_x_dropout, q_z_y_dropout, _ = model.infer_latent(
      [image_ph, label_ph, [image_ph, label_ph]], masks=[None, mask_y_ph, None])
  mu_qzy_dropout = q_z_y_dropout.density.mean()
  std_qzy_dropout = q_z_y_dropout.density.stddev()
  q_z_x_dropout = q_z_x_dropout.density.sample()
  q_z_y_dropout = q_z_y_dropout.density.sample()
  p_x_z_dropout, p_y_z_dropout = model.predict(z_ph)
  p_x_z_dropout = p_x_z_dropout.mean()
  p_y_z_dropout = p_y_z_dropout.mean()

  # This must only be done after the loss is built; that makes sure the
  # loss tensor and the *_dropout variables are created with is_training=True.
  encoders.set_is_training(False)
  decoders.set_is_training(False)
  model_eval = joint_vae.JointVae(encoders, decoders, prior, loss_object)

  q_z_x, q_z_y, _ = model_eval.infer_latent(
      [image_ph, label_ph, [image_ph, label_ph]], masks=[None, mask_y_ph, None])

  mu_qzy = q_z_y.density.mean()
  std_qzy = q_z_y.density.stddev()

  q_z_x = q_z_x.density.sample()
  q_z_y = q_z_y.density.sample()

  p_x_z, p_y_z = model_eval.predict(z_ph)
  p_x_z = p_x_z.mean()
  p_y_z = p_y_z.mean()

  #qj(
  #    np.sum([
  #        np.prod(qj(v.get_shape(), v.name, z=1))
  #        for v in tf.trainable_variables()
  #    ]), 'Total number of parameters')

  # Construct the train op and initialize variables.
  optimizer = tf.train.AdamOptimizer(0.001)
  train_op = optimizer.minimize(loss)

  init_fn = tf.global_variables_initializer()
  tf.set_random_seed(42)

  # Start the session
  sess = tf.Session()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess, coord=coord)

  # Run the experiment.
  losses = []
  y_imgs = []
  y_imgs_gauss = []
  y_imgs2x = []
  y_imgs2x_gauss = []
  y_imgs5x = []
  y_imgs5x_gauss = []
  y_mus = []
  y_sigs = []

  sess.run(init_fn)
  for step in xrange(0, num_steps + 1):
    _, loss_item = sess.run([train_op, loss])
    losses.append(loss_item)

    if (step < 2000 and step % 100 == 0) or step % 1000 == 0:
      print((step, loss_item), 'Step, loss')

      def process_ys(ys):
        if 2 in data_dim_y:
          img = np.zeros([ys.shape[0], 3])
          img[:, :-1] = ys
          ys = img
        return np.reshape(ys, [y_sample_steps, y_sample_steps, 3])

      ys = sess.run(p_y_z, {z_ph: sample_zys})
      y_imgs.append(process_ys(ys))

      ys2x = sess.run(p_y_z, {z_ph: sample_zys * 2})
      y_imgs2x.append(process_ys(ys2x))

      ys5x = sess.run(p_y_z, {z_ph: sample_zys * 5})
      y_imgs5x.append(process_ys(ys5x))

      if product_of_experts:
        z_mu_y, z_sig_y = sess.run(
            [mu_qzy, std_qzy], {label_ph: attribute_vecs,
                                mask_y_ph: mask_vecs})
        z_mu_y_dropout, z_sig_y_dropout = sess.run(
            [mu_qzy_dropout,
             std_qzy_dropout], {label_ph: attribute_vecs,
                                mask_y_ph: mask_vecs})

        # Only show prior on PoE UE models.
        if universal_expert:
          z_mu_y = np.concatenate((z_mu_y, [[0.0, 0.0]]), axis=0)
          z_sig_y = np.concatenate((z_sig_y, [[1.0, 1.0]]), axis=0)
          z_mu_y_dropout = np.concatenate(
              (z_mu_y_dropout, [[0.0, 0.0]]), axis=0)
          z_sig_y_dropout = np.concatenate(
              (z_sig_y_dropout, [[1.0, 1.0]]), axis=0)
      else:
        z_mu_y, z_sig_y = sess.run([mu_qzy, std_qzy],
                                   {label_ph: attribute_vecs})
        z_mu_y_dropout, z_sig_y_dropout = sess.run(
            [mu_qzy_dropout, std_qzy_dropout], {label_ph: attribute_vecs})
      y_mus.append(z_mu_y)
      y_sigs.append(z_sig_y)

      def add_gaussians_to_figure(figure,
                                  z_mu,
                                  z_sig,
                                  attribute_colors,
                                  filename=None,
                                  sample_range=sample_range,
                                  fade_bg=True,
                                  line_width=5,
                                  return_fig=True):
        """Add gaussians to a previously rendered figure."""
        ax = figure.gca()

        if fade_bg:
          box = patches.Rectangle(
              xy=(-sample_range, -sample_range),
              width=2 * sample_range,
              height=2 * sample_range,
              alpha=0.3,
              fill=True,
              facecolor='w',
              edgecolor=None)
          ax.add_artist(box)

        for mu, sigma, color in reversed(zip(z_mu, z_sig, attribute_colors)):
          ell = patches.Ellipse(
              xy=mu,
              width=sigma[0] * 4.0,  # 2 stdevs, or 95%
              height=sigma[1] * 4.0,
              angle=0)
          ell.set_facecolor('none')
          ell.set_edgecolor(color)
          ell.set_linewidth(line_width)
          ax.add_artist(ell)

        ax.set_autoscaley_on(False)
        ax.set_xlim([-sample_range, sample_range])
        ax.set_ylim([sample_range, -sample_range])
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.figure.tight_layout(pad=0)
        if filename:
          with tf.gfile.Open(filename, 'w') as f:
            ax.figure.savefig(
                f, dpi='figure', pad_inches=0, bbox_inches='tight')
            plt.close(ax.figure)

        if return_fig:
          return figure
        else:
          return image_utils.canvas_to_np(figure, rescale=True)

      y_img = image_utils.show_image(
          y_imgs[-1], filename=None, show=False, return_fig=True)
      y_img2x = image_utils.show_image(
          y_imgs2x[-1], filename=None, show=False, return_fig=True)
      y_img5x = image_utils.show_image(
          y_imgs5x[-1], filename=None, show=False, return_fig=True)

      y_imgs_gauss.append(
          add_gaussians_to_figure(
              y_img,
              z_mu_y,
              z_sig_y,
              attribute_colors,
              sample_range=sample_range,
              line_width=2,
              return_fig=False))
      y_imgs2x_gauss.append(
          add_gaussians_to_figure(
              y_img2x,
              z_mu_y,
              z_sig_y,
              attribute_colors,
              sample_range=sample_range * 2,
              line_width=2,
              return_fig=False))
      y_imgs5x_gauss.append(
          add_gaussians_to_figure(
              y_img5x,
              z_mu_y,
              z_sig_y,
              attribute_colors,
              sample_range=sample_range * 5,
              line_width=2,
              return_fig=False))

      plt.close(y_img)
      plt.close(y_img2x)
      plt.close(y_img5x)

      if sparse_image_writes and step != num_steps:
        # Only write full images at the end, since there are so many jobs now.
        continue

      ##########################################################################
      # Write images
      ##########################################################################

      xs, ys = sess.run([p_x_z, p_y_z], {z_ph: sample_zxs})
      xs_dropout, ys_dropout = sess.run([p_x_z_dropout, p_y_z_dropout],
                                        {z_ph: sample_zxs})

      if 'multi-vae' in architecture:
        image_utils.plot_images(
            xs,
            n=x_sample_steps,
            filename=os.path.join(base_dir, '%s_bwimgs_%05d' % (outname, step)))

      # Resize xs and ys for color imaging.
      if 2 in data_dim_y:
        img = np.zeros([ys.shape[0], 3])
        img[:, :-1] = ys
        ys = img

        img_dropout = np.zeros([ys_dropout.shape[0], 3])
        img_dropout[:, :-1] = ys_dropout
        ys_dropout = img_dropout

      xs = 1.0 - np.stack((xs,) * 3, axis=-1).squeeze()
      ys = ys[:, np.newaxis, np.newaxis, :]
      xs_dropout = 1.0 - np.stack((xs_dropout,) * 3, axis=-1).squeeze()
      ys_dropout = ys_dropout[:, np.newaxis, np.newaxis, :]

      def color_xs_with_ys(xs, ys, filename):
        """Color images according to attribute vectors."""
        # Black or blue ys.
        # The 0th axis holds the image indices that we need to modify.
        xs = xs.copy()
        black_ys = np.where(np.sum(ys[:, :, :, :2], axis=-1) < 0.3)[0]
        black_xs = 1.0 - xs[black_ys]
        xs *= ys
        xs[black_ys] += black_xs
        xs = np.minimum(1.0, xs)
        figure = image_utils.plot_images(
            xs, n=x_sample_steps, return_fig=True, filename=filename)
        return xs, figure

      filename = os.path.join(base_dir, '%s_images_%05d' % (outname, step))
      xs_py, figure = color_xs_with_ys(xs, ys, filename)
      filename = os.path.join(base_dir, '%s_images_gauss_%05d.png' % (outname,
                                                                      step))
      add_gaussians_to_figure(figure, z_mu_y, z_sig_y, attribute_colors,
                              filename)

      if dropout:
        filename = os.path.join(base_dir, '%s_images_dropout_%05d' % (outname,
                                                                      step))
        _, figure = color_xs_with_ys(xs_dropout, ys_dropout, filename)
        filename = os.path.join(base_dir, '%s_images_gauss_dropout_%05d.png' %
                                (outname, step))
        add_gaussians_to_figure(figure, z_mu_y_dropout, z_sig_y_dropout,
                                attribute_colors, filename)

      # Compute consistency(?) visualization for generated images and labels
      # by comparing against the full ground truth database's latent
      # distributions.
      if step == num_steps + 10:  # TODO(iansf)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        q_z_x, _, _ = model_eval.infer_latent(
            [all_image, all_label, [all_image, all_label]])
        mu_qzx = q_z_x.density.mean()
        std_qzx = q_z_x.density.stddev()

        gt_z_labels = []
        gt_z_sigs = []
        num_batches = 70000 // all_batch_size
        for batch in range(num_batches):
          if batch % 100 == 0:
            print('Sampling batch %d of %d...' % (batch, num_batches))
          z_mu_x, z_sig_x, z_label = sess.run([mu_qzx, std_qzx, all_label])
          gt_z_labels.extend([
              retrieval_utils.RetrievalDatapoint(
                  utils.Gaussian(np.array([mu]), np.array([sig])), lab, None)
              for mu, sig, lab in zip(z_mu_x, z_sig_x, z_label)
          ])
          gt_z_sigs.append(z_sig_x)
        mean_gt_z_sigs = np.mean(gt_z_sigs)
        print(mean_gt_z_sigs, 'mean_gt_z_sigs')

        gt_ys = np.zeros_like(ys)
        gt_ys_kl = np.zeros_like(ys)
        for q, sample_z in enumerate(sample_zxs):
          if q % 100 == 0:
            print('Calculating query %d of %d...' % (q, len(sample_zxs)))
          query = retrieval_utils.QueryDatapoint(
              utils.Gaussian(
                  np.array([sample_z]),
                  np.array([[mean_gt_z_sigs, mean_gt_z_sigs]])), None)
          dists = retrieval_utils.compute_query_images_distance(
              query, gt_z_labels, metric='l2')
          dists_kl = retrieval_utils.compute_query_images_distance(
              query, gt_z_labels, metric='kl')

          # argmin for distance metrics.
          best_dist_id = np.argmin(dists)
          best_label = gt_z_labels[best_dist_id].label
          gt_ys[q, :, :, :num_y] = best_label

          best_dist_kl_id = np.argmin(dists_kl)
          best_label_kl = gt_z_labels[best_dist_kl_id].label
          gt_ys_kl[q, :, :, :num_y] = best_label_kl

        l2_kl_agreement_score = (1 - np.sum(
            np.sum(gt_ys == gt_ys_kl, axis=-1) != 3) / float(ys.shape[0]))
        print(l2_kl_agreement_score, 'l2_kl_agreement_score')

        wrong_ys = (np.sum(gt_ys == np.round(ys), axis=-1) !=
                    3)[:, :, :, np.newaxis]
        agreement_score = 1 - np.sum(wrong_ys) / float(wrong_ys.shape[0])
        print(agreement_score, 'agreement_score')

        # L2 internal consistency images.
        filename = os.path.join(base_dir, '%s_images_gtlabl2_%05d' % (outname,
                                                                      step))
        _, figure = color_xs_with_ys(xs, gt_ys, filename)
        filename = os.path.join(base_dir, '%s_images_gtlabl2_gauss_%05d.png' %
                                (outname, step))
        add_gaussians_to_figure(figure, z_mu_y, z_sig_y, attribute_colors,
                                filename)

        # KL internal consistency images.
        filename = os.path.join(base_dir, '%s_images_gtlabkl_%05d' % (outname,
                                                                      step))
        _, figure = color_xs_with_ys(xs, gt_ys_kl, filename)
        filename = os.path.join(base_dir, '%s_images_gtlabkl_gauss_%05d.png' %
                                (outname, step))
        add_gaussians_to_figure(figure, z_mu_y, z_sig_y, attribute_colors,
                                filename)

        error_xs = (1 - wrong_ys) * xs_py + wrong_ys * (1 - xs_py)
        figure = image_utils.plot_images(
            error_xs,
            n=x_sample_steps,
            return_fig=True,
            filename=os.path.join(base_dir, '%s_errors_%05d' % (outname, step)))

        filename = os.path.join(base_dir, '%s_errors_gauss_%05d.png' % (outname,
                                                                        step))
        add_gaussians_to_figure(figure, z_mu_y, z_sig_y, attribute_colors,
                                filename)

        prob_xs = ((1 - np.abs(gt_ys - ys)) * xs_py + np.abs(gt_ys - ys) *
                   (1 - xs_py))
        figure = image_utils.plot_images(
            prob_xs,
            n=x_sample_steps,
            return_fig=True,
            filename=os.path.join(base_dir, '%s_probs_%05d' % (outname, step)))

        filename = os.path.join(base_dir, '%s_probs_gauss_%05d.png' % (outname,
                                                                       step))
        add_gaussians_to_figure(figure, z_mu_y, z_sig_y, attribute_colors,
                                filename)

  image_utils.plot_images(
      np.array(y_imgs),
      n=int(np.sqrt(len(y_imgs))),
      filename=os.path.join(base_dir, '%s_all_labels1x' % outname))

  image_utils.plot_images(
      np.array(y_imgs_gauss),
      n=int(np.sqrt(len(y_imgs_gauss))),
      filename=os.path.join(base_dir, '%s_all_labels1x_gauss' % outname))

  image_utils.plot_images(
      np.array(y_imgs2x),
      n=int(np.sqrt(len(y_imgs2x))),
      filename=os.path.join(base_dir, '%s_all_labels2x' % outname))

  image_utils.plot_images(
      np.array(y_imgs2x_gauss),
      n=int(np.sqrt(len(y_imgs2x_gauss))),
      filename=os.path.join(base_dir, '%s_all_labels2x_gauss' % outname))

  image_utils.plot_images(
      np.array(y_imgs5x),
      n=int(np.sqrt(len(y_imgs5x))),
      filename=os.path.join(base_dir, '%s_all_labels5x' % outname))

  image_utils.plot_images(
      np.array(y_imgs5x_gauss),
      n=int(np.sqrt(len(y_imgs5x_gauss))),
      filename=os.path.join(base_dir, '%s_all_labels5x_gauss' % outname))

  plt.plot(losses)
  with tf.gfile.Open(os.path.join(base_dir, '%s_losses.png' % outname), 'w') as f:
    plt.savefig(f, dpi='figure', pad_inches=0, bbox_inches='tight')
  plt.close()

  coord.request_stop()
  coord.join(threads)
  sess.close()


if __name__ == '__main__':
  tf.app.run(main)
