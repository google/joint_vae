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

"""A multimodal convolutional variational autoencoder model in Sonnet.

Assembles components for a multimodal joint variational autoencoder,
constructing a model along with a multimodal nelbo loss function.
More details on the multmodal NELBO loss function can be found at:
//research/vale/imagination/jmvae/multimodal_elbo_vae

Author: vrama@
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import sonnet as snt
import tensorflow as tf
from tensorflow.contrib.distributions.python.ops import mvn_linear_operator as mvn_linop

from datasets.deepmind_shapes import dataset_provider
# TODO(vrama): Add an mnist with attributes dataset provider.
from datasets.mnist_attributes import dataset_provider as affine_mnist_provider
from datasets.celeba import celeba_data_provider

from joint_vae import bivcca_loss
from joint_vae import jmvae_loss
from joint_vae import multimodal_elbo_loss
from joint_vae import joint_vae
from joint_vae.utils import LatentDensity
from joint_vae.utils import compute_likelihood
from experiments import celeba_encoder_decoder
from experiments import mnista_encoder_decoder

flags = tf.flags


FLAGS = tf.app.flags.FLAGS

def check_mean_or_sample(decoder, mean_or_sample):
  """Return sample or mean or both from a distribution.

  Args:
    decoder: tf.contrib.distribution object.
    mean_or_sample: str, 'sample', 'mean', or 'both'
  Returns:
    `Tensor` which is either a sample or mean from the distribution, or
     `Tensor` which is a concatentation of the mean and sample `Tensors`.
  Raises:
    ValueError: If invalid option is provided for mean_or_sample.
  """
  if mean_or_sample == 'sample':
    return tf.cast(decoder.sample(), tf.float32)
  elif mean_or_sample == 'mean':
    return decoder.mean()
  elif mean_or_sample == 'both':
    return tf.concat(
        [decoder.mean(), tf.cast(decoder.sample(), tf.float32)], axis=-1)
  else:
    raise ValueError('Invalid mean_or_sample: %s' % (mean_or_sample))


class ConvolutionalMultiVae(object):
  """A class for Convolutional Multimodal Variational Autoencoder."""

  def __init__(self, config, mode='train', split_name='train',
               add_summary=True):
    """Initialization.

    Args:
      config: A configuration class specifying various options for constructing
        the model. See
          //research/vale/imagination/scratch.multimodal_vae/configuration.py
        for more details.
      mode: One of "train", "val" or "test", or 'inference'.
            Used to set various options for graph construction.
      split_name: One of "train", "val" or "test", used to decide which split
        to load the data from for use in the model.
      add_summary: Bool, whether to add summaries to the graph or not.
    """
    logging.info('Creating the model in mode %s', mode)
    self._mode = mode

    self._split_name = split_name
    if self._mode != 'inference' and self._mode != self._split_name:
      logging.warning(
          'Note: split_name and mode are different. Please make sure this is '
          'not in error.')

    self._add_summary = add_summary

    self.config = config

    # [batch_size, width, height, channels]
    self._images = None

    # 4 * [batch_size], List of `Tensors` of size [batch_size]
    self._labels = None

    # Optional list of tensors containing the true factors of variation that
    # generated the labels. Only relevant for synthetic datasets.
    self._true_latents = None

    # Int, Number of samples in the split of the dataset.
    self._num_samples = None

    # Tuple, contains the number of categories corresponding to each attribute
    # in the dataset.
    self._num_classes_per_attribute = None

    self.product_of_experts = self.config.product_of_experts

  def _shuffle_or_not(self):
    """Decide whether to shuffle the input data or not."""
    if self._mode == 'train':
      return True
    return False

  def _check_inputs(self):
    """A check to verify that inputs are in expected formats."""
    assert isinstance(self._labels, list), 'Labels must be a list of Tensors.'
    assert self._images.dtype == tf.float32, 'Images must be float 32 Tensors.'
    assert isinstance(self._num_classes_per_attribute,
                      tuple), ('Num classes per attribute must be a tuple.')
    assert (
        self.config.num_classes_per_attribute is None or
        self._num_classes_per_attribute == self.config.num_classes_per_attribute
    ), ('Number of classes for the attribute is inconsistent.')
    if self._mode != 'inference':
      assert isinstance(self._num_samples, int), 'Num samples must be int.'

  def _build_inputs(self):
    """Construct inputs for the VAE model.

    Outputs:
      self._images: A [batch_size, height, width, channels] `Tensor`.
      self._labels: A list of `Tensors` of [batch_size]
      self._num_classes_per_attribute: tuple, number of classes for each
      attribute.
        For instance an attribute "big" might have two attributes "small or
        large".
        Then the Tensor would be [2], similarly for more attributes.
      self._num_samples: Number of samples in the split of the dataset.
    """
    if self.config.dataset == 'dm_shapes_with_labels':
      raise NotImplementedError
      #images, labels, latents, num_samples, num_classes_per_attribute = (
      #    dataset_provider.provide_data(
      #        self._split_name,
      #        self.config.batch_size,
      #        split_type=self.config.split_type,
      #        preprocess_options=self.config.preprocess_options,
      #        grayscale=self.config.grayscale,
      #        shuffle_data=self._shuffle_or_not()))
      #self._true_latents = latents
    elif self.config.dataset == 'affine_mnist':
      tf.logging.info('Loading the affine mnist datset with split type %s' % (self.config.split_type))
      images, labels, latents, num_samples, num_classes_per_attribute = (
          affine_mnist_provider.provide_data(
              self._split_name,
              self.config.batch_size,
              split_type=self.config.split_type,
              preprocess_options=self.config.preprocess_options,
              grayscale=self.config.grayscale,
              shuffle_data=self._shuffle_or_not()))
      self._true_latents = latents
    elif self.config.dataset == 'celeba':
      images, labels, num_samples, num_classes_per_attribute, _ = (
          celeba_data_provider.provide_data(
            FLAGS.dataset_dir,
            self._split_name,
            self.config.batch_size,
            split_type=self.config.split_type,
            image_size=self.config.image_size,
            preprocess_options=self.config.preprocess_options,
            grayscale=self.config.grayscale,
            shuffle_data=self._shuffle_or_not())
      )
    else:
      raise NotImplementedError('Only CUB, CelebA, DeepMind Shapes with labels, '
                                'and MNIST with attributes '
                                'are available for use currently.')
    # In inference mode, we prepare the model for interactive usage, with
    # ability to specify a class label or image via. placeholders.
    if self._mode == 'inference':
      num_samples = None
      inference_image_ph = tf.placeholder(
          dtype=tf.float32,
          shape=[None] + list(self.config.image_size))
      inference_label_ph = tf.placeholder(
          dtype=tf.int32, shape=[None, len(num_classes_per_attribute)])
      # Used for inference in a product of experts model.
      ignore_label_mask_ph = tf.placeholder(
          dtype=tf.float32, shape=[None, len(num_classes_per_attribute)])

      self._inference_image_ph = inference_image_ph
      self._inference_label_ph = inference_label_ph
      self._ignore_label_mask_ph = ignore_label_mask_ph

      labels = tf.unstack(inference_label_ph, axis=1)
      images = inference_image_ph

    # Override whatever was in the config with the true value from the dataset.
    self.config.num_classes_per_attribute = num_classes_per_attribute

    self._images = images
    self._labels = labels
    self._num_classes_per_attribute = num_classes_per_attribute
    self._num_samples = num_samples  # Used for evaluation.

    self._check_inputs()

  def get_mask(self):
    """Create a mask to train multimodal elbo.

    Essentially, this code masks out the elbo terms corresponding to
    class labels 'y' if is_multi_stage returns True. Otherwise, all
    three elbo terms, elbo(x), elbo(y), elbo(xy) are used.

    Returns:
      mask: a `Tensor` of [batch_size, 3] if 'is_multi_stage' or 'reweighted'
      'elbo is set to True, otherwise None.
    """
    mask = None
    if self.is_multi_stage:
      mask = tf.constant([[1, 0, 1]] * self.config.batch_size, dtype=tf.float32)
    elif self.config.loss_type == 'reweighted_elbo':
      mask = tf.constant(
          [[1, self.config.reweighted_elbo_y_scale, 1]] *
          self.config.batch_size,
          dtype=tf.float32)

    return mask

  def _build_generative(self):
    """Construct the Multimodal Variational Autoencoder.

    Assembles different comopnents for a multimodal variational autoencoder
    using a multimodal elbo objective.

    Outputs:
      TODO(vrama): Update the output.
      self._vae: an object of `multimodal_elbo_vae.MultimodalElbo`.
    """
    tf.logging.info("Constructing encoders and decoders.")
    tf.logging.info("Label decoder has product of experts: %s" % (str(self.product_of_experts)))
    # Get an image decoder with a DCGAN architecture.
    encoder_decoder_model = (celeba_encoder_decoder if self.config.dataset=='celeba' else mnista_encoder_decoder)
    if FLAGS.image_decoder_channels == '_REPLACE_ME':
      img_decoder_channels = 3
    else:
      img_decoder_channels = self.config.image_size[-1]
    encoders, decoders, prior, joint_distribution_index = encoder_decoder_model.get_nips_17_conv_multi_vae_networks(
        latent_dim=self.config.num_latent,
        product_of_experts_y=self.product_of_experts,
        decoder_output_channels=(1024, 512, 256, 128, img_decoder_channels),
        decoder_output_shapes=((4, 4), (8, 8), (16, 16), (32, 32), (self.config.image_size[0], self.config.image_size[1])),
        decoder_kernel_shapes=((4, 4), (5, 5), (5, 5), (5, 5), (5, 5)),
        decoder_strides=((1, 1), (2, 2), (2, 2), (2, 2), (2, 2)),
        decoder_paddings=(snt.VALID, snt.SAME, snt.SAME, snt.SAME, snt.SAME),
        encoder_output_channels=self.config.encoder_output_channels,
        encoder_kernel_shapes=self.config.encoder_kernel_shapes,
        encoder_strides=self.config.encoder_strides,
        encoder_paddings=self.config.encoder_paddings,
        activation_fn=self.config.activation_fn,
        use_batch_norm=self.config.use_batch_norm,
        mlp_layers=self.config.mlp_layers,
        output_distribution=self.config.image_likelihood,
        dropout=self.config.dropout,
        keep_prob=self.config.keep_prob,
        vocab_sizes=self._num_classes_per_attribute,
        label_embed_dims=self.config.label_embed_dims,
        mlp_dim=self.config.post_fusion_mlp_layers[0],  # TODO(vrama): Modify options properly.
        l1_pyz=self.config.label_decoder_regularizer,
        is_training=self.is_training)
    self.joint_distribution_index = joint_distribution_index
    # TODO(vrama): Remove unused options or make sure to mention they are
    # deprecated in configuration.py.
    # Unused options are: li_mlp_layers, post_fusion_mlp_layers,
    # li_post_fusion_mlp_layers

    tf.logging.info("Constructing loss %s" % self.config.loss_type)
    # We call the multimodal elbo loss class for forward kl to make sure that
    # all the variables for mutlimodal elbo are instantiated and are a part of
    # the graph.
    if self.config.loss_type in ['multimodal_elbo',  'fwkl']:
      loss = multimodal_elbo_loss.MultimodalElboLoss(
          encoders,
          decoders,
          prior,
          alpha_x=self.config.alpha_x,
          alpha_y=self.config.alpha_y,
          alpha_y_elbo_y=self.config.private_p_y_scaling,
          rescale_amodal_x=self.config.rescale_amodal_x,
          predict_both=self.config.melbo_predict_both,
          stop_gradient_y=self.config.stop_elboy_gradient,
          stop_gradient_x=self.config.stop_elbox_gradient,
          mode=self._mode,
          path_derivative=self.config.path_derivative)
    elif self.config.loss_type == 'bivcca':
      loss = bivcca_loss.BiVccaLoss(
          encoders,
          decoders,
          prior,
          alpha_x=self.config.alpha_x,
          alpha_y=self.config.alpha_y,
          mu_tradeoff=self.config.bivcca_mu,
          joint_distribution_index=joint_distribution_index,
          mode=self._mode)
    elif self.config.loss_type == 'jmvae':
      loss = jmvae_loss.JmvaeLoss(
          encoders,
          decoders,
          prior,
          alpha_x=self.config.alpha_x,
          alpha_y=self.config.alpha_y,
          jmvae_alpha=self.config.jmvae_alpha,
          joint_distribution_index=joint_distribution_index,
          mode=self._mode)

    self._vae = joint_vae.JointVae(encoders, decoders, prior, loss)

  def _build_label_inference_loss(self, loss_type='monte_carlo'):
    """Builds the loss to train q(z| y) in a multistage model."""
    empirical_gmm_mean = tf.placeholder(
        dtype=tf.float32,
        shape=[
            self.config.label_calibration_batch_size, self.config.num_latent
        ],
        name='empirical_gmm_mean')
    empirical_gmm_covariance = tf.placeholder(
        dtype=tf.float32,
        shape=[
            self.config.label_calibration_batch_size, self.config.num_latent,
            self.config.num_latent
        ],
        name='empirical_gmm_covariance')

    labels = tf.placeholder(
        dtype=tf.int32,
        shape=[
            self.config.label_calibration_batch_size,
            len(self._num_classes_per_attribute)
        ],
        name='labels')

    # Split the labels into a list of labels which the encoder needs.
    labels_list = tf.unstack(
        labels, num=len(self._num_classes_per_attribute), axis=1)
    empirical_gmm_scale = tf.cholesky(
        empirical_gmm_covariance, name='cholesky_scale')

    density, sample = self._vae._encoder_y(  # pylint: disable=protected-access
        labels_list, self.get_label_mask(as_list=False))
    label_inference_network = LatentDensity(density, sample)

    empirical_gmm_mean = tf.stop_gradient(empirical_gmm_mean)
    empirical_gmm_scale = tf.stop_gradient(empirical_gmm_scale)

    target_distribution = mvn_linop.MultivariateNormalLinearOperator(
        empirical_gmm_mean,
        scale=tf.contrib.linalg.LinearOperatorTriL(empirical_gmm_scale))

    if loss_type == 'monte_carlo':
      z = target_distribution.sample()
      # Since target distribution is multivariate, it produces just one number
      # for z.
      log_prob_target_distribution = target_distribution.log_prob(z)

      # Since label inference network is parameterized like a univariate normal,
      # it returns a number for the log prob of each data point, which needs to
      # be summed.
      sum_axis = range(1, z.get_shape().ndims)
      log_prob_inference_distribution = tf.reduce_sum(
          label_inference_network.density.log_prob(z), sum_axis)

      kl_loss = log_prob_target_distribution - log_prob_inference_distribution

    elif loss_type == 'analytical':
      # TODO(vrama): Why is analytical KL diverging??
      ## Split the tensor along batch dimension.

      # diag_std_tensors = tf.matrix_diag(label_inference_network.density.scale)
      # label_inference_network_mean = label_inference_network.density.loc

      # transformed_label_inference_density = (
      #    mvn_linop.MultivariateNormalLinearOperator(
      #        label_inference_network_mean,
      #        scale=tf.contrib.linalg.LinearOperatorTriL(diag_std_tensors)))

      # Analytical KL is available so use it.
      # kl_loss = tf.contrib.distributions.kl_divergence(target_distribution,
      #                                    transformed_label_inference_density)
      simplified_target_distribution = tf.contrib.distributions.Normal(
          loc=empirical_gmm_mean,
          scale=tf.sqrt(tf.matrix_diag_part(empirical_gmm_covariance)))
      kl_loss = tf.contrib.distributions.kl_divergence(
          simplified_target_distribution, label_inference_network.density)

    kl_loss = tf.reduce_mean(kl_loss)

    return kl_loss, empirical_gmm_mean, empirical_gmm_covariance, labels

  def build_forward_kl_loss(self):
    """Build the forward kl loss between KL(p(z| x) | q(z| x)."""
    # Simulate data from the prior.
    samples = self._vae.sample(FLAGS.batch_size, return_latent=True)
    x, y, z = samples
    # Make sure the gradients dont go back into the decoders.
    x = tf.stop_gradient(x)
    y = [tf.stop_gradient(y_i) for y_i in y]
    # Normally the prior is fixed and so does not need the stop gradient but
    # in special cases such as the vamp prior the `z` need not be fixed.
    z = tf.stop_gradient(z)
    # Ensure that the samples are all float 32 to pass through the model.
    x = tf.cast(x, tf.float32)
    # Learn to fit the data using the inference networks.
    latents = self._vae._encoders.infer_latent([x, y, [x, y]], v=None)  # pylint-disable-private
    latent_log_like = [compute_likelihood(lat.density, z) for i, lat in enumerate(latents) if i != self.joint_distribution_index]
    loss = -1 * tf.reduce_sum(tf.reduce_sum(tf.stack(latent_log_like, axis=1), axis=1))
    return loss

  def _build_loss(self):
    """Builds the loss.

    Outputs:
      self._loss: A 1-D loss `Tensor` which can be used for optimization.
    """
    logging.info('Using loss %s.', self.config.loss_type)
    if self.config.loss_type == 'fwkl':
      self._loss = self.build_forward_kl_loss()
    else:
      mask = self.get_mask()
      self._loss = self._vae.build_nelbo_loss(
          [self._images, self._labels, [self._images, self._labels]],
          mask=mask
      )

    # For multi stage, construct another loss for a label inference network.
    if self.is_multi_stage:
      label_inference_loss, mean_ph, var_ph, label_ph = (
          self._build_label_inference_loss(self.config.calibration_kl_type))
      self._label_inference_loss = label_inference_loss
      self._mean_ph = mean_ph
      self._var_ph = var_ph
      self._label_ph = label_ph

  def _setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name='global_step',
        trainable=False,
        dtype=tf.int64,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    self.global_step = global_step

  def build_model(self):
    """Connect the model, by constructing inputs, model and the loss."""
    self._build_inputs()
    self._build_generative()
    self._build_loss()
    if self._mode is not 'inference':
      self._setup_global_step()

  def setup_saver(self):
    """Set up the saver module."""
    saver = None
    if self._mode != 'inference':
      saver = tf.train.Saver()

    return saver

  def get_forward_kl_init_fn(self, checkpoint_dir):
    """Set up an initializer for the case where we want to train forward KL."""
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    # Restore just the terms which appear in elbo(x, y)
    variables_to_restore = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.split('/')[0] in ['inference_xy', 'decoder_y', 'decoder_x']]
    saver = tf.train.Saver(var_list=variables_to_restore)
    def init_fn(sess):
      tf.logging.info('Initializing the model from %s.' % (latest_checkpoint))
      saver.restore(sess, latest_checkpoint)

    return init_fn

  def _get_infer_latent_fns(self, modality=None):
    """Select inference modules based on modality.

    Args:
      modality: str, either 'label', or 'image' or 'image_and_label'
    Returns:
      A tf.contrib.distributions, `Tensor` tuple, containing the inferred
        density and a sample from the density respectively.
    Raises:
      ValueError: If modality is not in 'label', 'image', 'image_and_label'
    """
    label_mask_to_use = self.get_label_mask(as_list=False)

    encoder_image, encoder_label, encoder_image_label= self._vae.infer_latent(
        [self._images, self._labels, [self._images, self._labels]],
        masks=[None, label_mask_to_use, None])

    if modality is None or modality == 'label':
      return encoder_label
    elif modality == 'image':
      return encoder_image
    elif modality == 'image_and_label':
      return encoder_image_label
    else:
      raise ValueError('Invalid option: %s' % (modality))

  def label_inference_fn(self):
    return self._get_infer_latent_fns(modality='label').density

  def generate_image_conditioned_latent_utils(self, mean_or_sample='both'):
    latent_density_ph = tf.placeholder(
        dtype=tf.float32, shape=[None, self.config.num_latent])
    return latent_density_ph, check_mean_or_sample(
        self._get_conditional_sampling_fns(latent_density_ph, modality='image'),
        mean_or_sample)

  def _get_conditional_sampling_fns(self, latent, modality=None):
    """Select sampling functions based on modality.

    Args:
      latent: A `Tensor` of num_latent.
      modality: str, a modality for which we are getting the sampling
        distribution. One of either 'image' or 'label'
    Returns:
      A tf.contrib.distributions object for the sampling distribution.
    Raises:
      ValueError: If modality is not in 'image' or 'label'
    """
    sampling_image, sampling_label = self._vae.predict(latent)
    if modality is None or modality == 'image':
      return sampling_image
    elif modality == 'label':
      return sampling_label
    else:
      raise ValueError('Invalid option: %s' % (modality))

  def generate_label_conditioned_image(self, num_samples_predictive_dist=10):
    r"""Given an image, predict the label for it, via the latent.

    Given an image, we first sample a z ~ q(z | x), and then based on it
    pass it via. p(y| z), to get a categorical predictive density. Using
    this density, we make the \argmax p(y| z) MAP estimate for the class
    labels.

    Args:
      num_samples_predictive_dist: Number of samples of the latent state to
        draw before predicting the class label.

    Returns:
      attribute_preds: List of `Tensor` of [batch_size], containing the
        predictions for each attribute of interest in the minibatch.

    Raises:
      ValueError: If number of samples to use is less than 1.
    """
    # Infer latents conditioned on label.
    encoder_image = self._get_infer_latent_fns(modality='image')

    attribute_predictive_dist = []
    for s in xrange(num_samples_predictive_dist):
      # TODO(vrama): Check if this helps, make changes elsewhere.
      latent = encoder_image.density.sample()
      decoder_attributes_dist = self._get_conditional_sampling_fns(
          latent, modality='label')
      for index, attribute_pred in enumerate(decoder_attributes_dist):
        if s == 0:
          attribute_predictive_dist.append(attribute_pred.probs)
        else:
          attribute_predictive_dist[index] = (
              attribute_predictive_dist[index] * s + attribute_pred.probs) / (
                  s + 1)

    attribute_preds = []

    for attribute_dist in attribute_predictive_dist:
      attribute_preds.append(tf.arg_max(attribute_dist, dimension=1))
    return attribute_preds

  def generate_images_conditioned_label(self,
                                        num_images=1,
                                        mean_or_sample='sample'):
    """Given a set of attribute values, predict the image corresponding to it.

    Args:
      num_images: Number of images to generate for the given label.
      mean_or_sample: Whether to return a sample from the predictive density or
        the mean value.
    Returns:
      If num_images = 1, A `Tensor` indicating either a sample or the mean of
      the sampling distribution. If both, then the `Tensor` has extra channels,
      with mean and sample concatenated.
      And a list of `Tensors` otherwise.
    Raises:
      ValueError: If mean_or_sample is not in 'sample' or 'mean' or 'both'
    """
    decoded_images = []

    for _ in xrange(num_images):
      encoder_label = self._get_infer_latent_fns(modality='label')
      latent = encoder_label.density.sample()
      decoder_image = self._get_conditional_sampling_fns(
          latent, modality='image')
      decoded_images.append(check_mean_or_sample(decoder_image, mean_or_sample))

    if len(decoded_images) == 1:
      decoded_images = decoded_images[0]

    return decoded_images

  def get_label_mask(self, as_list=True):
    """Provides a label mask for use in product of inference model."""
    if self.product_of_experts and self._mode == 'inference':
      use_mask = self._ignore_label_mask_ph
    else:
      use_mask = tf.ones(
          [tf.shape(self._labels[0])[0],
           len(self._labels)], dtype=tf.float32)

    if as_list:
      use_mask = tf.unstack(use_mask, axis=1)

    return use_mask

  def introspection_eval_op(self, num_samples_predictive_dist=10):
    """Creates an op to introspect quality of image generation.

    The core philosophy is that given some attributest that we condition on, the
    model should be able to generate images which it can itself identify as
    corresponding to the attributes or semantics that we specified. This is done
    by first using a class label to generate an image, and then performing
    inference to get the latent representation for the image we conditioned on.
    Given this latent distribution, we draw multiple samples from the latent
    representation to get multiple candidate predictive densities. The final
    predictive density is the average of the predictive densities. Given the
    predictive density, we do argmax to predict a label and compute the hamming
    distance between the predicted label and the original label we provided.
    Approaches which have a low value for this metric are better.

    Args:
      num_samples_predictive_dist: Int, number of samples to draw from the
        latent state before making predictions.
    Returns:
      hamming_distance: [batch_size] `Tensor` of the hamming distance between
        true labels and predictions for each datapoint in the minibatch.
    """
    encoder_label = self._get_infer_latent_fns(modality='label')
    latent = encoder_label.density.sample()
    decoder_image = self._get_conditional_sampling_fns(latent, modality='image')
    decoder_image = check_mean_or_sample(decoder_image, 'sample')

    # Feed the decoded image back through the inference network.
    # pylint: disable=protected-access
    comprehended_latent_density, _ = self._vae._encoder_x(decoder_image)
    # pylint: enable=protected-access

    attribute_predictive_dist = []
    for s in xrange(num_samples_predictive_dist):
      latent = comprehended_latent_density.sample()
      decoder_attributes_dist = self._get_conditional_sampling_fns(
          latent, modality='label')
      for index, attribute_pred in enumerate(decoder_attributes_dist):
        if s == 0:
          attribute_predictive_dist.append(attribute_pred.probs)
        else:
          attribute_predictive_dist[index] = (
              attribute_predictive_dist[index] * s + attribute_pred.probs) / (
                  s + 1)

    hamming_distance = tf.zeros(tf.shape(self._labels[0]), dtype=tf.int32)

    label_mask_to_use = self.get_label_mask()
    for attribute_dist, gt_label, mask in zip(attribute_predictive_dist,
                                              self._labels, label_mask_to_use):
      pred_attribute = tf.to_int32(tf.arg_max(attribute_dist, dimension=1))
      hamming_distance += tf.to_int32(
          tf.not_equal(pred_attribute * mask, gt_label * mask))

    return hamming_distance

  def generate_image_conditioned_image_label(self, mean_or_sample='sample'):
    """Generate an image conditioned on image and label.

    This runs the model in a sort of "autoencoding mode" where we sample from
    z ~ q(z| x, y) and pass it through the generator to get a sampling density.

    Args:
      mean_or_sample: Whether to return a sample from the predictive density or
        the mean value.
    Returns:
      A `Tensor` indicating either a sample or the mean of the sampling
        distribution.
    Raises:
      ValueError: If mean_or_sample is not in 'sample' or 'mean' or 'both'
    """
    encoder_image_label = self._get_infer_latent_fns(modality='image_and_label')
    latent = encoder_image_label.density.sample()
    decoder_image = self._get_conditional_sampling_fns(latent, modality='image')

    return check_mean_or_sample(decoder_image, mean_or_sample)

  def generate_image_conditioned_image(self, mean_or_sample='sample'):
    """Generate an image conditioned on image.

    This runs the model in an "autoencoding mode" where we sample from
    z ~ q(z| x) and pass it through the generator to get a sampling density.

    Args:
      mean_or_sample: Whether to return a sample from the predictive density or
        the mean value.
    Returns:
      A `Tensor` indicating either a sample or the mean of the sampling
        distribution.
    Raises:
      ValueError: If mean_or_sample is not in 'sample' or 'mean' or 'both'
    """
    encoder_image = self._get_infer_latent_fns(modality='image')
    latent = encoder_image.density.sample()
    decoder_image = self._get_conditional_sampling_fns(latent, modality='image')

    return check_mean_or_sample(decoder_image, mean_or_sample)

  @property
  def joint_inference_network(self):
    return self._get_infer_latent_fns(modality='image_and_label')

  @property
  def latent_conditioned_image(self):
    """Return the density in the latent space conditioned on image.

    Returns:
      latent_z: tf.contrib.distributions.Normal object
    """
    return self._get_infer_latent_fns(modality='image')

  @property
  def latent_conditioned_label(self):
    """Return the density in the latent space conditioned on label.

    Returns:
      latent_z: tf.contrib.distributions.Normal object
    """
    return self._get_infer_latent_fns(modality='label')

  @property
  def is_training(self):
    return self._mode == 'train'

  @property
  def num_classes_per_attribute(self):
    return self._num_classes_per_attribute

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def is_multi_stage(self):
    return self.config.loss_type == 'multi_stage'

  @property
  def num_samples(self):
    return self._num_samples

  @property
  def inference_label_ph(self):
    return self._inference_label_ph

  @property
  def inference_image_ph(self):
    return self._inference_image_ph

  @property
  def true_latents(self):
    return self._true_latents

  @property
  def ignore_label_mask_ph(self):
    if self.product_of_experts is False:
      logging.warn('Product of experts is set to False, so the label mask '
                   'placeholder is unused.')
    return self._ignore_label_mask_ph

  @property
  def gmm_moment_and_label_placeholders(self):
    return self._mean_ph, self._var_ph, self._label_ph

  @property
  def label_inference_loss(self):
    return self._label_inference_loss

  @property
  def loss(self):
    return self._loss

  @property
  def model(self):
    return self._vae
