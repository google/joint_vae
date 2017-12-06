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

r"""Loss for a Multimodal Variational Autoencoder with multimodal-ELBO.

Describes the loss of a variational autoencoder for two modalities. The
objective function we optimize currently optimizes a convex combination of three
evidence lower-bounds (ELBOs) on the original likelihood functions.
Thus our objective function looks like:
  log p(x) + log p(y) + log p (x, y)

This is hard so we optimize a multimodal evidence lower bound:
  \alpha_xy E_{z~q(z| x, y)} [\alpha_x \log p(x|z) + \alpha_y \log p(y| z)
    - KL(q(z|x, y)|| p(z))]
    +  E_{z~q(z|x)} [\alpha_x \log p(x|z) - KL(q(z|x)||p(z))]
    +  E_{z~q(z|y)} [\alpha_y \log p(y|z) - KL(q(z|y)||p(z))]

alpha_x and alpha_y scale the conditional likelihoods p(x|z) and p(y|z) by
specified constants.

TODO(hmoraldo): update these doc, in particular, remove the Colab link or update
the notebook.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from joint_vae import utils
from joint_vae import loss



# TODO(hmoraldo): stop using x, y explicitly anywhere (just like in jmvae_loss).
class MultimodalElboLoss(loss.Loss):
  r"""A class for the loss of a Joint Multimodal Variational Autoencoder.

  The optimized objective to optimize is as follows:
  Given modalities x and y, we want to learn a joint probabilistic model of
  x, and y, marginalizing over latents 'z'. This code describes a variational
  approach to the problem, using inference networks q(z| x, y), q(z| x) and
  q(z| y).

  \alpha_xy E_{z~q(z| x, y)} [\log p(x|z) + \log p(y| z) - KL(q(z|x, y)|| p(z))]
    + \alpha_y E_{z~q(z|x)} [\log p(x|z) - KL(q(z|x)||p(z))]
    + \alpha_x E_{z~q(z|y)} [\log p(y|z) - KL(q(z|y)||p(z))]
  """

  def __init__(self,
               encoders,
               decoders,
               prior,
               alpha_x=1.0,
               alpha_y=1.0,
               alpha_y_elbo_y=0.0,
               rescale_amodal_x=False,
               predict_both=False,
               stop_gradient_y=False,
               stop_gradient_x=False,
               name='multimodal_elbo',
               mode='train',
               path_derivative=False,
               add_summary=True):
    """Initializes the components of the module.

    Args:
      encoders: An Encoders object.
      decoders: A Decoders object.
      prior: Callable returning `p(z|v)`.
      alpha_x: Scaling factor for likelihood corresponding to x.
      alpha_y: Scaling factor for likelihood corresponding to y.
      alpha_y_elbo_y: Extra scaling factor just for the likelihood p(y|q(z|y)).
      rescale_amodal_x: If true, for p(x|q(z|x)) use alpha_x/alpha_y_elbo_y
        as scaling factor.
      predict_both: Whether to predict both modalities when one of them
        is dropped out. That is, when predict_both is set to true,
        we sample z ~ q(z|y), and maximize likelihood of log p(x| z)
        and log p(y| z).
      stop_gradient_y: Boolean, specifies whether p(y| z) should get gradients
        from classifying z ~ q(z| y).
      stop_gradient_x: Boolean, specifies whether p(x| z) should get gradients
        from classifxing z ~ q(z| x).
      name: Name of the MultimodalElbo.
      mode: string, name of the mode we are using 'train', 'val', 'inference'
      path_derivative: Boolean, True uses path derivative False uses analytical
        KL based derivative.
      add_summary: Whether to spawn summary ops that record the value of
        each term in the loss.

    Raises:
      ValueError: If alpha_x, alpha_y or alpha_xy are negative.
    """
    alphas = [None]
    if alpha_y_elbo_y == 0.0:
      alpha_y_elbo_y = alpha_y

    super(MultimodalElboLoss, self).__init__(
        encoders, decoders, prior, alphas, name, mode, add_summary)
    self._alpha_x = alpha_x
    self._alpha_y = alpha_y
    self._alpha_y_elbo_y = alpha_y_elbo_y
    self._rescale_amodal_x = rescale_amodal_x
    self._path_derivative = path_derivative
    self._predict_both = predict_both
    self._stop_gradient_y = stop_gradient_y
    self._stop_gradient_x = stop_gradient_x

  def _log_pdf_elbo_components(self, x, y, v=None):
    r"""Calculates a components for the ELBO and for JMVAE-kl objectives.

    Args:
      x: Input observations for `x`, `Tensor` of size `[B, ...]``.
      y: Input observations for `y`, `Tensor` for size `[B, ...]``.
      v: The covariate to condition the inference over, e.g. labels.

    Returns:
      all_log_likelihoods: named tuple with the following entries
        log_p_x_qxy: Tensor of [batch_size], computing log p(x| z)
          where z~q(z|x,y)
        log_p_x_qx: Tensor of [batch_size], computing log p(x| z) where
          z ~ q(z| x)
        log_p_x_qy: Tensor of [batch_size], computing log p(x| z) where
          z ~ q(z| y)
        log_p_y_qxy: Tensor of [batch_size], computing log p(y| z)
          where z~q(z| x,y)
        log_p_y_qx: Tensor of [batch_size], computing log p(y| z)
          where z~q(z| x)
        log_p_y_qy: Tensor of [batch_size], computing log p(y| z)
          where z~q(z| y)
      all_kl_terms: named tuple with the following entries
        kl_qxy_p: Tensor of [batch_size], computing KL(q(z|x,y)||p(z))
        kl_qx_p: Tensor of [batch_size], computing KL(q(z|x)||p(z))
        kl_qy_p: Tensor of [batch_size], computing KL(q(z|y)||p(z))
    """
    with tf.name_scope('{}_log_pdf_elbo'.format(self.scope_name)):
      # Calculate sampling KL and keep z around.
      kls, zs = self._kl_and_z([x, y, [x, y]], v)
      kl_qx_p, kl_qy_p, kl_qxy_p = kls
      z_x, z_y, z_xy = zs

      # Evaluate log_p under z~qxy.
      px, py = self._decoders.predict(z_xy, v)
      log_p_x_qxy = utils.compute_likelihood(px, x)
      log_p_y_qxy = utils.compute_likelihood(py, y)

      # Evaluate log_p under z~qx.
      # TODO(vrama): Make changes to also add stop gradient for log-y
      px, py = self._decoders.predict(
          z_x, v, trainable_list=[not self._stop_gradient_x, True])
      log_p_x_qx = utils.compute_likelihood(px, x)

      log_p_y_qx = utils.compute_likelihood(py, y)

      # Evaluate log_p under z~qy.
      px, py = self._decoders.predict(
          z_y, v, trainable_list=[True, not self._stop_gradient_y])
      log_p_x_qy = utils.compute_likelihood(px, x)
      log_p_y_qy = utils.compute_likelihood(py, y)

      # Scale all likelihood terms by alpha_x or alpha_y.
      log_p_x_qxy *= self._alpha_x

      if self._rescale_amodal_x:
        log_p_x_qx *= (self._alpha_x / self._alpha_y_elbo_y)
      else:
        log_p_x_qx *= self._alpha_x

      log_p_x_qy *= self._alpha_x

      log_p_y_qxy *= self._alpha_y
      log_p_y_qx *= self._alpha_y
      log_p_y_qy *= self._alpha_y_elbo_y
      if self._alpha_y is None:
        # TODO(iansf): Add hparam to select between different elbo scalings.
        # TODO(vrama): Make sure all these scalings make sense.
        log_p_y_qy *= self._alpha_y

      all_log_likelihoods = loss.LogLikelihoods(
          log_p_x_qxy, log_p_x_qx, log_p_x_qy, log_p_y_qxy, log_p_y_qx,
          log_p_y_qy)

      all_kl_terms = loss.KLTerms(kl_qxy_p, kl_qx_p, kl_qy_p)

    return all_log_likelihoods, all_kl_terms

  def log_pdf_elbos(self, x, y, v=None):
    """Compute the evidence lower bound (ELBO).

    Args:
      x: `Tensor`, observed modality 'x'.
      y: `Tensor`, observed modality 'y'.
      v: `Tensor`, (optional) when conditioning generator on some other
        modality `v`.
    Returns:
      multimodal_pdf_elbo: [B, 3] `Tensor`, each column has elbo for [x, y, xy]
    """
    ll, kl = self._log_pdf_elbo_components(x, y, v)
    # Build combined evidence lower bound.
    if self._predict_both:
      multimodal_pdf_elbo = tf.stack(
          (ll.log_p_x_qx + ll.log_p_y_qx - kl.kl_qx_p,
           ll.log_p_y_qy + ll.log_p_x_qy - kl.kl_qy_p,
           ll.log_p_x_qxy + ll.log_p_y_qxy - kl.kl_qxy_p),
          axis=1,
          name='concat_elbo')
    else:
      multimodal_pdf_elbo = tf.stack(
          (ll.log_p_x_qx - kl.kl_qx_p,
           ll.log_p_y_qy - kl.kl_qy_p,
           ll.log_p_x_qxy  +
               ll.log_p_y_qxy - kl.kl_qxy_p),
          axis=1,
          name='concat_elbo')

    # Add summaries.
    if self._add_summary:
      utils.scalar_summary_with_scope(
          'elbo_x', tf.reduce_mean(multimodal_pdf_elbo[:, 0]),
          self._mode
      )
      utils.scalar_summary_with_scope(
          'elbo_y', tf.reduce_mean(multimodal_pdf_elbo[:, 1]),
          self._mode
      )
      utils.scalar_summary_with_scope(
          'elbo_xy', tf.reduce_mean(multimodal_pdf_elbo[:, 2]),
          self._mode
      )

    return multimodal_pdf_elbo

  def build_sg_loss(self, x, y):
    multimodal_pdf_elbo = self.log_pdf_elbos(x, y)
    elbo_x, elbo_y, elbo_xy = tf.unstack(multimodal_pdf_elbo, axis=1)

    loss_1 = -tf.reduce_mean(elbo_x + elbo_xy)
    loss_2 = -tf.reduce_mean(elbo_y)

    return loss_1, loss_2

  def build_nelbo_loss(self, inputs, mask=None, v=None):
    """Compute loss for the VAE, supports semi-supervised training.

    Compute the loss for a minibatch of training samples. Also provides
    the option to give a mask to specify which terms in the loss to
    use for each data point, allowing, say semi-supervised learning through
    the mask.
    Args:
      inputs: List of input observations, `Tensor` of size `[B, ...]``.
      mask: [B, 3] `Tensor`, mask specifying if x, y or both x and y are
        present. For example, if the training data point just has x, the
        mask would be [1.0 0.0 0.0], for x and y it would be [1.0 1.0 1.0].
      v: `Tensor`, (optional) when conditioning generator on some other
        modality `v`.
    Returns:
      multimodal_df_elbo: [B, 3] `Tensor`, each column has elbo for [x, y, xy]
    """
    x, y, _ = inputs

    multimodal_pdf_elbo = self.log_pdf_elbos(x, y)
    if self._stop_gradient_y is 'REPLACE_ME':
      return self.build_sg_loss(x, y)

    if mask is None:
      mask = tf.ones((tf.shape(x, out_type=tf.int32)[0], 3))

    # Normalize each row of the mask to sum to 1 after weighing
    # the losses.
    mask /= tf.reduce_sum(mask, axis=1, keep_dims=True)

    loss_tensor = -1 * tf.reduce_mean(
        tf.reduce_sum(multimodal_pdf_elbo * mask, axis=1))

    if self._add_summary:
      utils.scalar_summary_with_scope('NELBO', loss_tensor, self._mode)

    return loss_tensor
