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

"""Loss for a Joint Multimodal Variational Auto-Encoder (JMVAE).

Closely follows the implementation in the following paper:
Joint Multimodal Learning with Deep Generative Models
Masahiro Suzuki, Kotaro Nakayama, Yutaka Matsuo
ArXiv: https://arxiv.org/abs/1611.01891
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf

from joint_vae import utils
from joint_vae import loss

# pylint: disable=not-callable


# TODO(hmoraldo): share more code between the three Loss subclasses.
class JmvaeLoss(loss.Loss):
  r"""A class for the loss of a Joint Multimodal Variational Autoencoder.

  The objective to optimize is the following:
  Given modalities x and y, we want to learn a joint probabilistic model of
  some inputs, and marginalizing over latents 'z'. This code describes a
  variational approach to the problem, using inference networks.
  The overall objective called (JMVAE-kl), from Suzuki et.al. for networks
   q(z| x, y), q(z| x) and q(z| y) is as follows:

  \max E_{z~q(z| x, y}} [\log p(y| z) + \log p(x| z) - log q(z| x, y)/p(z)\
    - \alpha * (\log q(z| x, y)/q(z| x) + \log q(z| x, y)/q(z|y)
  """

  def __init__(
      self,
      encoders,
      decoders,
      prior,
      alpha_x=1.0,
      alpha_y=1.0,
      jmvae_alpha=0.1,
      joint_distribution_index=0,  # TODO(hmoraldo): should be list.
      name='jmvae',
      mode='train',
      add_summary=True):
    """Initialises the components of the module.

    JMVAEs are built from a set of three objects described below.

    Args:
      encoders: an Encoders object.
      decoders: a Decoders object.
      prior: callable returning `p(z|v)`.
      alphas: list of scaling factors, for the likelihood corresponding to each
        encoder / decoder.
      jmvae_alpha: scalar; scaling factor for KL divergence in elbo.
      joint_distribution_index: index of the encoder to use for sampling, and
        to treat specially as joint distribution when computing the loss.
      name: name of the Jmvae.
      mode: 'train', 'val', or 'test'.
      add_summary: Whether to spawn summary ops that record the value of
        each term in the loss.
    """
    alphas = [alpha_x, alpha_y]

    super(JmvaeLoss, self).__init__(encoders, decoders, prior, alphas, name,
                                    mode, add_summary)
    self.jmvae_alpha = jmvae_alpha
    self._cache = {}
    self.joint_distribution_index = joint_distribution_index

  def log_pdf_elbo(self, inputs, v=None):
    """Construct the jmvae kl objective function.

    Args:
      inputs: List of input observations, each a `Tensor` of size `[B, ...]`.
      v: Placeholder.

    Returns:
      jmave_kl_objective: Tensor of size [1], returns the objective value for
        the evidence lower bound augmented with KL divergence that jmvae-kl
        computes.
    """
    log_ps, kl_ps = self._log_pdf_elbo_components(inputs, v)

    log_ps_no_joint = [
        p for i, p in enumerate(log_ps) if i != self.joint_distribution_index
    ]
    kl_ps_no_joint = [
        kl for i, kl in enumerate(kl_ps) if i != self.joint_distribution_index
    ]

    # Build evidence lower bound.
    elbo = tf.reduce_mean(
        tf.reduce_sum(log_ps_no_joint, axis=0) -
        kl_ps[self.joint_distribution_index])
    kl = tf.reduce_mean(tf.reduce_sum(kl_ps_no_joint, axis=0))
    jmvae_kl_objective = elbo - self.jmvae_alpha * kl

    if self._add_summary:
      utils.scalar_summary_with_scope('elbo', elbo, self._mode)
      utils.scalar_summary_with_scope('kl', kl, self._mode)

    return jmvae_kl_objective

  # TODO(vrama): Clean this, handle mask argument better.
  def build_nelbo_loss(self, inputs, mask=None, v=None):
    """Construct the final loss to train the JMVAE.

    Args:
      inputs: List of input observations, `Tensor` of size `[B, ...]``.
      mask: Placeholder, does not do anything.
      v: Placeholder, does not do anything.
    Returns:
      loss: [1] `Tensor`, loss to train the JMVAE.
    """
    if mask is not None:
      logging.warn('Masking is not implemented for JMVAE.')

    elbo = self.log_pdf_elbo(inputs)
    loss_tensor = -elbo

    if self._add_summary:
      utils.scalar_summary_with_scope('NELBO', loss_tensor, self._mode)
    return loss_tensor

  def _log_pdf_elbo_components(self, inputs, v=None):
    """Calculates a components for the ELBO and for JMVAE-kl objectives.

    Args:
      inputs: List of input observations, each a `Tensor` of size `[B, ...]`.
      v: The covariate to condition the inference over, e.g. labels.

    Returns:
      log_ps: List of [B] `Tensor`, each representing log p(x | z).
      kls: List of [B] `Tensor`, each representing a KL divergence.
    """
    with tf.name_scope('{}_log_pdf_elbo'.format(self.scope_name)):
      # Calculate sampling KL and keep z around.
      kls, z = self._kl_and_z(inputs, v)

      # Evaluate log_p.
      predictions = self._decoders.predict(z, v)
      log_ps = [
          alpha * utils.compute_likelihood(p, inpt)
          for alpha, p, inpt in zip(self._alphas, predictions, inputs)
      ]

    return log_ps, kls

  def _kl_and_z(self, inputs, v=None):
    """Returns analytical or sampled KL divergence and a sample.

    This will return the analytical KL divergence if one is available (as
    registered with `kullback_leibler.RegisterKL`), and if this is not available
    then it will return a sampled KL divergence (in this case the returned
    sample is the one used for the KL divergence).

    Args:
      inputs: List of input observations, each a `Tensor` of size `[B, ...]`.
      v: The covariate to condition over, e.g. labels.

    Returns:
      Pair `(kl, z)`, where `kl` is a list of KL divergences (each a `Tensor`
      with shape `[B]`, where `B` is the batch size), and `z` is the sample
      from the latent space used to compute it.
    """
    prior = self._prior(v)
    latents = self._encoders.infer_latent(inputs, v)
    # Always sample from the specified distribution to compute expectation.
    z = latents[self.joint_distribution_index].density.sample()

    try:
      q_joint = latents[self.joint_distribution_index].density
      kls_q_p = [
          tf.contrib.distributions.kl_divergence(
              q_joint, (prior
                        if i == self.joint_distribution_index else q.density))
          for i, q, in enumerate(latents)
      ]
    except NotImplementedError:
      logging.warn('Analytic KLD not available, using sampling KLD instead.')
      log_p_z = prior.log_prob(z, name='log_p_z')
      q_joint = latents[self.joint_distribution_index].density.log_prob(
          z, name='log_q_z_%s' % i)

      kls_q_p = [
          q_joint - (log_p_z if i == self.joint_distribution_index else
                     q.density.log_prob(z, name='log_qxy_z_%s' % i))
          for i, q in enumerate(latents)
      ]

    # Reduce over all dimension except batch. Assumes all kls have same shape.
    sum_axis = range(1, kls_q_p[0].get_shape().ndims)
    kl = [tf.reduce_sum(k, sum_axis, name='kl_q_p_%s' % i) for k in kls_q_p]

    return kl, z
