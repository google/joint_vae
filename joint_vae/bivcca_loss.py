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

r"""Loss for a Multimodal Variational Autoencoder with bi-VCCA.

The bi-VCCA objective is inspired by the description from the following
paper:
  Wang, Weiran, Xinchen Yan, Honglak Lee, and Karen Livescu. 2016.
  ``Deep Variational Canonical Correlation Analysis.'' arXiv [cs.LG].
  arXiv. http://arxiv.org/abs/1610.03454.

The bi-VCCA objective is different from the other objectives described
in the paper in the sense that it describes how we can train a different
inference network for each modality of interest.

Author: vrama@
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from joint_vae import utils
from joint_vae import loss


# TODO(hmoraldo): stop using x, y explicitly anywhere (just like in jmvae_loss).
class BiVccaLoss(loss.Loss):
  r"""A class for the loss of a Joint Multimodal Variational Autoencoder.

  A joint multimodal variational autoencoder optimizes the following objective:
  Given modalities x and y, we want to learn a joint probabilistic model of
  x, and y, marginalizing over latents 'z'.

  This loss describes a variational
  approach to the problem, using inference networks q(z| x) and
  q(z| y).

  \mu E_{z~q(z| x)} [\log p(x|z) + \log p(y| z) - KL(q(z|x)|| p(z))]
    + (1-\mu) E_{z~q(z|x)} [\log p(x|z) + \log p(y|z) - KL(q(z|y)||p(z))]
  """

  def __init__(
      self,
      encoders,
      decoders,
      prior,
      alpha_x=1.0,
      alpha_y=1.0,
      mu_tradeoff=0.5,
      joint_distribution_index=0,  # TODO(hmoraldo): should be list.
      name='bi_vcca',
      mode='train',
      path_derivative=False,
      add_summary=True):
    """Initialises the components of the module.

    Args:
      encoders: An Encoders object.
      decoders: A Decoders object.
      prior: Callable returning `p(z|v)`.
      alpha_x: list of scaling factors, for the likelihood corresponding to each
        encoder / decoder.
      alpha_y: list of scaling factors, for the likelihood corresponding to each
      mu_tradeoff: Tradeoff between the likelihood weighting for q(z| x)
        inference network and the q(z| y) inference network.
      joint_distribution_index: index of the encoder to use for sampling, and
        to treat specially as joint distribution when computing the loss.
      name: Name of the BiVcca.
      mode: string, name of the mode we are using 'train', 'val', 'inference'
      path_derivative: Boolean, True uses path derivative False uses analytical
        KL based derivative.
      add_summary: Whether to spawn summary ops that record the value of
        each term in the loss.

    Raises:
      ValueError: If alpha_x, alpha_y are negative.
    """
    if mu_tradeoff > 1.0 or mu_tradeoff <= 0.0:
      raise ValueError('Invalid value for tradeoff parameter mu: %f',
                       mu_tradeoff)

    alphas = [alpha_x, alpha_y]  # Assumes that the order is 'x' and 'y'

    super(BiVccaLoss, self).__init__(
        encoders, decoders, prior, alphas, name, mode, add_summary)
    self._cache = {}
    self._mu_tradeoff = mu_tradeoff
    self._path_derivative = path_derivative
    self._joint_distribution_index = joint_distribution_index

  def _log_pdf_elbo_components(self, inputs, v=None):
    r"""Calculates components for the ELBO and for JMVAE-kl objectives.

    Args:
      inputs: List of `Tensors`, each of size size `[B, ...]` and with input
        observations for a sigle modality.
      v: The covariate to condition the inference over, e.g. labels.

    Returns:
      log_p_q: an array such that log_p_q[i][j] contains a `Tensor` with
        log p(var[j]| z), where z ~ q(z| var[i]).
      kls: a `Tensor` with KL divergence for inputs[i].
    """
    with tf.name_scope('{}_log_pdf_elbo'.format(self.scope_name)):
      # Calculate sampling KL and keep z around.
      kls, zs = self._kl_and_z(inputs, v)
      log_p_q = [[
          utils.compute_likelihood(p, inpt) * alpha
          for inpt, p, alpha in zip(inputs,
                                    self._decoders.predict(z, v), self._alphas)
      ] for z in zs]

    return log_p_q, kls

  def log_pdf_elbos(self, inputs, v=None):
    """Compute the evidence lower bound (ELBO).

    Args:
      inputs: list of `Tensors`, each an observed modality.
      v: `Tensor`, (optional) when conditioning generator on some other
        modality `v`.
    Returns:
      multimodal_pdf_elbo: [B, n] `Tensor`, each column has elbo for a given
        modality, and n = len(inputs).
    """
    log_likelihoods, kls = self._log_pdf_elbo_components(inputs, v)

    # Build combined evidence lower bound.
    bivcca_cols = [
        tf.reduce_sum(
            [
                l for j, l in enumerate(ls)
                if j != self._joint_distribution_index
            ],
            axis=0) - kl
        for i, (ls, kl) in enumerate(zip(log_likelihoods, kls))
        if i != self._joint_distribution_index
    ]
    bivcca = tf.stack(bivcca_cols, axis=1, name='concat_bivcca')

    # Add summaries.
    if self._add_summary:
      for i, _ in enumerate(bivcca_cols):
        utils.scalar_summary_with_scope('bivcca_%d' % i,
                                        tf.reduce_mean(bivcca[:, i]),
                                        self._mode)

    return bivcca

  def build_nelbo_loss(self, inputs, mask=None, v=None):
    """Compute loss for the VAE, supports semi-supervised training.

    Compute the loss for a minibatch of training samples. Also provides
    the option to give a mask to specify which terms in the loss to
    use for each data point, allowing, say semi-supervised learning through
    the mask.
    Args:
      inputs: List of input observations, `Tensor` of size `[B, ...]``.
      mask: Stacked tensor of  [B, 2] `Tensors`, each specifying the weight
        for a term in the bi-vcca objective. The values in the mask are scaled
        to lie between 0 to 1 before using.
      v: `Tensor`, (optional) when conditioning generator on some other
        modality `v`.
    Returns:
      loss: [1] `Tensor`
    """
    bivcca = self.log_pdf_elbos(inputs)

    if mask is None:
      batch_size = tf.shape(inputs[0], out_type=tf.int32)[0]
      mask = tf.ones((batch_size, 2))

    mask *= tf.constant(
        [self._mu_tradeoff, 1 - self._mu_tradeoff], dtype=tf.float32)
    # Normalize each row of the mask to sum to 1 after weighing
    # the losses.
    mask /= tf.reduce_sum(mask, axis=1, keep_dims=True)

    loss_tensor = -1 * tf.reduce_mean(tf.reduce_sum(bivcca * mask, axis=1))

    if self._add_summary:
      utils.scalar_summary_with_scope('NELBO', loss_tensor, self._mode)

    return loss_tensor
