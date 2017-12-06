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

"""Loss for a Variational Autoencoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging

import tensorflow as tf

from sonnet.python.modules import base
from joint_vae import utils


LogLikelihoods = collections.namedtuple('LogLikelihoods', [
    'log_p_x_qxy',
    'log_p_x_qx',
    'log_p_x_qy',
    'log_p_y_qxy',
    'log_p_y_qx',
    'log_p_y_qy',
])
KLTerms = collections.namedtuple('KLTerms', ['kl_qxy_p', 'kl_qx_p', 'kl_qy_p'])


class Loss(base.AbstractModule):
  """A class for the loss of a Variational Auto-encoder."""

  def __init__(self,
               encoders,
               decoders,
               prior,
               alphas=None,
               name='loss',
               mode='train',
               add_summary=True):
    """Initialises the components of the module.

    Args:
      encoders: An Encoders object.
      decoders: A Decoders object.
      prior: Callable returning `p(z|v)`.
      alphas: List of scaling factors, for likelihood corresponding to
        encoder or decoder i.
      name: Name of the MultimodalElbo.
      mode: string, name of the mode we are using 'train', 'val', 'inference'
      add_summary: Whether to spawn summary ops that record the value of
        each term in the loss.

    Raises:
      ValueError: If alphas is empty.
    """
    if not alphas:
      raise ValueError('alphas cannot be empty.')

    super(Loss, self).__init__(name=name)
    self._encoders = encoders
    self._decoders = decoders
    self._prior = prior
    self._alphas = alphas
    self._cache = {}
    self._add_summary = add_summary
    self._mode = mode

  def log_pdf(self, inputs, v=None):
    """Redirects to log_pdf_elbo with a warning."""
    logging.warn('log_pdf is actually a lower bound')
    return self.log_pdf_elbos(inputs, v)

  def _build(self):
    raise NotImplementedError('_build method not yet implemented.')

  def _kl_and_z(self, inputs, v=None):
    """Returns analytical or sampled KL divergence and a sample.

    This will return the analytical KL divergence if one is available (as
    registered with `kullback_leibler.RegisterKL`), and if this is not available
    then it will return a sampled KL divergence (in this case the returned
    sample is the one used for the KL divergence).

    Args:
      inputs: Input observations, list of `Tensors` of size `[B, ...]`.
      v: The covariate to condition over, e.g. labels.

    Returns:
      Pair `(kl, samples)`, where `kl` is a list of KL divergences (each a
      `Tensor` with shape `[B]`, where `B` is the batch size), and `samples` is
      the list of samples taken with each available encoder.
    """
    prior = self._prior(v)
    latents = self._encoders.infer_latent(inputs, v)
    samples = [q.density.sample() for q in latents]

    if self._path_derivative:
      # Use the gradient proposed in the following paper:
      # https://arxiv.org/pdf/1703.09194v1.pdf
      # An Asymptotially zero-variance estimator for variational inference
      # Reoder et.al.
      logging.info('Using low variance path derivative.')
      densities = [utils.stop_gradient_and_synthesize_density(q.density)
                   for q in latents]
      kls_q_p = [d.log_prob(z, name='log_q_z_%s' % i) -
                 prior.log_prob(z, name='log_p_z_%s' % i)
                 for i, (d, z) in enumerate(zip(densities, samples))]
    else:
      kls_q_p = [tf.contrib.distributions.kl_divergence(q.density, prior)
                 for q in latents]

    # Reduce over all dimensions except batch.
    sum_axis = range(1, kls_q_p[0].get_shape().ndims)
    kls_q_p = [tf.reduce_sum(kl, sum_axis, name='kl_q_p_%s' % i)
               for i, kl in enumerate(kls_q_p)]

    return kls_q_p, samples
