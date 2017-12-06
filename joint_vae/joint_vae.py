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

"""Joint Variational Auto-encoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from sonnet.python.modules import base
from sonnet.python.modules import basic


class JointVae(base.AbstractModule):
  r"""A class for a generic Joint Variational Autoencoder.

  A joint multimodal variational autoencoder optimizes the following objective:
  Given modalities x and y, we want to learn a joint probabilistic model of
  x, and y, marginalizing over latents 'z'. This code describes a variational
  approach to the problem, using inference networks q(z| x, y), q(z| x) and
  q(z| y).

  Users need to provide three encoders and two decoders as callable objects that
  return distributions. The encoders take as inputs the observations `x` or
  `y`, or both `x` and `y` as inputs and provide a distribution over the latent
  state as output. The decoders take a sample from the latent state and provide
  densities over the observed variables `x` and `y` as output.

  The components of the Variational Auto-Encoder can be created as follows:

  ```python
  n_z = 2
  n_enc = 128
  n_dec = 128
  size_x = 2
  size_y = 2

  def encoder_fn(x, unused_v):
    mlp_encoding = nets.MLP(
        name='mlp_encoder',
        output_sizes=[n_enc, 2*n_z],
        activation=tf.tanh)
    encoder_dist = gen.LogStddevNormal(mlp_encoding(x))

    return encoder_dist, encoder_dist.sample()

  def xy_encoder_fn(x, y, unused_v):
    mlp_encoding = nets.MLP(
        name='joint_mlp_encoder',
        output_sizes=[n_enc, 2*n_z],
        activation=tf.tanh)
    encoder_dist = gen.LogStddevNormal(mlp_encoding(x))

    return encoder_dist, encoder_dist.sample()

  # Three encoder components.
  encoder_x = snt.Module(encoder_fn, name='encoder_x')
  encoder_y = snt.Module(encoder_fn, name='encoder_y')
  encoder_xy = snt.Module(xy_encoder_fn, name='encoder_xy')

  encoders = encoder_decoder.Encoders(
    [encoder_x, encoder_y], encoder_xy , product_of_experts_list)
  ```

  where `product_of_experts` is a list of booleans, specifying whether to
  instantiate a product of experts label encoder for the corresponding encoder.

  A decoder with bernoulli likelihood can be created as follows:

  ```python
  def decoder_x_fn(latent, unused_v):
    x = nets.MLP(name='decoder_x',
                 output_sizes=[n_dec, size_x],
                 activation=tf.tanh)
    return tf.contrib.distributions.Bernoulli(logits=x)

  def decoder_y_fn(latent, unused_v):
    y = nets.MLP(name='decoder_y',
                 output_sizes=[n_dec, size_y],
                 activation=tf.tanh)
    return tf.contrib.distributions.Bernoulli(logits=y)

  # Two decoder components.
  decoder_x = snt.Module(decoder_x_fn, name='decoder_x')
  decoder_y = snt.Module(decoder_y_fn, name='decoder_y')

  decoders = encoder_decoder.Decoders([decoder_x, decoder_y])
  ```

  Given all these components, the Auto-Encoder can be created and used
  as follows:

  ```python
  vae = joint_vae.JointVae(encoders, decoders, prior, loss, name='joint_vae')
  ```

  Note that for the decoder, which provides the pixel distribution, the
  generative.QuantizedNormal distribution should be used instead of
  tf.contrib.distributions.Normal, to properly account for the fact that the
  data is discrete, if using a Gaussian likelihood.
  """

  def __init__(self,
               encoders,
               decoders,
               prior,
               loss,
               name='multimodal_elbo'):
    """Initialises the components of the module.

    Args:
      encoders: An Encoders object.
      decoders: A Decoders object.
      prior: Callable returning `p(z|v)`.
      loss: An object specifying the loss function for the
        Variational Auto-encoder.
      name: Name of the MultimodalElbo.
    """
    super(JointVae, self).__init__(name=name)
    self._encoders = encoders
    self._decoders = decoders
    self._prior = prior
    self._loss = loss
    self.name = name

  def sample(self, n, v=None, return_latent=False):
    """Draws samples from the learnt distributions p(x,y).

    Args:
      n: Number of samples to return.
      v: The covariate to condition the inference over, e.g. labels.
      return_latent: True, returns the latents used to sample each data point.

    Returns:
      A list of tensor, where element i has been produced by the corresponding
      decoder. Each sample `Tensor` comes from `p(x| z)` and has size
      `[B*N, ...]` where `N` is the number of samples, `B` is the batch size of
      the covariate `v` and `...` represents the shape of the observations.
    """
    with tf.name_scope('{}_sample'.format(self.name)):
      z = self.compute_prior(v).sample(n)  # [B] dimensional batch_size.
      if v is None:
        assert z.get_shape().as_list()[0] == n, (
            'In the unconditional case, the prior must correspond to a '
            'batch size of 1.')
      z = basic.MergeDims(start=0, size=2, name='merge_z')(z)

      samples = []
      for prediction in self.predict(z, v):
        this_sample = []
        if isinstance(prediction, list):
          this_sample = [p.sample() for p in prediction]
        else:
          this_sample = prediction.sample()
        samples.append(this_sample)

      if return_latent:
        samples.append(z)
    return samples

  def infer_latent(self, input_list, masks=None, v=None):
    """Performs inference over the latent variables.

    Args:
      input_list: List of tensors, each with size [batch, ...]. Each tensor
        represents an observed variable, that is input to the corresponding
        encoder.
      masks: List of tensors, each of size [B, len(y)]. masks[i] specifies
        which dimensions to ignore in a product of experts model with
        encoder self._encoders[i].
      v: The covariate to condition the inference over, e.g. labels.

    Returns:
      A list of samples, each produced by the corresponding encoder. Samples are
      of size `[N, B, latent_dim] where `B` is the batch size of `x`
      or `y`, and `N` is the number of samples asked and `latent_dim represents
      the size of the latent variables.
    """
    return self._encoders.infer_latent(input_list, masks, v)

  def predict(self, z, v=None, trainable_list=None):
    """Computes prediction over the observed variables.

    Args:
      z: latent variable, a `Tensor` of size `[B, ...]`.
      v: The covariate to condition the prediction over, e.g. labels.
      trainable_list: whether the returned network for decoders[i]
        should be trainable.

    Returns:
      The list of distributions `p(x|z)`, `p(y| z)`, ..., which on sample
      each one produces tensor of size `[N, B, ...]` where `N` is the number
      of samples asked.
    """
    return self._decoders.predict(z, v, trainable_list)

  def compute_prior(self, v=None):
    """Computes prior over the latent variables.

    Args:
      v: The covariate to condition the prior over, e.g. labels.

    Returns:
      The distribution `p(z)`, which on sample produces tensors of size
      `[N, ...]` where `N` is the number of samples asked and `...` represents
      the shape of the latent variables.
    """
    return self._prior(v)

  def build_nelbo_loss(self, inputs, mask=None, v=None):
    return self._loss.build_nelbo_loss(inputs, mask, v)

  def _build(self):
    raise NotImplementedError('_build method not yet implemented.')
