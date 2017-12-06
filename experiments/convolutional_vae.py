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

"""Construct a convolutional variational autoencoder model in Sonnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from joint_vae import encoder_decoder
from joint_vae import utils

import convolutional_multi_vae

flags = tf.flags
FLAGS = tf.app.flags.FLAGS


class ConvolutionalVae(convolutional_multi_vae.ConvolutionalMultiVae):
  """A class for a convolutional unimodal variational autoencoder."""

  def __init__(self, config, mode='train'):
    """Initialization.

    Args:
      config: A configuration class specifying various options for constructing
        the model. See
        //research/vale/imagination/scratch.multimodal_vae/configuration.py
        for more details.
      mode: One of "train", "val" or "test". Used to set various options for
        graph construction and data loading depending upon the chosen split.
    """
    super(ConvolutionalVae, self).__init__(config, mode)

  def _build_generative(self):
    """Construct the Unimodal Variational Autoencoder.

    Assembles different comopnents for a variational autoencoder.

    Outputs:
      self._vae: an object of `generative.Vae`.
    """
    conv_vae = conv_vae_components.ConvVaeComponents(
        num_latent=self.config.num_latent,
        output_channels=self.config.output_channels,
        kernel_shapes=self.config.kernel_shapes,
        strides=self.config.strides,
        paddings=self.config_paddings,
        activation_fn=self.config.activation_fn,
        use_batch_norm=self.config.use_batch_norm,
        mlp_layers=self.config.mlp_layers,
        dropout=self.config.dropout,
        keep_prob=self.config.keep_prob,
        is_training=self.is_training,)

    encoder = conv_vae.get_encoder_network()
    decoder = conv_vae.get_decoder_network(self.config.image_likelihood)
    prior = conv_vae.get_prior_network()

    self._vae = generative.Vae(encoder, decoder, prior)

  def _build_loss(self):
    self._loss = -1 * tf.reduce_mean(self._vae.log_prob_elbo(self._images))

  def q_z_x(self, x):
    return self._vae.infer_latent(x)

  def p_x_z(self, z):
    return self._vae.predict(z)

  def sample(self, num_samples=None, mean=None):
    return self._vae.sample(num_samples, mean=mean)
