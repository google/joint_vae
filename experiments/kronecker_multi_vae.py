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

"""A kronecker multimodal variational autoencoder model in Sonnet.

Assembles components for a multimodal joint variational autoencoder,
constructing a model along with a multimodal nelbo loss function.
More details on the multmodal NELBO loss function can be found at:
  //research/vale/imagination/jmvae/multimodal_elbo_vae

Author: vrama@
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from joint_vae import multimodal_elbo_loss
from joint_vae import encoder_decoder

import convolutional_multi_vae

flags = tf.flags
FLAGS = tf.app.flags.FLAGS


# TODO(vrama): Provide sampling utilities.
class KroneckerMultiVae(convolutional_multi_vae.ConvolutionalMultiVae):
  """A class for Kronecker Multimodal Variational Autoencoder."""

  def __init__(self, config, mode="train"):
    """Initialization.

    Args:
      config: A configuration class specifying various options for constructing
        the model. See //research/vale/imagination/scratch.multimodal_vae/configuration.py
        for more details.
      mode: One of "train", "val" or "test". Used to set various options for
        graph construction and data loading depending upon the chosen split.
    """
    super(KroneckerMultiVae, self).__init__(config, mode)

  def _build_generative(self):
    """Construct the Kronecker Multimodal Variational Autoencoder.

    Assembles different comopnents for a multimodal variational autoencoder
    using a multimodal elbo objective.

    Outputs:
      self._vae: an object of `multimodal_elbo_vae.MultimodalElbo`.
    """
    kron_vae = kronecker_vae_components.KroneckerVaeComponents(
        num_latent=self.config.num_latent,
        output_channels=self.config.output_channels,
        kernel_shapes=self.config.kernel_shapes,
        strides=self.config.strides,
        paddings=self.config.paddings,
        activation_fn=self.config.activation_fn,
        use_batch_norm=self.config.use_batch_norm,
        mlp_layers=self.config.mlp_layers,
        dropout=self.config.dropout,
        keep_prob=self.config.keep_prob,
        vocab_sizes=self._num_classes_per_attribute,
        label_embed_dims=self.config.label_embed_dims,
        post_fusion_mlp_layers=self.config.post_fusion_mlp_layers,
        is_training=self.is_training)

    # The Multimodal ELBO VAE objective needs three encoders and two decoders,
    # each encoder corresponds to a missing-ness pattern of interest.
    image_encoder = kron_vae.get_image_encoder_network()
    label_encoder = kron_vae.get_label_encoder_network()
    image_label_encoder = kron_vae.get_image_label_encoder_network()

    image_decoder = kron_vae.get_image_decoder_network(
        self.config.image_likelihood)
    label_decoder = kron_vae.get_label_decoder_network(
        self.config.label_likelihood)

    prior = kron_vae.get_prior_network()

    self._vae = multimodal_elbo_vae.MultimodalElbo(
        image_encoder, label_encoder, image_label_encoder, image_decoder,
        label_decoder, prior)
