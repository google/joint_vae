"""Encoder decoder architectures for CelebA and MNIST-affine datasets.

For both CelebA and MNIST-affine datasets we instantiate three encoder and two
decoder networks which we require for training multi modality VAEs. More details
on the encoders and decoders can be found below:

  1) Image encoder: A generic CNN architecture based on the discriminator in
    a DCGAN architecture (Radford et.al.)
  2) Label encoder: Given a categorical input of varying sizes we should be
    able to do inference to get to a latent space.
  3) Image Label encoder: Takes as input the image and all the labels and
    does inference.
  4) Image decoder: A DCGAN based image genration architecture.
  5) Label decoder: Label decoder is a categorical or bernoulli likelihood
    estimator which takes as input a latent space and outputs the corresponding
    tf contrib distributions object.

Author: Ramakrishna Vedantam (vrama@)
"""
import tensorflow as tf
from joint_vae.encoder_decoder import ConvolutionalDecoder
from joint_vae.encoder_decoder import ConvolutionalEncoder
from joint_vae.encoder_decoder import NIPS17ProductOfExpertsEncoder
from joint_vae.encoder_decoder import NIPS17MultiLayerPerceptronLabelEncoder
from joint_vae.encoder_decoder import NIPS17CategoricalLabelDecoder

from joint_vae.encoder_decoder import NormalPrior
from joint_vae.encoder_decoder import MultiLayerPerceptronMultiModalEncoder
from joint_vae.encoder_decoder import Encoders
from joint_vae.encoder_decoder import Decoders

def get_nips_17_conv_multi_vae_networks(latent_dim,
                                         product_of_experts_y,
                                         decoder_output_channels,
                                         decoder_output_shapes,
                                         decoder_kernel_shapes,
                                         decoder_strides,
                                         decoder_paddings,
                                         encoder_output_channels,
                                         encoder_kernel_shapes,
                                         encoder_strides,
                                         encoder_paddings,
                                         activation_fn,
                                         use_batch_norm,
                                         mlp_layers,
                                         output_distribution,
                                         dropout,
                                         keep_prob,
                                    vocab_sizes,
                                    label_embed_dims,
                                         mlp_dim,
                                         is_training,
                                         l1_pyz=0.0,
                                        swap_out_mlp_relu_for_elu=True, # We replace all relus with elus for MNISTaffine
                                         universal_expert=True):
  """Gives a set of networks for the Convolutional Multi VAE."""

  tf.logging.warn('Swap out mlp relu for elu set to : %s' % (swap_out_mlp_relu_for_elu))
  # Build the image encoder and prior (the only ones that actually matches
  # the convolutional multi VAE architecture).
  def build_x_encoder(name, last_layer_size, outputs_per_dim):
    return ConvolutionalEncoder(
        last_layer_size,
        encoder_output_channels=encoder_output_channels,
        encoder_kernel_shapes=encoder_kernel_shapes,
        encoder_strides=encoder_strides,
        encoder_paddings=encoder_paddings,
        activation_fn=activation_fn,
        use_batch_norm=use_batch_norm,
        mlp_layers=mlp_layers,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        outputs_per_dim=outputs_per_dim,
        swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
        name=name)

  # Build the label encoder which takes as input a list of [batch_size]
  # categorical (one-hot) encoded label lists.
  def build_y_encoder(name, last_layer_size, outputs_per_dim):
    # TODO(vrama): Complete the details of this encoder.
    y_encoder_module = (NIPS17ProductOfExpertsEncoder if product_of_experts_y else
                        NIPS17MultiLayerPerceptronLabelEncoder)

    return y_encoder_module(
        mlp_dim,
        last_layer_size,
        vocab_sizes,
        label_embed_dims,
        universal_expert=universal_expert,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        outputs_per_dim=outputs_per_dim,
        swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
        name=name)

  x_encoder = build_x_encoder('inference_x', latent_dim, outputs_per_dim=2)
  y_encoder = build_y_encoder('inference_y', latent_dim, outputs_per_dim=2)

  xy_encoder = MultiLayerPerceptronMultiModalEncoder(
      [
          build_x_encoder('inference_x_xy', mlp_dim, outputs_per_dim=1),
          build_y_encoder('inference_y_xy', mlp_dim, outputs_per_dim=1)
      ],
      mlp_dim,
      latent_dim,
      name='inference_xy')

  # Image decoder this is a DCGAN like architecture.
  x_decoder = ConvolutionalDecoder(
      latent_dim,
      decoder_output_channels=decoder_output_channels,
      decoder_output_shapes=decoder_output_shapes,
      decoder_kernel_shapes=decoder_kernel_shapes,
      decoder_strides=decoder_strides,
      decoder_paddings=decoder_paddings,
      activation_fn=activation_fn,
      use_batch_norm=use_batch_norm,
      mlp_layers=mlp_layers,
      dropout=dropout,
      keep_prob=keep_prob,
      is_training=True,
      output_distribution=output_distribution,
      swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
      name='decoder_x')

  prior = NormalPrior(latent_dim, name='prior')

  # Label decoder (gives a list of densities), which will be handled
  # appropriately when computing the likelihood.
  y_decoder = NIPS17CategoricalLabelDecoder(
      mlp_dim,
      data_dims=vocab_sizes,
      l1_pyz=l1_pyz,
      dropout=dropout,
      keep_prob=keep_prob,
      is_training=is_training,
      swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
      name='decoder_y')

  encoders = Encoders([x_encoder, y_encoder, xy_encoder],
                      [0, product_of_experts_y, 0])
  joint_distribution_index=2
  decoders = Decoders([x_decoder, y_decoder])
  return encoders, decoders, prior, joint_distribution_index
