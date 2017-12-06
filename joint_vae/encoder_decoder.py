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
"""Objects for handling groups of encoders and decoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy

import numpy as np
import sonnet as snt
import tensorflow as tf

from sonnet.python.modules import nets
from joint_vae import utils
from joint_vae import mlp


def _check_valid_labels(labels, data_dims):
  """Check if labels are consistent with expected classes per attribute.

  Args:
    labels: list of `Tensor` of [batch_size]
    data_dims: `Tuple` of ints, of same length as labels.
  """
  if len(labels) != len(data_dims):
    raise ValueError("Length of labels and number of attributes doesnt match.")

  if len(labels[0].get_shape().as_list()) != 1:
    raise ValueError("Expect densely-encoded labels as input.")

# TODO(hmoraldo): split this file in encoder.py, decoder.py.
class MultiModalEncoder(snt.AbstractModule):
  """Abstract class for encoding multiple modalities together."""

  def __init__(self,
               encoder_list,
               mlp_dim,
               latent_dim,
               dropout=False,
               keep_prob=0.5,
               is_training=True,
               swap_out_mlp_relu_for_elu=False,
               name='multi_modal_encoder'):
    """Initializes multi modality encoder.

    Args:
      encoder_list: list of Encoder objects.
      mlp_dim: scalar, dimension to use for the hidden layers of the combined
        MLP.
      latent_dim: scalar, dimension of the latent state for the JMVAE.
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (in expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      name: string, name for the sonnet module.
    """
    super(MultiModalEncoder, self).__init__(name=name)
    self.encoder_list = encoder_list
    self.mlp_dim = mlp_dim
    self.latent_dim = latent_dim
    self.dropout = dropout
    self.keep_prob = keep_prob
    self.is_training = is_training

    self.swap_out_mlp_relu_for_elu = swap_out_mlp_relu_for_elu
    self.mlp_activation_fn = tf.nn.relu
    if self.swap_out_mlp_relu_for_elu:
      self.mlp_activation_fn = tf.nn.elu

  def _build(self, modalities, sentinel=None, v=None):
    """Builds the encoder network function using a closure."""
    raise NotImplementedError


class Encoder(snt.AbstractModule):
  """Abstract class for single modality encoders."""

  def __init__(self,
               latent_dim,
               dropout=False,
               keep_prob=0.5,
               is_training=True,
               outputs_per_dim=2,
               dont_project=False,
               swap_out_mlp_relu_for_elu=False,
               name='encoder'):
    """Initializes the encoder.

    Args:
      latent_dim: scalar, Dimension of the latent state for the JMVAE.
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (in expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      outputs_per_dim: Integer, how many dimensions to output per latent
        dimension. Default of two causes the last layer of the logits to have
        2 * latent_dim elements (typically used for Gaussian mean, stddev).
      dont_project: Option to refrain from projecting the encoders output to the
        latent state, and just use the pre-latent state activations.
      swap_out_mlp_relu_for_elu: Option to swap out non linearity in MLPs from
        ReLU to ELU.
      name: string, name for the sonnet module.
    """
    super(Encoder, self).__init__(name=name)
    self.latent_dim = latent_dim
    self.dropout = dropout
    self.keep_prob = keep_prob
    self.is_training = is_training
    self.outputs_per_dim = outputs_per_dim
    self.dont_project = dont_project
    self.swap_out_mlp_relu_for_elu = swap_out_mlp_relu_for_elu
    self.mlp_activation_fn = tf.nn.relu
    if self.swap_out_mlp_relu_for_elu:
      self.mlp_activation_fn = tf.nn.elu

  def build_logits(self, modality, sentinel=None, v=None, activate_final=False):
    """Creates the logits of a unimodal inference network.

    Args:
      modality: tensor of size batch_size x SIZE_OF_MODALITY, This argument is
        modality agnostic, that is, depending upon whether we pass in `x` or
        `y` we can construct the corresponding unimodal inference network.
        Note that these can still be wrapped in different sonnet modules so
        they wont share parameters.
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
      activate_final: Whether to apply the activation function to the last
        layer.

    Returns:
      latent_normal: A tf.contrib.distributions.Normal object
      A sample from latent_normal. Note that to draw a new sample one would
       have to call the sample() routine of the distributions object.

    Raises: NotImplementedError in the abstract class.
    """
    raise NotImplementedError

  def _build(self, modality, sentinel=None, v=None):
    """Creates a unimodal inference network. See build_logits for args."""
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    if self.dont_project is True or self.outputs_per_dim == 1:
      raise ValueError('Cannot call build for encoders when dont project is'
                       ' set to True or outputs_per_dim is set to 1.')

    latent_normal = utils.LogStddevNormal(self.build_logits(modality, v))

    return latent_normal, latent_normal.sample()


class Decoder(snt.AbstractModule):
  """Abstract class for decoders."""

  def __init__(self,
               dropout=False,
               keep_prob=0.5,
               is_training=True,
               swap_out_mlp_relu_for_elu=False,
               name='decoder'):
    """Initializes the decoder.

    Args:
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (in expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      name: string, name for the sonnet module.
    """
    super(Decoder, self).__init__(name=name)
    self.dropout = dropout
    self.keep_prob = keep_prob
    self.is_training = is_training
    self.skip_vars = None

    self.swap_out_mlp_relu_for_elu = swap_out_mlp_relu_for_elu
    self.mlp_activation_fn = tf.nn.relu
    if self.swap_out_mlp_relu_for_elu:
      self.mlp_activation_fn = tf.nn.elu

  def _build(self,
             latent,
             sentinel=None,
             v=None,
             trainable=True,
             assertions=True):
    """Constructs a decoder.

    Args:
      latent: [batch_size, self.latent_dim] tensor with latent state.
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
      trainable: Bool. Whether px should be trainable.
      assertions: Bool. Whether to add assertions to the graph that any
        non-trainable networks maintain exactly the same weights as the
        trainable networks. The assertions will only be added if dropout isn't
        on, because this comparison is useless when the graph is
        non-deterministic.

    Returns:
      A decoder network..

    Raises:
      RuntimeError: if called with trainable=False the first time it's called.
    """
    raise NotImplementedError

  # TODO(hmoraldo): avoid making the user implement this function, as suggested
  # by iansf in cl/162000700.
  def _build_conditionally_trainable_net(self,
                                         data,
                                         build_fn,
                                         trainable=True,
                                         assertions=True):
    """Builds a network that is conditionally trainable.

    Args:
      data: A tensor that is the input to the network to build.
      build_fn: A function that takes an input tensor, and an optional argument
        name_suffix, and returns a pair subgraph, variables (the latter being
        a list of variables used in the constructed subgraph). The name suffix
        is appended to the names of all modules that are used during creation.
      trainable: Bool. Whether the decoder should be trainable.
      assertions: Bool. See Decoder._build for details.

    Returns:
      The constructed network (a tensor).

    Raises:
      RuntimeError: if called with trainable=False the first time it's called.
    """
    if not self.skip_vars and not trainable:
      raise RuntimeError('You must call this function with trainable=True at '
                         'least once before calling with trainable=False')

    # Modifying for build_fn which can give either a tensor or a list of
    # Tensors.
    was_not_list = False
    x, orig_vars = build_fn(data)

    if not isinstance(x, list):
      x = [x]
      was_not_list = True

    if trainable and not self.skip_vars:
      _, self.skip_vars = build_fn(data, name_suffix='_skip_grads')

      trainable_vars = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
      for sv in self.skip_vars:
        if sv in trainable_vars:
          trainable_vars.remove(sv)

    if not trainable:
      assign_ops = []
      for ov, sv in zip(orig_vars, self.skip_vars):
        assign_ops.append(tf.assign(sv, ov))

      assert_ops = []
      with tf.control_dependencies(assign_ops):
        x_skip, _ = build_fn(data, name_suffix='_skip_grads')

        if not isinstance(x_skip, list):
          x_skip = [x_skip]

        if assertions and not (self.dropout and self.is_training):
          for this_x, this_x_skip in zip(x, x_skip):
            assert_ops.append(
                tf.assert_less(
                    tf.reduce_max(tf.abs(this_x - this_x_skip)),
                    0.001,
                    data=[this_x, this_x_skip],
                    summarize=500))
          for ov, sv in zip(orig_vars, self.skip_vars):
            assert_ops.append(tf.assert_equal(ov, sv, summarize=500))

        with tf.control_dependencies(assert_ops):
          out_x = []
          for this_x_skip in x_skip:
            out_x.append(tf.identity(this_x_skip))
          x = out_x

    # If the original output from the build function was not a list, then
    # return the first member of the created list as output.
    if was_not_list:
      x = x[0]

    return x


# TODO(hmoraldo): get rid of product_of_experts_list by creating the default
# mask within the Product of Experts encoder, when needed.
_Encoders = collections.namedtuple('Encoders',
                                   'encoders,product_of_experts_list')


class Encoders(_Encoders):
  """Tuple containing multiple encoders.

  Attributes:
      encoders: List of encoders, each a callable returning tuple
        `q(z|x)`, z~q(z|x).
      product_of_experts_list: List of booleans. If an element is True, the
        corresponding encoder (in the list) uses a product of experts
        label encoder (modality y), False instantiates a regular encoder.
  """
  __slots__ = ()

  def infer_latent(self, input_list, masks=None, v=None):
    """See `JointVae.infer_latent`."""
    densities = []
    if masks is None:
      masks = [None] * len(input_list)
    if not (len(input_list) == len(masks) == len(self.product_of_experts_list)
            == len(self.encoders)):
      raise ValueError('Dimensions must match: input_list(%s), masks(%s), '
                       'product_of_experts_list(%s), encoders(%s)' %
                       (len(input_list), len(masks),
                        len(self.product_of_experts_list), len(self.encoders)))

    for modality, enc, poe, mask in zip(input_list, self.encoders,
                                        self.product_of_experts_list, masks):
      if poe:
        if mask is None:
          # if modality is a list, it contains the different distributions
          # in the product, at each entry, so use that information to figure out
          # the size of the mask. Otherwise create the size of the modality.
          if isinstance(modality, list):
            mask = tf.ones((tf.shape(modality[0])[0], len(modality)))
          else:
            mask = tf.ones_like(modality)
        density, sample = enc(modality, mask_labels=mask, v=v)
      else:
        density, sample = enc(modality, v=v)

      densities.append(utils.LatentDensity(density, sample))

    return densities

  def set_is_training(self, is_training):
    for encoder in self.encoders:
      encoder.is_training = is_training


_Decoders = collections.namedtuple('Decoders', 'decoders')


class Decoders(_Decoders):
  """Tuple containing multiple decoders.

  Attributes:
    decoders: List of callables returning `p(x|z)`.
  """
  __slots__ = ()

  def predict(self, z, v=None, trainable_list=None):
    """See `JointVae.predict`."""
    if trainable_list is None:
      trainable_list = [True] * len(self.decoders)
    if len(self.decoders) != len(trainable_list):
      raise ValueError(
          'Dimensions must match: decoders(%s), trainable_list(%s)' %
          (len(self.decoders), len(trainable_list)))

    decoded_values = []
    for dec, trainable in zip(self.decoders, trainable_list):
      decoded_values.append(dec(z, v=v, trainable=trainable))

    return decoded_values

  def set_is_training(self, is_training):
    for decoder in self.decoders:
      decoder.is_training = is_training


class Prior(snt.AbstractModule):
  """Abstract class for priors."""

  def __init__(self, latent_dim, name='prior'):
    """Initialize Prior.

    Args:
      latent_dim: scalar, Dimension of the latent state.
      name: string, name for the sonnet module.
    """
    super(Prior, self).__init__(name=name)
    self.latent_dim = latent_dim

  def _build(self, unused_v=None):
    """Builds the prior."""
    raise NotImplementedError


class NIPS17MnistaMultiLayerPerceptronMultiModalEncoder(MultiModalEncoder):
  """Multi-modal encoder that uses a multi layer perceptron."""

  def _build(self, modalities, sentinel=None, v=None):
    """Constructs a multimodal encoder network.

    Args:
      modalities: List of tensors, with same length as self.encoder_list.
        Each of these will be encoded with the corresponding encoder in
        encoder_list, before being processed further.
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
    Returns:
      latent_normal: A tf.contrib.distributions.Normal object.
      A sample from latent_normal. Note that to draw a new sample one would
       have to call the sample() routine of the distributions object.

    Raises:
      ValueError: if arguments after sentinel aren't specified by Name.
    """
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    encoded = [
        e.build_logits(m, v=v, activate_final=True)
        for e, m in zip(self.encoder_list, modalities)
    ]

    concat_modalities = tf.concat(encoded, axis=1, name='concat_modalities')

    mu_sigma = mlp.MLP(
        output_sizes=[self.mlp_dim, self.mlp_dim, 2 * self.latent_dim],
        dropout=self.dropout,
        keep_prob=self.keep_prob,
        activation=self.mlp_activation_fn)(
            concat_modalities, is_training=self.is_training)
    latent_normal = utils.LogStddevNormal(mu_sigma)
    return latent_normal, latent_normal.sample()


# Specific implementations of encoders, decoders, etc.
class MultiLayerPerceptronMultiModalEncoder(MultiModalEncoder):
  """Multi-modal encoder that uses a multi layer perceptron."""

  def _build(self, modalities, sentinel=None, v=None):
    """Constructs a multimodal encoder network.

    Args:
      modalities: List of tensors, with same length as self.encoder_list.
        Each of these will be encoded with the corresponding encoder in
        encoder_list, before being processed further.
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
    Returns:
      latent_normal: A tf.contrib.distributions.Normal object.
      A sample from latent_normal. Note that to draw a new sample one would
       have to call the sample() routine of the distributions object.

    Raises:
      ValueError: if arguments after sentinel aren't specified by Name.
    """
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    encoded = [
        e.build_logits(m, v=v, activate_final=True)
        for e, m in zip(self.encoder_list, modalities)
    ]

    concat_modalities = tf.concat(encoded, axis=1, name='concat_modalities')

    mu_sigma = mlp.MLP(
        output_sizes=[self.mlp_dim, 2 * self.latent_dim],
        dropout=self.dropout,
        keep_prob=self.keep_prob,
        activation=self.mlp_activation_fn)(
            concat_modalities, is_training=self.is_training)
    latent_normal = utils.LogStddevNormal(mu_sigma)
    return latent_normal, latent_normal.sample()

class NIPS17MnistaMultiLayerPerceptronLabelEncoder(Encoder):
  """Single modality encoder that uses a multi layer perceptron."""

  def __init__(self,
               mlp_dim,
               latent_dim,
               data_dims,
               embed_labels_dim,
               universal_expert=False,
               dropout=False,
               keep_prob=0.5,
               is_training=True,
               outputs_per_dim=2,
               dont_project=False,
               swap_out_mlp_relu_for_elu=False,
               name='mlp_encoder'):
    """Initializes the MLP encoder.

    Args:
      mlp_dim: scalar, Dimension to use for the hidden layers of MLPs. Can be
        None.
      latent_dim: scalar, Dimension of the latent state for the JMVAE.
      data_dims: tuple of number of classes for each attribute.
      embed_labels_dim: the dimension to project/embed labels to before using
        them in the network.
      universal_expert: Bool. True specifies that an additional expert will
        be created that matches the prior.
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (in expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      outputs_per_dim: Integer, how many dimensions to output per latent
        dimension. Default of two causes the last layer of the logits to have
        2 * latent_dim elements (typically used for Gaussian mean, stddev).
      dont_project: Boolean, tells us whether we want to project down to the
        dimensionality of the latent space or not.
      name: string, name for the sonnet module.
    """
    super(NIPS17MnistaMultiLayerPerceptronLabelEncoder, self).__init__(
        latent_dim=latent_dim,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        outputs_per_dim=outputs_per_dim,
        dont_project=dont_project,
        swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
        name=name)
    self.mlp_dim = mlp_dim
    self.outputs_per_dim = outputs_per_dim
    self.universal_expert = universal_expert
    self.data_dims = data_dims
    self.embed_labels_dim = embed_labels_dim

  def build_logits(self, labels, sentinel=None, v=None, activate_final=False):
    """Creates the logits of a unimodal inference network.

    Args:
      labels: list of `Tensor` of size [batch_size],
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
      activate_final: Whether to apply the activation function to the last
        layer.

    Returns:
      latent_normal: A tf.contrib.distributions.Normal object
      A sample from latent_normal. Note that to draw a new sample one would
       have to call the sample() routine of the distributions object.

    Raises:
      ValueError: if arguments after sentinel aren't specified by Name.
    """
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    _check_valid_labels(labels, self.data_dims)

    # Get one-hot labels from labels.
    embedded_labels = []
    for label, num_classes in zip(labels, self.data_dims):
      label_embedding = snt.Embed(vocab_size=num_classes, embed_dim=self.embed_labels_dim)(
              label)
      embedded_labels.append(
          mlp.MLP(output_sizes=[self.mlp_dim, self.mlp_dim],
                  dropout=self.dropout,
                  keep_prob=self.keep_prob,
                  activation=self.mlp_activation_fn,
                  activate_final=True)(label_embedding, is_training=self.is_training))

    output_sizes = [self.mlp_dim, self.mlp_dim] if self.mlp_dim else []
    if not self.dont_project:
      output_sizes.append(self.outputs_per_dim * self.latent_dim)

    # Iterate over the labels, creating corresponding embeddings.
    concat_embedded_labels = tf.concat(
        embedded_labels, axis=-1, name='concat_labels')
    return mlp.MLP(
        output_sizes=output_sizes,
        dropout=self.dropout,
        keep_prob=self.keep_prob,
        activation=self.mlp_activation_fn,
        activate_final=activate_final)(
            concat_embedded_labels, is_training=self.is_training)


class NIPS17MnistaProductOfExpertsEncoder(Encoder):
  """Single modality encoder that uses a Product of Experts."""

  def __init__(self,
               mlp_dim,
               latent_dim,
               data_dims,
               embed_labels_dim,
               universal_expert=False,
               dropout=False,
               keep_prob=0.5,
               is_training=True,
               outputs_per_dim=2,
               dont_project=False,
               swap_out_mlp_relu_for_elu=False,
               name='mlp_encoder'):
    """Initializes the Product of Experts encoder.

    Args:
      mlp_dim: scalar, Dimension to use for the hidden layers of MLPs. Can be
        None.
      latent_dim: scalar, Dimension of the latent state for the JMVAE.
      data_dims: tuple of number of classes for each attribute.
      embed_labels_dim: the dimension to project/embed labels to before using
        them in the network.
      universal_expert: Bool. True specifies that an additional expert will
        be created that matches the prior.
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (in expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      outputs_per_dim: Integer, how many dimensions to output per latent
        dimension. Default of two causes the last layer of the logits to have
        2 * latent_dim elements (typically used for Gaussian mean, stddev).
      name: string, name for the sonnet module.
    """
    super(NIPS17MnistaProductOfExpertsEncoder, self).__init__(
        latent_dim=latent_dim,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        outputs_per_dim=outputs_per_dim,
        dont_project=dont_project,
        swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
        name=name)
    self.mlp_dim = mlp_dim
    self.embed_labels_dim = embed_labels_dim
    self.universal_expert = universal_expert
    self.data_dims = data_dims

  def build_logit_list(self, labels, v=None, activate_final=False):
    """Just like build_logits, but returns a list instead of concat logits.

    labels: A list of [batch_size] tensors with dense indexing of
      labels.
    v: An optional conditioning input (always expected to be None)
    activate_final: whehter to activate the last layer of the MLP.
    """

    # Iterate over the labels, creating corresponding embeddings.
    _check_valid_labels(labels, self.data_dims)

    # Get one-hot labels from labels.
    embedded_labels = []
    for label, num_classes in zip(labels, self.data_dims):
      label_embedding = snt.Embed(vocab_size=num_classes, embed_dim=self.embed_labels_dim)(
              label)
      embedded_labels.append(
          mlp.MLP(output_sizes=[self.mlp_dim, self.mlp_dim],
                  dropout=self.dropout,
                  keep_prob=self.keep_prob,
                  activation=self.mlp_activation_fn,
                  activate_final=True)(label_embedding, is_training=self.is_training))


    logit_list = []

    output_sizes = [self.mlp_dim, self.mlp_dim] if self.mlp_dim else []
    if not self.dont_project:
      output_sizes.append(self.outputs_per_dim * self.latent_dim)

    for index, embed_label in enumerate(embedded_labels):
      expert = mlp.MLP(
          output_sizes=output_sizes,
          dropout=self.dropout,
          keep_prob=self.keep_prob,
          activation=self.mlp_activation_fn,
          activate_final=activate_final,
          name='_expert' + str(index))

      logit_list.append(expert(embed_label, is_training=self.is_training))

    return logit_list

  def build_logits(self,
                   modality,
                   sentinel=None,
                   v=None,
                   mask_labels=None,
                   activate_final=False):
    """Creates the logits of a unimodal inference network.

    Args:
      modality: list of tensors of [batch_size]
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
      mask_labels: For masking inputs in PoE models.
      activate_final: Whether to apply the activation function to the last
        layer.

    Returns:
      latent_normal: A tf.contrib.distributions.Normal object
      A sample from latent_normal. Note that to draw a new sample one would
       have to call the sample() routine of the distributions object.

    Raises:
      ValueError: if arguments after sentinel aren't specified by Name.
    """
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    logit_list = self.build_logit_list(
        modality, v=None, activate_final=activate_final)
    return tf.concat(logit_list, axis=1, name='concat_logits')

  def _build(self, modality, sentinel=None, v=None, mask_labels=None):
    """Creates a unimodal inference network. See build_logits for args."""
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    logit_list = self.build_logit_list(modality, v=None)

    variational_posteriors = [
        utils.LogStddevNormal(label) for label in logit_list
    ]

    final_variational_posterior = (
        utils.multiply_gaussian_with_universal_expert_pdfs(
            variational_posteriors,
            mask_labels,
            universal_expert=self.universal_expert))

    return final_variational_posterior, final_variational_posterior.sample()



class NIPS17MultiLayerPerceptronLabelEncoder(Encoder):
  """Single modality encoder that uses a multi layer perceptron."""

  def __init__(self,
               mlp_dim,
               latent_dim,
               data_dims,
               embed_labels_dim,
               universal_expert=False,
               dropout=False,
               keep_prob=0.5,
               is_training=True,
               outputs_per_dim=2,
               swap_out_mlp_relu_for_elu=False,
               name='mlp_encoder'):
    """Initializes the MLP encoder.

    Args:
      mlp_dim: scalar, Dimension to use for the hidden layers of MLPs. Can be
        None.
      latent_dim: scalar, Dimension of the latent state for the JMVAE.
      data_dims: tuple of number of classes for each attribute.
      embed_labels_dim: the dimension to project/embed labels to before using
        them in the network.
      universal_expert: Bool. True specifies that an additional expert will
        be created that matches the prior.
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (in expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      outputs_per_dim: Integer, how many dimensions to output per latent
        dimension. Default of two causes the last layer of the logits to have
        2 * latent_dim elements (typically used for Gaussian mean, stddev).
      name: string, name for the sonnet module.
    """
    super(NIPS17MultiLayerPerceptronLabelEncoder, self).__init__(
        latent_dim=latent_dim,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        outputs_per_dim=outputs_per_dim,
        swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
        name=name)
    self.mlp_dim = mlp_dim
    self.outputs_per_dim = outputs_per_dim
    self.universal_expert = universal_expert
    self.data_dims = data_dims
    self.embed_labels_dim = embed_labels_dim

  def build_logits(self, labels, sentinel=None, v=None, activate_final=False):
    """Creates the logits of a unimodal inference network.

    Args:
      labels: list of `Tensor` of size [batch_size],
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
      activate_final: Whether to apply the activation function to the last
        layer.

    Returns:
      latent_normal: A tf.contrib.distributions.Normal object
      A sample from latent_normal. Note that to draw a new sample one would
       have to call the sample() routine of the distributions object.

    Raises:
      ValueError: if arguments after sentinel aren't specified by Name.
    """
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    _check_valid_labels(labels, self.data_dims)

    # Get one-hot labels from labels.
    embedded_labels = []
    for label, num_classes in zip(labels, self.data_dims):
      embedded_labels.append(snt.Embed(vocab_size=num_classes, embed_dim=self.embed_labels_dim)(
              label))

    output_sizes = [self.mlp_dim, self.mlp_dim] if self.mlp_dim else []
    output_sizes.append(self.outputs_per_dim * self.latent_dim)

    # Iterate over the labels, creating corresponding embeddings.
    concat_embedded_labels = tf.concat(
        embedded_labels, axis=-1, name='concat_labels')
    return mlp.MLP(
        output_sizes=output_sizes,
        dropout=self.dropout,
        keep_prob=self.keep_prob,
        activation=self.mlp_activation_fn,
        activate_final=activate_final)(
            concat_embedded_labels, is_training=self.is_training)


class MultiLayerPerceptronEncoder(Encoder):
  """Single modality encoder that uses a multi layer perceptron."""

  def __init__(self,
               mlp_dim,
               latent_dim,
               universal_expert=False,
               dropout=False,
               keep_prob=0.5,
               is_training=True,
               outputs_per_dim=2,
               swap_out_mlp_relu_for_elu=False,
               name='mlp_encoder'):
    """Initializes the MLP encoder.

    Args:
      mlp_dim: scalar, Dimension to use for the hidden layers of MLPs. Can be
        None.
      latent_dim: scalar, Dimension of the latent state for the JMVAE.
      universal_expert: Bool. True specifies that an additional expert will
        be created that matches the prior.
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (in expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      outputs_per_dim: Integer, how many dimensions to output per latent
        dimension. Default of two causes the last layer of the logits to have
        2 * latent_dim elements (typically used for Gaussian mean, stddev).
      name: string, name for the sonnet module.
    """
    super(MultiLayerPerceptronEncoder, self).__init__(
        latent_dim=latent_dim,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        outputs_per_dim=outputs_per_dim,
        swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
        name=name)
    self.mlp_dim = mlp_dim
    self.outputs_per_dim = outputs_per_dim
    self.universal_expert = universal_expert

  def build_logits(self, modality, sentinel=None, v=None, activate_final=False):
    """Creates the logits of a unimodal inference network.

    Args:
      modality: tensor of size batch_size x SIZE_OF_MODALITY, This argument is
        modality agnostic, that is, depending upon whether we pass in `x` or
        `y` we can construct the corresponding unimodal inference network.
        Note that these can still be wrapped in different sonnet modules so
        they wont share parameters.
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
      activate_final: Whether to apply the activation function to the last
        layer.

    Returns:
      latent_normal: A tf.contrib.distributions.Normal object
      A sample from latent_normal. Note that to draw a new sample one would
       have to call the sample() routine of the distributions object.

    Raises:
      ValueError: if arguments after sentinel aren't specified by Name.
    """
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    output_sizes = [self.mlp_dim] if self.mlp_dim else []
    output_sizes.append(self.outputs_per_dim * self.latent_dim)

    # Iterate over the labels, creating corresponding embeddings.
    modality = tf.contrib.layers.flatten(modality)
    return mlp.MLP(
        output_sizes=output_sizes,
        dropout=self.dropout,
        keep_prob=self.keep_prob,
        activation=self.mlp_activation_fn,
        activate_final=activate_final)(
            modality, is_training=self.is_training)


class NIPS17ProductOfExpertsEncoder(Encoder):
  """Single modality encoder that uses a Product of Experts."""

  def __init__(self,
               mlp_dim,
               latent_dim,
               data_dims,
               embed_labels_dim,
               universal_expert=False,
               dropout=False,
               keep_prob=0.5,
               is_training=True,
               outputs_per_dim=2,
               swap_out_mlp_relu_for_elu=False,
               name='mlp_encoder'):
    """Initializes the Product of Experts encoder.

    Args:
      mlp_dim: scalar, Dimension to use for the hidden layers of MLPs. Can be
        None.
      latent_dim: scalar, Dimension of the latent state for the JMVAE.
      data_dims: tuple of number of classes for each attribute.
      embed_labels_dim: the dimension to project/embed labels to before using
        them in the network.
      universal_expert: Bool. True specifies that an additional expert will
        be created that matches the prior.
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (in expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      outputs_per_dim: Integer, how many dimensions to output per latent
        dimension. Default of two causes the last layer of the logits to have
        2 * latent_dim elements (typically used for Gaussian mean, stddev).
      name: string, name for the sonnet module.
    """
    super(NIPS17ProductOfExpertsEncoder, self).__init__(
        latent_dim=latent_dim,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        outputs_per_dim=outputs_per_dim,
        swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
        name=name)
    self.mlp_dim = mlp_dim
    self.embed_labels_dim = embed_labels_dim
    self.universal_expert = universal_expert
    self.data_dims = data_dims

  def build_logit_list(self, labels, v=None, activate_final=False):
    """Just like build_logits, but returns a list instead of concat logits.

    labels: A list of [batch_size] tensors with dense indexing of
      labels.
    v: An optional conditioning input (always expected to be None)
    activate_final: whehter to activate the last layer of the MLP.
    """

    # Iterate over the labels, creating corresponding embeddings.
    _check_valid_labels(labels, self.data_dims)

    # Get one-hot labels from labels.
    embedded_labels = []
    for label, num_classes in zip(labels, self.data_dims):
      embedded_labels.append(
          snt.Embed(vocab_size=num_classes, embed_dim=self.embed_labels_dim)(
              label))

    logit_list = []

    output_sizes = [self.mlp_dim, self.mlp_dim] if self.mlp_dim else []
    output_sizes.append(self.outputs_per_dim * self.latent_dim)

    for index, embed_label in enumerate(embedded_labels):
      expert = mlp.MLP(
          output_sizes=output_sizes,
          dropout=self.dropout,
          keep_prob=self.keep_prob,
          activation=self.mlp_activation_fn,
          activate_final=activate_final,
          name='_expert' + str(index))

      logit_list.append(expert(embed_label, is_training=self.is_training))

    return logit_list

  def build_logits(self,
                   modality,
                   sentinel=None,
                   v=None,
                   mask_labels=None,
                   activate_final=False):
    """Creates the logits of a unimodal inference network.

    Args:
      modality: list of tensors of [batch_size]
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
      mask_labels: For masking inputs in PoE models.
      activate_final: Whether to apply the activation function to the last
        layer.

    Returns:
      latent_normal: A tf.contrib.distributions.Normal object
      A sample from latent_normal. Note that to draw a new sample one would
       have to call the sample() routine of the distributions object.

    Raises:
      ValueError: if arguments after sentinel aren't specified by Name.
    """
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    logit_list = self.build_logit_list(
        modality, v=None, activate_final=activate_final)
    return tf.concat(logit_list, axis=1, name='concat_logits')

  def _build(self, modality, sentinel=None, v=None, mask_labels=None):
    """Creates a unimodal inference network. See build_logits for args."""
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    logit_list = self.build_logit_list(modality, v=None)

    variational_posteriors = [
        utils.LogStddevNormal(label) for label in logit_list
    ]

    final_variational_posterior = (
        utils.multiply_gaussian_with_universal_expert_pdfs(
            variational_posteriors,
            mask_labels,
            universal_expert=self.universal_expert))

    return final_variational_posterior, final_variational_posterior.sample()


class ProductOfExpertsEncoder(Encoder):
  """Single modality encoder that uses a Product of Experts."""

  def __init__(self,
               mlp_dim,
               latent_dim,
               universal_expert=False,
               dropout=False,
               keep_prob=0.5,
               is_training=True,
               outputs_per_dim=2,
               swap_out_mlp_relu_for_elu=False,
               name='mlp_encoder'):
    """Initializes the Product of Experts encoder.

    Args:
      mlp_dim: scalar, Dimension to use for the hidden layers of MLPs. Can be
        None.
      latent_dim: scalar, Dimension of the latent state for the JMVAE.
      universal_expert: Bool. True specifies that an additional expert will
        be created that matches the prior.
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (in expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      outputs_per_dim: Integer, how many dimensions to output per latent
        dimension. Default of two causes the last layer of the logits to have
        2 * latent_dim elements (typically used for Gaussian mean, stddev).
      name: string, name for the sonnet module.
    """
    super(ProductOfExpertsEncoder, self).__init__(
        latent_dim=latent_dim,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        outputs_per_dim=outputs_per_dim,
        swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
        name=name)
    self.mlp_dim = mlp_dim
    self.universal_expert = universal_expert

  def build_logit_list(self, modality, v=None, activate_final=False):
    """Just like build_logits, but returns a list instead of concat logits."""
    # Iterate over the labels, creating corresponding embeddings.
    labels = tf.unstack(modality, axis=-1)
    logit_list = []

    output_sizes = [self.mlp_dim] if self.mlp_dim else []
    output_sizes.append(self.outputs_per_dim * self.latent_dim)

    for index, label in enumerate(labels):
      label = tf.expand_dims(label, axis=-1)
      expert = mlp.MLP(
          output_sizes=output_sizes,
          dropout=self.dropout,
          keep_prob=self.keep_prob,
          activation=self.mlp_activation_fn,
          activate_final=activate_final,
          name='_expert' + str(index))

      logit_list.append(expert(label, is_training=self.is_training))

    return logit_list

  def build_logits(self,
                   modality,
                   sentinel=None,
                   v=None,
                   mask_labels=None,
                   activate_final=False):
    """Creates the logits of a unimodal inference network.

    Args:
      modality: tensor of size batch_size x SIZE_OF_MODALITY, This argument is
        modality agnostic, that is, depending upon whether we pass in `x` or
        `y` we can construct the corresponding unimodal inference network.
        Note that these can still be wrapped in different sonnet modules so
        they wont share parameters.
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
      mask_labels: For masking inputs in PoE models.
      activate_final: Whether to apply the activation function to the last
        layer.

    Returns:
      latent_normal: A tf.contrib.distributions.Normal object
      A sample from latent_normal. Note that to draw a new sample one would
       have to call the sample() routine of the distributions object.

    Raises:
      ValueError: if arguments after sentinel aren't specified by Name.
    """
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    logit_list = self.build_logit_list(
        modality, v=None, activate_final=activate_final)
    return tf.concat(logit_list, axis=1, name='concat_logits')

  def _build(self, modality, sentinel=None, v=None, mask_labels=None):
    """Creates a unimodal inference network. See build_logits for args."""
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    logit_list = self.build_logit_list(modality, v=None)

    variational_posteriors = [
        utils.LogStddevNormal(label) for label in logit_list
    ]

    final_variational_posterior = (
        utils.multiply_gaussian_with_universal_expert_pdfs(
            variational_posteriors,
            mask_labels,
            universal_expert=self.universal_expert))

    return final_variational_posterior, final_variational_posterior.sample()


class ConvolutionalEncoder(Encoder):
  """Single modality encoder that uses a convolutional network."""

  def __init__(self,
               latent_dim,
               encoder_output_channels,
               encoder_kernel_shapes,
               encoder_strides,
               encoder_paddings,
               activation_fn,
               use_batch_norm,
               mlp_layers,
               dropout,
               keep_prob,
               is_training=True,
               outputs_per_dim=2,
               dont_project=False,
               swap_out_mlp_relu_for_elu=False,
               name='convolutional_encoder'):
    """Initialize Multimodal VAE components.

    Args:
      latent_dim: int, Number of latent states `z` in the model. Each latent
        state has a N(0, 1) prior.
      encoder_output_channels: list, Number of feature maps in different layers
        of a CNN model.
      encoder_kernel_shapes: list, size of the kernel in each layer of CNN.
      encoder_strides: list, stride in each layer of CNN.
      encoder_paddings: list, snt.SAME or snt.VALID, for each layer.
      activation_fn: self.mlp_activation_fn, tf.nn.elu etc. Activation function to use as
        the non-linearity in the network. Shared across all non-linearities in
        the net.
      use_batch_norm: Boolean, True uses batch norm, False switches it off.
      mlp_layers: list, Number of units to use in each layer of MLPs constructed
        at various points in the network.
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (in expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      outputs_per_dim: Integer, how many dimensions to output per latent
        dimension. Default of two causes the last layer of the logits to have
        2 * latent_dim elements (typically used for Gaussian mean, stddev).
      name: string, name for the sonnet module.
    """
    super(ConvolutionalEncoder, self).__init__(
        latent_dim=latent_dim,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        outputs_per_dim=outputs_per_dim,
        dont_project=dont_project,
        swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
        name=name)

    # General properties.
    self.activation_fn = activation_fn
    self.use_batch_norm = use_batch_norm

    # Properties of the Convolutional VAE.
    self.encoder_output_channels = encoder_output_channels
    self.encoder_kernel_shapes = encoder_kernel_shapes
    self.encoder_strides = encoder_strides
    self.encoder_paddings = encoder_paddings

    # Properties of the MLP.
    self.mlp_layers = mlp_layers

  def build_logits(self, images, sentinel=None, v=None, activate_final=False):
    """Builds the logits of an image encoder network.

    Args:
      images: A [batch_size, height, width, channels] `Tensor`.
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
      activate_final: Whether to apply the activation function to the last
        layer.

    Returns:
      tf.contrib.distributions.Normal object.
      Tensor, a sample from the returned distribution.

    Raises:
      ValueError: if arguments after sentinel aren't specified by Name.
    """
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    if len(images.get_shape().as_list()) > 2:
      image_convnet_enc = nets.ConvNet2D(
          output_channels=self.encoder_output_channels,
          kernel_shapes=self.encoder_kernel_shapes,
          strides=self.encoder_strides,
          paddings=self.encoder_paddings,
          activation=self.activation_fn,
          use_batch_norm=self.use_batch_norm,
          name='_conv')
    else:
      image_convnet_enc = (
          lambda inputs, is_training, test_local_stats: tf.identity(inputs))

    image_flatten_enc = snt.BatchFlatten(name='_flatten')

    # TODO(hmoraldo): implement this same layer spec for all encoders and
    # decoders.
    mlp_layers = []
    if self.mlp_layers is not None:
      mlp_layers = copy.copy(self.mlp_layers)

    if not self.dont_project:
      mlp_layers.extend([self.outputs_per_dim * self.latent_dim])

    image_mlp_enc = mlp.MLP(
        output_sizes=mlp_layers,
        dropout=self.dropout,
        keep_prob=self.keep_prob,
        activation=self.activation_fn,
        activate_final=activate_final,
        name='_fc')

    variational_posterior = image_convnet_enc(
        images, is_training=self.is_training,
        test_local_stats=self.is_training)  # for batch norm.
    variational_posterior = self.activation_fn(variational_posterior)
    variational_posterior = image_flatten_enc(variational_posterior)
    return image_mlp_enc(variational_posterior)

class NIPS17MnistaCategoricalLabelDecoder(Decoder):
  """Decoder that generates categorical labels from a given latent code."""

  def __init__(self,
               mlp_dim,
               data_dims,
               num_layers_mlp=1,
               l1_pyz=0.0,
               dropout=False,
               keep_prob=0.5,
               is_training=True,
               swap_out_mlp_relu_for_elu=False,
               name='decoder'):
    """Initializes the decoder.

    Args
      mlp_dim: scalar, Dimension to use for the hidden layers of MLPs.
      data_dims: Tuple, Number of classes for each attribute.
      num_layers_mlp: Int, tells us how many layers to use in each label decoder
        MLP.
      l1_pyz: Float. Regularizer for label decoder.
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (in expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      name: string, name for the sonnet module.
    """
    super(NIPS17MnistaCategoricalLabelDecoder, self).__init__(
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
        name=name)
    self.mlp_dim = mlp_dim
    self.l1_pyz = l1_pyz
    self.data_dims = data_dims
    self.num_layers_mlp = num_layers_mlp

  def _build(self,
             latent,
             sentinel=None,
             v=None,
             trainable=True,
             assertions=False):
    """Constructs a decoder.

    Args:
      latent: [batch_size, self.latent_dim] tensor with latent state.
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
      trainable: Bool. Whether the decoder should be trainable.
      assertions: Bool. See Decoder._build for details.

    Returns:
      Bernoulli(x): a bernoulli distribution object with the logits as
        the decoded modality.

    Raises:
      RuntimeError: if called with trainable=False the first time it's called.
      ValueError: if arguments after sentinel aren't specified by name.
    """
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    regularizers = None
    if self.l1_pyz > 0:
      regularizers = {'w': tf.contrib.layers.l1_regularizer(self.l1_pyz)}

    def build_fn(data, name_suffix=''):
      """Builds the network, see Decoder._build for details."""
      mlp_object_list = []
      mlp_variables_list = []

      output_size_list = [self.mlp_dim]*self.num_layers_mlp

      # TODO(vrama): provide better names for build fn variables based on attributes.
      for index, num_classes in enumerate(self.data_dims):
        mlp_object = mlp.MLP(
            output_sizes=output_size_list + [num_classes],
            dropout=self.dropout,
            keep_prob=self.keep_prob,
            regularizers=regularizers,
            activation=self.mlp_activation_fn,
            name='mlp_' + str(index) + name_suffix)
        built_mlp = mlp_object(data, is_training=self.is_training)
        mlp_variables = mlp_object.get_variables()
        mlp_object_list.append(built_mlp)
        mlp_variables_list.extend(mlp_variables)

      return mlp_object_list, mlp_variables_list

    categorical_label_distributions = []

    decoded_label_list = self._build_conditionally_trainable_net(
        latent, build_fn, trainable=trainable, assertions=assertions)


    categorical_label_distributions = [
        tf.contrib.distributions.Categorical(decoded_label)
        for decoded_label in decoded_label_list
    ]
    return categorical_label_distributions

class NIPS17CategoricalLabelDecoder(Decoder):
  """Decoder that generates categorical labels from a given latent code."""

  def __init__(self,
               mlp_dim,
               data_dims,
               num_layers_mlp=1,
               l1_pyz=0.0,
               dropout=False,
               keep_prob=0.5,
               is_training=True,
               swap_out_mlp_relu_for_elu=False,
               name='decoder'):
    """Initializes the decoder.

    Args
      mlp_dim: scalar, Dimension to use for the hidden layers of MLPs.
      data_dims: Tuple, Number of classes for each attribute.
      num_layers_mlp: Int, tells us how many layers to use in each label decoder
        MLP.
      l1_pyz: Float. Regularizer for label decoder.
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (in expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      name: string, name for the sonnet module.
    """
    super(NIPS17CategoricalLabelDecoder, self).__init__(
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
        name=name)
    self.mlp_dim = mlp_dim
    self.l1_pyz = l1_pyz
    self.data_dims = data_dims
    self.num_layers_mlp = num_layers_mlp

  def _build(self,
             latent,
             sentinel=None,
             v=None,
             trainable=True,
             assertions=False):
    """Constructs a decoder.

    Args:
      latent: [batch_size, self.latent_dim] tensor with latent state.
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
      trainable: Bool. Whether the decoder should be trainable.
      assertions: Bool. See Decoder._build for details.

    Returns:
      Bernoulli(x): a bernoulli distribution object with the logits as
        the decoded modality.

    Raises:
      RuntimeError: if called with trainable=False the first time it's called.
      ValueError: if arguments after sentinel aren't specified by name.
    """
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    regularizers = None
    if self.l1_pyz > 0:
      regularizers = {'w': tf.contrib.layers.l1_regularizer(self.l1_pyz)}

    def build_fn(data, name_suffix=''):
      """Builds the network, see Decoder._build for details."""
      mlp_object_list = []
      mlp_variables_list = []

      # TODO(vrama): provide better names for build fn variables based on attributes.
      for index, num_classes in enumerate(self.data_dims):
        mlp_object = mlp.MLP(
            output_sizes=[self.mlp_dim, num_classes],
            dropout=self.dropout,
            keep_prob=self.keep_prob,
            regularizers=regularizers,
            activation=self.mlp_activation_fn,
            name='mlp_' + str(index) + name_suffix)
        built_mlp = mlp_object(data, is_training=self.is_training)
        mlp_variables = mlp_object.get_variables()
        mlp_object_list.append(built_mlp)
        mlp_variables_list.extend(mlp_variables)

      return mlp_object_list, mlp_variables_list

    categorical_label_distributions = []

    decoded_label_list = self._build_conditionally_trainable_net(
        latent, build_fn, trainable=trainable, assertions=assertions)


    categorical_label_distributions = [
        tf.contrib.distributions.Categorical(decoded_label)
        for decoded_label in decoded_label_list
    ]
    return categorical_label_distributions


class BernoulliDecoder(Decoder):
  """Decoder that generates data from a given latent code."""

  def __init__(self,
               mlp_dim,
               data_dim,
               batch_size,
               l1_pyz=0.0,
               dropout=False,
               keep_prob=0.5,
               is_training=True,
               swap_out_mlp_relu_for_elu=False,
               name='decoder'):
    """Initializes the decoder.

    Args:
      mlp_dim: scalar, Dimension to use for the hidden layers of MLPs.
      data_dim: list, The shape of the data observed. Excludes batch size.
      batch_size: Scalar. Batch size to use for training.
      l1_pyz: Float. Regularizer for label decoder.
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (in expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      name: string, name for the sonnet module.
    """
    super(BernoulliDecoder, self).__init__(
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
        name=name)
    self.mlp_dim = mlp_dim
    self.batch_size = batch_size
    self.l1_pyz = l1_pyz
    self.data_dim = np.array(data_dim)

  def _build(self,
             latent,
             sentinel=None,
             v=None,
             trainable=True,
             assertions=True):
    """Constructs a decoder.

    Args:
      latent: [batch_size, self.latent_dim] tensor with latent state.
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
      trainable: Bool. Whether the decoder should be trainable.
      assertions: Bool. See Decoder._build for details.

    Returns:
      Bernoulli(x): a bernoulli distribution object with the logits as
        the decoded modality.

    Raises:
      RuntimeError: if called with trainable=False the first time it's called.
      ValueError: if arguments after sentinel aren't specified by name.
    """
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    regularizers = None
    if self.l1_pyz > 0:
      regularizers = {'w': tf.contrib.layers.l1_regularizer(self.l1_pyz)}

    def build_fn(data, name_suffix=''):
      """Builds the network, see Decoder._build for details."""
      mlp_object = mlp.MLP(
          output_sizes=[self.mlp_dim, np.prod(self.data_dim)],
          dropout=self.dropout,
          keep_prob=self.keep_prob,
          regularizers=regularizers,
          activation=self.mlp_activation_fn,
          name='mlp' + name_suffix)
      built_mlp = mlp_object(data, is_training=self.is_training)
      mlp_variables = mlp_object.get_variables()
      return built_mlp, mlp_variables

    x = self._build_conditionally_trainable_net(
        latent, build_fn, trainable=trainable, assertions=assertions)
    x_shape = np.insert(self.data_dim, 0, self.batch_size)
    x = tf.reshape(x, x_shape)
    return tf.contrib.distributions.Bernoulli(x)


class ConvolutionalDecoder(Decoder):
  """Decoder that generates image data from a given latent code."""

  def __init__(self,
               latent_dim,
               decoder_output_channels,
               decoder_output_shapes,
               decoder_kernel_shapes,
               decoder_strides,
               decoder_paddings,
               activation_fn,
               use_batch_norm,
               mlp_layers,
               dropout,
               keep_prob,
               is_training=True,
               output_distribution='Bernoulli',
               swap_out_mlp_relu_for_elu=False,
               name='convolutional_decoder'):
    """Initialize Multimodal VAE components.

    Args:
      latent_dim: int, Number of latent states `z` in the model. Each latent
        state has a N(0, 1) prior.
      decoder_output_channels: tuple, number of feature maps in different layers
        of a CNN model.
      decoder_output_shapes: tuple of output shapes, each specifying the shape
        of the output of a given layer.
      decoder_kernel_shapes: tuple, size of the kernel in each layer of CNN.
      decoder_strides: tuple, stride in each layer of CNN.
      decoder_paddings: tuple, 'SAME' or 'VALID', for each layer.
      activation_fn: self.mlp_activation_fn, tf.nn.elu etc. Activation function to use as
        the non-linearity in the network. Shared across all non-linearities in
        the net.
      use_batch_norm: Boolean, True uses batch norm, False switches it off.
      mlp_layers: tuple, Number of units to use in each layer of MLPs constructed
        at various points in the network.
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (on expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      output_distribution: string, one of "Bernoulli", "Categorical", or
        "Gaussian".
      name: string, name for the sonnet module.
    """
    super(ConvolutionalDecoder, self).__init__(
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
        name=name)

    # Properties of the model.
    self.latent_dim = latent_dim
    self.mlp_layers = mlp_layers

    # General properties.
    self.activation_fn = activation_fn
    self.use_batch_norm = use_batch_norm
    self.output_distribution = output_distribution

    # Properties of the Convolutional VAE.
    self.decoder_output_channels = decoder_output_channels
    self.decoder_output_shapes = decoder_output_shapes
    self.decoder_kernel_shapes = decoder_kernel_shapes
    self.decoder_strides = decoder_strides
    self.decoder_paddings = decoder_paddings

  def _build(self,
             latent,
             sentinel=None,
             v=None,
             trainable=True,
             assertions=False):
    """Function for an image decoder network.

    Args:
      latent: A [batch_size, latent_dim] `Tensor`, a sample from the
        inference network's induced distribution.
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
      trainable: Bool. Whether the decoder should be trainable.
      assertions: Bool. See Decoder._build for details.

    Returns:
      A tf.contrib.distributions object corresponding to
      self.output_distribution.

    Raises:
      ValueError: if arguments after sentinel aren't specified by Name.
    """
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')
    #assert (type(self.decoder_output_channels) == 'tuple' and type(self.decoder_output_shapes) == 'tuple' and type(self.decoder_kernel_shapes)=='tuple' and
    #        type(self.decoder_strides) == 'tuple' and type(self.decoder_paddings) == 'tuple'), "Input Must be a tuple."

    final_output_channels = self.decoder_output_channels
    if self.output_distribution == 'Gaussian':
      # Predict mu and sigma for the gaussian case.
      modify_final_output_channels = list(final_output_channels)
      modify_final_output_channels[-1] *= 2
      final_output_channels = tuple(modify_final_output_channels)

    final_paddings = [(snt.SAME if p == 'SAME' else snt.VALID)
                      for p in self.decoder_paddings]

    def build_fn(data, name_suffix=''):
      """Builds the network, see Decoder._build for details."""
      image_convnet_dec = nets.ConvNet2DTranspose(
          output_channels=final_output_channels,
          output_shapes=self.decoder_output_shapes,
          kernel_shapes=self.decoder_kernel_shapes,
          strides=self.decoder_strides,
          paddings=final_paddings,
          activation=self.activation_fn,
          use_batch_norm=self.use_batch_norm,
          name='_conv' + name_suffix)

      # Make the latent dimensions (batch_size, 1, 1, latent_dims)
      reshaped_data = tf.expand_dims(data, 1)
      reshaped_data = tf.expand_dims(reshaped_data, 1)

      decoder = image_convnet_dec(
          reshaped_data,
          is_training=self.is_training,
          test_local_stats=self.is_training)

      return decoder, image_convnet_dec.get_variables()

    decoder = self._build_conditionally_trainable_net(
        latent, build_fn, trainable=trainable, assertions=assertions)
    output_module = utils.get_sampling_distribution(
        distribution_type=self.output_distribution)
    return output_module(decoder)


class LabelDecoder(Decoder):
  """Decoder that generates a label from a given latent code.

  Implements a decoder that is equivalent to that used in
  ConvMultiVaeComponentsSparseLabelDecoder for each individual label.
  """

  def __init__(self,
               activation_fn,
               vocab_size,
               regularizer,
               output_distribution,
               dropout,
               keep_prob,
               is_training=True,
               label_id='',
               swap_out_mlp_relu_for_elu=False,
               name='label_decoder'):
    """Initializes the decoder.

    Args:
      activation_fn: self.mlp_activation_fn, tf.nn.elu etc. Activation function to use as
        the non-linearity in the network. Shared across all non-linearities in
        the net.
      vocab_size: integer, number of classes for each attribute in the "label"
        modality.
      regularizer: float, adds an L-1 regularizer on the classifier weights
        for the label decoder. Set to None not to use L-1 regularization.
      output_distribution: string, one of "Bernoulli", "Categorical", or
        "Gaussian".
      dropout: Boolean, True uses dropout, False turns it off.
      keep_prob: If dropout is True, set the fraction of units to drop out
        (in expectation at training).
      is_training: Boolean, True indicates that model is in training mode.
      label_id: string, a name for the label to decode.
      name: string, name for the sonnet module.
    """
    super(LabelDecoder, self).__init__(
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        swap_out_mlp_relu_for_elu=swap_out_mlp_relu_for_elu,
        name=name)
    self.activation_fn = activation_fn
    self.vocab_size = vocab_size
    self.regularizer = regularizer
    self.output_distribution = output_distribution
    self.label_id = label_id

  def _build(self,
             latent,
             sentinel=None,
             v=None,
             trainable=True,
             assertions=True):
    """Constructs a decoder.

    Args:
      latent: [batch_size, self.latent_dim] tensor with latent state.
      sentinel: Makes sure the next arguments are only called by name.
      v: Unused.
      trainable: Bool. Whether the decoder should be trainable.
      assertions: Bool. See Decoder._build for details.

    Returns:
      The distribution object.

    Raises:
      RuntimeError: if called with trainable=False the first time it's called.
      ValueError: if arguments after sentinel aren't specified by name.
    """
    if sentinel is not None:
      raise ValueError('Arguments after sentinel should be specified by name.')

    regularizers = None
    if self.regularizer is not None:
      regularizers = {'w': tf.contrib.layers.l1_regularizer(self.regularizer)}

    def build_fn(data, name_suffix=''):
      """Builds the network, see Decoder._build for details."""
      # Output MLPs.
      # TODO(vrama): Provide options to set the output sizes programmatically.
      latent_decoder_module = mlp.MLP(
          output_sizes=[128],
          dropout=self.dropout,
          keep_prob=self.keep_prob,
          activation=self.activation_fn,
          regularizers=regularizers,
          name='latent_output_mlp_' + str(self.label_id) + name_suffix)
      latent_decoder_label = latent_decoder_module(
          data, is_training=self.is_training)

      decoder_label = self.activation_fn(latent_decoder_label)
      decoder_module = mlp.MLP(
          dropout=self.dropout,
          keep_prob=self.keep_prob,
          output_sizes=[128, self.vocab_size],
          activation=self.activation_fn,
          name='output_mlp_' + str(self.label_id) + name_suffix)
      decoder_label = decoder_module(
          decoder_label, is_training=self.is_training)

      all_variables = (latent_decoder_module.get_variables() +
                       decoder_module.get_variables())

      return decoder_label, all_variables

    decoder = self._build_conditionally_trainable_net(
        latent, build_fn, trainable=trainable, assertions=assertions)
    return utils.get_sampling_distribution(
        distribution_type=self.output_distribution)(decoder)


class VampPrior(Prior):
  def __init__(self, latent_dim, input_dim, initializer_fns, encoder, num_fake_inputs=20):
    super(VampPrior, self).__init(latent_dim)
    self.num_fake_inputs = num_fake_inputs
    self.encoder = encoder
    self.input_dims = input_dim
    self.initializer_fns = initializer_fns


  def _build(self, unused_v=None):
    # Instantiate the fake inputs.
    # TODO(vrama): Make the label encoder use fixed word embeddings which are
    # fed as input to the encoder.
    fake_inputs = []
    for index_fake_input in xrange(self.num_fake_inputs):
      this_fake_input = []
      for input_idx, (encoder_input_dims, init_fn) in enumerate(zip(self.input_dims, self.initializer_fns)):
        this_fake_input.append(
            tf.get_variable("fake_input_%d_%d" %
                            (index_fake_input, modality_index),
                                               shape=encoder_input_dims,
                            initializer=init_fn)
        )
      fake_inputs.append(this_fake_input)
    densities_in_mixture = [
        self.encoder(*this_fake_input) for this_fake_input in fake_inputs]
    return utils.NormalMixture(densities_in_mixture)


class NormalPrior(Prior):
  """Simple prior using the normal distribution."""

  def _build(self, unused_v=None):
    """Builds the prior."""
    return tf.contrib.distributions.Normal(
        tf.zeros([1, self.latent_dim], dtype=tf.float32),
        tf.ones([1, self.latent_dim], dtype=tf.float32))


# TODO(hmoraldo): possibly move get_jmvae_networks and
# get_convolutional_multi_vae_networks to vae_geom.py as these functions
# are somewhat specific to that experiment.
def get_jmvae_networks(mlp_dim,
                       latent_dim,
                       data_dim_x,
                       data_dim_y,
                       batch_size,
                       product_of_experts_y,
                       universal_expert=False,
                       l1_pyz=0.0,
                       use_sparse_label_decoder=False,
                       dropout=False,
                       keep_prob=0.5,
                       is_training=True):
  """Gives a set of networks for the JMVAE.

  The JMVAE architecture we instantiate here provides facility to use the same
  architecture for the encoder as well as decoder.

  Args:
    mlp_dim: scalar, Dimension to use for the hidden layers of MLPs.
    latent_dim: scalar, Dimension of the latent state for the JMVAE.
    data_dim_x: list, The shape of the data observed as modality 'x'. Excludes
      batch size.
    data_dim_y: list, The shape of the data observed as modality 'y'. Excludes
      batch size.
    batch_size: Batch size to use for training.
    product_of_experts_y: Bool. Whether to use a product of experts for
      labels (modality y).
    universal_expert: Bool.
    l1_pyz: Float. Regularizer for label decoder.
    use_sparse_label_decoder: Whether to use a sparse label decoder for y.
    dropout: Boolean, True uses dropout, False turns it off.
    keep_prob: If dropout is True, set the fraction of units to drop out
      (in expectation at training).
    is_training: Boolean, True indicates that model is in training mode.

  Returns:
    A triple with type (Encoders, Decoders, Prior).
  """

  def build_x_encoder(name):
    return MultiLayerPerceptronEncoder(
        mlp_dim,
        latent_dim,
        universal_expert=universal_expert,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        name=name)

  def build_y_encoder(name):
    y_encoder_module = (ProductOfExpertsEncoder if product_of_experts_y else
                        MultiLayerPerceptronEncoder)
    return y_encoder_module(
        mlp_dim,
        latent_dim,
        universal_expert=universal_expert,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        name=name)

  def build_simple_mlp(name):
    return MultiLayerPerceptronEncoder(
        None,
        mlp_dim,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        outputs_per_dim=1,
        name=name)

  x_encoder = build_x_encoder('inference_x')
  y_encoder = build_y_encoder('inference_y')

  xy_encoder = MultiLayerPerceptronMultiModalEncoder(
      [build_simple_mlp('inference_x_xy'),
       build_simple_mlp('inference_y_xy')],
      mlp_dim,
      latent_dim,
      dropout=dropout,
      keep_prob=keep_prob,
      is_training=is_training,
      name='inference_xy')

  # Image decoder.
  x_decoder = BernoulliDecoder(
      mlp_dim,
      data_dim=data_dim_x,
      batch_size=batch_size,
      l1_pyz=0.0,
      dropout=dropout,
      keep_prob=keep_prob,
      is_training=is_training,
      name='decoder_x')

  # Label decoder.
  if use_sparse_label_decoder:
    y_decoder = LabelDecoder(
        activation_fn=tf.nn.relu,
        vocab_size=2,
        regularizer=None,
        output_distribution='Gaussian',  # TODO(vrama): Why is this set to Gaussian???
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        name='decoder_y')
  else:
    y_decoder = BernoulliDecoder(
        mlp_dim,
        data_dim=data_dim_y,
        batch_size=batch_size,
        l1_pyz=l1_pyz,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        name='decoder_y')

  prior = NormalPrior(latent_dim, name='prior')

  encoders = Encoders([x_encoder, y_encoder, xy_encoder],
                      [0, product_of_experts_y, 0])
  decoders = Decoders([x_decoder, y_decoder])
  return encoders, decoders, prior


def get_convolutional_multi_vae_networks(latent_dim,
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
                                         is_training,
                                         mlp_dim,
                                         data_dim_y,
                                         batch_size,
                                         l1_pyz=0.0,
                                         universal_expert=True):
  """Gives a set of networks for the Convolutional Multi VAE."""

  # Build the image encoder and prior (the only ones that actually matches
  # the convolutional multi VAE architecture).
  # TODO(hmoraldo): finish implementation from conv_multi_vae_components. Once
  # that's done, get rid of args mlp_dim, data_dim_x, data_dim_y, batch_size,
  # product_of_experts, l1_pyz, that are only used for implementing the Jmvae
  # networks.
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
        name=name)

  def build_y_encoder(name, last_layer_size, outputs_per_dim):
    y_encoder_module = (ProductOfExpertsEncoder if product_of_experts_y else
                        MultiLayerPerceptronEncoder)

    return y_encoder_module(
        mlp_dim,
        last_layer_size,
        universal_expert=universal_expert,
        dropout=dropout,
        keep_prob=keep_prob,
        is_training=is_training,
        outputs_per_dim=outputs_per_dim,
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

  # Image decoder.
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
      name='decoder_x')

  prior = NormalPrior(latent_dim, name='prior')

  # Label decoder.
  y_decoder = BernoulliDecoder(
      mlp_dim,
      data_dim=data_dim_y,
      batch_size=batch_size,
      l1_pyz=l1_pyz,
      dropout=dropout,
      keep_prob=keep_prob,
      is_training=is_training,
      name='decoder_y')

  encoders = Encoders([x_encoder, y_encoder, xy_encoder],
                      [0, product_of_experts_y, 0])
  decoders = Decoders([x_decoder, y_decoder])
  return encoders, decoders, prior
