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
"""Some utility functions for constructing VAEs.

Author: vrama@
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import copy
import cPickle as pickle
import logging
import re

import numpy as np
import sonnet as snt
import tensorflow as tf

# A tuple to store inference network's output distributions, and a
# single sample from the distribution. This is done to make it convenient
# to implement autoregression (block-) conditional parameterization
# of the inference network later. That is, given q(z| x) = q(z[1:5]|x)*
# q(z[6:10]|x, z[1:5]), for example.
# NOTE: calling latent_xy.sample will only return one unique sample.
LatentDensity = namedtuple('LatentDensity', ['density', 'sample'])


class NormalMixture(object):
  """Provides a simplistic interface to a Gaussian Mixture."""

  def __init__(self, mixture_components, weights=None):
    if not all([
        isinstance(c, tf.contrib.distributions.Normal)
        for c in mixture_components
    ]):
      raise ValueError(
          'Inputs to Normal mixture must be a list of normal densities.')

    if weights is None:
      weights = np.ones(
          len(mixture_components), dtype=np.float32) * (
              1.0 / len(mixture_components))

    assert isinstance(weights, np.ndarray), "Weights must be a Numpy array."

    self.components = mixture_components
    self.num_components = len(mixture_components)
    self.weights = weights

  def log_pdf(self, datum, name=''):
    log_pdfs = [
        tf.expand_dims(density.log_prob(datum, name + str(index)) / weight, 0)
        for index, (density,
                    weight) in enumerate(zip(self.components, self.weights))
    ]
    log_pdfs = tf.concat(log_pdfs, 0)
    mixture_log_pdf = tf.reduce_sum(log_pdfs, axis=0)

    return mixture_log_pdf

  def log_prob(self, datum, name=''):
    return self.log_pdf(datum, name)

  def sample(self):
    """Sample from a gaussian mixture model."""
    # First pick a mixture component and then sample from that gaussian.
    # Overall this sampling process is not reparamerizable yet (because of the
    # discrete sampling step involved in the first step)
    selected_mixture = np.random.choice(
        range(len(self.components)), 1, p=self.weights)[0]
    return self.components[selected_mixture].sample()


def scalar_summary_with_scope(name, value, scope='train'):
  """Creates a summary with option to specify a scope."""
  name = re.sub('[^A-Za-z0-9]', '_', name)
  tf.summary.scalar(scope + '/' + name, value)


def multiply_gaussian_with_universal_expert_pdfs(gaussian_pdf_list,
                                                 mask,
                                                 universal_expert=True):
  """Multiply a set of gaussian pdfs to give a new gaussian.

  Assumes that the gaussians are all diagonal covariance.

  Args:
    gaussian_pdf_list: list, of tf.contrib.distributions.Normal objects.
    mask: A `Tensor` of [B, len(gaussian_pdf_list)] with a 0 to ignore
      a particular expert and a 1 to use an expert, where B is batch_size.
    universal_expert: Boolean, whether to use a universal expert or not.
      Set to True by default.

  Returns:
    product_gaussian_pdf: A tf.contrib.distributions.Normal object

  Raises:
    ValueError: if len(mask) != len(gaussian_pdf_list)
    ValueError: if mask is None
  """
  if mask is None:
    raise ValueError('Must provide a mask argument that is not None.')

  if mask.get_shape().as_list()[1] != len(gaussian_pdf_list):
    raise ValueError('Invalid number of gaussians.')

  # Convert the mask to float32 for use in the model.
  mask = tf.to_float(mask)

  if not gaussian_pdf_list:
    raise ValueError('Need a non-empty gaussian pdf list.')

  if universal_expert:
    expert_mu = tf.zeros_like(gaussian_pdf_list[0].loc)
    expert_loc = tf.ones_like(gaussian_pdf_list[0].scale)

    gaussian_pdf_list.append(
        tf.contrib.distributions.Normal(expert_mu, expert_loc))

    mask = tf.concat([mask, tf.ones((tf.shape(mask)[0], 1))], axis=1)

  assert_op = tf.assert_greater(
      tf.reduce_sum(mask, 1),
      tf.zeros(1),
      message='Atleast one of the gaussians must '
      'be selected in the expert of gaussian model.')

  with tf.control_dependencies([assert_op]):
    # Make the mask [NUM_GAUSSIANS, B, 1] for broadcasating.
    mask = tf.transpose(mask)
    mask = tf.expand_dims(mask, 2)

    # Compute the co-variance matrix of the new gaussian.
    # The diagonal covariance matrix for a product of gaussians is
    # 1/sigma^2 = \sum_k 1/sigma_k^2
    # where simga_i are vectors in R^D, representing the diagonal
    # entries of the covariance matrix.
    gaussian_pdf_means = [x.loc for x in gaussian_pdf_list]
    gaussian_pdf_stddev = [x.scale for x in gaussian_pdf_list]

    stacked_product_stddev = tf.stack(
        gaussian_pdf_stddev, axis=0, name='stack_gaussians')
    masked_product_inv_variance = 1 / tf.square(stacked_product_stddev) * mask
    product_stddev = tf.sqrt(tf.reduce_sum(masked_product_inv_variance, axis=0))
    product_stddev = 1 / product_stddev

    # Compute the mean of the product of gaussians, as:
    # \mu = \sum_k \mu_k/sigma_k ^ 2 * 1/(\sum_k 1/sigma_k^2)
    stacked_product_mean = tf.stack(
        gaussian_pdf_means, axis=0, name='stack_gaussian_means')
    stacked_product_mean *= masked_product_inv_variance
    stacked_product_mean = tf.reduce_sum(stacked_product_mean, axis=0)
    product_mean = (stacked_product_mean / tf.reduce_sum(
        masked_product_inv_variance, axis=0))

    product_gaussian_pdf = tf.contrib.distributions.Normal(
        product_mean, product_stddev)

  return product_gaussian_pdf


def stop_gradient_and_synthesize_density(gaussian_dist):
  """Stop gradient and return a new density with fixed parameters.

  This module is used to implement the asymptotically low-variance gradient
  estimator for the ELBO proposed by Reoder et.al.
  (https://arxiv.org/pdf/1703.09194v1.pdf)

  Args:
    gaussian_dist: An instance of tf.contrib.distributions.Normal
  Returns:
    new_dist: A tf.contrib.distributions.Normal object whose mean and variances
      have stop_gradient applied to them.
  """
  gaussian_mean = tf.stop_gradient(gaussian_dist.loc)
  gaussian_scale = tf.stop_gradient(gaussian_dist.scale)
  new_dist = tf.contrib.distributions.Normal(
      loc=gaussian_mean, scale=gaussian_scale)
  return new_dist


def compute_likelihood(densities, data):
  """Computes likelihood with a list of (or single) densities.

  Given a list of (or a single) densities, in general, and corresponding
  observations as a list, data,
  compute the likelihood of data[i] under density[i], and return
  the sum of likelihoods. Thus, this corresponds to an independence assumption
  across entries in data.

  Args:
    densities: list, each element is a `tf.contrib.distributions` object,
      must have a log_prob function.
    data: list, each element is a batch_size x data_dims `Tensor`.
  Returns:
    [batch_size] tensor with log-prob values, accumulated across all data.
  """
  if not isinstance(densities, list):
    densities = list([densities])
  if not isinstance(data, list):
    data = list([data])

  if len(densities) != len(data):
    assert ValueError('Data and densities must have same length.')

  # Stores batch_size number of values computing the likelihood.
  accumulated_log_prob = tf.zeros([tf.shape(data[0])[0]], dtype=tf.float32)

  for datum, density in zip(data, densities):
    log_prob = density.log_prob(datum)
    # Reduce all dimensions except batch.
    dims_datum = range(1, log_prob.get_shape().ndims)
    log_prob = tf.reduce_sum(log_prob, dims_datum)
    accumulated_log_prob += log_prob

  return accumulated_log_prob


def split_tensor_lastdim(ten):
  """Split last dimension into two tensors."""
  # TODO(vrama): Generalize this to other splits than 2.
  assert ten.get_shape().as_list()[-1] % 2 == 0, ('Last dimension'
                                                  'must be divisible by 2.')
  ten1, ten2 = tf.split(ten, 2, axis=len(ten.get_shape().as_list()) - 1)
  return ten1, ten2


def concat_gaussian(gauss1, gauss2):
  """Given two gaussians, return a new multivariate gaussian."""
  new_mu = tf.concat(
      (gauss1.mean(), gauss2.mean()), axis=gauss1.mean().get_shape().ndims - 1)
  new_sigma = tf.concat(
      (gauss1.variance(), gauss2.variance()),
      axis=len(gauss1.mean().get_shape().as_list()) - 1)
  return tf.contrib.distributions.Normal(new_mu, new_sigma)


def concat_bernoulli(b1, b2):
  new_logits = tf.concat((b1.logits, b2.logits), axis=-1)
  return tf.contrib.distributions.Bernoulli(logits=new_logits)


def concat_distributions(x1, x2):
  if 'Normal' in x1.name and 'Normal' in x2.name:
    x = concat_gaussian(x1, x2)
  elif 'Bernoulli' in x1.name and 'Bernoulli' in x2.name:
    x = concat_bernoulli(x1, x2)
  else:
    raise NotImplementedError('Only Normal and Bernoulli densities'
                              ' supported.')
  return x


def dims_to_targetshape(data_dims, batch_size=None, placeholder=False):
  """Prepends either batch size/None (for placeholders) to a data shape tensor.

  Args:
    data_dims: list, indicates shape of the data, ignoring the batch size.
      For an RGB image this could be [224, 224, 3] for example.
    batch_size: scalar, indicates the batch size for SGD.
    placeholder: bool, indicates whether the returned dimensions are going to
      be used to construct a TF placeholder.
  Returns:
    shape: A tensor with a batch dimension prepended to the data shape.
  """
  if batch_size is not None and batch_size != 0:
    shape = [batch_size]
  elif placeholder is True:
    shape = [None]
  else:
    shape = [-1]
  shape.extend(data_dims)

  return shape


_BERNOULLI = 'Bernoulli'
_GAUSSIAN = 'Gaussian'
_CATEGORICAL = 'Categorical'

Moments = namedtuple('moments', ['mean', 'variance'])
Gaussian = namedtuple('Gaussian', ['mean', 'std'])


def num_unique_rows(np_array):
  """Find unique rows in a 2D numpy array."""
  if np_array.ndim != 2:
    raise ValueError('Unique rows expects a 2D array.')

  np_array = [tuple(x) for x in np_array]
  unique_rows = len(set(np_array))

  return unique_rows


def add_simple_summary(summary_writer,
                       value_to_add,
                       tag,
                       iteration,
                       scope='val'):
  """Add a simple summary using a given summary writer."""
  summary = tf.Summary()
  value = summary.value.add()
  value.simple_value = value_to_add
  value.tag = scope + '/' + tag
  summary_writer.add_summary(summary, iteration)
  summary_writer.flush()


def apply_mask_to_label(label_vector, mask):
  """Applies a mask to labels to ignore some entries in a label vector.

  Args:
    label_vector: np.array of [batch_size, num_attributes]
    mask: np.array of [batch_size, num_attributes] with 1.0 and 0.0 values. 0.0
      denotes the values to ignore.
  Returns:
    masked_label_vector: A label vector with entries that we are not interested
      in set to -1.
  """
  # Ensure that there is no label value which is -1.0
  if np.any(np.equal(label_vector, -1.0)):
    raise ValueError('Labels have invalid value -1.0')

  masked_label_vector = copy.copy(label_vector)
  bool_mask = np.invert(mask.astype(np.bool))
  masked_label_vector[bool_mask] = -1.0

  return masked_label_vector


def gaussian_analytical_kl(gaussian_a, gaussian_b):
  r"""Computes the analytical KL between two (multivariate) diagonal gaussians.

  This code is for KL-divergence between numpy arrays. The implementation of
  KL divergence is based on the implementation in tf.contrib.distributions.

  Args:
    gaussian_a: a namedtuple of Gaussian, with mean and std fields.
      Mean and std are both numpy arrays of [batch_size, dimensions]
    gaussian_b: a namedtuple of Gaussian, with mean and std fields.
      Mean and std are both numpy arrays of [batch_size, dimensions]
  Returns:
    KL divergence between the two gaussians.
  """
  one = 1.0
  two = 2.0
  half = 0.5
  s_a_squared = np.square(gaussian_a.std)
  s_b_squared = np.square(gaussian_b.std)

  ratio = s_a_squared / s_b_squared
  kl = (np.square(gaussian_a.mean - gaussian_b.mean) /
        (two * s_b_squared) + half * (ratio - one - np.log(ratio)))

  # Reduce over all dimensions other than the batch.
  kl = np.sum(kl, axis=-1)

  return kl


def get_sampling_distribution(distribution_type='Bernoulli',
                              quantize_normal=True):
  """Closure to create a density to sample from, or to compute likelihood with.

  To be used "only" for generative networks p(x| z), where z is some hidden
  state, and 'x' is an observed modaltiy. Gives options to specify either
  Bernoulli or Gaussian or Categorical likelihoods.

  Args:
    distribution_type: Str, specifies the distribution for which we compute
      likelihood.
    quantize_normal: Boolean, flag to specify whether to quantize the Normal
      distribution for discrete outputs.
  Returns:
    A sonnet module for the likelihood / observed variable sampling
    distribution.
  """

  def output_distribution(act):
    """Route the activations through the corresponding density."""
    if distribution_type == _BERNOULLI:
      return tf.contrib.distributions.Bernoulli(logits=act)
    elif distribution_type == _GAUSSIAN and quantize_normal:
      # Use generative.QuantizedNormal to account for the fact that the images
      # are discrete values as opposed to continuous values.
      return QuantizedNormal(act)
    elif distribution_type == _GAUSSIAN:
      logging.warning('Using Gaussian likelihood without discretization.')
      return LogStddevNormal(act)
    elif distribution_type == _CATEGORICAL:
      return tf.contrib.distributions.Categorical(logits=act)
    else:
      raise ValueError('Distribution type must be one of %s or %s' %
                       (_BERNOULLI, _GAUSSIAN))

  return snt.Module(output_distribution)


def add_accuracy_ops(prediction_list,
                     label_list,
                     attribute_list,
                     mode='eval',
                     summary_prefix='Accuracy_'):
  """Add accuracy ops to graph to use with lists of labels.

  Args:
    prediction_list: List of `Tensor` of [batch_size]
    label_list: List of `Tensor` of [batch_size]
    attribute_list: List of `string`, names of each of the accuracy ops.
    mode: string, Mode prefix to add to the summary scope.
    summary_prefix: Prefix to use when creating the summary.

  Returns:
    all_summary_names: List of `string`, with names for each accuracy.
    update_op_list: List of Ops with value accuracy_list[i]
  Raises:
    ValueError: if prediction_list and label_list have different length
  """

  if len(prediction_list) != len(label_list):
    raise ValueError('Prediction list has length %d, label list has length %d' %
                     (len(prediction_list), len(label_list)))

  update_op_attributes = []
  all_summary_names = []

  for pred_label, gt_label, attribute in zip(prediction_list, label_list,
                                             attribute_list):
    _, update_op = tf.metrics.accuracy(gt_label, pred_label)

    update_op_attributes.append(update_op)
    all_summary_names.append(summary_prefix + attribute)

    scalar_summary_with_scope(all_summary_names[-1], update_op, mode)

  overall_accuracy = tf.constant(0.0, dtype=tf.float32)
  for item in update_op_attributes:
    overall_accuracy += item
  overall_accuracy /= len(update_op_attributes)

  update_op_attributes.append(overall_accuracy)
  all_summary_names.append(summary_prefix + 'overall')
  scalar_summary_with_scope(all_summary_names[-1], overall_accuracy, mode)

  assert len(all_summary_names) == len(update_op_attributes), (
      'Number of summaries and accuracy ops must be the same.')

  return all_summary_names, update_op_attributes


class SaveLoadMomentsPickle(object):
  """Class to load and save a pickle with moments."""

  def __init__(self, pickle_file):
    """Initialize a class to load and save pickle files."""
    self._pickle_file = pickle_file

  def to_pickle(self, data_to_pickle):
    """Dict of moment objects is serialized."""
    serializable_data = {}
    for label, moment in data_to_pickle.iteritems():
      serializable_data[label] = moment._asdict()

    with tf.gfile.Open(self._pickle_file, 'w') as f:
      pickle.dump(serializable_data, f)

  def from_pickle(self):
    """A dict of dict (with moments) is read and converted to Moments."""
    with tf.gfile.Open(self._pickle_file, 'r') as f:
      moment_data = pickle.load(f)

    for label, moment in moment_data.iteritems():
      moment_data[label] = Moments(**moment)

    return moment_data


def create_init_fn(checkpoint_path):
  """Return a function to restore variables.

  Args:
    checkpoint_dir: Directory with all the training checkpoints.
  Returns:
    restore_fn: An op to which we can pass a session to restore
      variables.
  Raises:
    ValueError: If invalid checkpoint name is found or there are no checkpoints
      in the directory specified.
  """
  saver = tf.train.Saver()
  logging.info('Checkpoint path: %s', (checkpoint_path))
  global_step_ckpt = int(checkpoint_path.split('-')[-1])

  # Checks for fraudulent checkpoints without a global_step.
  if global_step_ckpt == checkpoint_path:
    raise ValueError('Invalid checkpoint name %s.' % (checkpoint_path))

  def init_fn(sess):
    logging.info('Restoring the model from %s', (checkpoint_path))
    saver.restore(sess, checkpoint_path)

  return init_fn, global_step_ckpt


def create_restore_fn(checkpoint_path, saver):
  """Return a function to restore variables.

  Args:
    checkpoint_path: Path to the checkpoint to load.
    saver: tf.train.Saver object
  Returns:
    restore_fn: An op to which we can pass a session to restore
      variables.
    global_step_ckpt: Int, the global checkpoint number.
  Raises:
    ValueError: If invalid checkpoint name is found or there are no checkpoints
      in the directory specified.
  """
  logging.info('Checkpoint path: %s', (checkpoint_path))
  global_step_ckpt = int(checkpoint_path.split('-')[-1])

  # Checks for fraudulent checkpoints without a global_step.
  if global_step_ckpt == checkpoint_path:
    raise ValueError('Invalid checkpoint name %s.' % (checkpoint_path))

  def restore_fn(sess):
    logging.info('Restoring the model from %s', (checkpoint_path))
    saver.restore(sess, checkpoint_path)

  return restore_fn, global_step_ckpt


def np_array_to_tuple(array):
  """Convert a numpy array to a tuple after squeezing it."""
  array = tuple(np.squeeze(array))
  return array


def unbatchify_list(label_list):
  """Unbatchify a list.

  Given a list of [batch_size] numpy arrays, return a list of [list_size] numpy
  arrays of length batch_size.

  Args:
    label_list: list of [batch_size] numpy array of length [list_size].
  Returns:
    batched_list: [list_size] numpy array of [batch_size] tensors.
  """
  assert isinstance(label_list, list), 'Invalid input, must be list.'
  batched_list = []
  for item in xrange(label_list[0].shape[0]):
    batch_labels = []
    for label_index in xrange(len(label_list)):
      batch_labels.append(label_list[label_index][item])
    batched_list.append(batch_labels)
  return batched_list


class QuantizedNormal(tf.contrib.distributions.Normal):
  """Quantized normal for use with discrete / quantized signals.

  It is safe to use this class with any `params` tensor as it expects log_sigma
  to be provided, which will then be exp'd and passed as sigma into the standard
  tf.distribution.Normal distribution.
  """

  def __init__(self, params, bin_size=1 / 128.0, slice_dim=-1, name='Normal'):
    """Distribution constructor.

    Args:
      params: Tensor containing the concatenation of [mu, log_sigma] parameters.
        The shape of `params` must be known at graph construction (ie.
        params.get_shape() must work).
      bin_size: The quantization at which we are going to see output values.
      slice_dim: Dimension along which params will be sliced to retrieve mu
        and log_sigma. Negative values index from the last dimension.
      name: Name of the distribution.

    Raises:
      ValueError: If the `params` tensor cannot be split evenly in two along the
        slicing dimension.
    """
    mu, self._log_sigma = self._split_mu_log_sigma(params, slice_dim)
    sigma = tf.exp(self._log_sigma)
    self._bin_size = bin_size
    super(QuantizedNormal, self).__init__(
        loc=mu, scale=sigma, name=name, validate_args=False)

  def _split_mu_log_sigma(self, params, slice_dim):
    """Splits `params` into `mu` and `log_sigma` along `slice_dim`."""
    params = tf.convert_to_tensor(params)
    size = params.get_shape()[slice_dim].value
    if size % 2 != 0:
      raise ValueError('`params` must have an even size along dimension {}.'
                       .format(slice_dim))
    half_size = size // 2
    mu = snt.SliceByDim(
        dims=[slice_dim], begin=[0], size=[half_size], name='mu')(params)
    log_sigma = snt.SliceByDim(
        dims=[slice_dim], begin=[half_size], size=[half_size],
        name='log_sigma')(params)
    return mu, log_sigma

  def log_prob(self, x, name='log_prob'):
    """Compute log pdf under a quantized normal distribution.

    Args:
      x: `Tensor` of any size whose log-probability we wish to estimate.
    """
    x += tf.random_uniform(
        tf.shape(x),
        minval=-1 * self._bin_size * 0.5,
        maxval=self._bin_size * 0.5,
        name='quantization_noise')
    return super(QuantizedNormal, self).log_prob(x)


class LogStddevNormal(tf.contrib.distributions.Normal):
  """Diagonal Normal that accepts a concatenated[mu, log_sigma] tensor.

  It is safe to use this class with any `params` tensor as it expects log_sigma
  to be provided, which will then be exp'd and passed as sigma into the standard
  tf.distribution.Normal distribution.
  """

  def __init__(self, params, slice_dim=-1, name='Normal'):
    """Distribution constructor.

    Args:
      params: Tensor containing the concatenation of [mu, log_sigma] parameters.
        The shape of `params` must be known at graph construction (ie.
        params.get_shape() must work).
      slice_dim: Dimension along which params will be sliced to retrieve mu
        and log_sigma. Negative values index from the last dimension.
      name: Name of the distribution.

    Raises:
      ValueError: If the `params` tensor cannot be split evenly in two along the
        slicing dimension.
    """
    mu, self._log_sigma = self._split_mu_log_sigma(params, slice_dim)
    sigma = tf.exp(self._log_sigma)
    super(LogStddevNormal, self).__init__(
        loc=mu, scale=sigma, name=name, validate_args=False)

  def _split_mu_log_sigma(self, params, slice_dim):
    """Splits `params` into `mu` and `log_sigma` along `slice_dim`."""
    params = tf.convert_to_tensor(params)
    size = params.get_shape()[slice_dim].value
    if size % 2 != 0:
      raise ValueError('`params` must have an even size along dimension {}.'
                       .format(slice_dim))
    half_size = size // 2
    mu = snt.SliceByDim(
        dims=[slice_dim], begin=[0], size=[half_size], name='mu')(params)
    log_sigma = snt.SliceByDim(
        dims=[slice_dim], begin=[half_size], size=[half_size],
        name='log_sigma')(params)
    return mu, log_sigma
