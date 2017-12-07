#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 3.0 (the "License");

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

"""Configuration for a Unimodal/ Multimodal Variational Autoencoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sonnet as snt
import tensorflow as tf

from datasets import label_map

flags = tf.flags

FLAGS = tf.app.flags.FLAGS

# TODO(vrama): Organize options so that if we choose a given dataset outside of
# this interface we dont have to change a bunch of dependencies as well.
tf.app.flags.DEFINE_string(
    'dataset', 'celeba',
    'Name of the dataset to use for experiments. Can be set to one of '
    '\'celeba\', \'affine_mnist\', \'cub\', or \'dm_shapes_with_labels\'.')

# TODO(vrama): For debugging only, remove!!
tf.app.flags.DEFINE_string('image_decoder_channels', '', 'Set to _REPLACE_ME to set '
                   ' image decoder to a three channel output.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '', 'Path to the dataset we want to train our model on.')

tf.app.flags.DEFINE_string('split_type', 'iid', '\"comp\" or \"iid\" splits.')

tf.app.flags.DEFINE_string('loss_type', 'multimodal_elbo',
                    'Options: multimodal_elbo or jmvae or bivcca or fwkl.')

tf.app.flags.DEFINE_string('fwkl_init_dir', '', 'Directory with checkpoint to '
                           'warm start forward kl objective training from.')

tf.app.flags.DEFINE_boolean('product_of_experts', False,
                     'True uses a product of experts inference model q(z| y).')

tf.app.flags.DEFINE_integer('num_latent', 10, 'Number of latent dimensions to use'
                     ' in the model.')

tf.app.flags.DEFINE_float(
    'jmvae_alpha', 0.01,
    'Only used if loss_type is jmvae, sets the weighting between the '
    'ELBO and KL terms in JMVAE-kl')

tf.app.flags.DEFINE_float(
    'alpha_x', 1, 'How much to scale the likelihood terms for \log p(x),'
    'applies to all objectives.')

tf.app.flags.DEFINE_boolean(
    'rescale_amodal_x', False, 'If true, we rescale amodal x at'
    ' alpha_x/private_elbo_y_qy otherwise we reuse the value of alpha x for'
    ' amodal x')

tf.app.flags.DEFINE_float(
    'alpha_y', 50, 'How much to scale the likelihood terms for \log p(y),'
    'applies to all objectives.')

tf.app.flags.DEFINE_float('bivcca_mu', 0.5, 'How much to weigh the two elbo terms in '
                   'bi-vcca relative to each other.')

# Architecture choices.
tf.app.flags.DEFINE_string('classifier_type', 'spmlp',
                    'lmlp: uses a shallow classifier.\n'
                    'dmlp: uses a deep classifier.\n'
                    'spmlp: Uses a separate classifier for each latent to '
                    'attribute mapping.')

tf.app.flags.DEFINE_float('reweighted_elbo_y_scale', 5.0,
                   'How much to scale the elbo(y) term corresponding to '
                   'elbo(x) and elbo(x, y)')

tf.app.flags.DEFINE_float('label_decoder_regularizer', 0.0, 'L-1 Regularizer on the '
                   'weights of the p(y| z) network. Only used when classifier '
                   'type is set to lmlp.')

tf.app.flags.DEFINE_boolean('path_derivative', False, 'If False, use analytical '
                     'KL divergence based ELBO, if True use low-variance '
                     'path derivative.')

tf.app.flags.DEFINE_string(
    'image_likelihood', 'Gaussian',
    'Likelihood to use for images: Bernoulli or Gaussian or Categorical.')

tf.app.flags.DEFINE_string(
    'label_likelihood', 'Categorical',
    'Likelihood to use for labels: Bernoulli or Gaussian or Categorical.')

tf.app.flags.DEFINE_float('learning_rate', .0001, 'Learning rate for training.')
tf.app.flags.DEFINE_float('calibration_learning_rate', 0.0001,
                   'Calibration learning rate.')  # Only used for 'multi_stage'
tf.app.flags.DEFINE_integer('num_iters_calibration', 40000, 'Number of iterations of '
                     'calibration to run.')
tf.app.flags.DEFINE_integer('label_calibration_batch_size', 16, 'Batch size to use '
                     'for calibrating labels.')
tf.app.flags.DEFINE_string('calibration_kl_type', 'analytical', 'Only used for multi '
                    'stage loss. Sets whether we are doing analytical KL or '
                    'monte carlo KL')

tf.app.flags.DEFINE_boolean('dropout', False,
                     'True: apply dropout, False: dont apply dropout.')
tf.app.flags.DEFINE_boolean('use_batch_norm', True,
                     'True: apply batch norm., False: Dont apply batch norm.')
tf.app.flags.DEFINE_float('keep_prob', 0.9,
                   'Proability with which to retain activations in dropout.')

tf.app.flags.DEFINE_boolean('cub_categorical', True,
                     'Use categorical data from the cub sstable, rather than '
                     'the raw attributes.')
tf.app.flags.DEFINE_boolean('cub_classes_only', False,
                     'Only use the class in the attribute vector for CUB.')
tf.app.flags.DEFINE_boolean('cub_skip_classes', False,
                     'Don\'t include the CUB class in the attributes.')
tf.app.flags.DEFINE_boolean('cub_irv2_features', False,
                     'Use inception resnet v2 features for CUB images.')
tf.app.flags.DEFINE_float('cub_box_cox_lambda', 0.0,
                   'Box-Cox lambda_2 parameter to transform power-law-'
                   'distributed IRv2 features to normally-distributed '
                   'features. Set to > 0 to enable, and watch the histograms '
                   'output by vae_eval.')

tf.app.flags.DEFINE_float('private_p_y_scaling', 1.0,
                   'Scaling for alpha_y_elbo_y.')
tf.app.flags.DEFINE_boolean('stop_elboy_gradient', False,
                     'Whether to use stop gradients for elboy.')
tf.app.flags.DEFINE_boolean('stop_elbox_gradient', False,
                     'Whether to use stop gradients for elbox.')


# Top level function that gives out the configuration.

def get_configuration(dataset=None):
  """Provides an interaface to set the configuration.

  We need this interface because often we want to do things like load the
  configuration to run the model on say an ipython notebook or in some other
  context where we dont have explicit bash commands invoking the configuraiton.
  In such cases we do not want a bunch of parmeters to be set to the wrong
  value because we have no way of specifying what dataset we are working with
  and loading the corresponding configurations.
  """
  if dataset is None:
    dataset = FLAGS.dataset

  return Configuration(dataset)

class Configuration(object):
  """Class to define configurations for Variational Autoencoders."""

  def __init__(self, dataset):
    # Dataset dependent options, which need to change for each dataset.
    self.dataset = dataset

    self.split_type = FLAGS.split_type

    # Only used by CUB models, but need to be set for everyone.
    self.cub_classes_only = FLAGS.cub_classes_only
    self.cub_skip_classes = FLAGS.cub_skip_classes
    self.cub_irv2_features = FLAGS.cub_irv2_features
    self.cub_box_cox_lambda = FLAGS.cub_box_cox_lambda

    if self.dataset == 'affine_mnist':
      self.image_size = (64, 64, 1)

      self.num_classes_per_attribute = (10, 2, 3, 4)

      self.grayscale = True

      self.preprocess_options = 'binarize'  # Options to preprocess the image.

      # Note that we always restore from the IID checkpoint regardless of
      # whether the VAE is being trained on IID or compositional splits, since
      # we care about having the best classifier for eval.
      self.comprehensibility_ckpt = ('mnista_classifier_checkpoint/model.ckpt-50002')
      self.label_map_json = ('./data/mnist_with_attributes/iid_label_map.json')
      self.comprehensibility_hidden_units = 1024
    elif self.dataset == 'celeba':
      self.image_size = (64, 64, 3)
      self.num_classes_per_attribute = tuple([2]*18)
      self.preprocess_options= 'center'
      self.grayscale = False
      self.comprehensibility_chkpt = ''
      self.comprehensibility_hidden_units = 512
      self.label_map_json = os.path.join(FLAGS.dataset_dir, 'attribute_label_map.json')

    self.loss_type = FLAGS.loss_type

    self.image_likelihood = FLAGS.image_likelihood
    # Used in case loss_type is set to multimodal ELBO.
    self.path_derivative = FLAGS.path_derivative

    self.num_latent = FLAGS.num_latent

    self.label_likelihood = FLAGS.label_likelihood

    self.product_of_experts = FLAGS.product_of_experts

    # Convolutional sub-network parameters for the encoder.
    self.encoder_input_channels = self.image_size[-1]
    self.encoder_output_channels = [32, 64, 128, 16]
    # Try increasing number of feature maps.

    self.encoder_kernel_shapes = [5, 5, 5, 1]  # Kernel shapes in Conv[i] layer.

    self.encoder_strides = [1, 2, 2, 1]  # Strides in conv[i] layer.

    self.encoder_paddings = [snt.SAME, snt.SAME, snt.VALID, snt.VALID]

    self.activation_fn = tf.nn.elu

    self.use_batch_norm = FLAGS.use_batch_norm

    # MLP sub-network parameters.
    self.mlp_layers = [512, 512]

    self.li_mlp_layers = [512, 512]

    self.dropout = FLAGS.dropout

    self.keep_prob = FLAGS.keep_prob

    # For multi-modal vae models.
    self.label_embed_dims = 32

    self.post_fusion_mlp_layers = [512]

    self.li_post_fusion_mlp_layers = [512]

    # Label decoder options.
    self.label_decoder_regularizer = FLAGS.label_decoder_regularizer

    # Shallow vs. deep classifier for label MLP.
    self.label_classifier_mlp = FLAGS.classifier_type

    # Objective options JMVAE specific.
    self.jmvae_alpha = FLAGS.jmvae_alpha  # Alpha for JMVAE-kl objective.

    # Objective options bi-VCCA specific.
    self.bivcca_mu = FLAGS.bivcca_mu

    # Options specific to multimodal ELBO and reweighted elbo.
    self.alpha_y = FLAGS.alpha_y
    self.alpha_x = FLAGS.alpha_x
    self.rescale_amodal_x = FLAGS.rescale_amodal_x

    # True, compute likelihood of all modalities in each multimodal elbo term.
    self.melbo_predict_both = False

    # Options specific to multi_stage loss.
    self.label_calibration_batch_size = FLAGS.label_calibration_batch_size
    self.calibration_kl_type = FLAGS.calibration_kl_type

    # Options specific to reweighted ELBO.
    self.reweighted_elbo_y_scale = FLAGS.reweighted_elbo_y_scale

    # Multimodal_elbo-specific hparams.
    self.private_p_y_scaling = FLAGS.private_p_y_scaling
    self.stop_elboy_gradient = FLAGS.stop_elboy_gradient
    self.stop_elbox_gradient = FLAGS.stop_elbox_gradient

    self.modify_options_for_product_of_experts()

    self.check_options()

  def check_options(self):
    """Check that the provided options are consistent or are as expected."""

    if self.dataset == 'affine_mnist' and self.image_likelihood != 'Bernoulli':
      raise ValueError('Need bernoulli likelihood for dataset affine_mnist')

    if len(self.image_size) != 3:
      raise ValueError("Image size must be 3 dimensional H x W x C.")

    if (self.image_likelihood == 'Bernoulli' and
        self.preprocess_options != 'binarize'):
      raise ValueError('Bernoulli likelihood must have binarized images.')
    if self.num_latent % 2 != 0:
      raise ValueError('Number of latent dims must be divisible by 2.')
    if self.split_type not in ['comp', 'iid']:
      raise ValueError('Invalid split type %s' % (self.split_type))

    if self.loss_type == 'fwkl':
      assert (FLAGS.stop_elbox_gradient and FLAGS.stop_elboy_gradient), (
          "For forward KL the elbo(x, y) term must have been trained with "
          "stopgradients in elbo(x) and elbo(y) in the triple elbo pretraining."
      )

    if self.loss_type not in [
        'multimodal_elbo', 'jmvae', 'bivcca', 'multi_stage', 'reweighted_elbo',
        'fwkl'
    ]:
      raise ValueError('Invalid loss type %s' % (self.loss_type))
    if (self.dataset in ['affine_mnist'] and
        not self.comprehensibility_ckpt):
      raise ValueError('Needs path to pretrained comprehensibility network.')
    if (self.dataset in ['cub', 'affine_mnist'] and
        not self.label_map_json):
      raise ValueError('Needs as input a label map json file.')

  def modify_options_for_product_of_experts(self):
    """Modify options for product of experts q(z| y) model.

    Modify options for the product of experts model so that the number of
    parameters is approximately the same as the original model.
    """
    if self.product_of_experts:
      if self.dataset not in ['dm_shapes_with_labels', 'affine_mnist']:
        tf.logging.info('Need to calibrate parameters of product of experts for dataset %s. '
           'This model may not be comparable with non-PoE models.' %
           self.dataset)
      self.li_post_fusion_mlp_layers = [
          x / 4 for x in self.li_post_fusion_mlp_layers
      ]
      self.li_mlp_layers = [x / 4 for x in self.li_mlp_layers]


class TrainingConfig(object):

  def __init__(self):
    self.learning_rate = FLAGS.learning_rate
    self.calibration_learning_rate = FLAGS.calibration_learning_rate
    self.num_iters_calibration = FLAGS.num_iters_calibration
    self.optimizer = tf.train.AdamOptimizer
