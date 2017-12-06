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

"""A class to evaluate comprehensibility.

Provides options to evaluate the comprehensibility of image generation by
either checking if the generated images are recognized to be of the intended
class by a classifier or by checking for SSIM scores of the generated image
against ground truth images.
"""
import numpy as np
import tensorflow as tf

from datasets import multi_attribute_net

BINARIZE = 'binarize'


# TODO(vrama): Impelement SSIM comprehensibility.
class Comprehensibility(object):
  """Class to evaluate comprehensibility of generated images."""

  def __init__(self,
               checkpoint,
               config,
               num_classes_per_attribute,
               image_size,
               visualize_means,
               attribute_names=None,
               hidden_units=1024):
    """Initialize the comprehensibility evaluation.

    Args:
      checkpoint: str, path to the supervised classification model used to
        evaluate the quality of the generated images.
      config: Configuration object.
      num_classes_per_attribute: The number of classes for each attribute. This
        should be in the same order as attribute names, and in the same order as
        used in the supervised classification model. The supervised classifier
        is trained using the code at
        /research/vale/imagination/classify_deepmind_labels/multi_attribute_net
      image_size: Tuple, of ints describing the size of the input image.
      visualize_means: boolean -- True to visualize mean outputs, False to
        visualize samples.
      attribute_names: Tuple, of strings, describing the name of each attribute,
        each entry corresponds to the num_classes_per_attribute input.
      hidden_units: Number of hidden units in the FC layer of the classifier.
    """
    self.visualize_means = visualize_means
    self._g = tf.Graph()

    tf.logging.warn(
        'Please make sure that the checkpoint that is fed for '
        'comprehensibility eval has been trained on the same input '
        'preprocessing.'
    )

    with self._g.as_default():
      self.image_ph = tf.placeholder(
          shape=[None] + list(image_size), dtype=tf.float32)

      if config.cub_irv2_features:
        logits_list = multi_attribute_net.mlp_multi_attribute_net(
            self.image_ph,
            num_classes_per_attribute,
            attribute_names=attribute_names,
            hidden_units=hidden_units,
            is_training=False)
      else:
        logits_list = multi_attribute_net.conv_multi_attribute_net(
            self.image_ph,
            num_classes_per_attribute,
            attribute_names=attribute_names,
            hidden_units=hidden_units,
            is_training=False)
      predictions_list = [tf.argmax(logits, 1) for logits in logits_list]

      self.prediction_op = predictions_list

      # Setup saver.
      self.saver = tf.train.Saver()

      self.sess = tf.Session()
      tf.logging.info(
          'Restoring the comprehensibility evaluation classifier from %s',
          checkpoint)
      self.saver.restore(self.sess, checkpoint)

  # TODO(vrama): Add tolerance for comprehensibility.
  def evaluate(self, image, original_label, mask_for_label):
    """Run evaluation for a given image.

    The evaluation protocol is to pass the input image through the pretrained
    classifier given during init and to check if the prediction on the input
    image matches the ground truth. The error is the hamming distance between
    the original label set and the predicted label set.

    Args:
      image: np.array of size [c, h, w]
      original_label: np.array of [num_attributes]. The ground truth labels
        for the `image`.
      mask_for_label: np.array of [num_attributes]. Specifies which attributes
        are to be masked out during evaluation. The attributes which are masked
        out are not used in computing the hamming error.
    Returns:
      hamming_distance: An array of [num_attributes] with the hamming distance
        between predictions and ground truth for each attribute.
    Raises:
      ValueError: If input images are not between [0, 1], which corresponds to
        samples/means from bernoulli distribution.
    """
    if np.min(image) < 0.0:
      raise ValueError('Inputs must be between 0 to 1, corresponding to option '
                       '%s.', BINARIZE)

    with self._g.as_default():
      # The last channel of image contains two channels, first for mean second
      # for sample. We wish to pass the sample through the classifier.
      if image.shape[-1] == 2:
        assert self.visualize_means, 'Erroneous input passed.'
        # Sample Image.
        image_to_pass = image[:, :, :, 1, np.newaxis]
        # Mean Image.
        image_to_visualize = image[:, :, :, 0, np.newaxis]
      else:
        image_to_pass = image
        image_to_visualize = image

      predicted_labels_for_image = self.sess.run(self.prediction_op,
                                                 {self.image_ph: image_to_pass})
      predicted_labels_for_image = np.array(
          predicted_labels_for_image).squeeze()
      # Return hamming error between the original desired label and the
      # predicted label.
      hamming_distance = np.not_equal(
          predicted_labels_for_image * mask_for_label,
          original_label * mask_for_label).astype(np.int32).squeeze()

      predicted_labels_for_image = np.squeeze(predicted_labels_for_image)

      return hamming_distance, predicted_labels_for_image, image_to_visualize
