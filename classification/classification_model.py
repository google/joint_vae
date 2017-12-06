#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.
"""
Builds an Inception-Resnet model for classifying face images.
"""
import tensorflow as tf
import classification.inception_resnet_v2 as inception_resnet_v2
from classification.inception_resnet_v2 import default_image_size
from datasets.celeba import celeba_data_provider
from classification.inception_preprocessing import preprocess_image

flags = tf.app.flags

tf.app.flags.DEFINE_float('weight_decay', 0.00004,
                          'Weight decay to train the model.')

FLAGS = tf.app.flags.FLAGS


# Make sure we preprocess things based on what inception resnet wants.
class ClassifyFaces(object):

  def __init__(self,
               mode='train',
               split='train',
               finetune=True,
               from_scratch=False):
    """A class to finetune an image classification model on celeba.

    mode: Mode in which we want to run the model 'train' or 'eval'
    split: Name of the split of CELEBA we are running the classifier on, either
      'train', 'val', or 'test'.
    finetune: Bool, whether to finetune the earlier layers of the network or
      not.
    from_scratch: Whether we are training the underlying CNN from scratch or
      from an imagenet checkpoint.
    """
    if mode == 'train':
      tf.logging.info('Setting up image classifier in finetune=%s mode' %
                      (str(finetune)))

    self.finetune = finetune
    self.images = None
    self.labels = None
    self.mode = mode
    self.split = split
    self.from_scratch = from_scratch

  def build_inputs(self):
    images, labels, num_samples, num_classes_per_attribute, attribute_names = (
        celeba_data_provider.provide_data(
            FLAGS.dataset_dir,
            self.split,
            FLAGS.batch_size,
            split_type="iid",
            image_size=[64, 64],
            preprocess_options="center",
            grayscale=False,
            shuffle_data=self.is_training))

    if any(
        [x != num_classes_per_attribute[0] for x in num_classes_per_attribute]):
      raise RuntimeError('Code assumes that we have equal number of labels '
                         ' for each attribute.')
    # Preprocess images to fit inception resnet size, doing this outside of the
    # queuerunners since this will make it easy to run outputs from the model
    # throught the preprocessing as opposed to just making it a part of the
    # data loader.
    images = self.process_imagination_images_for_inception_resnet(images)
    self.images = images
    # Stack the labels together for convenience in computing the loss.
    self.labels = tf.stack(labels, axis=-1)
    self.num_samples = num_samples
    self.num_classes_per_attribute = num_classes_per_attribute
    self._attribute_names = attribute_names

  def build_classifier(self):
    inception_arg_scope_args = {'weight_decay': FLAGS.weight_decay}

    # Set the batch normalization layer to evaluation mode, and keep the moving
    # mean and variance from imagenet pretraining.
    if self.from_scratch:
      inception_arg_scope_args['batch_norm_decay'] = 1.0

    with tf.contrib.slim.arg_scope(
        inception_resnet_v2.inception_resnet_v2_arg_scope(
            **inception_arg_scope_args)):
      logits, _ = inception_resnet_v2.inception_resnet_v2(
          self.images,
          len(self.num_classes_per_attribute) *
          self.num_classes_per_attribute[0],
          is_training=self.is_training,
          create_aux_logits=False,
          from_scratch=self.from_scratch)

      self.logits = tf.reshape(logits, [
          FLAGS.batch_size,
          len(self.num_classes_per_attribute), self.num_classes_per_attribute[0]
      ])

  def build_loss(self):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.logits, labels=self.labels)
    tf.losses.add_loss(tf.reduce_mean(tf.reduce_sum(loss, axis=-1)))
    self.loss = tf.losses.get_total_loss(add_regularization_losses=True)
    tf.summary.scalar('total_loss', self.loss)

  def build_model(self):
    self.build_inputs()
    self.build_classifier()
    self.build_loss()

  def setup_saver(self):
    return tf.train.Saver()

  def get_init_fn(self, checkpoint_path):
    variables_to_restore = [
        x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if 'InceptionResnetV2' in x.name and 'Logits' not in x.name
    ]
    saver = tf.train.Saver(variables_to_restore)

    def init_fn(sess):
      tf.logging.info('Restoring variables from %s checkpoint.' %
                      (checkpoint_path))
      saver.restore(sess, checkpoint_path)

    return init_fn

  def get_trainable_vars(self):
    all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if not self.finetune:
      trainable_vars = [
          x for x in all_trainable_vars if 'InceptionResnetV2/Logits' in x.name
      ]
      return trainable_vars
    return all_trainable_vars

  def _shuffle_or_not(self):
    return self.is_training

  def process_imagination_images_for_inception_resnet(self, images):
    """Applies processing to imagination images (64x64) for inception resnet compatibility."""
    image_list = tf.unstack(images, axis=0)
    assert len(image_list) == FLAGS.batch_size, ("Image list must be of size"
                                                 " batch size.")
    processed_image_list = []
    for image in image_list:
      # Change the range of images, and for generated images, make sure that
      # we account for any overflow.
      # Currently this is clamped to always false to only keep simple
      # transformations.
      processed_image_list.append(
          preprocess_image(
              image,
              default_image_size,
              default_image_size,
              blur_image=FLAGS.blur_image,
              is_training=self.is_training and self.from_scratch))
    images = tf.stack(processed_image_list, axis=0)
    tf.summary.image('input_images', images)
    return images

  @property
  def attribute_names(self):
    return self._attribute_names

  @property
  def is_training(self):
    return self.mode == 'train'
