"""Provides a data provider for the CelebA dataset.

See celeba_dataset.py for more details of the CelebA dataset.

Author: Ramakrishna Vedantam (vrama@)
"""
from datasets.celeba import celeba_dataset

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import dataset_data_provider
from datasets import label_map

CENTER = 'center'

def provide_data(dataset_dir,
                 split_name,
                 batch_size,
                 split_type,
                 image_size,
                 preprocess_options=None,
                 grayscale=None,
                 shuffle_data=True,
                 num_preprocessing_threads=4,
                 num_readers=10,
                 add_summary=False):
  """TODO(vrama): Add documentation."""
  # Check that the options are all valid.
  if preprocess_options != CENTER:
    raise ValueError("Must preprocess celeba images to be -1 to 1.")

  dataset_celeba, metadata = celeba_dataset.get_split(
      split_name=split_name,
      split_type=split_type,
      dataset_dir=dataset_dir,
      image_length=image_size[0])
  attribute_names = label_map.LabelMap(metadata['label_map_json']).attributes

  # Plug it into a data provider.
  provider = dataset_data_provider.DatasetDataProvider(
      dataset_celeba,
      num_readers=num_readers,
      shuffle=shuffle_data,
      common_queue_capacity=4 * batch_size,
      common_queue_min=2 * batch_size)

  # Extract labels from the data provider.
  # Note that the celeba dataset is configured in a way that we extract the
  # center 64x64 pixels from a 148 x 148 center crop of the original celeba
  # image.
  # TODO(vrama): Also add blur as done in the vae-gan paper.
  [image, labels] = provider.get(['image', 'labels'])

  image = tf.to_float(image)

  if preprocess_options == CENTER:
    image = tf.subtract(image, 128.0)
    image = tf.div(image, 128.0)

  batch_image, batch_labels = tf.train.batch(
      [image, labels],
      batch_size=batch_size,
      num_threads=num_preprocessing_threads,
      capacity=2 * batch_size)

  batch_label_list = tf.unstack(batch_labels, axis=-1, name='unstack_labels')

  # Add image summary.
  if add_summary:
    tf.summary.image("input_images", batch_image)

  return batch_image, batch_label_list, dataset_celeba.num_samples, metadata[
      'num_classes_per_attribute'], attribute_names
