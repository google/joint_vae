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

"""Contains code for loading and preprocessing the Grayscale Shapes data."""

import tensorflow as tf

from tensorflow.contrib.slim.python.slim.data import dataset_data_provider

from datasets.mnist_attributes import affine_mnist_dataset_comp
from datasets.mnist_attributes import affine_mnist_dataset_iid
from datasets.mnist_attributes import affine_mnist_retrieval_set
import retrieval_set

BINARIZE = 'binarize'
CENTER = 'center'


def provide_data(split_name,
                 batch_size,
                 split_type='comp',
                 grayscale=True,
                 preprocess_options=CENTER,
                 dataset_dir=None,
                 shuffle_data=None):
  """Provides batches of Grayscale shapes dataset.

  Args:
    split_name: Name of the split of data, one of the keys in
      grayscale_shapes_dataset._SPLITS_TO_SIZES
    batch_size: The number of images in each batch.
    split_type: 'comp' or 'iid', comp loads a dataset
      with non-intersecting label sets during train and test. IID just
      loads a dataset that is split IID.
    grayscale: Boolean, True indicates that the images are grayscale,
      False indicates images are RGB.
    preprocess_options: 'binarize' or 'center', options for preprocessing
      images. 'binarize' converts the image into a float tensor with 0 and 1
      values, while 'center' does mean subtraction and scaling to make the
      input between -1 and 1.
    dataset_dir: The directory where the deepmind grayscale shapes
      data can be found.

  Returns:
    images: A `Tensor` of size [batch_size, 64, 64, 1] if grayscale is True.
      A `Tensor` of size [batch_size, 64, 64, 3] if grayscale is False.
    batch_label_list: A list of `Tensor` of size [batch_size], where each
      element has a value denoting the class label associated with the four kind
      of labels (shape, size, orientation, location). Thus, in this case, the
      list will be of length 4.
    latents: A `Tensor` of size [batch_size, _NUM_LATENTS], which is the true
      number of latent values underlying the model. These are available for
      grayscale_shapes_dataset and grayscale_shapes_iid_dataset.
    num_samples: The number of total samples in the dataset.
    num_classes_per_attribute: List, with number of labels for the classfication
      problem corresponding to each attribute.
    shuffle_data: Boolean, indicates whether to use a RandomShuffleQueue or a
      FIFO queue for loading data.
  Raises:
    ValueError: if the split_name is not either 'train' or 'val' or 'test'
  """
  if not shuffle_data:
    if split_name == 'train':
      shuffle_data = True
    else:
      shuffle_data = False

  # The retrieval split is a different split of the data, from
  # IID or compositional. The retrieval split is an IID split, still though.
  if split_name == 'retrieval':
    dataset, metadata = affine_mnist_retrieval_set.get_split(split_name=split_name, dataset_dir=dataset_dir)
  # If we are not requesting the retrieval split, then look for whether we want
  # IID or compositional split.
  elif split_type == 'comp':
    dataset, metadata = affine_mnist_dataset_comp.get_split(
        split_name, dataset_dir)
  elif split_type == 'iid':
    dataset, metadata = affine_mnist_dataset_iid.get_split(
        split_name, dataset_dir)
  else:
    raise ValueError('Invalid %s split type.', split_type)

  if split_name != 'retrieval':
    assert metadata['split_type'] == split_type, ('Called the wrong dataset'
                                                  ' specification file.')

  provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      common_queue_capacity=2 * batch_size,
      common_queue_min=batch_size,
      shuffle=shuffle_data)
  [image, labels, latents] = provider.get(['image', 'labels', 'latents'])

  image = tf.to_float(image)

  # Preprocess the images.
  if preprocess_options == CENTER:
    image = tf.subtract(image, 128.0)
    image = tf.div(image, 128.0)
  elif preprocess_options == BINARIZE:
    image = tf.div(image, 255.0)
  else:
    raise ValueError('Invalid argument for preprocess_options %s' %
                     (preprocess_options))

  # Creates a QueueRunner for the pre-fetching operation.
  images, batch_labels, batch_latents = tf.train.batch(
      [image, labels, latents],
      batch_size=batch_size,
      num_threads=1,
      capacity=5 * batch_size)

  if grayscale is True:
    images = tf.slice(
        images, [0, 0, 0, 0], [batch_size, 64, 64, 1], name='slice_image')

  batch_label_list = tf.unstack(batch_labels, axis=-1, name='unstack_labels')

  return images, batch_label_list, batch_latents, dataset.num_samples, metadata[
      'num_classes_per_attribute']
