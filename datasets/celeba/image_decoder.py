#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright Â© 2017 rvedantam3 <vrama91@vt.edu>
#
# Distributed under terms of the MIT license.

"""A class specifying preprocessing steps for CelebA images.

Author: Ramakrishna Vedantam
vrama@gatech.edu

The class is based on the example image decoder at:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/\
        slim/python/slim/data/tfexample_decoder.py#L270
"""
import tensorflow as tf
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops

from tensorflow.contrib.slim.python.slim.data import tfexample_decoder


class ImageDecodeProcess(tfexample_decoder.Image):
  """An image handler class which also preprocesses the image."""
  def __init__(self, image_length=64, central_crop_offset=(40, 15), target_length=(148, 148), *argv, **kwargs):
    """Class to handle loading and preprocessing of image data.

    First loads the image from the tfexample encoded / raw image byte stream
    and then resizes it into image dimensions. Then optionally applies
    transformations such as cropping, resizing, and filtering (TODO).

    Args:
      central_crop_offset: A tuple of (height, width) of the offset from the
        top left corner of the image from where we would like to start cropping
        the bounding box.
      target_length: A tuple of (height, width) of the size of the box. We
        can obtain the bottom left coordinates of the box for instance by doing
        central_crop_offset + target_length.
      image_length: Int, size of the resized box for the image.
    Returns:
      image_cropped_resized: `Tensor` of tf.float32 with the pixels from the
        cropped and resized image.
    """
    super(ImageDecodeProcess, self).__init__(*argv, **kwargs)
    self._image_length = image_length
    self._central_crop_offset = central_crop_offset
    self._target_length = target_length

  def _decode(self, image_buffer, image_format):
    """Decodes and optionally applies transformations to the image buffer.

    Args:
      image_buffer: The tensor representing the encoded string tensor.
      image_format: The image format for the image in `image_buffer`. If image
        format is `raw`, all images are expected to be in this format, otherwise
        this op can decode a mix of `jpg` and `png` formats.

    Returns:
      A `Tensor` that represents decoded image of shape self._shape or
        (?, ?, self._channels) if shape is not specified.

    """
    def decode_image():
      """Decodes a png or jpg based on the headers."""
      return image_ops.decode_image(image_buffer, self._channels)

    def decode_raw():
      """Decodes a raw image."""
      return parsing_ops.decode_raw(image_buffer, out_type=self._dtype)

    pred_fn_pairs = {
        math_ops.logical_or(
            math_ops.equal(image_format, 'raw'),
            math_ops.equal(image_format, 'RAW')): decode_raw,
    }

    image = control_flow_ops.case(
        pred_fn_pairs, default=decode_image, exclusive=True)
    image.set_shape([None, None, self._channels])

    if self._shape is not None:
      image = array_ops.reshape(image, self._shape)

    # Process image.
    image_cropped = tf.image.crop_to_bounding_box(image, self._central_crop_offset[0], self._central_crop_offset[1], self._target_length[0], self._target_length[1])
    assert image_cropped.get_shape().as_list()[0] == image_cropped.get_shape().as_list()[1], ("Cropped image must be square.")

    # Resize image to 64 x 64.
    image_cropped_resized = tf.image.resize_images(image_cropped, [self._image_length, self._image_length], method=tf.image.ResizeMethod.BICUBIC)

    # TODO(vrama): Get the gaussian blur based on image scaling thing to work.

    return image_cropped_resized
