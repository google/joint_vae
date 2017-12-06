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

"""A class to add perturbations to MNIST images.

The perturbations we add to the MNIST dataset include rotation,
scaling and translation.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np

import utils




def rotate_image(input_image, rotation_angle, interpolation=Image.BICUBIC):
  """Rotate a PIL Image and return the output image and size."""
  out_image = input_image.rotate(rotation_angle, interpolation, expand=True)
  image_size = utils.XY(*out_image.size)
  return out_image, image_size


def scale_image(input_image, image_size, scale, interpolation=Image.BICUBIC):
  """Scale a PIL Image and return the output image and size."""
  input_image.thumbnail((int(image_size.x * scale), int(image_size.y * scale)),
                        interpolation)
  image_size = utils.XY(*input_image.size)
  return input_image, image_size


def paste_image_on_black_canvas(image, location, canvas_size=64):
  """Paste input image at given location on a black canvas."""
  canvas = Image.fromarray(np.zeros((canvas_size, canvas_size, 3)).astype(np.uint8))

  canvas.paste(image, (location.x, location.y))
  return canvas


class PerturbImages(object):
  """Intitializes the perturb images class."""

  def __init__(self, valid_transformations, image_size=28, canvas_size=64):
    """Intializes the perturbations for images."""
    self._valid_transformations = valid_transformations
    self._interpolation_type = Image.BICUBIC
    self._image_size = image_size
    self._canvas_size = canvas_size
    self._initialize_canvas()

  def _initialize_canvas(self):
    """Initializes the canvas image on which we overlay a digit."""
    self._canvas = Image.fromarray(
        np.zeros((self._canvas_size, self._canvas_size, 3)))

  def bind_image(self, image):
    """Binds an mnist image to the canvas.

    We apply transformations to the binded image and then paste it
    on the canvas.
    """
    piece = np.repeat(image, 3, axis=-1)
    piece = Image.fromarray(piece)
    self._piece = piece

  def transform_image(self, rotation, scaling, location):
    # Peform transformations in the following order:
    # First do rotation, if specified.
    # Then do scaling
    # Then paste the image at some specified location.

    if scaling > 1.0:
      raise ValueError("Maximum allowed scale is 1.0.")

    if "rotate" in self._valid_transformations:
      self._piece = self._piece.rotate(
          rotation, self._interpolation_type, expand=False)

    if "scaling" in self._valid_transformations:
      self._piece.thumbnail((int(self._image_size * scaling), int(
          self._image_size * scaling)), self._interpolation_type)

    if "location" in self._valid_transformations:
      self._canvas.paste(self._piece, (location.x, location.y))

    return self._canvas
