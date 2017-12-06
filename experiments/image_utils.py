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

"""A few handy utilities for showing or saving images.

plot_images lays a set of images out on a grid, with optional labels.

show_image displays a single image, with optional border color.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import tensorflow as tf

gfile = tf.gfile


def get_color(orig_label=None,
              gt_label=None,
              target_label=None,
              adv_label=None,
              targeted_attack=False):
  """Get color according to different possible correctness combinations."""
  if ((gt_label is None or orig_label == gt_label) and
      (adv_label is None or adv_label == gt_label)):
    color = '#4d8021'  # 'green': correct prediction
  elif adv_label is None:
    color = '#b81e26'  # 'red': not adversarial, incorrect prediction
  elif adv_label == gt_label:
    color = '#a8db78'  # 'lightgreen': adv_label correct, but orig_label wrong
  elif adv_label == orig_label:
    color = '#b09b3e'  # 'yellow': incorrect, but no change to prediction
  elif not targeted_attack and adv_label is not None and adv_label != gt_label:
    color = '#b81e26'  # 'red': untargeted attack, adv_label changed, success
  elif targeted_attack and adv_label == target_label:
    color = '#b81e26'  # 'red': targeted attack, success
  elif targeted_attack and adv_label != target_label:
    color = '#d064a6'  # 'pink': targeted attack, changed prediction but failed
  else:
    color = '#6780d8'  # 'blue': unaccounted-for result
  return color


def plot_images(images,
                n=3,
                figure_width=8,
                filename='',
                filetype='png',  # pylint: disable=unused-argument
                orig_labels=None,
                gt_labels=None,
                target_labels=None,
                adv_labels=None,
                targeted_attack=False,
                skip_incorrect=True,
                blank_incorrect=True,
                blank_adv_correct=False,
                color_from_orig_label_if_blank_target=False,
                annotations=None,
                text_alpha=1.0,
                return_fig=False):
  """Plot images in a tight grid with optional labels."""
  is_adv = adv_labels is not None
  has_labels = (
      orig_labels is not None and gt_labels is not None and
      (not is_adv or not targeted_attack or target_labels is not None))

  orig_labels = (orig_labels
                 if orig_labels is not None else np.zeros(images.shape[0]))
  gt_labels = (gt_labels
               if gt_labels is not None else np.zeros(images.shape[0]))
  target_labels = (target_labels
                   if target_labels is not None else np.zeros(images.shape[0]))
  adv_labels = (adv_labels
                if adv_labels is not None else np.zeros(images.shape[0]))

  annotations = (annotations if annotations is not None else
                 [[] for _ in range(images.shape[0])])

  w = n
  h = int(np.ceil(images.shape[0] / float(w)))
  ih, iw = images.shape[1:3]
  channels = (images.shape[-1] if len(images.shape) > 3 else 1)

  figure_height = h * ih * figure_width / (iw * float(w))
  dpi = max(float(iw * w) / figure_width, float(ih * h) / figure_height)
  fig = plt.figure(figsize=(figure_width, figure_height), dpi=dpi)
  ax = fig.gca()

  minc = np.min(images)
  if minc < 0:
    # Assume images is in [-1, 1], and transform it back to [0, 1] for display
    images = images * 0.5 + 0.5

  figure = np.ones((ih * h, iw * w, channels)) * 0.5
  if channels == 1:
    figure = figure.squeeze(-1)

  ax.set_frame_on(False)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.axis('off')
  fig.tight_layout(pad=0)

  cur_image = 0
  for i in xrange(h):
    for j in xrange(w):
      if i * w + j >= images.shape[0]:
        # White out overflow slots, but not slots from skipping images.
        figure[i * ih:(i + 1) * ih, j * iw:(j + 1) * iw] = 1.0
        continue

      try:
        image = None
        while image is None:
          image = images[cur_image]
          orig_label = orig_labels[cur_image]
          gt_label = gt_labels[cur_image]
          target_label = target_labels[cur_image]
          adv_label = adv_labels[cur_image]
          annots = annotations[cur_image]
          cur_image += 1

          if has_labels and orig_label != gt_label:
            if skip_incorrect:
              image = None
            elif blank_incorrect:
              image *= 0.2

          if (image is not None and has_labels and is_adv and blank_adv_correct
              and adv_label == gt_label):
            image *= 0.0

      except IndexError:
        continue

      image = image.reshape([ih, iw, channels])
      if channels == 1:
        image = image.squeeze(-1)

      figure[i * ih:(i + 1) * ih, j * iw:(j + 1) * iw] = image

      if annots:
        for k, a in enumerate(annots):
          ax.annotate(
              str(a['label']),
              xy=(float(j) / w + 0.05 / w,
                  1.0 - float(i) / h - (0.05 + k * 0.13) / h),
              xycoords='axes fraction',
              color=a['color'],
              verticalalignment='top',
              alpha=text_alpha,
              fontsize=(0.12 * 72 * figure_height / h))
      elif has_labels:
        color = get_color(orig_label, gt_label if gt_label else None,
                          None, None, False)
        ax.annotate(
            str(orig_label),
            xy=(float(j) / w + 0.05 / w, 1.0 - float(i) / h - 0.05 / h),
            xycoords='axes fraction',
            color=color,
            verticalalignment='top',
            alpha=text_alpha,
            fontsize=(0.12 * 72 * figure_height / h)
        )

        if is_adv:
          if target_label or not color_from_orig_label_if_blank_target:
            color = get_color(orig_label, gt_label, target_label, adv_label,
                              targeted_attack and target_label)
          s = str(adv_label)
          if targeted_attack and target_label:
            s = '%s|%s' % (str(adv_label), str(target_label))
          ax.annotate(
              s,
              xy=(float(j) / w + 0.05 / w, 1.0 - float(i) / h - 0.18 / h),
              xycoords='axes fraction',
              color=color,
              verticalalignment='top',
              alpha=text_alpha,
              fontsize=(0.12 * 72 * figure_height / h)
          )

  if channels == 1:
    fig.figimage(figure, cmap='Greys')
  else:
    fig.figimage(figure)

  figure = canvas_to_np(fig)

  if filename:
    with tf.gfile.Open('{filename}.{filetype}'.format(**locals()), 'w') as f:
      fig.savefig(f, dpi='figure')
  else:
    fig.show(warn=False)

  if return_fig:
    return fig

  plt.close(fig)

  return figure


def show_image(
    image,
    color=None,
    filename=None,
    filetype='png',  # pylint: disable=unused-argument
    show=True,
    return_fig=False):
  """Draw a single image as a stand-alone figure with an optional border."""
  image = image[0] if image.ndim > 3 else image
  w = image.shape[1]
  h = image.shape[0]
  c = 3 if color else image.shape[-1]

  pad = int(w * 0.05) if color else 0
  wpad = 2 * pad + w
  hpad = 2 * pad + h

  figure_width = 2.0
  figure_height = figure_width * hpad / float(wpad)
  dpi = wpad / figure_width
  fig = plt.figure(figsize=(figure_width, figure_height), dpi=dpi)
  ax = fig.gca()

  minc = np.min(image)
  if minc < 0:
    # Assume image is in [-1, 1], and transform it back to [0, 1] for display
    image = image * 0.5 + 0.5

  ax.set_frame_on(False)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.axis('off')
  fig.tight_layout(pad=0)

  color = mpl.colors.colorConverter.to_rgb(color) if color else 0.0

  figure = np.ones((hpad, wpad, c)) * color
  if c == 1:
    figure = figure.squeeze(-1)

  figure[pad:pad + h, pad:pad + w] = image

  if c == 1:
    fig.figimage(figure, cmap='Greys')
  else:
    fig.figimage(figure)

  figure = canvas_to_np(fig)

  if filename:
    with tf.gfile.Open('{filename}.{filetype}'.format(**locals()), 'w') as f:
      fig.savefig(f, dpi='figure')

  if show:
    fig.show(warn=False)

  if return_fig:
    return fig

  plt.close(fig)

  return figure


def canvas_to_np(figure, rescale=False):
  """Turn a pyplt figure into an np image array of bytes or floats."""
  figure.canvas.draw()

  # Collect the pixels back from the pyplot figure.
  (buff, width_height) = figure.canvas.print_to_buffer()
  img = np.array(buff).reshape(list(width_height)[::-1] + [-1])
  if img.shape[-1] > 3:
    img = img[:, :, 0:3]

  img = img.astype(np.float32)
  if rescale:
    img /= 255.0

  return img
