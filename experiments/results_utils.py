#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright ÃÂÃÂ© 2017 rvedantam3 <vrama91@vt.edu>
#
# Distributed under terms of the MIT license.
"""Some utilities for processing and presenting results."""
import glob
import random
import numpy as np


def filter_result_files(file_list, cannot_contain=('entropy', 'metadata')):
  return [f for f in file_list if all([cc not in f for cc in cannot_contain])]


def compute_mean_std(per_datum_results, num_splits=5, split=0.8, seed=123):
  # Use the seed for repeatability.
  np.random.seed(seed)

  assert isinstance(per_datum_results[0],
                    float), "Result must be a 1-D float list."

  num_original_results = len(per_datum_results)
  subset_results = []

  for this_split in xrange(num_splits):  # pylint: disable=E0602
    subset_pick = np.random.choice(
        per_datum_results,
        int(split * num_original_results),
        p=[1.0 / num_original_results] * num_original_results)
    # Assumes the metric is linear in instances.
    subset_results.append(np.mean(subset_pick))
  return np.mean(subset_results), np.std(subset_results)



def latest_checkpoints(file_list, type_of_file='checkpoint'):
  """Grab the latest checkpoints for a given list of files."""
  latest_files = []
  for item in file_list:
    file_pattern = '_'.join(item.split("_")[:-1])
    file_extension = item.split('.')[-1]
    all_files = glob.glob(file_pattern + "*")
    if type_of_file == 'checkpoint':
      all_files = filter_result_files(all_files)
    else:
      all_files = filter_result_files(all_files, cannot_contain=('kl', 'entropy'))
    global_steps = [(int(f.split('_')[-1].split('.')[0]), f) for f in all_files]
    latest_checkpoint = sorted(global_steps, key=lambda x: -x[0])
    latest_files.append(latest_checkpoint[0][1])

  return list(set(latest_files))


def filename_to_method(filename, ban):
  fname= filename.split('/')[-1].replace(ban, '').rstrip('.p')
  return '_'.join(fname.split('_')[:-1])


def extract_global_step(filename):
  return filename.split('_')[-1].split('.')[0]
