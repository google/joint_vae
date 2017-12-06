"""Code for latent space interpolation.

The original version of the code is from https://github.com/dribnet/plat

Used in the paper:
  Sampling Generative Networks
  Tom White
  NIPS 2016
"""

import numpy as np
from scipy.stats import norm


def lerp(val, low, high):
  """Linear interpolation."""
  return low + (high - low) * val


def lerp_gaussian(val, low, high):
  """Linear interpolation with gaussian CDF."""
  low_gau = norm.cdf(low)
  high_gau = norm.cdf(high)
  lerped_gau = lerp(val, low_gau, high_gau)
  return norm.ppf(lerped_gau)


def slerp(val, low, high):
  """Spherical interpolation. val has a range of 0 to 1."""
  if val <= 0:
    return low
  elif val >= 1:
    return high
  elif np.allclose(low, high):
    return low
  omega = np.arccos(
      np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
  so = np.sin(omega)
  return np.sin(
      (1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def slerp_gaussian(val, low, high):
  """Spherical interpolation with gaussian CDF (generally not useful)."""
  offset = norm.cdf(np.zeros_like(low))  # offset is just [0.5, 0.5, ...]
  low_gau_shifted = norm.cdf(low) - offset
  high_gau_shifted = norm.cdf(high) - offset
  circle_lerped_gau = slerp(val, low_gau_shifted, high_gau_shifted)
  epsilon = 0.001
  clipped_sum = np.clip(circle_lerped_gau + offset, epsilon, 1.0 - epsilon)
  result = norm.ppf(clipped_sum)
  return result


def get_interpfn(spherical, gaussian):
  """Returns an interpolation function."""
  if spherical and gaussian:
    return slerp_gaussian
  elif spherical:
    return slerp
  elif gaussian:
    return lerp_gaussian
  else:
    return lerp


def do_interpolation(interpolation_fn, min_val, max_val, number_of_points):
  values = [0.0] + list(
      np.cumsum(np.ones((number_of_points)) * 1 / float(number_of_points)))
  interpolated_points = []
  for value in values:
    interpolated_points.append(interpolation_fn(value, min_val, max_val))
  return interpolated_points
