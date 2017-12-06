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

"""Package for building Joint VAE models in tensorflow and sonnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Top-level architecture
from joint_vae.joint_vae import JointVae
from joint_vae.loss import Loss

# Custom losses
from joint_vae.bivcca_loss import BiVccaLoss
from joint_vae.jmvae_loss import JmvaeLoss
from joint_vae.multimodal_elbo_loss import MultimodalElboLoss

# Encoder and decoder components
from joint_vae.encoder_decoder import MultiModalEncoder
from joint_vae.encoder_decoder import Encoder
from joint_vae.encoder_decoder import Decoder
from joint_vae.encoder_decoder import Encoders
from joint_vae.encoder_decoder import Decoders
from joint_vae.encoder_decoder import Prior
from joint_vae.encoder_decoder import MultiLayerPerceptronMultiModalEncoder
from joint_vae.encoder_decoder import MultiLayerPerceptronEncoder
from joint_vae.encoder_decoder import ProductOfExpertsEncoder
from joint_vae.encoder_decoder import ConvolutionalEncoder
from joint_vae.encoder_decoder import BernoulliDecoder
from joint_vae.encoder_decoder import ConvolutionalDecoder
from joint_vae.encoder_decoder import LabelDecoder
from joint_vae.encoder_decoder import NormalPrior
from joint_vae.encoder_decoder import get_jmvae_networks
from joint_vae.encoder_decoder import get_convolutional_multi_vae_networks
