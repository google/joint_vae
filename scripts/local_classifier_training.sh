#! /bin/sh
#
# local_classifier_training.sh
# Copyright (C) 2017 rvedantam3 <vrama91@vt.edu>
#
# Distributed under terms of the MIT license.
#
set -e

source preamble.sh

echo $CUDA_VISIBLE_DEVICES

DATASET="celeba"
DATASET_DIR=$GLOBAL_DATA_PATH/"celeba_for_tf_ig"

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/tmp/checkpoint

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
MODEL_NAME=inception_resnet_v2

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/inception_resnet_train

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
  tar -xvf inception_resnet_v2_2016_08_30.tar.gz
  mv inception_resnet_v2_2016_08_30.ckpt ${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt
  rm inception_resnet_v2_2016_08_30.tar.gz
fi

python classification/classifier_train.py \
	--dataset_dir ${DATASET_DIR}\
	--blur_image=1\
	--path_to_irv2_checkpoint ${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt\
	--finetune=1\
	--save_summaries_secs 60\
	--learning_rate 0.001\
	--save_interval_secs 60\
	--batch_size 32\

#python classification/classifier_eval.py\
#  --dataset_dir ${DATASET_DIR}\
#	--checkpoint_dir /tmp/inception_resnet_train\
#  --eval_dir /tmp/inception_resnet_train\
#	--split_name val\
#	--finetune=0\
#	--batch_size 128
