#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an Inception Resnet V2 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_resnet_v2_on_flowers.sh
set -e
source preamble.sh
DATASET_DIR=${GLOBAL_DATA_PATH}"/celeba_for_tf_ig/"

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=${GLOBAL_RUNS_PATH}"/pretrained"

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
BLUR_IMAGE=1
MODEL_NAME="inception_resnet_v2_blur_"${BLUR_IMAGE}

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=${GLOBAL_RUNS_PATH}"/celeba_ig_classification/"${MODEL_NAME}

# Sleep for 5000 seconds before finetuning.
FINETUNE_SLEEP=5000

if [ ! -d ${TRAIN_DIR} ]; then
	mkdir ${TRAIN_DIR}
fi

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

## Fine-tune only the new layers for 15000 steps.
#JOB_NAME="celeba_ig_train"
#MODE_STR="train"
#RUN_JOB_STR=${JOB_NAME}"_train_"
#echo "Setting up job ${JOB_NAME}"
#
#
#CMD_STRING="python classification/classifier_train.py \
#	--path_to_irv2_checkpoint ${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt\
#	--blur_image=${BLUR_IMAGE}\
#  --dataset_dir ${DATASET_DIR}\
#	--finetune=0\
#	--save_summaries_secs 600\
#	--learning_rate 0.001\
#	--save_interval_secs 600\
#	--train_log_dir ${TRAIN_DIR}\
#	--max_number_of_steps 15000\
#	--batch_size 32"

#sbatch utils/slurm_wrapper.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"

#JOB_NAME="celeba_ig_eval"
#MODE_STR="eval"
#RUN_JOB_STR=${JOB_NAME}"_eval_"
#echo "Setting up job ${JOB_NAME}"
#CMD_STRING="python classification/classifier_eval.py \
#  --dataset_dir ${DATASET_DIR}\
#	--blur_image=${BLUR_IMAGE}\
#	--checkpoint_dir ${TRAIN_DIR}\
#	--eval_dir ${TRAIN_DIR}\
#	--batch_size 128"
#
#sbatch utils/slurm_wrapper_cpu_only.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
#
#echo "Sleeping for ${FINETUNE_SLEEP} seconds before launching finetuning jobs"
#sleep ${FINETUNE_SLEEP}

JOB_NAME="celeba_ig_train_ft"
MODE_STR="train"
RUN_JOB_STR=${JOB_NAME}"_train_"
echo "Setting up job ${JOB_NAME}"
CMD_STRING="python classification/classifier_train.py \
	--path_to_irv2_checkpoint ${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt\
	--dataset_dir ${DATASET_DIR}\
	--blur_image=${BLUR_IMAGE}\
	--finetune=1\
	--save_summaries_secs 600\
	--learning_rate 0.0001\
	--save_interval_secs 600\
	--train_log_dir ${TRAIN_DIR}\
	--max_number_of_steps 200000\
	--batch_size 32"
sbatch utils/slurm_wrapper.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
