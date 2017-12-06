# Script to do large scale training of images on CELEBA faces dataset.
source preamble.sh

DATASET="affine_mnist"
# TODO(vrama): Put affine mnist at the global dataset path.
# This actually doesnt matter for affine mnist.
DATASET_DIR="/nethome/rvedantam3/data/mnista/"

ROOT_LOG_DIR=${GLOBAL_RUNS_PATH}"/imagination"
SPLIT_TYPE="comp"
EVAL_SPLIT_NAME="val"

ICLR_RESULTS_PATH=${GLOBAL_RUNS_PATH}/imagination/iclr_numbers_${SPLIT_TYPE}
EXP_PREFIX=${SPLIT_TYPE}_affine_mnist_iclr_fix
MODEL_TYPE=multi
num_training_steps=250000

loss_type=multimodal_elbo
alpha_x=1

if [ ! -e ${ROOT_LOG_DIR}/${EXP_PREFIX} ]; then
	mkdir ${ROOT_LOG_DIR}/${EXP_PREFIX}
fi

########################################### Best Comp Model.##########################################
stop_elbox_gradient=0
for num_latent in 10
do
	for stop_elboy_gradient in 1
	do
		for product_of_experts in 0
		do
			for private_py_scaling in 100
			do
				for alpha_y in 10 
				do
					for l1_reg in 0.0
					do
						JOB_NAME="affine_mnist_loss_"${loss_type}"_poe_"${product_of_experts}"_nl_"${num_latent}"_sg_"${stop_elboy_gradient}"_privatey_"${private_py_scaling}"_xsg_"${stop_elbox_gradient}"_ax_"${alpha_x}"_ay_"${alpha_y}"_l1_pxyz_"${l1_reg}
						echo "Setting up job" $JOB_NAME

						TRAIN_DIR=${ROOT_LOG_DIR}/${EXP_PREFIX}/${JOB_NAME}

						if [ ! -e ${TRAIN_DIR} ]; then
							mkdir ${TRAIN_DIR}
						fi

						MODE_STR="_train_"
						RUN_JOB_STR=${JOB_NAME}"_train_"
						CMD_STRING="python experiments/vae_train.py\
							--dataset affine_mnist \
							--dataset_dir ${DATASET_DIR}\
							--split_type ${SPLIT_TYPE}\
							--loss_type ${loss_type}\
							--image_likelihood Bernoulli\
							--label_likelihood Categorical \
							--alpha_x=${alpha_x}\
							--alpha_y=${alpha_y}\
							--product_of_experts=${product_of_experts}\
							--train_log_dir ${TRAIN_DIR}\
							--model_type ${MODEL_TYPE}\
							--num_latent ${num_latent}\
							--stop_elboy_gradient=${stop_elboy_gradient}\
							--stop_elbox_gradient=${stop_elbox_gradient}\
							--max_number_of_steps ${num_training_steps}\
							--private_py_scaling ${private_py_scaling}\
							--label_decoder_regularizer ${l1_reg}\
							--alsologtostderr"
						
						if [[ ${1} == '' ]] || [[ ${1} == 'train' ]]; then
						  sbatch utils/slurm_wrapper.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
					  fi

						MODE_STR="_eval_"
						RUN_JOB_STR=${JOB_NAME}${MODE_STR}
						CMD_STRING="python experiments/vae_eval.py\
							--dataset affine_mnist \
							--dataset_dir ${DATASET_DIR} \
							--split_type ${SPLIT_TYPE}\
							--split_name val \
							--num_result_datapoints 25 \
							--image_likelihood Bernoulli \
							--label_likelihood Categorical \
							--product_of_experts=${product_of_experts}\
							--model_type ${MODEL_TYPE}\
							--num_latent ${num_latent}\
							--checkpoint_dir ${TRAIN_DIR}\
							--eval_dir ${TRAIN_DIR}\
							--alsologtostderr"

						if [[ ${1} == '' ]] || [[ ${1} == 'eval' ]]; then
						  sbatch utils/slurm_wrapper_cpu_only.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
					  fi

						MODE_STR="_imeval_"
						RUN_JOB_STR=${JOB_NAME}${MODE_STR}

						CMD_STRING="python experiments/vae_imagination_eval.py\
							--input_queries=${PWD}/query_files/mnist_with_attributes_dataset_${SPLIT_TYPE}/query_${EVAL_SPLIT_NAME}_poe_${product_of_experts}.p\
							--model_type multi\
							--dataset affine_mnist \
							--dataset_dir ${DATASET_DIR}\
							--split_name ${EVAL_SPLIT_NAME}\
							--split_type ${SPLIT_TYPE}\
							--image_likelihood Bernoulli\
							--label_likelihood Categorical \
							--product_of_experts=${product_of_experts}\
							--iclr_results_path=${ICLR_RESULTS_PATH}\
							--num_latent ${num_latent}\
							--checkpoint_dir ${TRAIN_DIR}\
							--eval_dir ${TRAIN_DIR}\
							--alsologtostderr"

						 if [[ ${1} == '' ]] || [[ ${1} == 'imeval' ]]; then
						   sbatch utils/slurm_wrapper_cpu_only.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
						 fi
					done
				done
			done
		done
	done
done
########################################### Different settings for KL(q, p) SGx model.################
stop_elbox_gradient=1
for num_latent in 10
do
	for stop_elboy_gradient in 1
	do 
		for product_of_experts in 1
		do
			for private_py_scaling in 1
			do
				for alpha_y in 10 50
				do
					for l1_reg in 5e-5
					do
						JOB_NAME="affine_mnist_loss_"${loss_type}"_poe_"${product_of_experts}"_nl_"${num_latent}"_sg_"${stop_elboy_gradient}"_privatey_"${private_py_scaling}"_xsg_"${stop_elbox_gradient}"_ax_"${alpha_x}"_ay_"${alpha_y}"_l1_pxyz_"${l1_reg}
						echo "Setting up job" $JOB_NAME

						TRAIN_DIR=${ROOT_LOG_DIR}/${EXP_PREFIX}/${JOB_NAME}

						if [ ! -e ${TRAIN_DIR} ]; then
							mkdir ${TRAIN_DIR}
						fi

						MODE_STR="_train_"
						RUN_JOB_STR=${JOB_NAME}"_train_"
						CMD_STRING="python experiments/vae_train.py\
							--dataset affine_mnist \
							--dataset_dir ${DATASET_DIR}\
							--split_type ${SPLIT_TYPE}\
							--loss_type ${loss_type}\
							--image_likelihood Bernoulli\
							--label_likelihood Categorical \
							--alpha_x=${alpha_x}\
							--alpha_y=${alpha_y}\
							--product_of_experts=${product_of_experts}\
							--train_log_dir ${TRAIN_DIR}\
							--model_type ${MODEL_TYPE}\
							--num_latent ${num_latent}\
							--stop_elboy_gradient=${stop_elboy_gradient}\
							--stop_elbox_gradient=${stop_elbox_gradient}\
							--max_number_of_steps ${num_training_steps}\
							--private_py_scaling ${private_py_scaling}\
							--label_decoder_regularizer ${l1_reg}\
							--alsologtostderr"

						if [[ ${1} == '' ]] || [[ ${1} == 'train' ]]; then
						  sbatch utils/slurm_wrapper.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
						fi

						MODE_STR="_eval_"
						RUN_JOB_STR=${JOB_NAME}${MODE_STR}
						CMD_STRING="python experiments/vae_eval.py\
							--dataset affine_mnist \
							--dataset_dir ${DATASET_DIR} \
							--split_type ${SPLIT_TYPE}\
							--split_name val \
							--num_result_datapoints 25 \
							--image_likelihood Bernoulli \
							--label_likelihood Categorical \
							--product_of_experts=${product_of_experts}\
							--model_type ${MODEL_TYPE}\
							--num_latent ${num_latent}\
							--checkpoint_dir ${TRAIN_DIR}\
							--eval_dir ${TRAIN_DIR}\
							--alsologtostderr"

						if [[ ${1} == '' ]] || [[ ${1} == 'eval' ]]; then
						  sbatch utils/slurm_wrapper_cpu_only.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
						fi

						MODE_STR="_imeval_"
						RUN_JOB_STR=${JOB_NAME}${MODE_STR}

						CMD_STRING="python experiments/vae_imagination_eval.py\
							--input_queries=${PWD}/query_files/mnist_with_attributes_dataset_${SPLIT_TYPE}/query_${EVAL_SPLIT_NAME}_poe_${product_of_experts}.p\
							--model_type multi\
							--dataset affine_mnist \
							--dataset_dir ${DATASET_DIR}\
							--split_name ${EVAL_SPLIT_NAME}\
							--split_type ${SPLIT_TYPE}\
							--image_likelihood Bernoulli\
							--label_likelihood Categorical \
							--product_of_experts=${product_of_experts}\
							--iclr_results_path=${ICLR_RESULTS_PATH}\
							--num_latent ${num_latent}\
							--checkpoint_dir ${TRAIN_DIR}\
							--eval_dir ${TRAIN_DIR}\
							--alsologtostderr"

						 if [[ ${1} == '' ]] || [[ ${1} == 'imeval' ]]; then
						   sbatch utils/slurm_wrapper_cpu_only.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
						 fi
					done
				done
			done
		done
	done
done
