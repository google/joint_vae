# Script to do large scale training of images on CELEBA faces dataset.
source preamble.sh

DATASET="affine_mnist"
# TODO(vrama): Put affine mnist at the global dataset path.
# This actually doesnt matter for affine mnist.
DATASET_DIR="/nethome/rvedantam3/data/mnista/"

ROOT_LOG_DIR=${GLOBAL_RUNS_PATH}"/imagination"
SPLIT_TYPE="iid"
EVAL_SPLIT_NAME="val"

ICLR_RESULTS_PATH=${GLOBAL_RUNS_PATH}/imagination/iclr_mnista_fresh_${SPLIT_TYPE}
EXP_PREFIX=iclr_mnista_fresh_${SPLIT_TYPE}

MODEL_TYPE=multi
num_training_steps=500000

alpha_x=1
product_of_experts=1
if [ ! -e ${ROOT_LOG_DIR}/${EXP_PREFIX} ]; then
	mkdir ${ROOT_LOG_DIR}/${EXP_PREFIX}
fi

########################################## KL(q| p).################
loss_type=multimodal_elbo
for num_latent in 10
do
	for stop_elboy_gradient in 1
	do 
		for stop_elbox_gradient in 1
		do
			for private_py_scaling in 1 50 100
			do
				for alpha_y in 1 10 50
				do
					for l1_reg in 5e-6
					do
						JOB_NAME="affine_mnist_loss_"${loss_type}"_poe_"${product_of_experts}"_nl_"${num_latent}"_sg_"${stop_elboy_gradient}"_privatey_"${private_py_scaling}"_xsg_"${stop_elbox_gradient}"_ax_"${alpha_x}"_ay_"${alpha_y}"_l1_pxyz_"${l1_reg}
						echo "Setting up job" $JOB_NAME

						TRAIN_DIR=${ROOT_LOG_DIR}/${EXP_PREFIX}/${JOB_NAME}

						if [ ! -e ${TRAIN_DIR} ]; then
							mkdir ${TRAIN_DIR}
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
							--evaluate_once\
							--interplate_queries=0\
							--alsologtostderr"
						eval ${CMD_STRING}
						# TODO(vrama): Remove that interpolate queries flag.

					done
				done
			done
		done
	done
done

################JmVAE models #####################################
loss_type=jmvae
for num_latent in 10
do
	for jmvae_alpha in 1.0  #TODO(vrama): Check if we can do for 0.1 0.01
	do
		for alpha_y in 1 10 50
		do
		  for l1_reg in 5e-6
		  do

			JOB_NAME="affine_mnist_loss_"${loss_type}"_poe_"${product_of_experts}"_nl_"${num_latent}"_jmalpha_"${jmvae_alpha}"_ax_"${alpha_x}"_ay_"${alpha_y}"_l1_pxyz_"${l1_reg}

			TRAIN_DIR=${ROOT_LOG_DIR}/${EXP_PREFIX}/${JOB_NAME}

			if [ ! -e ${TRAIN_DIR} ]; then
				mkdir ${TRAIN_DIR}
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
				--evaluate_once\
				--alsologtostderr"
			eval ${CMD_STRING}
		 done
	 done
  done
done


###### BiVCCA models ######
loss_type=bivcca
for num_latent in 10
do
	for bivcca_mu in 0.7  #TODO(vrama): Check if we can do for 0.1 0.01
	do
		for alpha_y in 1 10 50
		do
		  for l1_reg in 5e-6
		  do

			JOB_NAME="affine_mnist_loss_"${loss_type}"_poe_"${product_of_experts}"_nl_"${num_latent}"_bmu"${bivcca_mu}"_ax_"${alpha_x}"_ay_"${alpha_y}"_l1_pxyz_"${l1_reg}

			TRAIN_DIR=${ROOT_LOG_DIR}/${EXP_PREFIX}/${JOB_NAME}

			if [ ! -e ${TRAIN_DIR} ]; then
				mkdir ${TRAIN_DIR}
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
				--evaluate_once\
				--alsologtostderr"
			eval ${CMD_STRING}
		 done
	 done
  done
done
