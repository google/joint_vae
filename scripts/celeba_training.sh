# Script to do large scale training of images on CELEBA faces dataset.
source preamble.sh

DATASET="celeba"
DATASET_DIR=${GLOBAL_DATA_PATH}"/celeba_for_tf_ig/"
ROOT_LOG_DIR=${GLOBAL_RUNS_PATH}"/imagination"

EXP_PREFIX=celeba_qn_ic
MODEL_TYPE=multi
num_training_steps=200000

loss_type=multimodal_elbo
alpha_x=1
alpha_y=50
product_of_experts=1

if [ ! -e ${ROOT_LOG_DIR}/${EXP_PREFIX} ]; then
	mkdir ${ROOT_LOG_DIR}/${EXP_PREFIX}
fi

########################################### Triple elbo models.################
#for num_latent in 18
#do
#	for stop_elboy_gradient in 0 1
#	do 
#		for private_py_scaling in 50 100
#		do
#			alpha_y=${private_py_scaling}
#			JOB_NAME="celeba_loss_"${loss_type}"_poe_"${product_of_experts}"_nl_"${num_latent}"_sg_"${stop_elboy_gradient}"_privatey_"${private_py_scaling}"_ax_"${alpha_x}"_ay_"${alpha_y}
#			echo "Setting up job" $JOB_NAME
#
#			TRAIN_DIR=${ROOT_LOG_DIR}/${EXP_PREFIX}/${JOB_NAME}
#
#			MODE_STR="_train_"
#			RUN_JOB_STR=${JOB_NAME}"_train_"
#			CMD_STRING="python experiments/vae_train.py --dataset celeba --dataset_dir ${DATASET_DIR} --loss_type ${loss_type} --image_likelihood Gaussian --label_likelihood Categorical --alpha_x=${alpha_x} --alpha_y=${alpha_y} --product_of_experts=${product_of_experts} --train_log_dir ${TRAIN_DIR} --model_type ${MODEL_TYPE}  --num_latent ${num_latent} --stop_elboy_gradient=${stop_elboy_gradient} --max_number_of_steps ${num_training_steps} --private_py_scaling ${private_py_scaling} --alsologtostderr"
#
#			sbatch utils/slurm_wrapper.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
#
#			MODE_STR="_eval_"
#			RUN_JOB_STR=${JOB_NAME}${MODE_STR}
#			CMD_STRING="python experiments/vae_eval.py --dataset celeba --dataset_dir ${DATASET_DIR} --split_name val --num_result_datapoints 25 --image_likelihood Gaussian --label_likelihood Categorical --product_of_experts=${product_of_experts} --model_type ${MODEL_TYPE}  --num_latent ${num_latent}   --checkpoint_dir ${TRAIN_DIR} --eval_dir ${TRAIN_DIR}  --alsologtostderr"
#
#			sbatch utils/slurm_wrapper_cpu_only.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
#		done
#	done
#done

########################################### Without likelihood reweighting.################
# Baseline without likelihood reweighting
#private_py_scaling=1
#alpha_x=1 # melbo_x1 is always set to 1
#alpha_y=1
#for num_latent in 18 100
#do
#	for stop_elboy_gradient in 0 1
#	do
#		JOB_NAME="celeba_loss_"${loss_type}"_nl_"${num_latent}"_sg_"${stop_elboy_gradient}"_privatey_"${private_py_scaling}"_ax_"${alpha_x}"_ay_"${alpha_y}
#		echo "Setting up job" $JOB_NAME
#
#		TRAIN_DIR=${ROOT_LOG_DIR}/${EXP_PREFIX}/${JOB_NAME}
#
#		MODE_STR="_train_"
#		RUN_JOB_STR=${JOB_NAME}"_train_"
#		CMD_STRING="python experiments/vae_train.py --dataset celeba --dataset_dir ${DATASET_DIR} --loss_type ${loss_type} --image_likelihood Gaussian --label_likelihood Categorical  --train_log_dir ${TRAIN_DIR} --model_type ${MODEL_TYPE}  --num_latent ${num_latent}  --stop_elboy_gradient=${stop_elboy_gradient} --alpha_x=${alpha_x} --alpha_y=${alpha_y} --max_number_of_steps ${num_training_steps} --private_py_scaling ${private_py_scaling} --alsologtostderr"
#
#		sbatch utils/slurm_wrapper.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
#
#		#MODE_STR="_eval_"
#		#RUN_JOB_STR=${JOB_NAME}${MODE_STR}
#		#CMD_STRING="python experiments/vae_eval.py --dataset celeba --dataset_dir ${DATASET_DIR} --split_name val --num_result_datapoints 25 --image_likelihood Gaussian --label_likelihood Categorical --model_type ${MODEL_TYPE}  --num_latent ${num_latent}   --checkpoint_dir ${TRAIN_DIR} --eval_dir ${TRAIN_DIR}  --alsologtostderr"
#
#		#sbatch utils/slurm_wrapper_cpu_only.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
#	done
#done

########################### Jmvae models #####################################################
#loss_type=jmvae
#for num_latent in 18
#do
#	# for jmvae_alpha in 0.01 0.1 1.0
#	for jmvae_alpha in 0.1 1.0
#	do 
#		# for alpha_y in 1 50 100
#		for alpha_y in 50 100
#		do
#			JOB_NAME="celeba_loss_"${loss_type}"_poe_"${product_of_experts}"_nl_"${num_latent}"_jmalpha_"${jmvae_alpha}"_ax_"${alpha_x}"_ay_"${alpha_y}
#			echo "Setting up job" $JOB_NAME
#
#			TRAIN_DIR=${ROOT_LOG_DIR}/${EXP_PREFIX}/${JOB_NAME}
#

#			MODE_STR="_train_"
#			RUN_JOB_STR=${JOB_NAME}"_train_"
#			CMD_STRING="python experiments/vae_train.py --dataset celeba --dataset_dir ${DATASET_DIR} --loss_type ${loss_type} --image_likelihood Gaussian --label_likelihood Categorical --alpha_x=${alpha_x} --alpha_y=${alpha_y} --product_of_experts=${product_of_experts} --train_log_dir ${TRAIN_DIR} --model_type ${MODEL_TYPE} --num_latent ${num_latent} --jmvae_alpha=${jmvae_alpha} --max_number_of_steps ${num_training_steps} --alsologtostderr"
#
#			sbatch utils/slurm_wrapper.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
#
#			MODE_STR="_eval_"
#			RUN_JOB_STR=${JOB_NAME}${MODE_STR}
#			CMD_STRING="python experiments/vae_eval.py --dataset celeba --dataset_dir ${DATASET_DIR} --split_name val --num_result_datapoints 25 --image_likelihood Gaussian --label_likelihood Categorical --product_of_experts=${product_of_experts} --model_type ${MODEL_TYPE}  --num_latent ${num_latent}   --checkpoint_dir ${TRAIN_DIR} --eval_dir ${TRAIN_DIR}  --alsologtostderr"
#
#			sbatch utils/slurm_wrapper_cpu_only.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
#		done
#	done
#done

########################################### Triple elbo models with stopgradient x.################
stop_elbox_gradient=1
for num_latent in 18
do
	for stop_elboy_gradient in 1
	do 
		for private_py_scaling in 1 50 100
		do
			alpha_y=${private_py_scaling}
			JOB_NAME="celeba_loss_"${loss_type}"_poe_"${product_of_experts}"_nl_"${num_latent}"_sg_"${stop_elboy_gradient}"_xsg_"${stop_elbox_gradient}"_privatey_"${private_py_scaling}"_ax_"${alpha_x}"_ay_"${alpha_y}
			echo "Setting up job" $JOB_NAME

			TRAIN_DIR=${ROOT_LOG_DIR}/${EXP_PREFIX}/${JOB_NAME}

			if [ -e ${TRAIN_DIR} ]; then
				echo "Files exist at ${TRAIN_DIR}. Continue and delete them? [y/n]"
				read yn
				if [ ${yn} == "y" ]; then
					rm -r ${TRAIN_DIR}
				fi
			else
				mkdir ${TRAIN_DIR}
			fi

			MODE_STR="_train_"
			RUN_JOB_STR=${JOB_NAME}"_train_"
			CMD_STRING="python experiments/vae_train.py\
				--dataset celeba\
       --dataset_dir ${DATASET_DIR}\
				--loss_type ${loss_type}\
				--image_likelihood Gaussian\
				--label_likelihood Categorical\
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
				--alsologtostderr"

			sbatch utils/slurm_wrapper.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"

			MODE_STR="_eval_"
			RUN_JOB_STR=${JOB_NAME}${MODE_STR}
			CMD_STRING="python experiments/vae_eval.py\
				--dataset celeba\
       --dataset_dir ${DATASET_DIR}\
				--split_name val\
				--num_result_datapoints 25\
				--image_likelihood Gaussian\
				--label_likelihood Categorical\
				--product_of_experts=${product_of_experts}\
				--model_type ${MODEL_TYPE}\
				--num_latent ${num_latent}\
			  --checkpoint_dir ${TRAIN_DIR}\
				--eval_dir ${TRAIN_DIR}\
				--alsologtostderr"

			sbatch utils/slurm_wrapper_cpu_only.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
		done
	done
done

######################### KL(q, p) stopgradient x, y, private_py_scaling = 1 ##########################
stop_elbox_gradient=1
private_py_scaling=1
product_of_experts=1
for num_latent in 18
do
	for stop_elboy_gradient in 1
	do 
		for alpha_y in 50 100
		do
			JOB_NAME="celeba_loss_"${loss_type}"_poe_"${product_of_experts}"_nl_"${num_latent}"_sg_"${stop_elboy_gradient}"_xsg_"${stop_elbox_gradient}"_privatey_"${private_py_scaling}"_ax_"${alpha_x}"_ay_"${alpha_y}
			echo "Setting up job" $JOB_NAME

			TRAIN_DIR=${ROOT_LOG_DIR}/${EXP_PREFIX}/${JOB_NAME}

			if [ -e ${TRAIN_DIR} ]; then
				echo "Files exist at ${TRAIN_DIR}. Continue and delete them? [y/n]"
				read yn
				if [ ${yn} == "y" ]; then
					rm -r ${TRAIN_DIR}
				fi
			else
				mkdir ${TRAIN_DIR}
			fi

			MODE_STR="_train_"
			RUN_JOB_STR=${JOB_NAME}"_train_"
			CMD_STRING="python experiments/vae_train.py\
				--dataset celeba\
       --dataset_dir ${DATASET_DIR}\
				--loss_type ${loss_type}\
				--image_likelihood Gaussian\
				--label_likelihood Categorical\
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
				--alsologtostderr"

			sbatch utils/slurm_wrapper.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"

			MODE_STR="_eval_"
			RUN_JOB_STR=${JOB_NAME}${MODE_STR}
			CMD_STRING="python experiments/vae_eval.py\
				--dataset celeba\
       --dataset_dir ${DATASET_DIR}\
				--split_name val\
				--num_result_datapoints 25\
				--image_likelihood Gaussian\
				--label_likelihood Categorical\
				--product_of_experts=${product_of_experts}\
				--model_type ${MODEL_TYPE}\
				--num_latent ${num_latent}\
			  --checkpoint_dir ${TRAIN_DIR}\
				--eval_dir ${TRAIN_DIR}\
				--alsologtostderr"

			sbatch utils/slurm_wrapper_cpu_only.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}"
		done
	done
done


########################################### BiVCCA Models.################
loss_type="bivcca"
for num_latent in 18
do
	for product_of_experts in 1
	do
		for alpha_y in 50
		do
			for bivcca_mu in 0.3 0.7
			do
				JOB_NAME="celeba_loss_"${loss_type}"_poe_"${product_of_experts}"_nl_"${num_latent}"_bmu_"${bivcca_mu}"_ax_"${alpha_x}"_ay_"${alpha_y}
				echo "Setting up job" $JOB_NAME

				TRAIN_DIR=${ROOT_LOG_DIR}/${EXP_PREFIX}/${JOB_NAME}

				MODE_STR="_train_"
				RUN_JOB_STR=${JOB_NAME}"_train_"
				CMD_STRING="python experiments/vae_train.py\
					--dataset celeba\
          --dataset_dir ${DATASET_DIR}\
					--loss_type ${loss_type}\
					--image_likelihood Gaussian\
					--label_likelihood Categorical\
					--alpha_x=${alpha_x}\
					--alpha_y=${alpha_y}\
					--product_of_experts=${product_of_experts}\
					--train_log_dir ${TRAIN_DIR}\
					--model_type ${MODEL_TYPE}\
					--num_latent ${num_latent}\
					--bivcca_mu=${bivcca_mu}\
					--max_number_of_steps ${num_training_steps}\
					--alsologtostderr"

			  sbatch utils/slurm_wrapper.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}" ""

				MODE_STR="_eval_"
				RUN_JOB_STR=${JOB_NAME}${MODE_STR}
				CMD_STRING="python experiments/vae_eval.py\
		 			--dataset celeba\
          --dataset_dir ${DATASET_DIR}\
					--split_name val\
					--num_result_datapoints 25\
					--image_likelihood Gaussian\
					--label_likelihood Categorical\
					--product_of_experts=${product_of_experts}\
					--model_type ${MODEL_TYPE}\
					--num_latent ${num_latent}\
					--checkpoint_dir ${TRAIN_DIR}\
					--eval_dir ${TRAIN_DIR}\
					--alsologtostderr"

			 sbatch utils/slurm_wrapper_cpu_only.sh "${CMD_STRING}" "${RUN_JOB_STR}" "${MODE_STR}" "${TRAIN_DIR}" ""
			done
		done
	done
done
