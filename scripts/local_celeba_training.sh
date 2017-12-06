# Script to train an imagination model on Celebrities with Attributes (CELEBA) dataset.

# First source the preamble file.
source preamble.sh

DATASET="celeba"
DATASET_DIR=${GLOBAL_DATA_PATH}"/celeba_for_tf_ig/"

#python experiments/vae_train.py \
#	--dataset_dir ${DATASET_DIR}\
#	--model_type multi\
#	--batch_size 64\
#	--dataset celeba\
#	--loss_type multimodal_elbo\
#	--num_latent 40\
#	--image_likelihood Gaussian\
#	--label_likelihood Categorical\
#	--stop_elboy_gradient True\
#	--alsologtostderr

EXP_PREFIX=celeba
MODEL_TYPE=multi
alpha_x=1
alpha_y=50
num_training_steps=100000
num_latent=20
stop_elboy_gradient=1
stop_elbox_gradient=1
private_py_scaling=50
melbo_x2=50
TRAIN_DIR=/tmp/test_celeba

python experiments/vae_train.py\
	--dataset ${DATASET}\
	--dataset_dir ${DATASET_DIR}\
	--loss_type bivcca\
	--product_of_experts=0 \
	--image_likelihood Gaussian \
	--label_likelihood Categorical \
	--melbo_x1=1.0 \
	--train_log_dir ${TRAIN_DIR}\
	--model_type ${MODEL_TYPE}  \
	--num_latent ${num_latent}\
	--stop_elbox_gradient=${stop_elbox_gradient}\
	--stop_elboy_gradient=${stop_elboy_gradient}\
	--melbo_c=${melbo_x2}\
	--melbo_x2=${melbo_x2}\
	--max_number_of_steps ${num_training_steps}\
	--private_py_scaling ${private_py_scaling}\
	--alsologtostderr

#python experiments/vae_eval.py \
#	--dataset celeba\
#	--dataset_dir ${DATASET_DIR}\
#	--num_latent 40\
#	--image_likelihood Gaussian\
#	--label_likelihood Categorical\
#	--model_type multi\
#	--checkpoint_dir ~/runs/unimodal_vae \
#	--eval_dir ~/runs/unimodal_vae \
#	--split_name val \
#	--alsologtostderr
