# Script to download MNIST, and create MNIST-A dataset with attributes.

# Dataset Generation does not need GPU computations.
export CUDA_VISIBLE_DEVICES=""

PATH_TO_MNIST="/tmp/mnist_valid"
PATH_TO_MNISTA="/tmp/mnist_with_attributes/"

if [ ! -e ${PATH_TO_MNIST} ]; then
	mkdir ${PATH_TO_MNIST}
fi

if [ ! -e ${PATH_TO_MNISTA} ]; then
	mkdir ${PATH_TO_MNISTA}
fi

python datasets/mnist/download_and_convert_mnist.py\
	--dataset_dir ${PATH_TO_MNIST}\
	--create_validation

# Create IID as well as Compositional splits of mnist with attributes (MNIST-A).

for SPLIT_TYPE in "iid" "comp"
do
	echo "Generating dataset for split "${SPLIT_TYPE}
	python datasets/mnist_attributes/create_dataset.py \
		--output_train_tfexample "${PATH_TO_MNISTA}/${SPLIT_TYPE}_train"\
		--output_val_tfexample "${PATH_TO_MNISTA}/${SPLIT_TYPE}_val"\
		--output_test_tfexample "${PATH_TO_MNISTA}/${SPLIT_TYPE}_test"\
		--path_to_original_mnist ${PATH_TO_MNIST}\
		--label_split_json "${PATH_TO_MNISTA}/${SPLIT_TYPE}_label_split.json"\
		--label_map_json "${PATH_TO_MNISTA}/${SPLIT_TYPE}_label_map.json"\
		--image_split_json "${PATH_TO_MNISTA}/${SPLIT_TYPE}_num_images_split.json"\
		--replication 10\
		--binarize \
		--split_type ${SPLIT_TYPE}
done
