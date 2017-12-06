#! /bin/sh
#
# create_affine_mnist.sh
# Copyright (C) 2017 rvedantam3 <vrama91@vt.edu>
#
# Distributed under terms of the MIT license.
#

# Download and preprocess MNIST dataset.
PATH_TO_MNIST="/tmp/mnist_valid"
if [ ! -e ${PATH_TO_MNIST} ]; then
	mkdir ${PATH_TO_MNIST}
fi

python datasets/mnist/download_and_convert_mnist.py --dataset_dir ${PATH_TO_MNIST} --create_validation

# Create affine MNIST.
for SPLIT_TYPE in "iid"
do
	echo "Generating dataset for split "${SPLIT_TYPE}
	python datasets/mnist_attributes/create_dataset.py \
		--output_train_tfexample "${PATH_TO_AFFINE_MNIST}/${SPLIT_TYPE}_train@20"\
		--output_val_tfexample "${PATH_TO_AFFINE_MNIST}/${SPLIT_TYPE}_val@5"\
		--output_test_tfexample "${PATH_TO_AFFINE_MNIST}/${SPLIT_TYPE}_test@5"\
		--label_split_json "${PATH_TO_AFFINE_MNIST}/${SPLIT_TYPE}_label_split.json"\
		--label_map_json "${PATH_TO_AFFINE_MNIST}/${SPLIT_TYPE}_label_map.json"\
		--image_split_json "${PATH_TO_AFFINE_MNIST}/${SPLIT_TYPE}_num_images_split.json"\
		--replication 10\
		--binarize \
		--split_type ${SPLIT_TYPE} \
		--path_to_original_mnist ${PATH_TO_MNIST}
done
