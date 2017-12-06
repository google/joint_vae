# Download and process the celeba dataset into TFRecords for consumption by tensorflow.
#
# This script uses a subset of 18 visually distinctive attributes from the 40
# attributes present in the original CelebA dataset.
# The subset of 18 attributes is taken from:
#
# Perarnau, Guim, Joost van de Weijer, Bogdan Raducanu, and Jose M. Alvarez. 2016. 
# Invertible Conditional GANs for Image Editing.
# arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1611.06355.
#
# See datasets/celeba/attribute_subset_icgan.txt for the list of 18 attributes.

export CUDA_VISIBLE_DEVICES=""

# Stores TFRecords for Training.
CELEBA_DATASET_FOLDER="/srv/share/datasets/celeba_for_tf_icgan"

# Raw images from CELEBA.
RAW_CELEBA_FOLDER="/srv/share/datasets/celeba"

# Process Train split.
python datasets/celeba/download_and_process_celeba.py \
	--dataset_dir ${RAW_CELEBA_FOLDER}\
	--output_directory ${CELEBA_DATASET_FOLDER}\
	--attribute_label_map ${CELEBA_DATASET_FOLDER}/"attribute_label_map.json"\
	--attribute_subset_list "datasets/celeba/attribute_subset_icgan.txt"\
	--split_name "train"\
	--shards 20\
	--num_threads 5

# Process Val split.
python datasets/celeba/download_and_process_celeba.py\
	--dataset_dir ${RAW_CELEBA_FOLDER}\
	--output_directory ${CELEBA_DATASET_FOLDER}\
	--attribute_label_map ${CELEBA_DATASET_FOLDER}/"attribute_label_map.json"\
	--attribute_subset_list "datasets/celeba/attribute_subset_icgan.txt"\
	--split_name "val"\
	--shards 5\
	--num_threads 1

# Process test split.
python datasets/celeba/download_and_process_celeba.py\
	--dataset_dir ${RAW_CELEBA_FOLDER}\
	--output_directory ${CELEBA_DATASET_FOLDER}\
	--attribute_label_map ${CELEBA_DATASET_FOLDER}/"attribute_label_map.json"\
	--attribute_subset_list "datasets/celeba/attribute_subset_icgan.txt"\
	--split_name "test"\
	--shards 5\
	--num_threads 1
