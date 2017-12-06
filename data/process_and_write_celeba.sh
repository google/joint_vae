# Download and process the celeba dataset into TFRecords for consumption by tensorflow.

export CUDA_VISIBLE_DEVICES=""

# Raw CELEBA dataset.
CELEBA_DATASET_FOLDER="/srv/share/datasets/celeba_for_tf"
RAW_CELEBA_FOLDER="/srv/share/datasets/celeba"

# Process Train split.
#python celeba/download_and_process_celeba.py \
#	--dataset_dir ${RAW_CELEBA_FOLDER}\
#	--output_directory ${CELEBA_DATASET_FOLDER}\
#	--attribute_label_map ${CELEBA_DATASET_FOLDER}/"attribute_label_map.json"\
#	--split_name "train"\
#	--shards 20\
#	--num_threads 5
#
## Process Val split.
#python celeba/download_and_process_celeba.py\
#	--dataset_dir ${RAW_CELEBA_FOLDER}\
#	--output_directory ${CELEBA_DATASET_FOLDER}\
#	--attribute_label_map ${CELEBA_DATASET_FOLDER}/"attribute_label_map.json"\
#	--split_name "val"\
#	--shards 5\
#	--num_threads 1
#
## Process test split.
#python celeba/download_and_process_celeba.py\
#	--dataset_dir ${RAW_CELEBA_FOLDER}\
#	--output_directory ${CELEBA_DATASET_FOLDER}\
#	--attribute_label_map ${CELEBA_DATASET_FOLDER}/"attribute_label_map.json"\
#	--split_name "test"\
#	--shards 5\
#	--num_threads 1

# Subset of CELEBA dataset.
CELEBA_DATASET_FOLDER="/srv/share/datasets/celeba_for_tf_ig"
RAW_CELEBA_FOLDER="/srv/share/datasets/celeba"

# Process Train split.
python celeba/download_and_process_celeba.py \
	--dataset_dir ${RAW_CELEBA_FOLDER}\
	--output_directory ${CELEBA_DATASET_FOLDER}\
	--attribute_label_map ${CELEBA_DATASET_FOLDER}/"attribute_label_map.json"\
	--attribute_subset_list "celeba/attribute_subset_icgan.txt"\
	--split_name "train"\
	--shards 20\
	--num_threads 5

# Process Val split.
python celeba/download_and_process_celeba.py\
	--dataset_dir ${RAW_CELEBA_FOLDER}\
	--output_directory ${CELEBA_DATASET_FOLDER}\
	--attribute_label_map ${CELEBA_DATASET_FOLDER}/"attribute_label_map.json"\
	--attribute_subset_list "celeba/attribute_subset_icgan.txt"\
	--split_name "val"\
	--shards 5\
	--num_threads 1

# Process test split.
python celeba/download_and_process_celeba.py\
	--dataset_dir ${RAW_CELEBA_FOLDER}\
	--output_directory ${CELEBA_DATASET_FOLDER}\
	--attribute_label_map ${CELEBA_DATASET_FOLDER}/"attribute_label_map.json"\
	--attribute_subset_list "celeba/attribute_subset_icgan.txt"\
	--split_name "test"\
	--shards 5\
	--num_threads 1
