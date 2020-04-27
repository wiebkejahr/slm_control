#!/bin/bash
# /*
#  * Created on Wednesday, 6th November 2019 9:47:12
#  * @author: hmcgovern
#  */

# -e
# -o

# you can change this if you like, make sure to redirect the path
# in main.py
OUTPUT_DIR='.'

# creates dataset, model, and run dirs in your specified output dir
mkdir -p $OUTPUT_DIR/datasets
mkdir -p $OUTPUT_DIR/models
mkdir -p $OUTPUT_DIR/runs

DATA_DIR=$OUTPUT_DIR/datasets
MODEL_DIR=$OUTPUT_DIR/models
LOG_DIR=$OUTPUT_DIR/runs

####################### 1. MAKE DATASET #############################
NUM_POINTS=2 # will do 90/10 train/validation split
TEST_NUM=2 # number of additional test samples to create
NAME="4.27.20_3d_offset_sted_TEST2" # make this as descriptive as possible
# don't touch this
DATASET="${DATA_DIR}/${NAME}.hdf5"

# To see all options, run 'python create_train_data.py --help'. Output copied below.
#
# DATASET PARAMETERS
# positional arguments:
#   num_points            number of data points for the dataset, will be split 90/10 training/validation
#   data_dir              path to where you want the dataset stored
#   test_num              number of points to use in test set

# optional arguments:
#   -h, --help            show this help message and exit
#   -r, --resolution      resolution of training example psf image. Default is 64 (x 64)
#   --multi               (FLAG) whether or not to use cross-sections
#   --offset              (FLAG) whether or not to incorporate offset
#   --mode {fluor,sted,z-sted} which mode of data to create

if [ ! -f ${DATASET} ]; then
python3 create_train_data.py ${NUM_POINTS} ${TEST_NUM} ${DATASET} -r 64 --multi --offset --mode 'sted'
else
echo "Dataset already exists"
fi

######################### 2. TRAIN ##################################
# HYPERPARAMETERS 
LR=0.001 # learning rate
NUM_EPOCHS=15
BATCH_SIZE=64 
MODEL_NAME="${NAME}_eps_${NUM_EPOCHS}_lr_${LR}_bs_${BATCH_SIZE}"

# don't touch these
MODEL_STORE_PATH="${MODEL_DIR}/${MODEL_NAME}.pth"
LOGDIR=${LOG_DIR}/${MODEL_NAME}

# To see all options, run 'python train.py --help'. Output copied below.
#
# TRAINING PARAMETERS
# positional arguments:
#   lr                    learning rate
#   num_epochs            number of epochs to run
#   batch_size            batch size
#   dataset               path to dataset on which to train
#   model_store_path      path to where you want to save model checkpoints

# optional arguments:
#   -h, --help            show this help message and exit
#   --logdir              path to logging dir for optional tensorboard visualization
#   --warm_start          path to a previous checkpoint dir to continue training from a previous run

if [ ! -f ${MODEL_STORE_PATH} ]; then
python3 train.py ${LR} ${NUM_EPOCHS} ${BATCH_SIZE} ${DATASET} ${MODEL_STORE_PATH} --multi --offset --logdir ${LOGDIR}
else
echo "Model already exists"
fi
####################### 3. EVALUATE ################################
# To see all options, run 'python evaluate.py --help'. Output copied below.
#
# EVALUATION
# positional arguments:
#   test_dataset_dir  path to dataset
#   model_store_path  path to model checkpoint dir

# optional arguments:
#   -h, --help        show this help message and exit
#   --logdir          path to logging dir for optional tensorboard visualization

python3 evaluate.py ${DATASET} ${MODEL_STORE_PATH} --multi --offset --logdir ${LOGDIR}

# ./utils/tensorboard.sh