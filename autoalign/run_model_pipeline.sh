#!/bin/bash
# /*
#  * Project: deep-adaptive-optics
#  * Created on: Wednesday, 6th November 2019 9:47:12 am
#  * --------
#  * @author: hmcgovern
#  */

# -e
# -o

######################### USER PARAMETERS ################################
CREATE_DATA=false # do you want to create a new dataset?
TRAIN=true # do you want to train a new model?
WARM_START=false # do you want to continue training from a previous run?
MULTI=true # do you want to train with 3 cross-sections?
TENSORBOARD=false # do you want to monitor the training in tensorboard?
EVAL=false # do you want to test a trained model?

NUM_POINTS=1000 # will do 90/10 train/validation split
TEST_NUM=10 # number of additional test samples to create
RESOLUTION=64
 NAME="04.02.20_xsection"
# NAME="27.01.20_xsection" # the name of your dataset, will be given a .hdf5 extension, make it descriptive
# OUTPUT_DIR='/Users/hope/SeaDrive'
OUTPUT_DIR="/c/Users/hmcgover/Seafile/My Library" # ideally a remote location

# HYPERPARAMETERS 
LEARNING_RATE=0.001 # there is a range that works well for each optimizer
NUM_EPOCHS=20 # how many times it will go through the entire dataset
BATCH_SIZE=64 # after how many examples it will update the model weights
OPTIM="Adam"

###################### DO NOT MODIFY BELOW THIS LINE ######################

DATASET_NAME="${NAME}.hdf5"
# TEST_NUM=2000 # from Zhang's paper

# TODO: add flag to create subdirectories Datasets, Models, and Runs if they don't exist
DATASET_DIR="${OUTPUT_DIR}/Datasets/${DATASET_NAME}"

# creates a dataset file if CREATE_DATA = True
if [ "$CREATE_DATA" = true ]; then
    python create_train_data.py ${NUM_POINTS} "${DATASET_DIR}" ${RESOLUTION} ${TEST_NUM} --multi $MULTI
fi


MODEL_NAME="${NAME}_${NUM_EPOCHS}_epochs_${OPTIM}_lr_${LEARNING_RATE}_batchsize_${BATCH_SIZE}"
MODEL_STORE_PATH="${OUTPUT_DIR}/Models/${MODEL_NAME}.pth"
# Currently hardcoded
WARM_START_PATH="${OUTPUT_DIR}/Models/04.01.20_multi_test_30_epochs_Adam_lr_0.001_batchsize_64.pth"
LOGDIR="${OUTPUT_DIR}/Runs/${MODEL_NAME}"

# starts the tensorboard
if [ "${TENSORBOARD}" = true ]; then
    tensorboard --logdir="${OUTPUT_DIR}/Runs/"
fi

# runs the model
if [ "$TRAIN" = true ]; then
    echo "Training ${MODEL_NAME}"
    python train.py ${NUM_EPOCHS} ${BATCH_SIZE} ${LEARNING_RATE} "${DATASET_DIR}" "${LOGDIR}" \
    "${MODEL_STORE_PATH}" --multi $MULTI --warm_start_path "${WARM_START_PATH}"
fi

# evaluates the model
if [ "$EVAL" = true ]; then
    echo "Testing ${MODEL_NAME}"
    python evaluate.py "${DATASET_DIR}" "${LOGDIR}" "${MODEL_STORE_PATH}"
fi

#eof
