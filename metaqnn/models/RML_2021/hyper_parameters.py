from helpers.data_loader import *
import numpy as np

MODEL_NAME = 'RML_2021'
# DATASET_PATH = '/home/ashwin/datasets/custom-data/MatGenData.h5'
TRAINED_MODEL_DIR = 'trained_models'

# Number of output neurons
# for MatGenData
NUM_CLASSES = 27  # Number of output neurons

# Input Size
INPUT_SIZE = 1024

# Batch Queue parameters
TRAIN_BATCH_SIZE = 2048  # Batch size for training (scaled linearly with number of gpus used)
EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE  # Batch size for validation

TEST_INTERVAL_EPOCHS = 1  # Num epochs to test on, should really always be 1
MAX_EPOCHS = 15  # Max number of epochs to train model

# Training Parameters
OPTIMIZER = 'Adam'  # Optimizer (should be in caffe format string)
MAX_LR = 1e-4  # The max LR (scaled linearly with number of gpus used)

# Reward small parameter
# This rewards networks smaller than this number of trainable parameters
MAX_TRAINABLE_PARAMS_FOR_REWARD = 50000

# TRAIN_DATA, TEST_DATA, TRAIN_LABELS, TEST_LABELS, TRAIN_SNRS, TEST_SNRS =  load_data(DATASET_PATH)
# VAL_DATA, TEST_DATA, VAL_LABELS, TEST_LABELS, VAL_SNRS, TEST_SNRS = val_data_split(TEST_DATA, TEST_LABELS, TEST_SNRS)
# BEST_SNR_TEST_DATA, BEST_SNR_TEST_LABELS, BEST_SNR_TEST_SNRS  = best_snr_data(TEST_DATA, TEST_LABELS, TEST_SNRS)
