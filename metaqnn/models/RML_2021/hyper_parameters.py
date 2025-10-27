from data_loader import *
import numpy as np

MODEL_NAME = 'RML_2021'
DATASET_PATH = '/home/ashwin/datasets/RADIOML_2021_07_INT8/RADIOML_2021_07_INT8.hdf5'

# Number of output neurons
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
MIN_ACCURACY = 0.5

TRAIN_DATA, TEST_DATA, TRAIN_LABELS, TEST_LABELS, TRAIN_SNRS, TEST_SNRS =  load_data(DATASET_PATH)
VAL_DATA, TEST_DATA, VAL_LABELS, TEST_LABELS, VAL_SNRS, TEST_SNRS = val_data_split(TEST_DATA, TEST_LABELS, TEST_SNRS)
BEST_SNR_VAL_DATA, BEST_SNR_VAL_LABELS = best_snr_data(VAL_DATA, VAL_LABELS, VAL_SNRS)

TRAINED_MODEL_DIR = 'trained_models/'