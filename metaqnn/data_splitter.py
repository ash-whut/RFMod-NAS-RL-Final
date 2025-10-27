import argparse
import sys
import os
import h5py as h5
import numpy as np
from sklearn.model_selection import train_test_split
from helpers.data_loader import *

def data_splitter(dataset_path, split_data_path, matlab_gen_data):
    DATASET_PATH = dataset_path

    TRAIN_DATA, TEST_DATA, TRAIN_LABELS, TEST_LABELS, TRAIN_SNRS, TEST_SNRS =  load_data(DATASET_PATH, matlab_gen_data)
    VAL_DATA, TEST_DATA, VAL_LABELS, TEST_LABELS, VAL_SNRS, TEST_SNRS = val_data_split(TEST_DATA, TEST_LABELS, TEST_SNRS)
    BEST_SNR_TEST_DATA, BEST_SNR_TEST_LABELS, BEST_SNR_TEST_SNRS  = best_snr_data(TEST_DATA, TEST_LABELS, TEST_SNRS)

    if not os.path.exists(split_data_path):
        os.makedirs(split_data_path)

    np.save(os.path.join(split_data_path, "X_train"), TRAIN_DATA)
    np.save(os.path.join(split_data_path, "X_test"), TEST_DATA)
    np.save(os.path.join(split_data_path, "X_val"), VAL_DATA)
    np.save(os.path.join(split_data_path, "X_test_best"), BEST_SNR_TEST_DATA)

    np.save(os.path.join(split_data_path, "Y_train"), TRAIN_LABELS)
    np.save(os.path.join(split_data_path, "Y_test"), TEST_LABELS)
    np.save(os.path.join(split_data_path, "Y_val"), VAL_LABELS)
    np.save(os.path.join(split_data_path, "Y_test_best"), BEST_SNR_TEST_LABELS)

    np.save(os.path.join(split_data_path, "Z_train"), TRAIN_SNRS)
    np.save(os.path.join(split_data_path, "Z_test"), TEST_SNRS)
    np.save(os.path.join(split_data_path, "Z_val"), VAL_SNRS)
    np.save(os.path.join(split_data_path, "Z_test_best"), BEST_SNR_TEST_SNRS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'dataset_path',
        help='Path to the dataset'
    )

    parser.add_argument(
        'split_data_path',
        help='Output path to split dataset'
    )

    parser.add_argument(
        'matlab_gen_data',
        help='Matlab generated data or not (default = False)'
    )

    args = parser.parse_args()

    data_splitter(args.dataset_path, args.split_data_path, args.matlab_gen_data)