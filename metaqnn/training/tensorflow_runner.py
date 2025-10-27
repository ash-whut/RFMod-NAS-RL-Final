from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List
import numpy as np
from os import path
from tqdm import tqdm

from grammar.state_enumerator import State
from training.one_cycle_lr import OneCycleLR

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
# for MatGenData
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.metrics import classification_report

import os

## TODO improve for matgendata
class TensorFlowRunner(object):
    def __init__(self, state_space_parameters, hyper_parameters):
        self.data_path = '/home/ashwin/repos/RFMod-NAS-RL-Final/metaqnn/data/'
        self.ssp = state_space_parameters
        self.hp = hyper_parameters
        self.features = np.load(f'{self.data_path}X_train.npy')
        # for MatGenData
        # self.labels = np.load(f'{self.data_path}Y_train.npy')
        self.labels = to_categorical(np.load(f'{self.data_path}Y_train.npy'))
        self.test_features = np.load(f'{self.data_path}X_test.npy')
        # for MatGenData
        # self.test_labels = np.load(f'{self.data_path}Y_test.npy')
        self.test_labels = to_categorical(np.load(f'{self.data_path}Y_test.npy'))
        self.validation_features = np.load(f'{self.data_path}X_val.npy')
        # for MatGenData
        # self.validation_labels = np.load(f'{self.data_path}Y_val.npy')
        self.validation_labels = to_categorical(np.load(f'{self.data_path}Y_val.npy'))
        self.best_snr_test_features = np.load(f'{self.data_path}X_test_best.npy')
        # for MatGenData
        # self.best_snr_test_labels = np.load(f'{self.data_path}Y_test_best.npy')
        self.best_snr_test_labels = to_categorical(np.load(f'{self.data_path}Y_test_best.npy'))

    @staticmethod
    def compile_model(state_list: List[State], loss, metric_list, lr):
        _optimizer = Adam(learning_rate = lr)
        if len(state_list) < 1:
            raise Exception("Illegal neural net")

        model = tf.keras.Sequential()
        for state in state_list:
            model.add(state.to_tensorflow_layer())
        model.compile(optimizer=_optimizer, 
                      loss=loss, 
                      metrics=metric_list)
        return model

    @staticmethod
    def clear_session():
        K.clear_session()

    @staticmethod
    def count_trainable_params(model):
        return np.sum([K.count_params(w) for w in model.trainable_weights])

    def train_and_predict(self, model, model_iteration, parallel_no=1):
        model.fit(
            x=self.features,
            y=self.labels,
            batch_size=self.hp.TRAIN_BATCH_SIZE * parallel_no,
            epochs=self.hp.MAX_EPOCHS,
            validation_data=(self.validation_features, 
                            self.validation_labels),
            verbose=1,
            callbacks=[
                OneCycleLR(
                    max_lr=self.hp.MAX_LR * parallel_no, end_percentage=0.2, scale_percentage=0.1,
                    maximum_momentum=None,
                    minimum_momentum=None, verbose=True)
                ]
        ) 

        return (
            model.predict(self.test_features),
            model.evaluate(x=self.validation_features, y=self.validation_labels, batch_size=self.hp.EVAL_BATCH_SIZE),
            model.evaluate(x=self.best_snr_test_features, y=self.best_snr_test_labels, batch_size=self.hp.EVAL_BATCH_SIZE)
        )