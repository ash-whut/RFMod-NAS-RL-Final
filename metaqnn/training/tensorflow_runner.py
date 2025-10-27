from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List
import numpy as np
from os import path
from tqdm import tqdm

from grammar.state_enumerator import State
from training.one_cycle_lr import OneCycleLR

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn import preprocessing
from sklearn.metrics import classification_report

import psutil
import shutil
import os

class TBLogMover(Callback):
    def __init__(self, log_dir, log_dir_dest, iteration, batch_size):
        self.log_dir = log_dir
        self.log_dir_dest = log_dir_dest
        self.iteration = iteration
        self.batch_size = batch_size
        os.makedirs(log_dir_dest, exist_ok=True)
        
    def on_train_end(self, logs=None):
        if (self.iteration % self.batch_size) == 0 and self.iteration != 0:
            print(os.listdir(self.log_dir))
            print("Iteration value is: ", self.iteration)
            batch_path = os.path.join(self.log_dir_dest, str(self.iteration))
            os.mkdir(batch_path)
            print(f"Batch path {self.iteration} made")

            start_point = max(0, self.iteration - self.batch_size)
            
            for iteration_step in range(start_point, self.iteration):
                data_point_path = os.path.join(self.log_dir, str(iteration_step))
                try:
                    shutil.move(data_point_path, batch_path)
                except FileNotFoundError:
                    print("Did not find file (or path): ", data_point_path)
                    break

class TensorFlowRunner(object):
    def __init__(self, state_space_parameters, hyper_parameters):
        self.ssp = state_space_parameters
        self.hp = hyper_parameters
        self.features = self.hp.TRAIN_DATA
        self.labels = self.hp.TRAIN_LABELS
        self.test_features = self.hp.TEST_DATA
        self.test_labels = self.hp.TEST_LABELS
        self.validation_features = self.hp.VAL_DATA
        self.validation_labels = self.hp.VAL_LABELS
        self.best_snr_val_features = self.hp.BEST_SNR_VAL_DATA
        self.best_snr_val_labels = self.hp.BEST_SNR_VAL_LABELS

    @staticmethod
    def compile_model(state_list: List[State], loss, metric_list, lr):
        _optimizer = Adam(learning_rate = lr)
        if len(state_list) < 1:
            raise Exception("Illegal neural net")  # TODO create clearer/better exception (class)

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
        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss / 1024**3:.2f} GB")
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
                    minimum_momentum=None, verbose=True),
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    verbose=1),
                TensorBoard(
                    log_dir=path.join('learner_logs/models', str(model_iteration)),
                    histogram_freq=1,
                    profile_batch=0),
                TBLogMover('learner_logs/models', 
                           'learner_logs/batches', 
                           model_iteration, 
                           100)
                ]
        ) 

        return (
            model.predict(self.test_features),
            model.evaluate(x=self.validation_features, y=self.validation_labels, batch_size=self.hp.EVAL_BATCH_SIZE),
            model.evaluate(x=self.best_snr_val_features, y=self.best_snr_val_labels, batch_size=self.hp.EVAL_BATCH_SIZE)
        )