import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import gc

import models.RML_2021.hyper_parameters as hp
import models.RML_2021.state_space_parameters as ssp
import grammar.cnn
from grammar.state_string_utils import StateStringUtils
from training.tensorflow_runner import TensorFlowRunner

import time

candidate_models_idx = [2435, 562]
model_strings = []
total_training_times = []

replay_database = pd.read_csv("./learner_logs_old/replay_database.csv")

for candidate_model_idx in candidate_models_idx:
    model_string = replay_database.loc[replay_database['ix_q_value_update'] == candidate_model_idx]['net'].iloc[0]
    model_strings.append(model_string)

def model_generator(net_strings):
    models = []
    for string in net_strings:
        stt_str_utls = StateStringUtils(ssp)
        cnn_parsed = grammar.cnn.parse("net", string)
        states = stt_str_utls.convert_model_string_to_states(cnn_parsed)
            
        tfr = TensorFlowRunner(ssp, hp)
        model = tfr.compile_model(states, loss='categorical_crossentropy', metric_list=['accuracy'], lr=1e-4)
        models.append(model)
    return models

def model_trainer(models):
    trained_models = []
    for idx, model in enumerate(models):
            print(model.summary())
            start_time = time.time()
            model.fit(
                x=hp.TRAIN_DATA,
                y=hp.TRAIN_LABELS,
                batch_size=1024,
                epochs=100,
                validation_data=(hp.VAL_DATA, hp.VAL_LABELS),
                verbose=1,
                callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath = f'./model_{candidate_models_idx[idx]}.h5', 
                                                 verbose = 1,
                                                 save_best_only=True, 
                                                 save_weights_only=False,
                                                 mode='auto')]
            ) 
            end_time = time.time()
            total_time = end_time - start_time
            trained_models.append(model)
            total_training_times.append(total_time)
            K.clear_session()
            gc.collect()
            
    return trained_models

models = model_generator(model_strings)
trained_models = model_trainer(models)

with open("training_times.txt", "w") as file:
    file.write(str(total_training_times))

# for idx, trained_model in trained_models:
#     trained_model.save(f"model_{idx}.h5")
#     print(total_training_times[idx])