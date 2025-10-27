import argparse
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import gc
import json

from helpers.confusion_matrix_plot import plot_confusion_matrix as normal_plot
from helpers.confusion_matrix_plot_subset import plot_confusion_matrix as subset_plot
from helpers.models_ranker import model_ranker
from additional_analysis.score_distribution_plot import score_plot
import models.RML_2021.hyper_parameters as hp
import models.RML_2021.state_space_parameters as ssp
import grammar.cnn
from grammar.state_string_utils import StateStringUtils
from training.tensorflow_runner import TensorFlowRunner

import numpy as np
import time
import os

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

def model_trainer(models, candidate_models_idx, retrained_models_path, cm_plots_path, split_data_path):
    trained_models = []
    total_training_times = []
    for idx, model in enumerate(models):
            print(model.summary())
            start_time = time.time()
            model.fit(
                x=np.load(os.path.join(split_data_path, "X_train.npy")),
                y=np.load(os.path.join(split_data_path, "Y_train.npy")),
                batch_size=1024,
                epochs=1,
                validation_data=(np.load(os.path.join(split_data_path, "X_val.npy")), np.load(os.path.join(split_data_path, "Y_val.npy"))),
                verbose=1,
                callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(retrained_models_path, 
                                                                                      f"model_{candidate_models_idx[idx]}.h5"), 
                                                verbose = 1,
                                                save_best_only=True, 
                                                save_weights_only=False,
                                                mode='auto')]
            )

            end_time = time.time()
            total_time = end_time - start_time
            trained_models.append(model)
            total_training_times.append(total_time)
            
            normal_plot(os.path.join(cm_plots_path, f"cm-usable-model-{candidate_models_idx[idx]}"),
                        os.path.join(retrained_models_path, f"model_{candidate_models_idx[idx]}.h5"),
                        split_data_path,
                        False)
            normal_plot(os.path.join(cm_plots_path, f"cm-best-model-{candidate_models_idx[idx]}"),
                        os.path.join(retrained_models_path, f"model_{candidate_models_idx[idx]}.h5"),
                        split_data_path,
                        True)
            subset_plot(os.path.join(f"{cm_plots_path}-subset", f"cm-usable-model-{candidate_models_idx[idx]}"),
                        os.path.join(retrained_models_path, f"model_{candidate_models_idx[idx]}.h5"),
                        split_data_path,
                        False)
            subset_plot(os.path.join(f"{cm_plots_path}-subset", f"cm-best-model-{candidate_models_idx[idx]}"),
                        os.path.join(retrained_models_path, f"model_{candidate_models_idx[idx]}.h5"),
                        split_data_path,
                        True)
            K.clear_session()
            gc.collect()

    with open("training_times.txt", "w") as tt:
        tt.write(str(total_training_times))
            
    return trained_models

def main():  
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'retrained_models_path',
        help='Path where retrained models are stored'
    )
    parser.add_argument(
        'cm_plots_path',
        help='Model being tested'
    )
    parser.add_argument(
        'data_path',
        help='Path to pre-split dataset'
    )
    parser.add_argument(
        'learner_logs_path',
        help='Path to learner logs'
    )

    args = parser.parse_args()

    candidate_models_idx = model_ranker(os.path.join(args.learner_logs_path, 
                                                    "replay_database.csv"),
                                        10)
    json_dict = defaultdict(float)

    for idx, model_id in enumerate(candidate_models_idx):
        json_dict[int(candidate_models_idx[idx][0])] = candidate_models_idx[idx][1]

    with open("candidate_models_and_scores.json", "w") as model_score_file:
        json.dump(json_dict, model_score_file)

    candidate_models_idx = list(json_dict.keys())
    print(candidate_models_idx)

    model_strings = []
    total_training_times = []

    replay_database = pd.read_csv(os.path.join(args.learner_logs_path, "replay_database.csv"))

    for candidate_model_idx in candidate_models_idx:
        model_string = replay_database.loc[replay_database['ix_q_value_update'] == candidate_model_idx]['net'].iloc[0]
        model_strings.append(model_string)

    if not os.path.exists(args.retrained_models_path):
        os.makedirs(args.retrained_models_path)

    if not os.path.exists(args.cm_plots_path):
        os.makedirs(args.cm_plots_path)   

    if not os.path.exists(f"{args.cm_plots_path}-subset"):
        os.makedirs(f"{args.cm_plots_path}-subset")   

    models = model_generator(model_strings)
    trained_models = model_trainer(models, 
                                   candidate_models_idx, 
                                   args.retrained_models_path, 
                                   args.cm_plots_path,
                                   args.data_path)
    
    score_plot(args.learner_logs_path)

    ## TODO Make a thing for classification reports too

if __name__ == '__main__':
    main()