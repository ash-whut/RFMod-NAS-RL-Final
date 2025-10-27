import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime
from os import path
import importlib
import tensorflow as tf

import pandas as pd

from grammar import q_learner
from training.tensorflow_runner import TensorFlowRunner

import psutil
import gc

class TermColors(object):
    HEADER = '\033[95m'
    YELLOW = '\033[93m'
    OKBLUE = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class QCoordinator(object):
    def __init__(self,
                 list_path,
                 state_space_parameters,
                 hyper_parameters,
                 epsilon=None,
                 number_models=None,
                 reward_small=False):

        print("\n\nRun started at: {}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))

        self.replay_columns = [
            'net',  # Net String
            'accuracy',  # Accuracy of the network
            'best_accuracy',  # Best accuracy of the network
            'trainable_parameters',  # Amount of trainable params of the network
            'ix_q_value_update',  # Iteration for q value update
            'epsilon',  # For epsilon greedy
            'time_finished'  # UNIX time
        ]

        self.list_path = list_path

        self.replay_dictionary_path = os.path.join(list_path, 'replay_database.csv')
        self.replay_dictionary, self.q_training_step = self.load_replay()

        self.schedule_or_single = False if epsilon else True
        if self.schedule_or_single:
            self.epsilon = state_space_parameters.epsilon_schedule[0][0]
            self.number_models = state_space_parameters.epsilon_schedule[0][1]
        else:
            self.epsilon = epsilon
            self.number_models = number_models if number_models else 10000000000
        self.state_space_parameters = state_space_parameters
        self.hyper_parameters = hyper_parameters

        self.number_q_updates_per_train = 100
        self.reward_small = reward_small

        self.list_path = list_path
        self.qlearner = self.load_qlearner()
        self.tf_runner = TensorFlowRunner(self.state_space_parameters, self.hyper_parameters)

        self.best_accuracies_and_iterations = []
        
        while not self.check_reached_limit():
            self.train_new_net()

        print('{}{}Experiment Complete{}'.format(TermColors.BOLD, TermColors.OKGREEN, TermColors.RESET))
    
    def train_new_net(self):
        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss / 1024**3:.2f} GB")
        net, net_to_run, iteration = self.generate_new_network()     
        print('{}Training net:\n{}\nIteration {:d}, Epsilon {:f}: [Network {:d}/{:d}]{}'.format(
            TermColors.OKBLUE, net_to_run, iteration, self.epsilon, self.number_trained_unique(self.epsilon),
            self.number_models, TermColors.RESET
        ))

        predictions, (test_loss, test_accuracy), (best_case_loss, best_case_accuracy), trainable_params, model = self._train_and_predict(
            self.tf_runner,
            net,
            self.hyper_parameters.MODEL_NAME,
            iteration,
            self.hyper_parameters.MAX_LR
        )

        # if not self.best_accuracies_and_iterations or test_accuracy > self.best_accuracies_and_iterations[-1][1]:
        #     self.remove_least_accurate_model_if_needed()
        #     self.save_best_performing_model(model, iteration, test_accuracy)

        self.incorporate_trained_net(
            net_to_run, 
            float(test_accuracy),
            float(best_case_accuracy),
            trainable_params, 
            float(self.epsilon), 
            [iteration]
        )
            
        TensorFlowRunner.clear_session()
        gc.collect()

        
    @staticmethod
    def _train_and_predict(tf_runner, net, model_name, iteration, lr):
        model = tf_runner.compile_model(net, loss='categorical_crossentropy', metric_list=['accuracy'], lr=lr)
        model.summary()
        trainable_params = tf_runner.count_trainable_params(model)

        predictions, (test_loss, test_accuracy), (best_case_loss, best_case_accuracy) = tf_runner.train_and_predict(model, iteration)

        return predictions, (test_loss, test_accuracy), (best_case_loss, best_case_accuracy), trainable_params, model

    def load_replay(self):
        if os.path.isfile(self.replay_dictionary_path):
            print('Found replay dictionary')
            replay_dic = pd.read_csv(self.replay_dictionary_path)
            q_training_step = max(replay_dic.ix_q_value_update)
        else:
            replay_dic = pd.DataFrame(columns=self.replay_columns)
            q_training_step = 0
        return replay_dic, q_training_step

    def load_qlearner(self):
        # Load previous q_values
        if os.path.isfile(os.path.join(self.list_path, 'q_values.csv')):
            print('Found q values')
            qstore = q_learner.QValues()
            qstore.load_q_values(os.path.join(self.list_path, 'q_values.csv'))
        else:
            qstore = None

        ql = q_learner.QLearner(self.hyper_parameters,
                                self.state_space_parameters,
                                self.epsilon,
                                qstore=qstore,
                                replay_dictionary=self.replay_dictionary,
                                reward_small=self.reward_small)

        return ql

    @staticmethod
    def filter_replay_for_first_run(replay):
        """ Order replay by iteration, then remove duplicate nets keeping the first"""
        temp = replay.sort_values(['ix_q_value_update']).reset_index(drop=True).copy()
        return temp.drop_duplicates(['net'])

    def number_trained_unique(self, epsilon=None):
        """Epsilon defaults to the minimum"""
        replay_unique = self.filter_replay_for_first_run(self.replay_dictionary)
        eps = epsilon if epsilon else min(replay_unique.epsilon.values)
        replay_unique = replay_unique[replay_unique.epsilon == eps]
        return len(replay_unique)

    def check_reached_limit(self):
        """ Returns True if the experiment is complete"""
        if len(self.replay_dictionary):
            completed_current = self.number_trained_unique(self.epsilon) >= self.number_models

            if completed_current:
                if self.schedule_or_single:
                    # Loop through epsilon schedule, If we find an epsilon that isn't trained, start using that.
                    completed_experiment = True
                    for epsilon, num_models in self.state_space_parameters.epsilon_schedule:
                        if self.number_trained_unique(epsilon) < num_models:
                            self.epsilon = epsilon
                            self.number_models = num_models
                            self.qlearner = self.load_qlearner()
                            completed_experiment = False

                            break

                else:
                    completed_experiment = True

                return completed_experiment

            else:
                return False

    def generate_new_network(self):
        try:
            (net_string, net, accuracy, best_accuracy, trainable_params) = self.qlearner.generate_net()

            # We have already trained this net
            if net_string in self.replay_dictionary.net.values:
                self.q_training_step += 1
                self.incorporate_trained_net(
                    net_string,
                    accuracy,
                    best_accuracy,
                    trainable_params,
                    self.epsilon,
                    [self.q_training_step]
                )
                return self.generate_new_network()
            else:
                self.q_training_step += 1
                return net, net_string, self.q_training_step

        except Exception:
            print(traceback.print_exc())
            sys.exit(1)

    def incorporate_trained_net(self, net_string, accuracy, best_accuracy,
                                trainable_params, epsilon, iterations):

        try:
            # If we sampled the same net many times, we should add them each into the replay database
            for train_iter in iterations:
                self.replay_dictionary = pd.concat([
                    self.replay_dictionary,
                    pd.DataFrame({
                        'net': [net_string],
                        'accuracy': [accuracy],
                        'best_accuracy': [best_accuracy],
                        'trainable_parameters': [trainable_params],
                        'ix_q_value_update': [train_iter],
                        'epsilon': [epsilon],
                        'time_finished': [time.time()]
                    })
                ])
                self.replay_dictionary.to_csv(self.replay_dictionary_path, index=False, columns=self.replay_columns)

            self.qlearner.update_replay_database(self.replay_dictionary)
            for train_iter in iterations:
                self.qlearner.sample_replay_for_update(train_iter)
            self.qlearner.save_q(self.list_path)

            print('{}Incorporated net, acc: {:f}, best_acc: {:f}, net(trainable_params={}):\n{}{}'.format(
                TermColors.YELLOW, accuracy, best_accuracy, trainable_params, net_string, TermColors.RESET
            ))
        except Exception:
            print(traceback.print_exc())
            
    def remove_least_accurate_model_if_needed(self):
        os.remove(path.normpath(f"{self.tf_runner.hp.TRAINED_MODEL_DIR}/{self.hyper_parameters.MODEL_NAME}_{self.best_accuracies_and_iterations[0][0]:04}.keras")) and \
            self.best_accuracies_and_iterations.pop(0) if (len(self.best_accuracies_and_iterations) == 5) else None

    def save_best_performing_model(self, model, iteration, accuracy):
            self.best_accuracies_and_iterations.append((accuracy, iteration))
            model.save(path.normpath(f"{self.tf_runner.hp.TRAINED_MODEL_DIR}/{self.hyper_parameters.MODEL_NAME}_{iteration:04}.keras"))

def main():
    parser = argparse.ArgumentParser()

    model_pkgpath = '/home/ashwin/repos/RL-SCA/metaqnn/models'
    model_choices = next(os.walk(model_pkgpath))[1]

    parser.add_argument(
        'model',
        help='Model package name. Package should have a hyper_parameters.py and a state_space_parameters.py file.',
        choices=model_choices
    )

    parser.add_argument(
        '--reward-small',
        help='Reward having a network with little trainable parameters',
        action='store_true'
    )
    parser.add_argument('-eps', '--epsilon', help='For Epsilon Greedy Strategy', type=float)
    parser.add_argument('-nmt', '--number_models_to_train', type=int,
                        help='How many models for this epsilon do you want to train.')

    args = parser.parse_args()

    _model = importlib.import_module("models." + args.model)

    

    factory = QCoordinator(
        "learner_logs",
        _model.state_space_parameters,
        _model.hyper_parameters,
        args.epsilon,
        args.number_models_to_train,
        args.reward_small
    )


if __name__ == '__main__':
    main()
