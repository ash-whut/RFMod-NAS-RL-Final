import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score


model = tf.keras.models.load_model("/home/ashwin/repos/RFMod-NAS-RL-Final/candidate_models_retrained/model_572.h5")

# print(model.summary())

data = np.load("/home/ashwin/repos/RFMod-NAS-RL-Final/mat_gen data/X_test.npy")
labels = np.load("/home/ashwin/repos/RFMod-NAS-RL-Final/mat_gen data/Y_test.npy")

# data = np.load("/home/ashwin/repos/RFMod-NAS-RL-Final/mat_gen data/X_test_best.npy")
# labels = np.load("/home/ashwin/repos/RFMod-NAS-RL-Final/mat_gen data/Y_test_best.npy")

classes = ["BPSK", "QPSK", "8PSK",
           "16QAM", "32QAM", "64QAM",
           "128QAM", "256QAM", "16APSK",
           "32APSK", "64APSK", "128APSK",
           "FM", "AM-DSB-SC", "AM-SSB-SC"]

print(classification_report(labels, np.argmax(model.predict(data), axis=1), target_names = classes))
# test_y_hat = np.argmax(model.predict(data), axis = 1)
# print(model.evaluate(data, labels))
# print(accuracy_score(labels, test_y_hat))

# print(np.shape(data))
# print(np.shape(labels))