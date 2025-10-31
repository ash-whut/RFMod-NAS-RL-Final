import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

import argparse

def plotting_helper(cm, figname, cmap=plt.cm.Blues, labels=[]):
    plt.rcParams['font.size'] = '15'
    plt.figure(figsize = (15,12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    label_len = np.shape(labels)[0]
    tick_marks = np.arange(label_len)
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()    
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{figname}.pdf", format='pdf', dpi=1200, bbox_inches='tight')
    plt.savefig(f"{figname}.eps", format='eps', bbox_inches='tight')

def plot_confusion_matrix(output_file_name: str, input_model_file_name: str, data_path: str, least_noise = False, matGenData = False) -> None:
    model = tf.keras.models.load_model(input_model_file_name)

    if matGenData:
        classes = ["BPSK", "QPSK", "8PSK",
                   "16QAM", "32QAM", "64QAM",
                   "128QAM", "256QAM", "16APSK",
                   "32APSK", "64APSK", "128APSK",
                   "FM", "AM-DSB-SC", "AM-SSB-SC"]
    else:
        classes = ["OOK","4ASK","8ASK",
            "BPSK","QPSK","8PSK","16PSK","32PSK",
            "16APSK","32APSK","64APSK","128APSK",
            "16QAM","32QAM","64QAM","128QAM","256QAM",
            "AM-SSB-WC","AM-SSB-SC","AM-DSB-WC","AM-DSB-SC","FM",
            "GMSK","OQPSK","BFSK","4FSK","8FSK"]

    data = np.load(f'{data_path}/X_test.npy') if not least_noise else np.load(f'{data_path}/X_test_best.npy')

    if matGenData:
        labels = to_categorical(np.load(f'{data_path}/Y_test.npy')) if not least_noise else to_categorical(np.load(f'{data_path}/Y_test_best.npy'))        
    else:
        labels = np.load(f'{data_path}/Y_test.npy') if not least_noise else np.load(f'{data_path}/Y_test_best.npy')

    test_Y_hat = model.predict(data, batch_size=2048)
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0,data.shape[0]):
        j = list(labels[i,:]).index(1)
        k = int(np.argmax(test_Y_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

    plotting_helper(confnorm, output_file_name, labels=classes)