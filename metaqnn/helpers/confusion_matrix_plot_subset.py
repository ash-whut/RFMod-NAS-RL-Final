import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
    plt.savefig(f"{figname}-subset.pdf", format='pdf', dpi=1200, bbox_inches='tight')
    plt.savefig(f"{figname}-subset.eps", format='eps', bbox_inches='tight')

def plot_confusion_matrix(output_file_name: str, input_model_file_name: str, data_path: str, least_noise = False) -> None:
    model = tf.keras.models.load_model(input_model_file_name)
    
    classes = ["OOK","4ASK","8ASK",
        "BPSK","QPSK","8PSK","16PSK","32PSK",
        "16APSK","32APSK","64APSK","128APSK",
        "16QAM","32QAM","64QAM","128QAM","256QAM",
        "AM-SSB-WC","AM-SSB-SC","AM-DSB-WC","AM-DSB-SC","FM",
        "GMSK","OQPSK","BFSK","4FSK","8FSK"]
    # index 0, 1, 2, 3, 4, 8, 9, 20, 21, 22, 23, 24, 25, 26

    data = np.load(f'{data_path}/X_test.npy') if not least_noise else np.load(f'{data_path}/X_test_best.npy')
    labels = np.load(f'{data_path}/Y_test.npy') if not least_noise else np.load(f'{data_path}/Y_test_best.npy')

    subset_mod_class_idx = [0, 1, 2, 3, 4, 8, 9, 20, 21, 22, 23, 24, 25, 26]
    index_mapping = dict()

    index_mapping = {idx:index for index, idx in enumerate(subset_mod_class_idx)}

    print(index_mapping)

    indices = np.where(np.isin(np.argmax(labels, axis=1), subset_mod_class_idx))[0]

    data, labels = data[indices], labels[indices]

    classes = [classes[idx] for idx in subset_mod_class_idx]

    # classes[np.argmax(labels[index])]

    test_Y_hat = model.predict(data, batch_size=2048)
    conf = np.zeros([len(classes), len(classes) + 1])
    confnorm = np.zeros([len(classes), len(classes) + 1])
    counter = 0
    for i in range(0,data.shape[0]):
        j = index_mapping[list(labels[i,:]).index(1)]

        try:
            k = index_mapping[int(np.argmax(test_Y_hat[i,:]))]
        except KeyError:
            k = len(classes)
            counter+=1

        conf[j, k] += 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

    print("Counter: ", counter)
    plotting_helper(confnorm, output_file_name, labels=classes + ["Other"])