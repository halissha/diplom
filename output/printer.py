import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from error_control.errors import check_arg_absence
from output.saver import save_attack_results
import seaborn as sns
import tensorflow as tf
matplotlib.use('Agg')

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
mnist_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

@check_arg_absence
def print_attack_results(model, data, attack, dataset):
    y1 = model.predict(data.X_test)
    y2 = model.predict(data.X_adv)
    z0 = np.argmax(data.y_test, axis=1)
    z1 = np.argmax(y1, axis=1)
    z2 = np.argmax(y2, axis=1)

    print('\nPlotting results')
    fig = plt.figure(figsize=(10, 2.2))
    gs = gridspec.GridSpec(2, 10, wspace=0.05, hspace=0.05)

    for i in range(10):
        ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
        ind = np.random.choice(ind)
        xcur = [data.X_test[ind], data.X_adv[ind]]
        ycur = y2[ind]
        zcur = z2[ind]
        x_train = np.expand_dims(x_train, axis=-1)
        x_train = tf.image.resize(x_train, [32, 32])
        for j in range(2):
            img = np.squeeze(xcur[j])
            ax = fig.add_subplot(gs[j, i])
            ax.imshow(img, cmap='gray', interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])
        match dataset:
            case "mnist":
                zcur = mnist_classes[zcur]
            case "cifar10":
                zcur = cifar10_classes[zcur]
        ax.set_xlabel('{0} ({1:.2f})'.format(zcur, ycur[zcur]), fontsize=12)
        gs.tight_layout(fig)
    save_attack_results(model=model.__class__.__name__, attack=attack, dataset=dataset)

def print_confusion_matrix(model, data):
    # Predict the values from the validation dataset
    y_pred = model.predict(data.X_test)
    # Convert predictions classes to one hot vectors
    y_pred_classes = np.argmax(y_pred, axis=1)
    # Convert validation observations to one hot vectors
    y_true = np.argmax(data.y_test, axis=1)
    # compute the confusion matrix
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(12, 9))
    c = sns.heatmap(confusion_mtx, annot=True, fmt='g')
    c.set(xticklabels=classes, yticklabels=classes)
