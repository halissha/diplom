import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from input_output.tech.errors import check_arg_absence
import seaborn as sns
import tensorflow as tf
from PIL import Image
import time
matplotlib.use('Agg')

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
mnist_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def make_showcase_view(model, data, dataset):
    y1 = model.predict(data.X_test)
    y1 = tf.nn.softmax(y1)
    y2 = model.predict(data.X_adv)
    y2 = tf.nn.softmax(y2)
    z0 = np.argmax(data.y_test, axis=1)
    z1 = np.argmax(y1, axis=1)
    z2 = np.argmax(y2, axis=1)
    fig = plt.figure(figsize=(14, 2.2))
    gs = gridspec.GridSpec(2, 10, wspace=0.9, hspace=0.05)

    for i in range(10):
        ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
        ind = np.random.choice(ind)
        xcur = [data.X_test[ind], data.X_adv[ind]]
        ycur = y2[ind]
        zcur = z2[ind]
        x_train = np.expand_dims(data.X_train, axis=-1)
        x_train = tf.image.resize(data.X_train, [32, 32])
        for j in range(2):
            img = np.squeeze(xcur[j])
            ax = fig.add_subplot(gs[j, i])
            ax.imshow(img, cmap='gray', interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])
        z_cur = 0
        match dataset:
            case "mnist":
                z_cur = mnist_classes[zcur]
            case "cifar10":
                z_cur = cifar10_classes[zcur]
        ax.set_xlabel('{0} ({1:.2f})'.format(z_cur, ycur[zcur]), fontsize=12)
        gs.tight_layout(fig)


def make_confusion_matrix(model, dataset, data):
    y_pred = model.predict(data.X_test)
    y_pred = tf.nn.softmax(y_pred)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(data.y_test, axis=1)
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(12, 9))
    c = sns.heatmap(confusion_mtx, annot=True, fmt='g')
    match dataset:
        case "mnist":
            c.set(xticklabels=cifar10_classes, yticklabels=cifar10_classes)
        case "cifar10":
            c.set(xticklabels=cifar10_classes, yticklabels=cifar10_classes)
    figure = c.get_figure()
    return figure
