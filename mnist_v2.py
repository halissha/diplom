import os
import numpy as np
import matplotlib
import data

# Matplotlib backend for saving plots to files. Only compatible with UNIX systems ('Agg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from deepfool import deepfool

IMG_SIZE = 28
IMG_CHAN = 1
N_CLASSES = 10

class Mnist:

    def __init__(self, val_split):
        self.X_adv = None
        self.model = self.init_model()

    @staticmethod
    def init_model():
        model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                                    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
                                    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                                    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dropout(0.25),
                                    tf.keras.layers.Dense(10, activation='softmax')])
        return model

    @staticmethod
    def init_data(val_split):
        Xtrain, ytrain, Xtest, ytest = data.get_mnist_data()
        Xtrain, ytrain, Xtest, ytest = data.prepare_data(Xtrain, ytrain, Xtest, ytest, IMG_SIZE, IMG_CHAN)
        Xtrain, ytrain, Xvalid, yvalid = data.split_data(Xtrain, ytrain, val_split)
        return Xtrain, ytrain, Xtest, ytest, Xvalid, yvalid

    def compile(self, optimizer: str, loss: str, metrics: list[str]):
        self.model.compile(optimizer=optimizer,
             loss=loss,
             metrics=metrics)

    def train(self, batch_size: int, epochs: int, train_data: tuple, validation_data: tuple):
        self.model.fit(train_data[0], train_data[1], batch_size=batch_size, epochs=epochs,
                       validation_data=validation_data)

    def predict(self, sample_data):
        return self.model.predict(sample_data)




    @staticmethod
    def mnist_deepfool(self, X_data, epochs=1, eps=0.01, batch_size=128):
        print('\nMaking adversarials via DeepFool')
        n_sample = X_data.shape[0]
        n_batch = int((n_sample + batch_size - 1) / batch_size)
        Xadv = np.empty_like(X_data)

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            adv = deepfool(self.model, X_data[start:end], epochs=epochs)
            Xadv[start:end] = adv
        return Xadv

    def evaluate(self, eval_data: tuple, title=""):
        results = self.model.evaluate(eval_data[0], eval_data[1])
        print(title, "loss, acc:", results, sep=' ')

# print('\nEvaluating on clean data')
#
# results = model.evaluate(X_test, y_test)
# print("CLEAN: test loss, test acc:", results)
#
# print('\nGenerating adversarial data')
#
# X_adv = mnist_deepfool(X_test, epochs=3)
#
# print('\nEvaluating on adversarial data')
#
# results = model.evaluate(X_adv, y_test)
#
# print("ADVERSARIAL: test loss, test acc:", results)
# print('\nRandomly sample adversarial data from each category')
#
# y1 = model.predict(X_test)
# y2 = model.predict(X_adv)
#
# z0 = np.argmax(y_test, axis=1)
# z1 = np.argmax(y1, axis=1)
# z2 = np.argmax(y2, axis=1)
# print(z0, z1, z2)
#
# print('\nPlotting results')
# fig = plt.figure(figsize=(10, 2.2))
# gs = gridspec.GridSpec(2, 10, wspace=0.05, hspace=0.05)
#
# for i in range(10):
#     print('Target {0}'.format(i))
#     ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
#     ind = np.random.choice(ind)
#     xcur = [X_test[ind], X_adv[ind]]
#     ycur = y2[ind]
#     zcur = z2[ind]
#
#     for j in range(2):
#         img = np.squeeze(xcur[j])
#         ax = fig.add_subplot(gs[j, i])
#         ax.imshow(img, cmap='gray', interpolation='none')
#         ax.set_xticks([])
#         ax.set_yticks([])
#     ax.set_xlabel('{0} ({1:.2f})'.format(zcur, ycur[zcur]), fontsize=12)
#
# print('\nSaving figure')
# gs.tight_layout(fig)
# plt.savefig('img/deepfool_mnist.png')