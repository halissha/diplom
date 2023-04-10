import numpy as np
import matplotlib
from models.base_model.base_model import Model
import os

# Matplotlib backend for saving plots to files. Only compatible with UNIX systems ('Agg')
matplotlib.use('Agg')
import tensorflow as tf
from attacks.deepfool import deepfool

class CustomModel(Model):

    def __init__(self):
        super().__init__()
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

    def mnist_deepfool(self, X_data, epochs=1, batch_size=128):
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
