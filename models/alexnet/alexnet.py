import numpy as np
import matplotlib
import os

# Matplotlib backend for saving plots to files. Only compatible with UNIX systems ('Agg')
matplotlib.use('Agg')
import tensorflow as tf
from attacks.deepfool import deepfool
from models.base_model.base_model import Model

class AlexNet(Model):

    def __init__(self):
        super().__init__()
        self.model = self.init_model()
        self.callback = self.get_callback()

    @staticmethod
    def init_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Resizing(
                227, 227, interpolation="bicubic", crop_to_aspect_ratio=False),
            tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                                input_shape=(227, 227, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        return model

    def compile(self, optimizer: str, loss: str, metrics: list[str]):
        self.model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.001),
                           loss=loss,
                           metrics=metrics)

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