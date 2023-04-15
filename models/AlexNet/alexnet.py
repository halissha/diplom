import numpy as np
import matplotlib
import os

# Matplotlib backend for saving plots to files. Only compatible with UNIX systems ('Agg')
matplotlib.use('Agg')
import tensorflow as tf
from attacks.deepfool import deepfool
from attacks.fast_gradient import fgm
from models.base_model.base_model import Model

class AlexNet(Model):

    def __init__(self):
        super().__init__()
        self.model = self.init_model()

    @staticmethod
    def init_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Resizing(
                224, 224, interpolation="bicubic", crop_to_aspect_ratio=False),
            tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                                   input_shape=(224, 224, 3)),
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
            tf.keras.layers.Dense(10, activation='linear')
        ])
        return model

