import numpy as np
import matplotlib
from models.base_model.base_model import Model
import os

# Matplotlib backend for saving plots to files. Only compatible with UNIX systems ('Agg')
matplotlib.use('Agg')
import tensorflow as tf
from attacks.deepfool import deepfool
from attacks.fast_gradient import fgm

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
                                    tf.keras.layers.Dense(10, activation='linear')])
        return model
