from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import os

class Model(ABC):

    def __init__(self):
        self.model = self.init_model()
        self.callback = self.get_callback()

    @staticmethod
    @abstractmethod
    def init_model():
        pass

    def get_callback(self):
        return tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    def compile(self, optimizer: str, loss: str, metrics: list[str]):
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

    def train(self, batch_size: int, epochs: int, train_data: tuple, validation_data: tuple, weights: bool):
        path = f"./models/{self.__class__.__name__.lower()}"
        if weights:
            self.model.load_weights(path + "/weights/weights.hdf5")
        else:
            self.model.fit(train_data[0], train_data[1], batch_size=batch_size, epochs=epochs,
                           validation_data=validation_data, callbacks=[self.callback])
            self.save_weights(path)

    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.model.save_weights(path + "/weights/weights.hdf5")

    def predict(self, sample_data):
        return self.model.predict(sample_data)

    def evaluate(self, eval_data: tuple, title=""):
        results = self.model.evaluate(eval_data[0], eval_data[1])
        print(title, "loss, acc:", results, sep=' ')
