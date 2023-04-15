from abc import ABC, abstractmethod
import tensorflow as tf
from attacks.deepfool import deepfool
from attacks.fast_gradient import fgm
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

    def train(self, epochs: int, batch_size, train_data: tuple, validation_data, weights: bool, dataset: str):
        path = f"./models/{self.__class__.__name__.lower()}/weights/{dataset}/"
        if weights:
            self.model.load_weights(path)
            print(f"Loaded pre-computed weights for {self.__class__.__name__}")
        else:
            print(f"Fitting model {self.__class__.__name__}")
            self.model.fit(train_data[0], train_data[1], epochs=epochs,
                           validation_data=validation_data, batch_size=batch_size, validation_freq=1)
            self.save_weights(path)

    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.model.save_weights(path)

    def predict(self, data):
        return self.model.predict(data, verbose=0)

    def evaluate(self, data, labels, title=""):
        print(f"\nEvaluating {self.__class__.__name__} on {title} data")
        results = self.model.evaluate(data, labels)

    def make_deepfool(self, X_data, epochs=1, batch_size=128):
        n_sample = X_data.shape[0]
        n_batch = int((n_sample + batch_size - 1) / batch_size)
        Xadv = np.empty_like(X_data)
        for batch in range(batch_size):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            adv = deepfool(model=self, x=X_data[start:end], epochs=epochs)
            Xadv[start:end] = adv
        return Xadv

    def make_fgsm(self, X_data, epochs=10, eps=0.01, batch_size=32):
        n_sample = X_data.shape[0]
        n_batch = int((n_sample + batch_size - 1) / batch_size)
        Xadv = np.empty_like(X_data)
        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            adv = fgm(model=self, x=X_data[start:end], eps=eps, epochs=epochs)
            Xadv[start:end] = adv
        return Xadv

    def model_call(self, data, logits=False):
        logit = self.model(data)
        predicted = tf.nn.softmax(logit)
        if logits:
            return predicted, logit
        return predicted
