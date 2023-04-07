import numpy as np
import tensorflow as tf


class ModelData:

    def __init__(self, model, val_split):
        self.X_adv = None
        match model.__class__.__name__:
            case "Mnist":
                self.X_train, self.y_train, self.X_test, self.y_test, self.img_size, self.img_chan = get_mnist_data()
                self.prepare_data()
                self.X_valid, self.y_valid = self.split_data(val_split=val_split)

    def prepare_data(self):
        self.X_train = np.reshape(self.X_train, [-1, self.img_size, self.img_size, self.img_chan])
        self.X_train = self.X_train.astype(np.float32) / 255
        self.X_test = np.reshape(self.X_test, [-1, self.img_size, self.img_size, self.img_chan])
        self.X_test = self.X_test.astype(np.float32) / 255
        to_categorical = tf.keras.utils.to_categorical
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    def split_data(self, val_split):
        if 0.0 >= val_split >= 1:
            raise ValueError("Validation split coefficient must be between 0 and 1")
        idx = np.random.permutation(self.X_train.shape[0])
        Xtrain, ytrain = self.X_train[idx], self.y_train[idx]
        n = int(Xtrain.shape[0] * (1 - val_split))
        X_valid = self.X_train[n:]
        y_valid = self.y_train[n:]
        self.X_train = self.X_train[:n]
        self.y_train = self.y_train[:n]
        return X_valid, y_valid



def get_mnist_data():
    (Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.mnist.load_data()
    img_size, img_chan = 28, 1
    return Xtrain, ytrain, Xtest, ytest, img_size, img_chan

def prepare_data(Xtrain, ytrain, Xtest, ytest, img_size, img_chan):
    Xtrain = np.reshape(Xtrain, [-1, img_size, img_size, img_chan])
    Xtrain = Xtrain.astype(np.float32) / 255
    Xtest = np.reshape(Xtest, [-1, img_size, img_size, img_chan])
    Xtest = Xtest.astype(np.float32) / 255
    to_categorical = tf.keras.utils.to_categorical
    ytrain = to_categorical(ytrain)
    ytest = to_categorical(ytest)
    return Xtrain, ytrain, Xtest, ytest
