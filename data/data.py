import numpy as np
import tensorflow as tf
from input_output.tech.saver import load_adversarial_data, save_adversarial_data


class ModelData:
    def __init__(self, dataset, model, attack, adv_present):
        self.X_adv, self.X_valid, self.y_valid = None, None, None
        self.X_train, self.y_train, self.X_test, self.y_test, self.img_size, self.img_chan = self.get_dataset(dataset)
        if adv_present:
            print(f"Loaded pre-computed adversarial data for {model.__class__.__name__}")
            self.X_adv = load_adversarial_data(dataset=dataset, model=model, attack=attack)
        self.prepare_data()

    def prepare_data(self):
        self.X_train = np.reshape(self.X_train, [-1, self.img_size, self.img_size, self.img_chan])
        self.X_test = np.reshape(self.X_test, [-1, self.img_size, self.img_size, self.img_chan])
        self.X_train = self.X_train.astype(np.float32) / 255
        self.X_test = self.X_test.astype(np.float32) / 255
        self.y_train = tf.keras.utils.to_categorical(self.y_train)
        self.y_test = tf.keras.utils.to_categorical(self.y_test)

    def split_data(self, val_split):
        if 0.0 >= val_split >= 1:
            raise ValueError("Validation split coefficient must be between 0 and 1")
        idx = np.random.permutation(self.X_train.shape[0])
        self.X_train, self.y_train = self.X_train[idx], self.y_train[idx]
        n = int(self.X_train.shape[0] * (1 - val_split))
        self.X_valid = self.X_train[n:]
        self.y_valid = self.y_train[n:]
        self.X_train = self.X_train[:n]
        self.y_train = self.y_train[:n]

    def generate_adversarial_data(self, model, attack, dataset, adv_present: bool):
        if not adv_present:
            print(f"\nMaking {attack} attack...")
            match attack:
                case "deepfool":
                    self.X_adv = model.make_deepfool(X_data=self.X_test, epochs=5, batch_size=128)
                case "fgsm":
                    self.X_adv = model.make_fgsm(X_data=self.X_test, epochs=20)
            save_adversarial_data(X_adv=self.X_adv, dataset=dataset, model=model, attack=attack)
            print(f"Successfully made {attack} attack")


    @staticmethod
    def get_dataset(dataset):
        match dataset:
            case "mnist":
                (Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.mnist.load_data()
                img_size, img_chan = 28, 1
            case "cifar10":
                (Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.cifar10.load_data()
                img_size, img_chan = 32, 3
            case _:
                raise Exception(f"Trying to download unknown dataset '{dataset}'")
        return Xtrain, ytrain, Xtest, ytest, img_size, img_chan

