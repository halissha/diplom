from deepfool import deepfool
import argparse
from mnist_v2 import Mnist

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--model', type=str,
                            help='Model type')
        self.parser.add_argument('--attack', type=str,
                            help='Attack type')
        self.parser.add_argument('--version', type=str,
                            help='Tensorflow version')
        self.parser.add_argument('--val_split', type=float,
                            help='Validation split', default=0.1)
        self.args = self.get_args()

    def get_args(self):
        return self.parser.parse_args()

    def get_model(self):
        match self.args.model:
            case "mnist":
                mnist = Mnist(val_split=self.get_val_split())
                return mnist
    def get_attack(self):
        match self.args.attack:
            case "deepfool":
                return

    def get_val_split(self):
        return self.args.val_split
