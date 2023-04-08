import argparse
from models.custom.custom_model import CustomModel

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--model', type=str,
                            help='Model type')
        self.parser.add_argument('--dataset', type=str,
                                 help='Dataset type')
        self.parser.add_argument('--attack', type=str,
                            help='Attack type')
        self.parser.add_argument('--val_split', type=float,
                            help='Validation split', default=0.1)
        self.parser.add_argument('--weights', type=bool,
                                 help='Is there weights for this model', default=False)
        self.__args = self.get_args()

    def get_args(self):
        return self.parser.parse_args()

    def get_model(self):
        match self.__args.model:
            case "custom":
                mnist = CustomModel()
                return mnist

    def get_attack(self):
        return self.__args.attack

    def get_val_split(self):
        return self.__args.val_split

    def get_dataset_name(self):
        return self.__args.dataset

    def get_weights(self):
        return self.__args.weights
