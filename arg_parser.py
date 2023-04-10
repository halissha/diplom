import argparse
from models.custommodel.custom_model import CustomModel
from models.alexnet.alexnet import AlexNet

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
        self.parser.add_argument('--epochs', type=int,
                                 help='Number of epochs to train model', default=5)
        self.parser.add_argument('--batches', type=int,
                                 help='Number of epochs to train model', default=32)
        self.__args = self.get_args()

    def get_args(self):
        return self.parser.parse_args()

    def get_model(self):
        match self.__args.model:
            case "custom":
                model = CustomModel()
                return model
            case "alexnet":
                model = AlexNet()
                return model

    def get_attack(self):
        return self.__args.attack

    def get_val_split(self):
        return self.__args.val_split

    def get_dataset_name(self):
        return self.__args.dataset

    def get_weights(self):
        return self.__args.weights

    def get_epochs(self):
        return self.__args.epochs

    def get_batches(self):
        return self.__args.batches