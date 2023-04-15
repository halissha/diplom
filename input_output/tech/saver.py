import os
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from input_output.tech.printer import make_showcase_view, make_confusion_matrix
from input_output.tech.errors import check_arg_absence
import glob

@check_arg_absence
def save_attack_results(model, data, attack, dataset, save):
    if save:
        save_perturbed_pics(X_adv=data.X_adv, model=model, dataset=dataset, attack=attack, count=1000)
    make_showcase_view(model=model.model, data=data, dataset=dataset)
    save_attack_showcase(model=model.__class__.__name__, attack=attack, dataset=dataset)
    figure = make_confusion_matrix(model=model, data=data, dataset=dataset)
    save_confusion_matrix(model=model.__class__.__name__, attack=attack, dataset=dataset, figure=figure)

def save_perturbed_pics(X_adv, model, dataset, attack, count=1000):
    path = f'./input_output/results/{model.__class__.__name__}/{dataset}/{attack}/pics'
    delete_perturbed_pics(path=path)
    for i in range(count - 1):
        time.sleep(0.005)
        adv = X_adv[i: (i + 1)]
        adv = np.squeeze(adv)
        adv = (adv * 255).round().astype(np.uint8)
        match dataset:
            case "mnist":
                img = Image.fromarray(adv, mode='L')
            case "cifar10":
                img = Image.fromarray(adv, mode='RGB')
        img.save(f'{path}/dummy_pic_{i + 1}.jpg')
    print("Successfully saved perturbed pics to the directory %s " % path)

def save_attack_showcase(model, attack, dataset):
    path = f'./input_output/results/{model}/{dataset}/{attack}'
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print("Creation of the directory %s failed" % path)
    plt.savefig(path + f"/{attack}_{dataset}.jpg")
    print("Successfully saved attack showcase to the directory %s " % path)

def save_confusion_matrix(model, attack, dataset, figure):
    path = f'./input_output/results/{model}/{dataset}/{attack}'
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print("Creation of the directory %s failed" % path)
    figure.savefig(path + f"/confusion_matrix.png")
    print("Successfully saved confusion matrix to the directory %s \n" % path)

def save_adversarial_data(X_adv, model, dataset, attack):
    path = f"./models/{model.__class__.__name__}/adversarial_data/{dataset}/{attack}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(f'{path}/data.npy', 'wb') as file:
        np.save(file, X_adv)

def load_adversarial_data(model, dataset, attack):
    path = f"./models/{model.__class__.__name__}/adversarial_data/{dataset}/{attack}"
    if not os.path.exists(path):
        raise Exception(f'Cannot load adversarial data from {path}. Such path does not exist')
    with open(f'{path}/data.npy', 'rb') as file:
        X_adv = np.load(file)
    return X_adv

def delete_perturbed_pics(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print("Creation of the directory %s failed" % path)
    if os.listdir(path):
        files = glob.glob(f'{path}/*')
        for f in files:
            os.remove(f)
        print("\nSuccessfully deleted perturbed pics from the directory %s " % path)