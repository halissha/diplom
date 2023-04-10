import os
import matplotlib.pyplot as plt

def save_attack_results(model, attack, dataset):
    path = f"./results/{dataset}/{model}/{attack}"
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print("Creation of the directory %s failed" % path)
    plt.savefig(path + f"/{attack}_{dataset}.jpg")
    print("Successfully saved results the directory %s " % path)