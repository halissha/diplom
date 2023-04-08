import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from error_control.errors import check_arg_absence
from output.saver import save_attack_results
matplotlib.use('Agg')

@check_arg_absence
def print_attack_results(model, data, attack, dataset):
    y1 = model.predict(data.X_test)
    y2 = model.predict(data.X_adv)
    z0 = np.argmax(data.y_test, axis=1)
    z1 = np.argmax(y1, axis=1)
    z2 = np.argmax(y2, axis=1)

    print('\nPlotting results')
    fig = plt.figure(figsize=(10, 2.2))
    gs = gridspec.GridSpec(2, 10, wspace=0.05, hspace=0.05)

    for i in range(10):
        ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
        ind = np.random.choice(ind)
        xcur = [data.X_test[ind], data.X_adv[ind]]
        ycur = y2[ind]
        zcur = z2[ind]
        for j in range(2):
            img = np.squeeze(xcur[j])
            ax = fig.add_subplot(gs[j, i])
            ax.imshow(img, cmap='gray', interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel('{0} ({1:.2f})'.format(zcur, ycur[zcur]), fontsize=12)
        gs.tight_layout(fig)
    save_attack_results(model=model.__class__.__name__, attack=attack, dataset=dataset)
