import math
import pandas as pd
import numpy as np
import argparse
import os

from matplotlib import pyplot as plt
from collections import defaultdict

def tensorboard_smooth(scalars: list[float], weight: float) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed


def showplots(mask_prc25, mask_median, mask_prc75,
              no_mask_prc25, no_mask_median, no_mask_prc75,
              steps, save, performance=True):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.title("")
    plt.xlabel('Training Steps/100')
    plt.ylabel('Median Performance') if performance else plt.ylabel('Median Reward')

    plt.plot(steps, mask_prc25, alpha=0)
    plt.plot(steps, mask_median, color='blue', label='PPAM')
    plt.plot(steps, mask_prc75, alpha=0)
    plt.fill_between(steps, mask_median, mask_prc25, color='blue', alpha=0.25)
    plt.fill_between(steps, mask_median, mask_prc75, color='blue', alpha=0.25)

    plt.plot(steps, no_mask_prc25, alpha=0)
    plt.plot(steps, no_mask_median, color='red', label='Vanilla')
    plt.plot(steps, no_mask_prc75, alpha=0)
    plt.fill_between(steps, no_mask_median, no_mask_prc25, color='red', alpha=0.25)
    plt.fill_between(steps, no_mask_median, no_mask_prc75, color='red', alpha=0.25)

    plt.yticks(np.arange(-50, 105, 10.0)) if performance else plt.yticks(np.arange(-60, 55, 10.0))
    plt.legend()
    plt.grid()

    if save is not None:
        plt.savefig(save)
        print('File saved: ', save)

    plt.show()

def plotdata(datafolders = None, save = None, performance = True):
    columns = ["Step", "Value"]
    data, mask, no_mask, steps = dict(), defaultdict(lambda: []), defaultdict(lambda: []), []
    for folder in datafolders:
        folder_name = 'Vanilla' if 'no-mask' in folder else 'PPAM'
        data[folder_name] = []
        for file in os.listdir(folder):
            filename = os.fsdecode(file)
            if len(steps) == 0:
                steps = pd.read_csv(folder + '/' + filename, usecols=columns).Step
            if (performance and 'performance' not in filename) or\
                (not performance and 'reward' not in filename):
                continue
            data[folder_name] += [pd.read_csv(folder + '/' + filename, usecols=columns).Value]
            data[folder_name][-1] = tensorboard_smooth(data[folder_name][-1], 0.99)

            for step in range(len(steps)):
                if 'no-mask' in folder:
                    no_mask[step].append(data[folder_name][-1][step])
                else:
                    mask[step].append(data[folder_name][-1][step])

    mask_prc25, mask_median, mask_prc75 = [], [], []
    no_mask_prc25, no_mask_median, no_mask_prc75 = [], [], []

    for mask_data, no_mask_data in zip(mask.values(), no_mask.values()):
        mask_prc25.append(np.percentile(mask_data, 25))
        mask_median.append(np.median(mask_data))
        mask_prc75.append(np.percentile(mask_data, 75))

        no_mask_prc25.append(np.percentile(no_mask_data, 25))
        no_mask_median.append(np.median(no_mask_data))
        no_mask_prc75.append(np.percentile(no_mask_data, 75))

    showplots(mask_prc25, mask_median, mask_prc75, no_mask_prc25, no_mask_median, no_mask_prc75, steps, save, performance)


if __name__ == '__main__':
    '''
        Note that the tensorboard visualizer should be used to plot the data you've generated.  
        This file can be used to plot the data in the 'experimental_data' folder. 
    '''
    parser = argparse.ArgumentParser(description='Plot results')
    parser.add_argument('-save', type=str, help='save figure on specified file', default=None)
    parser.add_argument('--performance', action=argparse.BooleanOptionalAction, help='Used to set which data to plot. "True" plots the performance function, "False" plots the (observable) reward function (default="True")', default=True)

    parser.add_argument('-datafolders', nargs=2, help='Folders containing the files with the data to plot (must be precisely 2)')

    args = parser.parse_args()

    plotdata(args.datafolders, args.save, args.performance)
