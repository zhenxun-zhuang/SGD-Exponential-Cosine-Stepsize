"""
Script for drawing comparison between different optimization methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os
import argparse

good_colors = [
    [0, 0, 1.0000],
    [1.0000, 0, 0],
    [0.1, 0.9500, 0.1],
    [0, 0, 0.1724],
    [1.0000, 0.1034, 0.7241],
    [1.0000, 0.8276, 0],
    [0.7241, 0.3103, 0.8276],
    [0.5172, 0.5172, 1.0000],
    [0.6207, 0.3103, 0.2759],
    [0, 1.0000, 0.7586],
    [0, 0.5172, 0.5862],
    [0, 0, 0.4828],
    [0.5862, 0.8276, 0.3103],
    [0.9655, 0.6207, 0.8621],
    [0.8276, 0.0690, 1.0000],
    [0.4828, 0.1034, 0.4138]
]

def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data, axis=0), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def readFile(folder, fileName, stat_names):
    """From folder read fileName, which contains fields in stat_names."""
    fullFileName = os.path.join(folder, fileName)
    with open(fullFileName, 'r') as f:
        results = {}
        counter = 0
        for line in f:
            if not line.startswith('['):
                continue
            stat_name = stat_names[counter]
            value_array = np.array(eval(line))
            results[stat_name] = np.vstack((results.get(stat_name,
                                                        np.zeros((0, len(value_array)))),
                                            value_array))
            counter = (counter + 1) % 4

    return results


def draw_comps(logs_folder, fig_type):
    linewidth = 5
    fontsize = 18

    stat_names = ['Training Loss', 'Training Accuracy', 'Test Loss', 'Test Accuracy']

    if fig_type == 'stagewise':    
        # Stagewise decay.
        labels = {'SGD':'SGD Constant Step Size',
                  'SGD_Exp_Decay':'Exponential Step Size',
                  'SGD_ReduceLROnPlateau':'ReduceLROnPlateau'}
    
        colors = {'SGD':0, 'SGD_ReduceLROnPlateau':6, 'SGD_Exp_Decay':3,
                  'SGD_Stage_Decay':1}
    
        draw_methods = ['SGD', 'SGD_ReduceLROnPlateau',
                        'SGD_Stage_Decay', 'SGD_Exp_Decay']

        legend_order = [1, 0, 3, 4, 2]

    elif fig_type == 'others':
        # Other stepsize schemes.
        labels = {'Adam':'Adam',
                  'SGD_1t_Decay':'O(1/t) Step Size',
                  'SGD_1sqrt_Decay':'O(1/sqrt(t)) Step Size',
                  'SGD_Exp_Decay':'Exponential Step Size',
                  'SGD_Cosine_Decay':'Cosine Decay',
                  'SLS-Armijo0':'SGD+Armijo',
                  'SLS-Armijo1':'SGD+Armijo',
                  'SLS-Armijo2':'SGD+Armijo'}

        colors = {'Adam':0, 'SGD_1sqrt_Decay':1, 'SGD_1t_Decay':2,
                  'SGD_Exp_Decay':3, 'SGD_Cosine_Decay':6, 'SLS-Armijo0':8,
                  'SLS-Armijo1':8, 'SLS-Armijo2':8}
    
        draw_methods = ['Adam', 'SGD_Exp_Decay', 'SGD_1sqrt_Decay', 'SGD_1t_Decay',
                        'SGD_Cosine_Decay', 'SLS-Armijo0', 'SLS-Armijo1', 'SLS-Armijo2']

        legend_order = [4, 3, 1, 2, 0, 5]

    fig_folder = './figs/'
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[15, 7.5])
    fig.subplots_adjust(left=0.08, bottom=0.26, right=0.98, top=0.97, wspace=0.22, hspace=None)

    files = os.listdir(logs_folder)
    for i, fileName in enumerate(files):
        method = fileName[fileName.find('_') + 1 : fileName.find('_Eta0')]
	
        if method not in draw_methods:
            continue

        stats = readFile(logs_folder, fileName, stat_names)
        dataset = fileName[:fileName.find('_')]

        if method == 'SGD_Stage_Decay':
            miles = fileName[fileName.find('Milestones_') + 11 : fileName.find('_Epoch')]
            num_miles = len(miles.split('_'))
            label = ('Stagewise Step Decay - %d Milestone%s' % (num_miles, 's' if num_miles > 1 else ''))
            color = good_colors[colors[method] + (num_miles - 1)]
        else:
            label = labels[method] if method in labels else method
            color = good_colors[colors[method]] if method in colors else good_colors[-(i+1)]

        if len(stats['Training Loss']) > 1:
            train_mean, train_h = mean_confidence_interval(stats['Training Loss'])
            test_mean, test_h = mean_confidence_interval(stats['Test Accuracy'])
        elif len(stats['Training Loss']) == 1:
            train_mean, train_h = stats['Training Loss'][0], np.zeros_like(stats['Training Loss'][0])
            test_mean, test_h = stats['Test Accuracy'][0], np.zeros_like(stats['Test Accuracy'][0])
        epochs = list(range(1, len(train_mean) + 1))
        ax1.plot(epochs, train_mean, linewidth=linewidth, label=label, color=color)
        ax1.fill_between(epochs, (train_mean - train_h),
                         (train_mean + train_h), color=color, alpha=0.1)
        ax2.plot(epochs, test_mean, linewidth=linewidth, label=label, color=color)
        ax2.fill_between(epochs, (test_mean - test_h),
                         (test_mean + test_h), color=color, alpha=0.1)


    ax1.set_xlabel('Number of epochs', fontsize=fontsize)
    ax1.set_ylabel('Training Loss', fontsize=fontsize)
    ax2.set_xlabel('Number of epochs', fontsize=fontsize)
    ax2.set_ylabel('Test Accuracy', fontsize=fontsize)

    # if dataset == 'FashionMNIST':
    #     ax1.set(ylim=[-0.005,0.1])
    #     ax2.set(ylim=[0.88, 0.935])
    # elif dataset == 'CIFAR10':
    #     ax1.set(ylim=[0, 0.8])
    #     ax2.set(ylim=[0.77, 0.93])
    # elif dataset == 'CIFAR100':
    #     ax1.set(ylim=[0, 2.5])
    #     ax2.set(ylim=[0.36, 0.76])

    ax1.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize-2)

    handles, labels = ax1.get_legend_handles_labels()
    if len(handles) != len(legend_order):
        legend_order = list(range(len(handles)))
    fig.legend([handles[idx] for idx in legend_order],[labels[idx] for idx in legend_order], fontsize=fontsize, loc='lower left', bbox_to_anchor=[0.005,0.03,0.98,0.25], ncol=3, mode='expand', borderaxespad=0.)

    plt.savefig(fig_folder + '/' + dataset + '_' + fig_type + '.png')

    plt.show()


def load_args():
    parser = argparse.ArgumentParser(description='Draw comparison between different optimization methods.')

    parser.add_argument('--logs-folder', type=str, default='./logs/CIFAR10',
                        help='log folder path')
    parser.add_argument('--fig-type', type=str, default='stagewise',
                        choices=['stagewise', 'others'],
                        help='which figure to draw')

    return parser.parse_args()


args = load_args()
draw_comps(args.logs_folder, args.fig_type)