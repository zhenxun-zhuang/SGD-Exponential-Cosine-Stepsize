"""
Script for drawing comparison between different optimization methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os
import argparse
from collections import defaultdict

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


def readFile(folder, fileName):
    """From folder read fileName, which contains fields in stat_names."""
    fullFileName = os.path.join(folder, fileName)
    with open(fullFileName, 'r') as f:
        results = defaultdict(list)
        for line in f:
            if line.startswith('finalgrep : accuracy test : '):
                results['Final Test Accuracy'].append(eval(line[line.find('test : ') + 7:]) / 100.0)
            if not line.startswith('Epoch'):
                continue
            epoch = eval(line[7 : line.find(' (')])
            if epoch == 0 or epoch == 1000000:
                continue
            dataset = line[line.find('(') + 1 : line.find(')')].rstrip()
            loss = eval(line[line.find('Loss ') + 5 : line.find('   Acc')])
            acc = eval(line[line.find('Acc ') + 4 :]) / 100.0
            results[(dataset + ' Loss')].append(loss)
            results[(dataset + ' Accuracy')].append(acc)

    return results

def get_results(folder):
    specs = os.listdir(folder)
    all_results = {}
    for spec in specs:
        stats = readFile(folder, os.path.join(spec, 'log.txt'))
        if 'Train Loss' not in stats:
            continue
        method = spec[14 : spec.find('--svm')]
        if method not in all_results:
            all_results[method] = defaultdict(list)
        for dataset in ['Train ', 'Valid ']:
            for item in ['Loss', 'Accuracy']:
                all_results[method][dataset + item].append(stats[dataset + item])
        all_results[method]['Final Test Accuracy'].append(stats['Final Test Accuracy'])
    
    for method in all_results:
        for item in all_results[method]:
            all_results[method][item] = np.array(all_results[method][item])

    return all_results

def draw_comps(logs_folder):
    all_results = get_results(logs_folder)

    linewidth = 5
    fontsize = 20

    # stepsize schemes.
    labels = {'adam':'Adam', 'adagrad':'AdaGrad', 'amsgrad':'AMSGrad', 'bpgrad':'BPGrad',
              'dfw':'DFW', 'sgd':'SGD', 'SGD_Exp_Decay':'Exponential Step Size',
              'SGD_Cosine_Decay':'Cosine Decay'}

    colors = {'adam':0, 'adagrad':1, 'amsgrad':2, 'SGD_Exp_Decay':3, 'dfw':4,
              'sgd':5, 'SGD_Cosine_Decay':6, 'bpgrad':7}

    draw_methods = ['adam', 'adagrad', 'amsgrad', 'SGD_Cosine_Decay', 
                    'dfw', 'bpgrad', 'SGD_Exp_Decay']

    fig_folder = './fig'
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[15, 7.5])
    fig.subplots_adjust(left=0.07, bottom=0.28, right=0.98, top=0.97,
                        wspace=0.22, hspace=None)

    for i, method in enumerate(draw_methods):
        if len(all_results[method]['Final Test Accuracy']) > 1 and all_results[method]['Final Test Accuracy'][0]:
            test_mean, test_h = mean_confidence_interval(all_results[method]['Final Test Accuracy'])
            print('%s -- Test accuracy: %g - %g\n' % (method, test_mean, test_h))
        elif len(all_results[method]['Final Test Accuracy']) == 1 and all_results[method]['Final Test Accuracy'][0]:
            test_mean, test_h = all_results[method]['Final Test Accuracy'][0][0], 0
            print('%s -- Test accuracy: %g\n' % (method, test_mean))

        label = labels[method] if method in labels else method
        color = good_colors[colors[method]] if method in colors else good_colors[-(i+1)]     
        if len(all_results[method]['Train Loss']) > 1:
            train_mean, train_h = mean_confidence_interval(all_results[method]['Train Loss'])
            valid_mean, valid_h = mean_confidence_interval(all_results[method]['Valid Accuracy'])
        elif len(all_results[method]['Train Loss']) == 1:
            train_mean, train_h = all_results[method]['Train Loss'][0], np.zeros_like(all_results[method]['Train Loss'][0])
            valid_mean, valid_h = all_results[method]['Valid Accuracy'][0], np.zeros_like(all_results[method]['Valid Accuracy'][0])

        epochs = list(range(1, len(train_mean) + 1))
        ax1.plot(epochs, train_mean, linewidth=linewidth, label=label, color=color)
        ax1.fill_between(epochs, (train_mean - train_h), (train_mean + train_h), color=color, alpha=0.1)
        ax2.plot(epochs, valid_mean, linewidth=linewidth, label=label, color=color)
        ax2.fill_between(epochs, (valid_mean - valid_h), (valid_mean + valid_h), color=color, alpha=0.1)

    ax1.set_xlabel('Number of epochs', fontsize=fontsize)
    ax1.set_ylabel('Training Loss', fontsize=fontsize)
    ax2.set_xlabel('Number of epochs', fontsize=fontsize)
    ax2.set_ylabel('Validation Accuracy', fontsize=fontsize)

    # ax1.set(ylim=[0, 2.5])
    # ax2.set(ylim=[0.36, 0.76])

    ax1.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize-2)

    fig.legend(fontsize=fontsize, loc='lower left', bbox_to_anchor=[0.002,0.03,0.98,0.25], ncol=4, mode='expand', borderaxespad=0.)

    plt.savefig(fig_folder + '/nlp.png')

    plt.show()


parser = argparse.ArgumentParser(description='Draw comparison between different optimization methods.')
parser.add_argument('--logs-folder', type=str, default='./logs',
                        help='log folder path')
args = parser.parse_args()
draw_comps(args.logs_folder)
