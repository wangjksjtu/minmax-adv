import os
import numpy as np
import glob
import re
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import get_interval

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def plot_w(indir='w_save/mnist'):
    font = {'size'   : 22}
    matplotlib.rc('font', **font)

    files = glob.glob(os.path.join(indir, "w_*.npy"))
    files = sorted(files, key=natural_keys)

    ws = []
    for filename in files:
        w = np.load(filename)
        ws.append(w)
        # print (w)
        print (np.mean(w, axis=0))
    
    ws = np.stack(ws, axis=0)
    ticks = [str(i) for i in range(0, ws.shape[0], 5)]

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
    
    plt.figure(figsize=(16, 8))

    medianprops = dict(linestyle='-', linewidth=2.5, color='red')

    colors = ['r', 'g', 'b']
    arch = list('ABC')

    for i in range(w.shape[-1]):
        data = ws[:,:,i]
        positions = np.array(range(len(data)))*3.0 + 0.7*(i-1)
        data = np.transpose(data)
        # print (data)
        # print (data.shape, len(data))
        # print (positions, len(positions))
        bp = plt.boxplot(data, positions=positions, sym='', patch_artist=True, medianprops=medianprops)
        # set_box_color(bp, color[i]) # colors are from http://colorbrewer2.org/
        plt.plot([], c=colors[i], label='Model ' + arch[i])

        for patch in bp['boxes']:
            patch.set_facecolor(colors[i])
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))

    plt.legend(loc='upper left')
    plt.xticks(range(0, len(ticks) * 3 * 5, 3 * 5), ticks)
    plt.xlim(-3, len(ticks)*3 * 5)
    plt.ylim(0, 1)
    plt.xlabel('Number of iterations')
    plt.ylabel('$w_i$')
    plt.tight_layout()
    # plt.show()
    plt.savefig('w_dist.pdf')
    

def plot_loss(indir='logs_mnist_avg'):
    sns.set(font_scale=1.4)
    # sns.set_context("talk")
    # sns.set_style("whitegrid")

    files = glob.glob(os.path.join(indir, "ens_model*.out"))
    results = dict()
    for filename in files:
        log = open(filename, 'r').readlines()
        epochs = int(log[0].split(', ')[6].split("=")[-1])
        result = []
        for line in log[5:8]:
            adv_loss, adv_acc = line.split("\t")
            adv_loss = float(adv_loss.split(": ")[-1])
            adv_acc = float(adv_acc.split(": ")[-1])
            result += [adv_loss, adv_acc]
        
        result += [float(log[-1].split(": ")[-1])]
        results[epochs] = result

    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = ["loss1", "acc1", "loss2", "acc2", "loss3", "acc3", "foolrate"]

    fig, ax1 = plt.subplots(figsize=(9,6))
    iters = np.arange(0, 40)
    
    ax1.plot(iters, df['loss1'], 'b^-.', alpha=0.8)
    ax1.plot(iters, df['loss2'], 'bo:', alpha=0.8)
    ax1.plot(iters, df['loss3'], 'bs--', alpha=0.8)


    ax1.set_xlabel('Number of iterations')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('loss', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()

    ax2.plot(iters, df['acc1'], 'm^-.', alpha=0.8)
    ax2.plot(iters, df['acc2'], 'mo:', alpha=0.8)
    ax2.plot(iters, df['acc3'], 'ms--', alpha=0.8)

    legend_elements = [Line2D([0], [0], marker='^', linestyle='-.', lw=2, c='r', label='Model A'),
                       Line2D([0], [0], marker='o', linestyle=':', lw=2, c='r', label='Model B'),
                       Line2D([0], [0], marker='s', linestyle='--', lw=2, c='r', label='Model C'),
                       Line2D([0], [0], marker='x', linestyle='-', lw=2, c='r', label='success rate')]

    l7 = ax2.plot(iters, df['foolrate'], 'rx-', linewidth=2.0)

    plt.legend(handles=legend_elements, loc='lower right')

    ax2.set_ylabel('test accuracy', color='m')
    ax2.tick_params('y', colors='m')
    ax2.grid(False)
    fig.tight_layout()
    # plt.show()
    plt.savefig('loss_acc_avg.pdf')
    

def plot_eps(indir="logs_mnist_eps"):
    sns.set(font_scale=1.4)
    files = glob.glob(os.path.join(indir, "ens_model*.out"))

    avg_asr = []
    avg_eps = []
    mm_asr = []
    mm_eps = []

    for filename in files:
        log = open(filename, 'r').readlines()
        avg = (log[0].split(', ')[2].split("=")[-1])
        eps = (log[0].split(', ')[7].split("=")[-1])
        foolrate = float(log[-1].split(": ")[-1])

        if avg == 'True': 
            avg_asr.append(foolrate)
            avg_eps.append(eps)
        else: 
            mm_asr.append(foolrate)
            mm_eps.append(eps)

    mm_asr = [x for _, x in sorted(zip(mm_eps, mm_asr))]
    avg_asr = [x for _, x in sorted(zip(avg_eps, avg_asr))]

    mm_eps = sorted(mm_eps)
    avg_eps = sorted(avg_eps)

    plt.plot(avg_eps, avg_asr, 'o-', label='average case', linewidth=2.0)
    plt.plot(mm_eps, mm_asr, '^-', label='minmax optimization', linewidth=2.0)
    plt.legend()
    plt.xlabel('$\\epsilon$')
    plt.ylabel('Attack success rate')
    plt.tight_layout()
    # plt.show()
    plt.savefig('compare_mnist.pdf')

# draw box plot between weight and eps
def plot_w_norm(indir='save_mnist_w'):
    font = {'size'   : 22}
    matplotlib.rc('font', **font)

    files = glob.glob(os.path.join(indir, "w_*.npy"))
    files = sorted(files, key=natural_keys)

    ws = []
    for filename in files:
        w = np.load(filename)
        ws.append(w)
    
    ws = np.stack(ws, axis=0)
    ticks = [str(i) for i in range(0, ws.shape[0], 50)]

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
    
    plt.figure(figsize=(16, 8))
    medianprops = dict(linestyle='-', linewidth=1.0, color='red')
    colors = ['r', 'g', 'b']
    arch = ["$\\ell_\\infty$","$\\ell_2$", "$\\ell_1$"]

    for i in range(ws.shape[-1]):
        data = ws[:,:,i]
        positions = np.array(range(len(data)))*3.0 + 0.7*(i-1)
        data = np.transpose(data)
        bp = plt.boxplot(data, positions=positions, widths=0.5, sym='', patch_artist=True, medianprops=medianprops)
        plt.plot([], c=colors[i], label=arch[i])

        for patch in bp['boxes']:
            patch.set_facecolor(colors[i])
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))

    plt.legend(loc='upper left')
    plt.xticks(range(0, len(ticks) * 3 * 5, 3 * 5), ticks)
    plt.xlim(-3, len(ticks)*3 * 5)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Number of iterations')
    plt.ylabel('$w_i$')
    plt.tight_layout()
    # plt.show()
    plt.savefig('w_dist.pdf')
 
# draw box plot between weights and eps
def plot_w_eps(indir='save_mnist_w'):
    font = {'size': 26}
    matplotlib.rc('font', **font)

    files = glob.glob(os.path.join(indir, "w_349.npy"))
    files = sorted(files, key=natural_keys)[1:8]
    ws = []
    eps = []
    for filename in files:
        w = np.load(filename)
        e = re.findall(r'\d+\.\d+', filename)[1]
        ws.append(w)
        eps.append(e)
    ws = np.stack(ws, axis=0)

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure(figsize=(8, 6))
    medianprops = dict(linestyle='-', linewidth=0.5, color='red')
    colors = ['blue', 'orange', 'b']
    arch = ["$\\ell_\\infty$","$\\ell_2$"]

    for i in range(ws.shape[-1]):
        data = ws[:,:,i]
        positions = np.array(range(len(data)))*15.0
        data = np.transpose(data)
        bp = plt.violinplot(data, positions=positions, widths=3.5, showmeans=True)
        plt.plot([], c=colors[i], label=arch[i])

        for pc in bp['bodies']:
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
            pc.set_alpha(0.5)

    plt.legend(loc='upper right')
    plt.xticks(range(0, len(eps) * 3 * 5, 3 * 5), eps)
    plt.xlim(-15, len(eps)*3 * 5)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('$\epsilon (\ell_2)$')
    plt.ylabel('$w_i$')
    plt.tight_layout()
    # plt.show()
    plt.savefig('w_dist.pdf')


# draw box plots on data transformation between weight and epoch
def plot_trans_w(indir='save_mnist_w'):
    font = {'size'   : 22}
    matplotlib.rc('font', **font)
    ws = []

    files = glob.glob(os.path.join(indir, "w_*.npy"))
    files = sorted(files, key=natural_keys)
    arch = ["fliplr", "adjust_brightness", "rotate_random", "useless"]
    for filename in files:
        w = np.load(filename)
        ws.append(w)
    ws = np.stack(ws, axis=0)
    ticks = [str(i) for i in range(0, ws.shape[0], 50)]

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
    
    plt.figure(figsize=(16, 8))
    #medianprops = dict(linestyle='-', linewidth=1, color='red')
    colors = ['r', 'g', 'b', 'y']
    for i in range(w.shape[-1]):
        data = ws[:,:,i]
        positions = np.array(range(len(data)))*3.0 + 0.7*(i-1)
        data = np.transpose(data)
        bp = plt.boxplot(data, positions=positions, sym='', patch_artist=True)
        plt.plot([], c=colors[i], label=arch[i])

    plt.legend(loc='upper left')
    plt.xticks(range(0, len(ticks) * 3 * 5, 3 * 5), ticks)
    plt.xlim(-3, len(ticks)*3 * 5)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('epochs')
    plt.ylabel('$w_i$')
    plt.tight_layout()
    # plt.show()
    plt.savefig('w_dist.pdf')


if __name__ == "__main__":
    # plot_w_norm()
    # plot_trans_w()
    # plot_loss()
    # plot_eps_new()
