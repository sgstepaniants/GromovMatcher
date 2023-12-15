import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy.io as sio

params = {
    'axes.linewidth': 0.3,
    'xtick.major.size': 1.5,
    'xtick.minor.size': 1,
    'xtick.major.width': 0.3,
    'xtick.minor.width': 0.3,
    'ytick.major.size': 1.5,
    'ytick.minor.size': 1,
    'ytick.major.width': 0.3,
    'ytick.minor.width': 0.3,
}
mpl.rcParams.update(params)
pti = 0.3528 / 25.4

diag_size = 10
out_size = 20
alpha = 0.3

size = (6,6)
dpi = 500
xw
os.chdir(sys.path[0])

max_fun = lambda Pi: Pi * ((Pi > 0) & (Pi >= np.maximum(np.max(Pi, axis=1)[:, None], np.max(Pi, axis=0)[None, :])))

overlaps = ["0.25", "0.5", "0.75"]
for overlap in overlaps:
    # True matching
    true = np.load(f"../../Coupling_matrices/TRUE_[{overlap}, 0.01, 0.5, 0.5, 'smooth', 1].npy")
    true = true.astype('float64')
    true[true > 0] = 1

    true_positives = np.where(true == 1)

    plt.figure(figsize=size)
    plt.scatter(true_positives[1]+0.5, true_positives[0]+0.5, color="forestgreen", marker=".", s=out_size)
    plt.xlim(0, true.shape[0])
    plt.ylim(0, true.shape[1])
    plt.xticks([])
    plt.yticks([])
    plt.gca().axis('off')
    plt.savefig(f'True_{overlap}.png', dpi = 100, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.show()


    # GromovMatcher matching
    fileGM = f"../../Coupling_matrices/GMcouple_topRT_[{overlap}, 0.01, 0.5, 0.5, 'smooth', 1]_[0.01, 'MAD', 4].npy"
    coupling = np.load(fileGM, allow_pickle = True)

    coupling = coupling.astype('float64')
    coupling = coupling/np.max(coupling)
    coupling = max_fun(coupling)
    matching = coupling > 0

    true_positives = np.where((matching == 1) & (true == 1))
    false_positives = np.where((matching == 1) & (true == 0))
    false_negatives = np.where((matching == 0) & (true == 1))

    plt.figure(figsize=size)
    plt.scatter(true_positives[1]+0.5, true_positives[0]+0.5, color="forestgreen", marker=".", s=out_size)
    plt.scatter(false_negatives[1]+0.5, false_negatives[0]+0.5, color="gainsboro", marker=".", s=diag_size, alpha = alpha)
    plt.scatter(false_positives[1]+0.5, false_positives[0]+0.5, color="red", marker=".", s=out_size)
    plt.xlim(0, matching.shape[0])
    plt.ylim(0, matching.shape[1])
    plt.xticks([])
    plt.yticks([])
    plt.gca().axis('off')
    plt.savefig(f'GMcoupling_{overlap}.png', dpi = dpi, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.show()


    # Thresholded GromovMatcher matching
    thr_matching = coupling >= 0.3

    true_positives = np.where((thr_matching == 1) & (true == 1))
    false_positives = np.where((thr_matching == 1) & (true == 0))
    false_negatives = np.where((thr_matching == 0) & (true == 1))

    plt.figure(figsize=size)
    plt.scatter(true_positives[1]+0.5, true_positives[0]+0.5, color="forestgreen", marker=".", s=out_size)
    plt.scatter(false_negatives[1]+0.5, false_negatives[0]+0.5, color="gainsboro", marker=".", s=diag_size, alpha = alpha)
    plt.scatter(false_positives[1]+0.5, false_positives[0]+0.5, color="red", marker=".", s=out_size)
    plt.xlim(0, matching.shape[0])
    plt.ylim(0, matching.shape[1])
    plt.xticks([])
    plt.yticks([])
    plt.gca().axis('off')
    plt.savefig(f'GMTcoupling_{overlap}.png', dpi = dpi, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.show()


    # MetabCombiner matching
    fileMC = f'../../Coupling_matrices/mC_couple_[{overlap}, 0.01, 0.5, 0.5, smooth, 1].npy'
    coupling = np.load(fileMC, allow_pickle = True)

    coupling = coupling.astype('float64')
    coupling = coupling/np.max(coupling)
    matching = max_fun(coupling)
    thr_matching = matching.copy()
    matching[matching>0] = 1
    thr_matching[thr_matching>0] = 1


    true_positives = np.where((matching == 1) & (true == 1))
    false_positives = np.where((matching == 1) & (true == 0))
    false_negatives = np.where((matching == 0) & (true == 1))


    plt.figure(figsize=size)
    plt.scatter(true_positives[1]+0.5, true_positives[0]+0.5, color="forestgreen", marker=".", s=out_size)
    plt.scatter(false_negatives[1]+0.5, false_negatives[0]+0.5, color="gainsboro", marker=".", s=diag_size, alpha = alpha)
    plt.scatter(false_positives[1]+0.5, false_positives[0]+0.5, color="red", marker=".", s=out_size)
    plt.xlim(0, matching.shape[0])
    plt.ylim(0, matching.shape[1])
    plt.xticks([])
    plt.yticks([])
    plt.gca().axis('off')
    plt.savefig(f'mCcoupling_{overlap}.png', dpi = dpi, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.show()


    # M2S matching
    fileM2S = f'../../Coupling_matrices/M2S_couple_[{overlap}, 0.01, 0.5, 0.5, smooth, 1].mat'
    coupling = sio.loadmat(fileM2S)['matching'].astype('float64')

    coupling = coupling/np.max(coupling)
    matching = max_fun(coupling)
    thr_matching = matching.copy()
    matching[matching>0] = 1


    true_positives = np.where((matching == 1) & (true == 1))
    false_positives = np.where((matching == 1) & (true == 0))
    false_negatives = np.where((matching == 0) & (true == 1))


    plt.figure(figsize=size)
    plt.scatter(true_positives[1]+0.5, true_positives[0]+0.5, color="forestgreen", marker=".", s=out_size)
    plt.scatter(false_negatives[1]+0.5, false_negatives[0]+0.5, color="gainsboro", marker=".", s=diag_size, alpha = alpha)
    plt.scatter(false_positives[1]+0.5, false_positives[0]+0.5, color="red", marker=".", s=out_size)
    plt.xlim(0, matching.shape[0])
    plt.ylim(0, matching.shape[1])
    plt.xticks([])
    plt.yticks([])
    plt.gca().axis('off')
    plt.savefig(f'M2Scoupling_{overlap}.png', dpi = dpi, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.show()