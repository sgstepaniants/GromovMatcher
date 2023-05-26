import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches

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

os.chdir(sys.path[0])

data_type1 = "CSPOS"
data_type2 = "LIVERPOS"

# GromovMatcher
Pi_hat = np.load(f"../../EPIC_couplings/GMBSLogNorm_Data{data_type1}_Data{data_type2}.npy")
#Pi_hat[Pi_hat < 0.3*np.max(Pi_hat)] = 0
matching = (Pi_hat > 0) & (Pi_hat >= np.maximum(np.max(Pi_hat, axis=1)[:, None], np.max(Pi_hat, axis=0)[None, :]))
p1, p2 = matching.shape
p_matched = np.sum(matching)

# Distance matrices
D1 = np.load(f"../../EPIC_distance_matrices/CorrMat_Data{data_type1}.npy")
D2 = np.load(f"../../EPIC_distance_matrices/CorrMat_Data{data_type2}.npy")

# Hand matches
hand_matching = np.load(f"../../EPIC_couplings/true_{data_type2}_{data_type1}.npy")
hand_matched_inds1, hand_matched_inds2 = np.where(hand_matching == 1)
matched_inds1, matched_inds2 = np.where(matching)




all_matched_inds1 = np.union1d(hand_matched_inds1, matched_inds1)
all_matched_inds2 = np.union1d(hand_matched_inds2, matched_inds2)

#lvTmp = np.linspace(0.0, 1.0, 999)
#cmTmp = mpl.cm.bwr(lvTmp)
#cmap = mpl.colors.ListedColormap(cmTmp)
cmap = "RdBu_r"

edgecolor = "black"

factor = 1

idx = matching[hand_matched_inds1, hand_matched_inds2]
inds1 = np.hstack([hand_matched_inds1[idx], hand_matched_inds1[~idx]])
fig = plt.figure(figsize=(pti*factor*len(inds1), pti*factor*len(inds1)))
im = plt.pcolormesh(D1[inds1][:, inds1], vmin=-1, vmax=1, cmap=cmap)
line = patches.Polygon(np.array([[sum(idx), 0], [sum(idx), sum(idx)], [0, sum(idx)]]), linewidth=0.7, edgecolor=edgecolor, facecolor="none", alpha=1, linestyle="-", closed=False)
plt.xticks([])
plt.yticks([])
plt.gca().add_patch(line)
plt.gca().invert_yaxis()
plt.gca().xaxis.set_ticks_position('top')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("D1_manual.png", bbox_inches="tight", pad_inches=0.01, transparent=True, dpi=1200)
plt.show()

inds2 = np.hstack([hand_matched_inds2[idx], hand_matched_inds2[~idx]])
fig = plt.figure(figsize=(pti*factor*len(inds2), pti*factor*len(inds2)))
plt.pcolormesh(D2[inds2][:, inds2], vmin=-1, vmax=1, cmap=cmap)
line = patches.Polygon(np.array([[sum(idx), 0], [sum(idx), sum(idx)], [0, sum(idx)]]), linewidth=0.7, edgecolor=edgecolor, facecolor="none", alpha=1, linestyle="-", closed=False)
plt.xticks([])
plt.yticks([])
plt.gca().add_patch(line)
plt.gca().invert_yaxis()
plt.gca().xaxis.set_ticks_position('top')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("D2_manual.png", bbox_inches="tight", pad_inches=0.01, transparent=True, dpi=1200)
plt.show()

factor = 0.2

shared_idx = np.in1d(matched_inds1, hand_matched_inds1[idx])
inds1 = np.hstack([matched_inds1[shared_idx], matched_inds1[~shared_idx]])
fig = plt.figure(figsize=(pti*factor*len(inds1), pti*factor*len(inds1)))
plt.pcolormesh(D1[inds1][:, inds1], vmin=-1, vmax=1, cmap=cmap)
line = patches.Polygon(np.array([[sum(shared_idx), 0], [sum(shared_idx), sum(shared_idx)], [0, sum(shared_idx)]]), linewidth=0.7, edgecolor=edgecolor, facecolor="none", alpha=1, linestyle="-", closed=False)
plt.gca().add_patch(line)
plt.xticks([])
plt.yticks([])
plt.gca().invert_yaxis()
plt.gca().xaxis.set_ticks_position('top')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("D1_gw.png", bbox_inches="tight", pad_inches=0.01, transparent=True, dpi=1200)
plt.show()

inds2 = np.hstack([matched_inds2[shared_idx], matched_inds2[~shared_idx]])
fig = plt.figure(figsize=(pti*factor*len(inds2), pti*factor*len(inds2)))
plt.pcolormesh(D2[inds2][:, inds2], vmin=-1, vmax=1, cmap=cmap)
line = patches.Polygon(np.array([[sum(shared_idx), 0], [sum(shared_idx), sum(shared_idx)], [0, sum(shared_idx)]]), linewidth=0.7, edgecolor=edgecolor, facecolor="none", alpha=1, linestyle="-", closed=False)
plt.gca().add_patch(line)
plt.xticks([])
plt.yticks([])
plt.gca().invert_yaxis()
plt.gca().xaxis.set_ticks_position('top')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("D2_gw.png", bbox_inches="tight", pad_inches=0.01, transparent=True, dpi=1200)
plt.show()


factor = 10
# colorbar for distance matrices
fig = plt.figure(figsize=(pti*factor*1, pti*factor*10))
cbar = fig.colorbar(im, ticks=[-1, 0, 1], cax=plt.gca())
cbar.ax.set_yticklabels([])
plt.savefig("distance_colorbar.pdf", bbox_inches="tight", pad_inches=0.1, transparent=True)




# Plot precisions and recalls for metabCombiner, M2S, and GromovMatcher

factor = 80

# CS & HCC positive mode (precision)
fig = plt.figure(figsize=(pti*factor, pti*factor))
x = np.array([0, 1, 2])
y = np.array([0.961, 0.967, 0.989])
y_lower = np.array([0.868, 0.908, 0.939])
y_upper = np.array([0.993, 0.991, 0.999])
plt.bar(x, y, width=0.6, color=["#F8766D", "#7CAE01", "#02BFC4"])
plt.errorbar(x, y, yerr=[y-y_lower, y_upper-y], capsize=2, elinewidth=0.5, capthick=0.5, fmt="none", color="k")
plt.scatter(x, y, s=1, color="k")
plt.xticks([])
plt.yticks([0.7, 0.8, 0.9, 1], [])
plt.ylim(0.7, 1.02)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig("CS_HCC_pos_precision.pdf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()

# CS & HCC positive mode (recall)
fig = plt.figure(figsize=(pti*factor, pti*factor))
x = np.array([0, 1, 2])
y = np.array([0.544, 0.978, 0.978])
y_lower = np.array([0.442, 0.923, 0.923])
y_upper = np.array([0.643, 0.996, 0.996])
plt.bar(x, y, width=0.6, color=["#F8766D", "#7CAE01", "#02BFC4"])
plt.errorbar(x, y, yerr=[y-y_lower, y_upper-y], capsize=2, elinewidth=0.5, capthick=0.5, fmt="none", color="k")
plt.scatter(x, y, s=1, color="k")
plt.xticks([])
plt.yticks([0, 0.5, 1], [])
plt.ylim(0, 1.02)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig("CS_HCC_pos_recall.pdf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()


# CS & HCC negative mode (precision)
fig = plt.figure(figsize=(pti*factor, pti*factor))
x = np.array([0, 1, 2])
y = np.array([0.875, 1, 0.95])
y_lower = np.array([0.529, 0.824, 0.764])
y_upper = np.array([0.993, 1, 0.997])
plt.bar(x, y, width=0.6, color=["#F8766D", "#7CAE01", "#02BFC4"])
plt.errorbar(x, y, yerr=[y-y_lower, y_upper-y], capsize=2, elinewidth=0.5, capthick=0.5, fmt="none", color="k")
plt.scatter(x, y, s=1, color="k")
plt.xticks([])
plt.yticks([0.5, 0.75, 1], [])
plt.ylim(0.5, 1.02)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig("CS_HCC_neg_precision.pdf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()

# CS & HCC negative mode (recall)
fig = plt.figure(figsize=(pti*factor, pti*factor))
x = np.array([0, 1, 2])
y = np.array([0.368, 0.947, 1])
y_lower = np.array([0.191, 0.754, 0.832])
y_upper = np.array([0.590, 0.997, 1])
plt.bar(x, y, width=0.6, color=["#F8766D", "#7CAE01", "#02BFC4"])
plt.errorbar(x, y, yerr=[y-y_lower, y_upper-y], capsize=2, elinewidth=0.5, capthick=0.5, fmt="none", color="k")
plt.scatter(x, y, s=1, color="k")
plt.xticks([])
plt.yticks([0.0, 0.5, 1], [])
plt.ylim(0.0, 1.02)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig("CS_HCC_neg_recall.pdf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()


# CS & PC positive mode (precision)
fig = plt.figure(figsize=(pti*factor, pti*factor))
x = np.array([0, 1, 2])
y = np.array([0.967, 0.855, 0.903])
y_lower = np.array([0.833, 0.759, 0.813])
y_upper = np.array([0.998, 0.917, 0.952])
plt.bar(x, y, width=0.6, color=["#F8766D", "#7CAE01", "#02BFC4"])
plt.errorbar(x, y, yerr=[y-y_lower, y_upper-y], capsize=2, elinewidth=0.5, capthick=0.5, fmt="none", color="k")
plt.scatter(x, y, s=1, color="k")
plt.xticks([])
plt.yticks([0.7, 0.8, 0.9, 1], [])
plt.ylim(0.7, 1.02)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig("CS_PC_pos_precision.pdf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()

# CS & PC positive mode (recall)
fig = plt.figure(figsize=(pti*factor, pti*factor))
x = np.array([0, 1, 2])
y = np.array([0.439, 0.985, 0.985])
y_lower = np.array([0.326, 0.919, 0.919])
y_upper = np.array([0.559, 0.999, 0.999])
plt.bar(x, y, width=0.6, color=["#F8766D", "#7CAE01", "#02BFC4"])
plt.errorbar(x, y, yerr=[y-y_lower, y_upper-y], capsize=2, elinewidth=0.5, capthick=0.5, fmt="none", color="k")
plt.scatter(x, y, s=1, color="k")
plt.xticks([])
plt.yticks([0, 0.5, 1], [])
plt.ylim(0, 1.02)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig("CS_PC_pos_recall.pdf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()



# CS & PC negative mode (precision)
fig = plt.figure(figsize=(pti*factor, pti*factor))
x = np.array([0, 1, 2])
y = np.array([1, 0.931, 0.929])
y_lower = np.array([0.845, 0.780, 0.774])
y_upper = np.array([1, 0.988, 0.987])
plt.bar(x, y, width=0.6, color=["#F8766D", "#7CAE01", "#02BFC4"])
plt.errorbar(x, y, yerr=[y-y_lower, y_upper-y], capsize=2, elinewidth=0.5, capthick=0.5, fmt="none", color="k")
plt.scatter(x, y, s=1, color="k")
plt.xticks([])
plt.yticks([0, 0.5, 1], [])
plt.ylim(0, 1.02)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig("CS_PC_neg_precision.pdf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()

# CS & PC negative mode (recall)
fig = plt.figure(figsize=(pti*factor, pti*factor))
x = np.array([0, 1, 2])
y = np.array([0.75, 0.964, 0.929])
y_lower = np.array([0.566, 0.823, 0.774])
y_upper = np.array([0.873, 0.998, 0.987])
plt.bar(x, y, width=0.6, color=["#F8766D", "#7CAE01", "#02BFC4"])
plt.errorbar(x, y, yerr=[y-y_lower, y_upper-y], capsize=2, elinewidth=0.5, capthick=0.5, fmt="none", color="k")
plt.scatter(x, y, s=1, color="k")
plt.xticks([])
plt.yticks([0, 0.5, 1], [])
plt.ylim(0, 1.02)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig("CS_PC_neg_recall.pdf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()