import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy.io import savemat
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import splrep, splev
import weighted

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

frac = 0.5
noise = 0.01
drift = "smooth"
folder = f"../../Overlap_{frac}"
df1 = pd.read_csv(f"{folder}/Data1_{frac}_{noise}_{drift}.csv")
df2 = pd.read_csv(f"{folder}/Data2_{frac}_{noise}_{drift}.csv")

MZ1 = np.array(df1.iloc[0, 1:]).astype(float)
RT1 = np.array(df1.iloc[1, 1:]).astype(float)
FI1 = np.array(df1.iloc[2:, 1:]).T.astype(float)
MZ2 = np.array(df2.iloc[0, 1:]).astype(float)
RT2 = np.array(df2.iloc[1, 1:]).astype(float)
FI2 = np.array(df2.iloc[2:, 1:]).T.astype(float)

D1 = euclidean_distances(FI1, squared=False)
D2 = euclidean_distances(FI2, squared=False)

coupling_file = f'true_{frac}'
coupling = np.load(f'{folder}/{coupling_file}.npy')

matched_inds1 = np.where(np.sum(coupling, axis=1) > 0)[0]
unmatched_inds1 = np.setdiff1d(np.arange(coupling.shape[0]), matched_inds1)
matched_inds2 = np.argmax(coupling[matched_inds1, :], axis=1)
unmatched_inds2 = np.setdiff1d(np.arange(coupling.shape[0]), matched_inds2)

p1 = 10
p2 = 14
num_matches = 7

seed = 3998963 #np.random.choice(int(1e7))
np.random.seed(seed=seed)

sub_matched_inds = np.random.choice(np.arange(len(matched_inds1)), num_matches, replace=False)
inds1 = np.hstack((matched_inds1[sub_matched_inds], np.random.choice(unmatched_inds1, p1-num_matches, replace=False)))
inds2 = np.hstack((matched_inds2[sub_matched_inds], np.random.choice(unmatched_inds2, p2-num_matches, replace=False)))
coupling_sub = np.zeros((p1, p2))
coupling_sub[:num_matches, :num_matches] = np.eye(num_matches)

perm = np.random.permutation(np.arange(p2))
coupling_sub = coupling_sub[:, perm]
inds2 = inds2[perm]

D1_sub = D1[inds1][:, inds1]
D2_sub = D2[inds2][:, inds2]

vmax = 10*np.ceil(max(np.max(D1_sub), np.max(D2_sub))/10)

factor = 7

## Panel b
# distance matrix of dataset 1
fig = plt.figure(figsize=(pti*factor*p1, pti*factor*p1))
plt.pcolormesh(D1_sub, vmin=0, vmax=vmax, cmap="cubehelix")
plt.gca().set_yticks(np.arange(p1) + 0.5)
plt.gca().set_yticklabels([])
plt.gca().set_xticks(np.arange(p1) + 0.5)
plt.gca().set_xticklabels([])
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("D1.pdf", bbox_inches="tight", pad_inches=0, transparent=True)

# distance matrix of dataset 2
fig = plt.figure(figsize=(pti*factor*p2, pti*factor*p2))
im = plt.pcolormesh(D2_sub, vmin=0, vmax=vmax, cmap="cubehelix")
plt.gca().set_yticks(np.arange(p2) + 0.5)
plt.gca().set_yticklabels([])
plt.gca().set_xticks(np.arange(p2) + 0.5)
plt.gca().set_xticklabels([])
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("D2.pdf", bbox_inches="tight", pad_inches=0, transparent=True)

# colorbar for distance matrices
fig = plt.figure(figsize=(pti*factor*1, pti*factor*p2))
cbar = fig.colorbar(im, ticks=[0, vmax/2, vmax], cax=plt.gca())
cbar.ax.set_yticklabels([])
#plt.savefig("distance_colorbar.pdf", bbox_inches="tight", pad_inches=0.1, transparent=True)


## Panel c
# distance matrix of dataset 1 (with ticks on right)
fig = plt.figure(figsize=(pti*factor*p1, pti*factor*p1))
plt.pcolormesh(D1_sub, vmin=0, vmax=vmax, cmap="cubehelix")
plt.gca().set_yticks(np.arange(p1) + 0.5)
plt.gca().set_yticklabels([])
plt.gca().set_xticks(np.arange(p1) + 0.5)
plt.gca().set_xticklabels([])
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.gca().yaxis.tick_right()
plt.gca().set_aspect('equal', adjustable='box')
#plt.savefig("D1_reordered.pdf", bbox_inches="tight", pad_inches=0, transparent=True)

# distance matrix of dataset 2 (reordered)
perm_inv = np.argsort(perm)
reorder = np.hstack((perm_inv[:num_matches], np.sort(perm_inv[num_matches:])))

fig = plt.figure(figsize=(pti*factor*p2, pti*factor*p2))
im = plt.pcolormesh(D2_sub[reorder][:, reorder], vmin=0, vmax=vmax, cmap="cubehelix")
plt.gca().set_yticks(np.arange(p2) + 0.5)
plt.gca().set_yticklabels([])
plt.gca().set_xticks(np.arange(p2) + 0.5)
plt.gca().set_xticklabels([])
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.gca().set_aspect('equal', adjustable='box')
#plt.savefig("D2_reordered.pdf", bbox_inches="tight", pad_inches=0, transparent=True)

# mock coupling Pi_tilde estimated by GW
factor = 10

Pi_tilde = gaussian_filter(coupling_sub, sigma=0.6)
fig = plt.figure(figsize=(pti*factor*p2, pti*factor*p1))
im = plt.pcolormesh(Pi_tilde, vmin=0, vmax=1, cmap="Greys")
plt.gca().set_yticks(np.arange(p1) + 0.5)
plt.gca().set_yticklabels([])
plt.gca().set_xticks(np.arange(p2) + 0.5)
plt.gca().set_xticklabels([])
#plt.gca().invert_yaxis()
#plt.gca().xaxis.tick_top()
#plt.savefig("Pi_tilde.pdf", bbox_inches="tight", pad_inches=0)

# colorbar for coupling matrix
fig = plt.figure(figsize=(pti*factor*0.5, pti*factor*p1))
cbar = fig.colorbar(im, ticks=[0, 0.5, 1], cax=plt.gca(), orientation="vertical")
cbar.ax.set_yticklabels([])
#plt.savefig("coupling_colorbar.pdf", bbox_inches="tight", pad_inches=0.1, transparent=True)

def pad_range(xs, pl, pr):
    xmin = np.min(xs)
    xmax = np.max(xs)
    dx = xmax - xmin
    return xmin-pl*dx, xmax+pr*dx

## Panel d
xs = RT2[inds2]
ys = RT1[inds1]
factor = 9

# GW coupling matrix ordered by retention times
rt_sorted_inds1 = np.argsort(ys)
rt_sorted_inds2 = np.argsort(xs)

fig = plt.figure(figsize=(pti*factor*p2, pti*factor*p1))
im = plt.pcolormesh(Pi_tilde[rt_sorted_inds1][:, rt_sorted_inds2], vmin=0, vmax=1, cmap="Greys")
plt.gca().set_yticks(np.arange(p1) + 0.5)
plt.gca().set_yticklabels([])
plt.gca().set_xticks(np.arange(p2) + 0.5)
plt.gca().set_xticklabels([])
#plt.gca().invert_yaxis()
#plt.gca().xaxis.tick_top()
#plt.savefig("Pi_tilde_rt_ordered.pdf", bbox_inches="tight", pad_inches=0)


# retention time plot
xmin, xmax = pad_range(xs, 0.1, 0.1)
ymin, ymax = pad_range(ys, 0.25, 0.5)
xy_prod = np.array([*itertools.product(xs, ys)])

sorted_inds = np.argsort(xy_prod[:, 0])
spl = splrep(xy_prod[sorted_inds, 0], xy_prod[sorted_inds, 1], w=Pi_tilde.T.flatten()[sorted_inds], xb=xmin, xe=xmax, k=2)
xmin2, xmax2 = pad_range(xs, 0.05, 0.05)
xs2 = np.linspace(xmin2, xmax2, 100)
ys2 = splev(xs2, spl)

f_hat = splev(xs, spl)
MAD = weighted.median(np.abs((ys[:, None] - f_hat[None, :]).flatten()), Pi_tilde.flatten())
sigma = 1.5

fig = plt.figure(figsize=(pti*factor*p1*6/5, pti*factor*p1))
cmap = mpl.cm.get_cmap("Greys")
plt.fill_between(xs2, ys2-sigma*MAD, ys2+sigma*MAD, color='green', alpha=0.1)
plt.hlines(ys, xmin=xmin, xmax=xmax, linewidth=0.1, color="lightgray")
plt.vlines(xs, ymin=ymin, ymax=ymax, linewidth=0.1, color="lightgray")
plt.plot(xs2, ys2, color="k", linewidth=0.5)
#plt.scatter(xs[np.argsort(perm)[:num_matches]], ys[:num_matches], color="r", s=5)
colors = [(0, 0, 0, alpha) for alpha in Pi_tilde.T.flatten()]
plt.scatter(xy_prod[:, 0], xy_prod[:, 1], c=colors, s=1, zorder=10)
plt.margins(x=0, y=0)
plt.gca().set_yticks(ys)
plt.gca().set_yticklabels([])
plt.gca().set_xticks(xs)
plt.gca().set_xticklabels([])
plt.xlim([xmin, xmax])
#plt.ylim([ymin, ymax])
#plt.savefig("RT_drift.pdf", bbox_inches="tight", pad_inches=0, transparent=True)


def CustomCmap(from_rgb,to_rgb):
    # from color r,g,b
    r1,g1,b1 = from_rgb
    # to color r,g,b
    r2,g2,b2 = to_rgb
    cdict = {'red': ((0, r1, r1),
                   (1, r2, r2)),
           'green': ((0, g1, g1),
                    (1, g2, g2)),
           'blue': ((0, b1, b1),
                   (1, b2, b2))}
    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

# GW coupling matrix ordered by retention times with outliers removed
rt_sorted_inds1 = np.argsort(ys)
rt_sorted_inds2 = np.argsort(xs)

kept_inds1, kept_inds2 = np.where(np.abs(ys[:, None] - f_hat[None, :]) < sigma*MAD)

Pi_tilde_outlier = np.zeros((p1, p2))
Pi_tilde_outlier[kept_inds1, kept_inds2] = Pi_tilde[kept_inds1, kept_inds2]

mask = np.zeros((p1, p2))
mask[kept_inds1, kept_inds2] = 1

fig = plt.figure(figsize=(pti*factor*p2, pti*factor*p1))
cmap_green = CustomCmap([1, 1, 1], [0, 0.5, 0])
im = plt.pcolormesh(Pi_tilde_outlier[rt_sorted_inds1][:, rt_sorted_inds2], vmin=0, vmax=1, cmap="Greys")
plt.pcolormesh(mask[rt_sorted_inds1][:, rt_sorted_inds2] > 0, vmin=0, vmax=1, cmap=cmap_green, alpha=0.1)
plt.gca().set_yticks(np.arange(p1) + 0.5)
plt.gca().set_yticklabels([])
plt.gca().set_xticks(np.arange(p2) + 0.5)
plt.gca().set_xticklabels([])
#plt.gca().invert_yaxis()
#plt.gca().xaxis.tick_top()
plt.savefig("Pi_tilde_outlierrem.pdf", bbox_inches="tight", pad_inches=0)


# final matching Pi_hat
tau = 0.3
thresh = tau * np.max(Pi_tilde_outlier)
Pi_hat = np.zeros((p1, p2))
Pi_hat[Pi_tilde_outlier >= thresh] = Pi_tilde_outlier[Pi_tilde_outlier >= thresh]
for i in range(p1):
    for j in range(p2):
        if Pi_hat[i, j] < max(np.max(Pi_hat[i, :]), np.max(Pi_hat[:, j])):
            Pi_hat[i, j] = 0

fig = plt.figure(figsize=(pti*factor*p2, pti*factor*p1))
im = plt.pcolormesh(Pi_hat, vmin=0, vmax=1, cmap="Greys")
plt.gca().set_yticks(np.arange(p1) + 0.5)
plt.gca().set_yticklabels([])
plt.gca().set_xticks(np.arange(p2) + 0.5)
plt.gca().set_xticklabels([])
#plt.gca().invert_yaxis()
#plt.gca().xaxis.tick_top()
plt.savefig("Pi_tilde_final.pdf", bbox_inches="tight", pad_inches=0)

Pi_hat[Pi_hat > 0] = 1

fig = plt.figure(figsize=(pti*factor*p2, pti*factor*p1))
im = plt.pcolormesh(Pi_hat, vmin=0, vmax=1, cmap="Greys")
plt.gca().set_yticks(np.arange(p1) + 0.5)
plt.gca().set_yticklabels([])
plt.gca().set_xticks(np.arange(p2) + 0.5)
plt.gca().set_xticklabels([])
#plt.gca().invert_yaxis()
#plt.gca().xaxis.tick_top()
plt.savefig("Pi_hat.pdf", bbox_inches="tight", pad_inches=0)
