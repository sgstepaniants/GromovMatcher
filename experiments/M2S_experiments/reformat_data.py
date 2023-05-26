import os
import numpy as np
import pandas as pd
from scipy.io import savemat
from matplotlib import pyplot as plt

frac = '0.75'
folder = f'Overlap_{frac}'
all_files = os.listdir(f'../{folder}')
files = []
for file in all_files:
    if file.endswith(".csv") and not file.endswith("_reformatted.csv"):
        files.append(file[:-4])

for file in files:
    print(file)
    df = pd.read_csv(f'../{folder}/{file}.csv')
    MZ = df.iloc[0, 1:]
    RT = df.iloc[1, 1:]
    FI = df.iloc[2:, 1:].median(axis=0)

    df_short = pd.DataFrame([RT, MZ, FI]).T
    df_short.to_csv(f'../{folder}/{file}_reformatted.csv', index=False, header=False)


coupling_file = f'true_{frac}'
coupling = np.load(f'../{folder}/{coupling_file}.npy')
mdict = {"coupling": coupling}
savemat(f'../{folder}/{coupling_file}.mat', mdict)
