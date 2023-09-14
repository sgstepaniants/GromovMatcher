##################################################################################################
# Computes precision and recall for the results obtained with metabCombiner
##################################################################################################

import numpy as np
import pandas as pd


def max_fun(couple):
    '''Takes the coupling matrix as input and only keeps the highest coupling coefficient for each column (sets the non max coefficients to 0)'''
    match = np.zeros(np.shape(couple))
    n2 = np.shape(couple)[1]
    match[np.argmax(couple, axis = 0),np.linspace(0,n2-1,n2,dtype = int)] = np.max(couple, axis = 0)
    return(match)

def PrecRec(prediction, true_coupling):
    '''Computes precision and recall given a coupling matrix, and the ground truth.
    Input:
    prediction: numpy array of size (features in dataset 1, features in dataset 2), coupling matrix as output by GM or mC
    true_coupling: numpy array of size (features in dataset 1, features in dataset 2), true coupling matrix with true pairs as coded as >0, and non-matched pairs as 0.
    '''
    # Code matches as 1 for convenience, non-matches as 0
    true = true_coupling.copy()
    true[true > 0] = 1
    pred = prediction.copy()
    pred[pred > 0] = 1
    # Recall
    subset_matched = np.where(true == 1)
    TP = np.sum(pred[subset_matched])
    FN = np.sum(pred[subset_matched] == 0)
    Rec = TP/(TP+FN)
    # Precision
    P = np.sum(pred)
    Prec = TP/P
    return(np.array([Prec, Rec]))


def Recap_MC_fun(config, n_sim):
    '''Computes average performance of mC for a given [overlap, m/z noise, RT noise, FI noise, RT drift shape] configuration.
    Results from mC must be stored in a RES folder and named 'couple_{configuration}.npy'. 
    Ground truth coupling matrices must be stored in a DATA folder and named 'TRUE_{configuration}.npy'.
    Input:
    config: numpy array [overlap, m/z noise, RT noise, FI noise, RT drift shape]
    n_sim: int, the number of simulations ran for one configuration
    Output:
    numpy array [average precision over the n_sim simulations, average recall avor the n_sim simulations]'''
    n_samples = np.linspace(1,n_sim,n_sim, dtype = int)
    o, sigM, sigRT, sigFI, d = config
    Average_PR = 0
    config2 = config.copy()
    config2.append(0)
    for N in n_samples :
        config2[-1] = N
        # load the mC results
        file_name = 'RES/couple_'+str(config2).replace("'","")+'.npy'
        coupling = np.load(file_name, allow_pickle = True)
        # load the ground truth matrix
        file_name_true = '../DATA/TRUE_'+str(config2)+'.npy'
        true_coupling = np.load(file_name_true, allow_pickle = True)
        true_coupling[true_coupling > 0] = 1
        # keep only the most likely match for each feature
        coupling = max_fun(max_fun(coupling).T).T
        # compute precision and recall, and sum over all the n_samples simulations
        Average_PR = Average_PR + PrecRec(np.transpose(coupling), np.transpose(true_coupling))
    [prec, rec] = Average_PR/len(n_samples)
    return([prec, rec])

# Run the previous function for every configuration of interest

CONFIG = []

for o in [0.25, 0.5, 0.75] : 
    for sigmaM in [0.01]:
        for sigmaRT in [0.2, 0.5, 1]:
            for sigmaFI in [0.1, 0.5, 1]:
                CONFIG.append([o, sigmaM, sigmaRT, sigmaFI, 'smooth'])
                           

PRs = []

for config in CONFIG:
    PRs.append([*config, binGap, Recap_MC_fun(config, 20)[0],Recap_MC_fun(config, binGap)[1]])

PRs = np.array(PRs)

np.save('mC_PrecisionRecall.npy', PRs)


