#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:30:54 2021

@author: mariebreeur
"""

import numpy as np
import pandas as pd
import math
import scipy.linalg as scl
from sklearn import preprocessing

import sys
import os
sys.path.append(r'datasets/')


def readCorrMat(data):
    #rownames = np.array(data['Unnamed: 0'])
    colnames = data.keys()[1:]
    
    col_ms = list()
    col_rs = list()
    for cname in colnames:
        tag = cname.split('.')
        col_ms.append(float(tag[0][1:] + '.' + tag[1]))
        col_rs.append(float(tag[2] + '.' + tag[3]))
    
    ms = np.array(col_ms)
    rs = np.array(col_rs)
    cnct = np.array(data[colnames])
    
    return ms, rs, cnct


## Drift functions
def smooth(x):
    return(1.1*x + 1.3*np.sin(1.2*x**(1/2)))

def piecewise_linear(x, breaks, b, coeffs):
    x0, x1, x2, x3 = breaks
    k0, k1, k2, k3, k4 = coeffs
    condlist = [x < x0, (x >= x0) & (x < x1), (x >= x1) & (x < x2), (x >= x2) & (x < x3), x >= x3]
    funclist = [lambda x: k0*x + b, lambda x: k0*x + b + k1*(x-x0), \
                lambda x: k0*x + b + k1*(x-x0) + k2*(x - x1),\
                lambda x:k0*x + b + k1*(x-x0) + k2*(x - x1) + k3*(x - x2),\
                lambda x:k0*x + b + k1*(x-x0) + k2*(x - x1) + k3*(x - x2) + k4*(x - x3)]
    return np.piecewise(x, condlist, funclist)

def piecewise2(x):
    return(piecewise_linear(x,[2,5,7,8],0,[1,-0.6,1,-0.8,0.3]))




def generate_dataset_pair(config, data):
    ms, rs, samples = readCorrMat(data)
    numsamples = samples.shape[0]
    num_mets = ms.size
    mets_tracker = np.linspace(0,num_mets-1,num_mets)
    # permute the metabolites and keep track of them
    full_perm = np.random.permutation(num_mets)
    mets_tracker = mets_tracker[full_perm]
    ms = ms[full_perm]
    rs = rs[full_perm]
    samples = samples[:, full_perm]
    # add (relative) noise to samples
    [overlap, sigmaM, sigmaRT, sigmaFI, rho, drift, N, norm] = config
    setting = 0
    
    #Assign mets to either one of the datasets
    num_mets1 = math.floor(overlap*num_mets + (1+setting)*(num_mets*((1-overlap)/2)))
    num_mets2 = math.floor(overlap*num_mets + (1-setting)*(num_mets*((1-overlap)/2)))
    num_mets_shared = num_mets1 + num_mets2 - num_mets
    
    mets_tracker1 = mets_tracker[0:num_mets1]
    mets_tracker2 = mets_tracker[(num_mets-num_mets2):num_mets]
    ms1 = ms[0:num_mets1]
    ms2 = ms[(num_mets-num_mets2):num_mets]
    rs1 = rs[0:num_mets1]
    rs2 = rs[(num_mets-num_mets2):num_mets]
    
    #True matching
    true_matching = np.zeros((num_mets1, num_mets2))
    x = np.linspace(num_mets1-num_mets_shared, num_mets1-1, num_mets_shared, dtype=int)
    y = np.linspace(0, num_mets_shared-1, num_mets_shared, dtype=int)
    true_matching[x, y] = 1
      
    #split into two dataset
    halfsamples = math.floor(numsamples/2)
    samples1 = samples[0:halfsamples, 0:num_mets1]
    samples2 = samples[halfsamples:numsamples, (num_mets-num_mets2):num_mets]
    
    #Add noise
    #ms1 = ms1 + np.random.uniform(-0.5*sigmaM,0.5*sigmaM,size = num_mets1)
    ms2 = ms2 + np.random.uniform(-sigmaM,sigmaM,size = num_mets2)
    
    if drift == 'smooth':
        rs2 = smooth(rs2)
    elif drift == 'piecewise2':
        rs2 = piecewise2(rs2)
        
    rs2 = rs2 + sigmaRT * np.random.uniform(-1,1,size = num_mets2)
    rs2[rs2 < 0.1] = 0.1 
    
    corr = generate_corrmat(num_mets1, 100, rho)
    noise1 = np.random.multivariate_normal(np.zeros(num_mets1), corr, halfsamples)
    samples1 = samples1 + sigmaFI * np.std(samples1) * noise1
    noise2 = np.random.multivariate_normal(np.zeros(num_mets2), np.eye(num_mets2), numsamples - halfsamples)
    samples2 = samples2 + sigmaFI * np.std(samples2) * noise2
    
    #Put in desired format
    Data1 = np.zeros((samples1.shape[0]+3, samples1.shape[1]))
    Data1[0,:]=mets_tracker1
    Data1[1,:]=ms1
    Data1[2,:]=rs1
    if norm:
        Data1[3:,:]= preprocessing.scale(samples1, axis = 0)
    else:
        Data1[3:,:]=samples1
    
    Data2 = np.zeros((samples2.shape[0]+3, samples1.shape[1]))
    Data2[0,:]=mets_tracker2
    Data2[1,:]=ms2
    Data2[2,:]=rs2
    if norm:
        Data2[3:,:]= preprocessing.scale(samples2, axis = 0)
    else:
        Data2[3:,:]=samples2
    
    return Data1, Data2, true_matching



def generate_corrmat(N, block_size, rho):
    n = N//block_size
    block = (1-rho)*np.eye(block_size) + rho*np.ones((block_size, block_size))
    if N%block_size != 0:
        last_block = (1-rho)*np.eye(N%block_size) + rho*np.ones((N%block_size,N%block_size))
        return(scl.block_diag(*([block] * n), last_block))
    else:
        return(scl.block_diag(*([block] * n)))

def matlab_print_array(arr):
    np.set_printoptions(threshold=sys.maxsize)

    n = arr.shape[0]
    for i in range(n):
        line = ' '.join([str(v) for v in arr[i, :]])
        sys.stdout.write(line)
        if i < n-1:
            sys.stdout.write("; ")

if __name__ == '__main__':
    print(os.getcwd())
    data = pd.read_excel(sys.argv[1])

    overlap = float(sys.argv[2])
    sigmaM = float(sys.argv[3])
    sigmaRT = float(sys.argv[4])
    sigmaFI = float(sys.argv[5])
    rho = float(sys.argv[6])
    drift = sys.argv[7]
    trial = int(sys.argv[8])
    norm = sys.argv[9] == "True"
    config = [overlap, sigmaM, sigmaRT, sigmaFI, rho, drift, trial, norm]

    np.random.seed(trial)
    Data1, Data2, true_matching = generate_dataset_pair(config, data)

    np.set_printoptions(threshold=sys.maxsize)
    sys.stdout.write("Data1: ")
    matlab_print_array(Data1)
    sys.stdout.write('\n')
    sys.stdout.write("Data2: ")
    matlab_print_array(Data2)
    sys.stdout.write('\n')
    sys.stdout.write("Matching: ")
    matlab_print_array(true_matching)