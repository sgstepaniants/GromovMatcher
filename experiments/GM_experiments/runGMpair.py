import numpy as np
from sklearn import preprocessing

from src.GromovMatcher import GM

def runGMpair(config, param, norm = False):
    '''
    Runs GromovMatcher on a pair of simulated datasets

    Parameters
    ----------
    config : TYPE - numpy array
        DESCRIPTION. config = [overlap value, m/z noise value, RT noise value, 
                               FI noise value, drift shape, sim number]
    param : TYPE - numpy array
        DESCRIPTION. param = [m/z tolerance value, RT outlier detection method,
                              number of ourlier detection steps]
    
    norm : TYPE - bool
        DESCRIPTION. Whether to center and scale the data

    Returns
    -------
    RES : TYPE - numpy array
        DESCRIPTION. Coupling matrix

    '''
    mgap, rtfiltr, koutliers = param
    
    verbose = False
    # Set to True to see detailsof the GM run-through
    
    # Upload the 2 simulated datasets to align
    DATA1 = np.load('../DATA/DATA1_'+str(config)+'.npy', allow_pickle=True)
    DATA2 = np.load('../DATA/DATA2_'+str(config)+'.npy', allow_pickle=True)
    
    # Remove the first line that corresponds to met identifiers
    Data1 = DATA1[1:,:]
    Data2 = DATA2[1:,:]
    
    # Data is formatted for GM already, with features in column
    # [0,:] is m/z
    # [1,:] is RT
    # [2:,:] are feature intensities
    
    # FIs log-transformed beforehand
    
    if norm:
        # Center and scale. Falls back to cosine distance
        Data1[2:,:] = preprocessing.scale(Data1[2:,:], axis = 0)
        Data2[2:,:] = preprocessing.scale(Data2[2:,:], axis = 0)
        file_name = "RES/GMcouple_Norm_"+str(config)+"_"+str(param)
    else:
        file_name = "RES/GMcouple_"+str(config)+"_"+str(param)
    
    coupling = GM(Data1, Data2, mgap = mgap, verbose = verbose, RT_fit = 'all', RT_filter = rtfiltr, K_outliers = koutliers)
    
    RES = np.array(coupling)
    np.save(file_name, RES)
    
    return (RES)



runGMpair([0.75, 0.01, 0.5, 0.5,'smooth',1], [0.01,'MAD', 2])
# For loop on the different simulation configurations to get full results