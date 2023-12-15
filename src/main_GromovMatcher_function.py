import numpy as np
import os
import math
import torch

import pandas as pd
from sklearn import metrics
import scipy.interpolate as si
import matplotlib.pyplot as plt

from filtering import *
from ugw_sinkhorn_solver import UGWSinkhornSolver

def max_fun(couple):
    match = np.zeros(np.shape(couple))
    n2 = np.shape(couple)[1]
    match[np.argmax(couple, axis = 0),np.linspace(0,n2-1,n2,dtype = int)] = np.max(couple, axis = 0)
    return(match)

def keep_max(coupling):
    return(max_fun(max_fun(coupling).transpose()).transpose())

def GM(Data1, Data2, D1 = None, D2 = None, w = 'm/z', mgap = 0.01, 
        lmbda = 0, otcost = None, mu = 0.5, quadratic = True,
        RT_pred = True, RT_fit = 'all', RT_filter = 'MAD', inner_filter = 'PI', RT_thr = 0.1, K_outliers = 2,
        output_RT_pred = False,
        verbose = False, plot_list = [], plot_path = None, 
        rho = 5e-2, ent = 5e-3, nits=50, nits_sinkhorn=1000, tol=1e-15, tol_sinkhorn=1e-7, timeout=1200):
    
    '''
    Main function aligning features across datasets.
    
    Data1/2 = input data, shape (n1/n2 + 2)*p1/p2. First row is m/z, second row is RTs, then one row per sample.
    
    
    OPTIMAL TRANSPORT RELATED INPUTS:
    
    
    D1/D2 = distance matrices shape p1*p1/p2*p2 reflecting the spatial organization of the features in dataset 1/2. 
    If None, default euclidean distance matrices will be used.
    
    w = weights constricting the GW matching, either None, 'm/z' or custom p1*p2 matrix.
    None means unconstricted matching, might affect accuracy.
    'm/z' is default option, where the constraint is based on m/z compatibility
    custom weights can be provided.
    
    mgap = m/z tolerance when w is 'm/z'. Default is 0.01, can be set to None if custom weights are provided and if otcost is not 'mixed_hard'.
    
    lmbda = mixture between GW and OT cost, in [0,1]. 0 means GW only, 1 is OT only.
    
    otcost = how is the OT cost computed for lmbda>0. Can be 'RT', 'mixed_hard', 'mixed_soft'
    'RT' is OT cost based on RT dist only.
    'mixed_hard' is a mixture between RT dist and m/z with a hard constraint, i.e if m/z difference > mgap cost is set to 100
    'mixed_soft' is a mixture between RT dist and m/z eucldean dist
    
    mu = governs the OT cost mixture if otcost is set to 'mixed_...', in [0,1]. 0 is RT only, 1 is m/z only.
    
    quadratic = DO NOT TOUCH.
    
    
    RT DRIFT ESTIMATION RELATED INPUTS:
    
    
    RT_pred = whether to predict RT drift and with which model. Can be either None, 'quantiles' or 'CV'
    None means no RT drift prediction will be carried out
    Otherwise 'X' will be passed on to the filtering.SimplifiedWRT_adjustment function.
    
    RT_fit = How the anchor set is computed, can be either 'all', 'max' or 'top'. 'all' takes all the recovered pairs in the matching into account to estimated the RT_drift with weighted b-splines. 'max' only takes into account the best match for each feature. 'top' takes only the top 10% pairs.
    
    RT_filter = whether to filter the coupling based on the estimated RT drift. Can be either 'None', 'MSE', 'MAD', 'mean', 'PI' or 'hard_thr'
    Will be passed on to the filtering.SimplifiedWRT_adjustment function.
    
    RT_thr = hard RELATIVE threshold for RT tolerance around the estimated RT deviation. Use with caution.
    
    
    MISC:
    
    
    verbose = whether to detail the steps during execution of the function
    
    plot_list = list of intermediary plots. Must contain either 'Distance patterns','Weights','Coupling','RT drift', 'Outliers'
    
    plot_path = path to where to put the requested plots. If None, the plots will only be displayed but not saved.
    '''
    
    #Data1 is row 1 : m/z, row 2 : RTS, following rows : samples
    #Transform to the data has to be applied before hand, i.e np.log(1 + Data1[2:,:]) and samples1 = preprocessing.scale(samples1, axis = 0)
    
    if verbose:
        print('Checking arguments:')
        print()

       
    
    ################################### Check dist arg
    
    if (D1 is not None)&(D2 is None):
        raise Exception('Only the distance matrix of dataset 1 has been provided.')
    elif (D1 is None)&(D2 is not None):
        raise Exception('Only the distance matrix of dataset 2 has been provided.')
        
    if verbose:
        if D1 is not None:
            print('Custom distance matrices have been provided for GW. Entropy parameter might misbehave.')
        else:
            print('Using default distance matrices.')


    ################################### Check OT args
    
    if otcost not in [None,'RT','mixed_hard', 'mixed_soft']:
        raise Exception('otcost must be either None, RT, mixed_hard or mixed_soft.')
    
    if (lmbda != 0)&(otcost is None):
        raise Exception('otcost is None but lmbda is not zero. Set otcost to either RT or mixed to compute hybrid cost ((1-lmbda)*GW + lmbda*OT).')
    
    if (otcost == 'mixed_hard')&(mgap is None):
        raise Exception('mgap must be set to compute hard mcost. Try default value 0.01.')
    
    if verbose:
        if (lmbda == 0):#Everything is set for the OT cost
            print('Matching will be computed using GW cost only.')
            if otcost is not None:
                print("Warning: otcost is not None but will not be used due to lmbda = 0.")
        elif lmbda == 1:
            print("Warning: lmbda = 1 drops the GW cost completely, the method's accuracy might be severely impacted.")
            if otcost == 'RT':
                print('Matching will be computed using RT OT cost only.')
            else:
                print('Matching will be computed using a mixed OT cost with mu =', mu)
        else: #0<lmbda<1
            if otcost == 'RT':
                print('Mixing GW and RT OT cost with lmbda =', lmbda)
            else:
                print('Mixing GW and mixed OT cost with lmbda =', lmbda,'and mu =', mu)
                
        
    ################################### Check weight arg
    
    if lmbda != 1:
        # We only need to check the GW related arguments if GW is taken into account
        if w == 'm/z':
            if mgap is None:
                raise Exception('mgap must be set to compute m/z constricted weights. Try default value 0.01.')
            if verbose:
                print('Restricting GW to m/z compatible pairs, using mgap =', mgap)
        elif w is not None:
            #Check dimensions
            if verbose:
                print("Restricting GW with custom weights. Entropy parameter might misbehave.")
            if mgap is not None:
                print("Warning: custom weights are provided but mgap is not None.")
                print("Disregarding the mgap argument for the weights computation.")
        else: #w is None
            if verbose:
                print("Unconstrained GW coupling. This might affect the method's accuracy.")
            
    ################################## Check if RT related args are consistent
    
    
    if output_RT_pred and not RT_pred:
        print('Warning: RT_pred is included in outputs but set to False. Setting output_RT_pred to False as well.')
        output_RT_pred = False
        
    if RT_fit not in ['all', 'max', 'top']:
        raise Exception('RT fit must be either all, max or top.')
        
    if RT_filter not in ['MSE', 'PI', 'MAD', 'hard_thr', 'mean', None]:
        raise Exception('RT filter must be either None, MSE, MAD, mean, PI or hard_thr.')
    
    if inner_filter not in ['PI','MSE','MAD','mean']:
        raise Exception('inner_filter must be either MSE, MAD, mean or PI.')
    
    if RT_filter is not None:
        if (RT_pred is None):
            raise Exception("RT can not be filtered without estimating their deviation. Set RT_pred to either 'quantiles' or 'CV'.")
        if (lmbda != 0):
            print('Warning: lmbda is not null and RT_filter is not None.')
            print('RT might be accounted for twice, depending in the OT cost mixture paremeter mu.')
    
    if verbose:
        if RT_pred:
            if RT_filter is not None:
                print('Estimating RT deviation and filtering. Filtered coupling will be returned.')
            else:
                print('Estimating RT deviation without filtering. Crude coupling will be returned.')
        else:
            print('Crude coupling will be returned.')
        print()
            
    ################################### Check if plot args are consistent
    
    if ('RT drift' in plot_list) & (RT_pred is None):
        print("Warning: RT_pred is None but 'RT drift' is in the plot list")
        print("Removing 'RT drift' from the plot list and carrying on with the execution.")
        plot_list.remove('RT drift') #carry on with the execution without plotting anything.
    
    if ('Outliers' in plot_list) & (RT_pred is None):
        print("Warning: RT_pred is None but 'Outliers' is in the plot list")
        print("Removing 'RT outliers' from the plot list and carrying on with the execution.")
        plot_list.remove('RT outliers') #carry on with the execution without plotting anything.
    
    if ('RT drift' not in plot_list) & (RT_pred is not None) & (not RT_filter):
        print("Warning: RT deviation will be estimated but niether used not plotted.")
        
    if not all(x in ['Distance patterns', 'Weights', 'Coupling','RT drift', 'Outliers'] for x in plot_list):
        print('Warning: Unrecognized plot instructions. Removing the following plots from plot_list:')
        for p in plot_list:
            if p not in ['Distance patterns', 'Weights', 'Coupling','RT drift','Outliers']:
                print('-',p)
        plot_list = np.intersect1d(plot_list, np.array(['Distance patterns', 'Weights', 'Coupling','RT drift','Outliers']))

    
    if plot_path is not None and plot_path != '':
        if(not os.path.exists(plot_path)):
            print('Warning: user-specified plot_path leads to a folder that does not exist.')
            print('Creating the '+plot_path+' folder now.')
            os.makedirs(plot_path)
    
    ################################## Heading into GW
    
    if verbose:
        print()
        print("Going ahead with the GW-based coupling:")
        print()
    
    ms1 = Data1[0,:]
    rs1 = Data1[1,:]
    samples1 = Data1[2:,:]
    
    ms2 = Data2[0,:]
    rs2 = Data2[1,:]
    samples2 = Data2[2:,:]
    
    (n1, num_mets1) = np.shape(samples1)
    (n2, num_mets2) = np.shape(samples2)
    
    # Compute pairwise distances
    if D1 is None:
        #Means that D2 as weel
        D1 = metrics.pairwise_distances(samples1.T) / math.sqrt(n1)
        D2 = metrics.pairwise_distances(samples2.T) / math.sqrt(n2)
        # normalize distance matrices
        const = np.sqrt(np.sqrt(np.mean(D1**2)) * np.sqrt(np.mean(D2**2)))
        D1 = D1 / const
        D2 = D2 / const
        if verbose:
            print('Default normalized euclidian distance matrices computed.')
    else:
        if verbose:
            print('Custom distance matrices are set.')
    
    if 'Distance patterns' in plot_list:
        plt.figure()
        plt.title('D1', fontweight='bold')
        plt.matshow(D1)
        plt.savefig(plot_path+'D1', dpi = 300)
        plt.figure()
        plt.title('D2', fontweight='bold')
        plt.matshow(D2)
        if plot_path is not None:
            plt.savefig(plot_path+'D2', dpi = 300)
    
    mu1 = np.ones(num_mets1) / num_mets1
    mu2 = np.ones(num_mets2) / num_mets2
    
    # mass constraint and otcost
    if w == 'm/z':
        ms12 = np.abs(ms1[:, np.newaxis] - ms2)
        ms12_dists = np.array([[100 if x > mgap else 0 for x in row] for row in ms12])
        weights = np.exp(ms12_dists)
        weights=torch.from_numpy(weights)
        if verbose:
            print("Default weights computed.")
    elif w is not None:
        weights = w
        weights=torch.from_numpy(weights)
        if verbose:
            print("Custom weights are set.")
    else:
        weights = w
        
    if 'Weights' in plot_list:
        plt.figure()
        plt.title('Weights', fontweight='bold')
        plt.pcolormesh(weight)
        plt.xlabel('metabolites 2')
        plt.ylabel('metabolites 1')
        plt.colorbar()
        if plot_path is not None:
            plt.savefig(plot_path+'Weights', dpi = 300)
        
    
    if lmbda != 0:
        if otcost == 'RT':
            otcost = np.abs(rs1[:, np.newaxis] - rs2)#RT gaps
        elif otcost == 'mixed_hard':
            ms12_dists = np.array([[100 if x > mgap else 0 for x in row] for row in np.abs(ms1[:, np.newaxis] - ms2)])
            rs12_dists = np.abs(rs1[:, np.newaxis] - rs2)
            otcost = mu*ms12_dists + (1-mu)*rs12_dists
        else:# mcost == 'soft':
            ms12_dists = np.abs(ms1[:, np.newaxis] - ms2)
            rs12_dists = np.abs(rs1[:, np.newaxis] - rs2)
            otcost = mu*ms12_dists + (1-mu)*rs12_dists
        otcost /= np.sqrt(np.mean(otcost**2))
        otcost=torch.from_numpy(otcost)
    else:
        otcost = None
    
    if verbose:
        if otcost is None:
            print('Matching cost is set to Gromov-Wasserstein only.')
        elif quadratic:
            print('Mixing Gromov-Wasserstein cost with a quadratic optimal transportation cost on rentention times.')
        else:
            print('Mixing Gromov-Wasserstein cost with a linear optimal transportation cost on rentention times.')
    
    solver = UGWSinkhornSolver(nits=nits, nits_sinkhorn=nits_sinkhorn, 
                               tol=tol, tol_sinkhorn=tol_sinkhorn, timeout=timeout)
    
    GW_coupling, _ = solver.ugw_sinkhorn(a = torch.from_numpy(mu1),Cx =  torch.from_numpy(D1),
                                      b = torch.from_numpy(mu2), Cy = torch.from_numpy(D2),
                                      rho = rho, eps = ent,
                                      lmbda=lmbda, otcost=otcost, quadratic=quadratic,
                                      weights=weights)
    
    coupling = GW_coupling.numpy()
    
    if np.any(np.isnan(coupling)):
            raise Exception(
                f"Solver got NaN plan with params (ent, rho) "
                f" = {ent, rho}. Try increasing argument ent."
            )
    
    
    if verbose:
        print('GW coupling successfully computed.')
        print(np.count_nonzero(coupling), 'pairs recovered in the crude coupling.')
    
    if 'Coupling' in plot_list:
        plt.figure()
        plt.title('Crude coupling', fontweight='bold')
        plt.pcolormesh(np.log(coupling))
        plt.xlabel('metabolites 2')
        plt.ylabel('metabolites 1')
        plt.colorbar()
        if plot_path is not None:
            plt.savefig(plot_path+'Coupling.png', dpi=300)
               
    ################################## RT deviation estimation and filtering
    
    if RT_pred:
        
        if verbose:
            print()
            print('Estimating RT deviation with model =', RT_pred, ':')
            
        if 'Outliers' in plot_list:
            p = True
        else:
            p = False
            
        if RT_fit == 'all':
            adj_spl, thr,_,_,_ = WRT_adjustment(rs1, rs2, coupling, filtr = RT_filter, inner_filtr = inner_filter, K_outliers = K_outliers, plot = p, plot_path = plot_path)
        elif RT_fit == 'max':
            pseudo_coupling = max_fun(max_fun(coupling.T).T)
            adj_spl, thr,_,_,_ = WRT_adjustment(rs1, rs2, pseudo_coupling, filtr = RT_filter, inner_filtr = inner_filter, K_outliers = K_outliers, plot = p, plot_path = plot_path)
        else: #RT_fit == 'top'
            pseudo_coupling = coupling.copy()
            thr = np.quantile(pseudo_coupling[np.nonzero(pseudo_coupling)], .9)
            pseudo_coupling[pseudo_coupling<thr] = 0
            adj_spl, thr,_,_,_ = WRT_adjustment(rs1, rs2, pseudo_coupling, filtr = RT_filter, inner_filtr = inner_filter, K_outliers = K_outliers, plot = p, plot_path = plot_path)
            
               
        if 'RT drift' in plot_list :
            
            x = np.array(sorted(rs1))
            y = si.splev(x, adj_spl)
            plt.figure(figsize = (5,3),dpi = 300)
            plt.title('RT drift', fontweight='bold')
            plt.scatter(rs1, rs2, s = 0.1, c = 'black')
            plt.plot(x,y,c = 'blue', lw = 0.8)
            plt.xlabel('RTs in dataset 1 (min)')
            plt.ylabel('RTs in dataset 2 (min)')
            if plot_path is not None:
                plt.savefig(plot_path+'RT_drift.png', dpi = 300)
            
        
        if verbose:
            print('Done.')
            print()
        
        if RT_filter is not None:
            
            if verbose:
                print('Filtering the crude coupling based on the estimated RT deviation:')
            
            pairs = np.nonzero(coupling)
            rt1 = rs1[pairs[0]]
            rt2 = rs2[pairs[1]]
            rt2_pred = si.splev(rt1, adj_spl)
            
            if RT_filter == 'PI':
                low_pred, upper_pred = thr
                sel = (rt2>low_pred) & (rt2<upper_pred)
            elif RT_filter == 'MSE':
                error = (rt2_pred - rt2)**2
                sel = error<thr
            elif RT_filter == 'MAD':
                error = np.sqrt((rt2_pred - rt2)**2)
                sel = error<thr
            else: #RT_filter == 'hard_thr'
                error = np.abs(rt2_pred - rt2)
                error = error/((rt2_pred + rt2)/2)
                sel = error < RT_thr
    
            pairs = np.array(pairs)[:,sel]
    
            couple = np.zeros(np.shape(coupling))
            couple[pairs[0,:],pairs[1,:]] = coupling[pairs[0,:],pairs[1,:]]
            
            if verbose:
                print('Done.')
                print(np.count_nonzero(couple), 'pairs in the filtered coupling.')
    
    else:
        couple = coupling

    couple = couple/np.max(couple)
    
    if output_RT_pred:
        return(couple, adj_spl)
    else:
        return(couple)



def link_feature(Data1, Data2, coupling, one_to_one = True):
    '''Transforms the GM output into  list of pairs with their associated coupling coefficient
    
    Input: 
    
    Data1/2 = input data, shape (n1/n2 + 2)*p1/p2. First row is m/z, second row is RTs, then one row per sample. both numpy arrays and pandas dataframes are supported.
    If numpy arrays are provided, since they do not contain feature names, each feature will be attributed a name in the form of "m/z@RT"
    
    coupling: coupling matrix (numpy array of size (p1, p2)) as output by GM
    
    one_to_one: (bool) whether to keep the most likely match for each feature
    '''
    if one_to_one:
        coupling = max_fun(max_fun(coupling).transpose()).transpose()
    if isinstance(Data1, np.ndarray):
        index1 = np.apply_along_axis(lambda d: str(round(d[0], 4)) + '@' + str(round(d[1], 4)), 0, Data1[0:2,:])
    else:
        index1 = np.array(Data1.columns)
    if isinstance(Data2, np.ndarray):
        index2 = np.apply_along_axis(lambda d: str(round(d[0], 4)) + '@' + str(round(d[1], 4)), 0, Data2[0:2,:])
    else:
        index2 = np.array(Data2.columns)
    
    couple = pd.DataFrame(coupling, index = index1, columns = index2)
    pair_list = pd.DataFrame(couple[couple != 0].stack())
    
    return(pair_list)
