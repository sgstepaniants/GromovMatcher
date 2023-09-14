import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


def max_fun(couple):
    match = np.zeros(np.shape(couple))
    n2 = np.shape(couple)[1]
    match[np.argmax(couple, axis = 0),np.linspace(0,n2-1,n2,dtype = int)] = np.max(couple, axis = 0)
    return(match)

def PrecRec(prediction, true_coupling):
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

CONFIG = []

for o in [0.25, 0.5, 0.75] : 
    for sigmaM in [0.01]:
        for sigmaRT in [0.2, 0.5, 1]:
            for sigmaFI in [0.1, 0.5, 1]:
                CONFIG.append([o, sigmaM, sigmaRT, sigmaFI, 'smooth'])
            

            
GAPS = [0.01]          
            
def Recap_MC_fun(config, binGap):
    n_samples = np.linspace(1,20,20, dtype = int)
    o, sigM, sigRT, sigFI, d = config
    AUC = 0
    config2 = config.copy()
    config2.append(0)
    for N in n_samples :
        config2[-1] = N
        file_name = 'RES_unsup/binGap'+str(binGap)+'/couple_'+str(config2).replace("'","")+'.npy'
        coupling = np.load(file_name, allow_pickle = True)
        file_name_true = '../DATA/TRUE_'+str(config2)+'.npy'
        true_coupling = np.load(file_name_true, allow_pickle = True)
        true_coupling[true_coupling > 0] = 1
        coupling = max_fun(max_fun(coupling).T).T
        AUC = AUC + PrecRec(np.transpose(coupling), np.transpose(true_coupling))
    [prec, rec] = AUC/len(n_samples)
    return([prec, rec])


CM = []

for config in CONFIG:
    for binGap in GAPS :
        CM.append([*config, binGap, Recap_MC_fun(config, binGap)[0],Recap_MC_fun(config, binGap)[1]])

CM = np.array(CM)


np.save('mC_unsup_PR.npy', CM)



########################################################################################################
#PLOTS
########################################################################################################

## COURBE À SIGMA ET LAMBDA FIXÉ

# overlap = 0.8
# setting = 0
# #sigma = 0.01
# THR = np.linspace(0, 0.00018, 50)

# Mgap = [0.005, 0.01, 0.05]
# RTgap = [1, 5]

# #### FORMAT k, mg, thr, sens, spec

# # COLORS = ['darkmagenta','darkblue','darkturquoise']
# # COLORS = ['black']
# matplotlib.rc('xtick', labelsize=6) 

# fig, axs = plt.subplots(2, 3, sharex='col', sharey='row')
# fig.set_size_inches(15,10)
# fig.subplots_adjust(hspace=0.16, wspace=0.05)
# # i = 0
# j = 0
# for mgap in Mgap:
#     i = 0
#     Recap1 = Recap
#     Recap1 = Recap[Recap[:,0] == mgap]
#     CM1 = CM[CM[:,0] == mgap]
#     for rtg in RTgap :
#         # print(c)
#         x = Recap1[Recap1[:,1] == rtg][:,2].astype(np.float)
#         # print(x)
#         y1 = Recap1[Recap1[:,1] == rtg][:,3].astype(np.float)
#         # print(y1)
#         y2 = Recap1[Recap1[:,1] == rtg][:,4].astype(np.float)
#         # axs[i,j].rc('axes', labelsize=1) 
#         axs[i,j].set_ylim([0.5, 1.05])
#         # axs.set_xlim([0, 0.00018])
#         axs[i,j].plot(x, y1, c = 'darkblue', label = 'rtg ='+str(rtg))
#         axs[i,j].axhline(y = CM1[0][1], c = 'red')
#         axs[i,j].plot(x, y2, c = 'darkblue', ls = 'dashed')
#         axs[i,j].axhline(y = CM1[0][2], ls = 'dashed', c = 'red')
#         # axs[i,j].axvline(x = thresh/len(n_sample),lw = 0.3, c = COLORS[c], ls = 'dotted' )
#         axs[i,j].set_title('m = '+str(mgap)+', rt = '+str(rtg*0.1))
#         i += 1
#     j += 1
# # handles, labels = axs.get_legend_handles_labels()
# # fig.legend(handles, labels, loc='right',fontsize = 'x-small')
# plt.savefig('/scratch/breeurm/Untargeted_mets_matching/Simulated_data/Third_round/SSV2_MRT.png', dpi = 300)




# RecapNorm = np.load('/scratch/breeurm/Untargeted_mets_matching/Simulated_data/Third_round/SS_V2_normalized.npy', allow_pickle = True)
# Recap = np.load('/scratch/breeurm/Untargeted_mets_matching/Simulated_data/Third_round/SS_V2.npy', allow_pickle = True)

# CM = np.load('/scratch/breeurm/Untargeted_mets_matching/Simulated_data/metabCombiner/SS.npy', allow_pickle = True)

########## COURBE À SIGMA ET LAMBDA FIXÉ

# overlap = 0.8
# setting = 0
# #sigma = 0.01
# THR = np.linspace(0, 0.00018, 50)



# Mgap = [0.01]
# RTgap = [5]

# #### FORMAT k, mg, thr, sens, spec

# COLORS = ['darkmagenta','darkblue','darkturquoise']
# # COLORS = ['black']

# fig, axs = plt.subplots(1, 1, sharex='col', sharey='row')
# fig.set_size_inches(10,7)
# fig.subplots_adjust(hspace=0.16, wspace=0.05)
# # i = 0
# j = 0
# rtg = 5
# for mgap in Mgap:
#     i = 0
#     Recap1 = Recap[Recap[:,0] == mgap]
#     # CM1 = CM[CM[:,0] == mgap]
#     x = Recap1[Recap1[:,1] == rtg][:,2].astype(np.float)
#     y1 = Recap1[Recap1[:,1] == rtg][:,3].astype(np.float)
#     y2 = Recap1[Recap1[:,1] == rtg][:,4].astype(np.float)
#     axs.set_ylim([0, 1.05])
#     axs.plot(x, y1, c = 'darkturquoise',label = 'Non-centered')
#     axs.plot(x, y2, c = 'darkturquoise', ls = 'dashed')
#     Recap1Norm = RecapNorm[RecapNorm[:,0] == mgap]
#     # CM1 = CM[CM[:,0] == mgap]
#     xNorm = Recap1Norm[Recap1Norm[:,1] == rtg][:,2].astype(np.float)
#     y1Norm = Recap1Norm[Recap1Norm[:,1] == rtg][:,3].astype(np.float)
#     y2Norm = Recap1Norm[Recap1Norm[:,1] == rtg][:,4].astype(np.float)
#     axs.set_ylim([0, 1.05])
#     axs.plot(xNorm, y1Norm, c = 'darkblue', label = 'Centered')
#     axs.plot(xNorm, y2Norm, c = 'darkblue', ls = 'dashed')
#     axs.set_title('Mgap = '+str(mgap)+', RTgap = 5')
#     j += 1
# handles, labels = axs.get_legend_handles_labels()
# fig.legend(handles, labels, loc='right') # ,fontsize = 'small')
# plt.savefig('/scratch/breeurm/Untargeted_mets_matching/Simulated_data/Third_round/SSV2Norm.png', dpi = 300)

   
