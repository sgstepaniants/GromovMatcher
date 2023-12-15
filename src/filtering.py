import numpy as np
import statistics
import scipy.interpolate as si
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
import scipy.stats as stats
import matplotlib.pyplot as plt


def WRT_adjustment(RT1, RT2, couple, filtr = 'MSE', K_outliers=0, inner_filtr = 'PI', 
                   kn = 10, k_iter = 5, plot=False, plot_path = ''):
    
    '''RT drift prediction function.
    Uses quantiles for the inner outlier detection, CV for the final fit.
    
    RT1 (resp. RT2) 1*N1 (resp. 1*N2) arrays with N1 (resp.N2) the number of features in dataset 1 (resp. dataset 2) 
    couple = N1*N2 array with coupling probabilities as computed from GW
    filtr = 'MSE' (mean-squared error) or 'PI' (prediction interval), outlier detection method
    K_outliers = number of internal filtering steps. Outliers are discarded at each of thos steps to improve fit.
    inner_filtr = 'MSE' or 'PI', internal outlier detection method.
    plot = None, 'RT_drift' or 'All' to plot either nothing, the final drift estimation, or the internal outlier detection steps + final estimation.
    
    '''
    
    #Select coupled pairs with associated weight 
    pairs = np.nonzero(couple)
    w = couple.copy()
    w = w/np.max(w)
    w = w[pairs]
    rt1 = RT1[pairs[0]]
    rt2 = RT2[pairs[1]]
    #Order them 
    order = np.argsort(rt1)
    x = rt1[order]
    y = rt2[order]
    w = w[order]
    
    x1 = x.copy()
    y1 = y.copy()
    w1 = w.copy()
    
    # Discard outliers
    for i in range(K_outliers):
        knots = np.linspace(0,1,kn + 1, endpoint = False)[1:]
        #print(knots)
        spl = si.splrep(x1, y1, w=w1, t=np.quantile(x1,knots))
        pred = si.splev(x1, spl)
        if inner_filtr == 'MSE':
            error = (pred - y1)**2
            mse = metrics.mean_squared_error(y_pred=pred, y_true=y1)
            sel = error<mse
        elif inner_filtr == 'WMSE':
            ### Not yet proofed, not accessible with default settings
            error = (pred - y1)**2
            mse = metrics.mean_squared_error(y_pred=pred, y_true=y1, 
                                         sample_weight=w1, multioutput = 'uniform_average')
            sel = error<mse
        elif inner_filtr == 'MAD':
            error = np.sqrt((pred - y1)**2)
            mad = np.median(error) + 2*stats.median_abs_deviation(error)
            sel = error<mad
        elif inner_filtr == 'mean':
            error = np.sqrt((pred - y1)**2)
            mad = np.mean(error)
            sel = error<mad
            thr = mad
        else: #inner_filtr = 'PI':
            std = np.std(pred - y1)
            low_pred=pred-1.96*std
            upper_pred=pred+1.96*std
            sel = (y1>low_pred) & (y1<upper_pred)
            
        if plot:
            y = si.splev(x1, spl)
            col = np.where(sel,'black','red')
            colors = ['red', 'black']
            plt.figure(figsize = (5,3),dpi = 200)
            plt.title('Outlier detection, step '+str(i), fontweight='bold')
            plt.scatter(x1, y1, s = 0.1, c=col)
            plt.plot(x1,y,c = 'blue', lw = 0.8)
            plt.show()
            plt.savefig(plot_path+'Outliers_step'+str(i+1)+'.png', dpi = 300)
            
        x1 = x1[sel]
        y1 = y1[sel]
        w1 = w1[sel]
    
    k_fold = KFold(n_splits=k_iter)
    lambs = np.linspace(start = 4, stop = 10, num = 7)
    mse = []
    for i in lambs:
        knots = np.linspace(0,1,int(i) + 1, endpoint = False)[1:]
        error = []
        for trg, tst in k_fold.split(x1):
            spl = si.splrep(x1[trg], y1[trg], w=w1[trg], t=np.quantile(x1[trg],knots))
            pred = si.splev(x1[tst],spl)
            true = y1[tst]
            error.append(metrics.mean_squared_error(y_pred=pred, y_true=true))
        mse.append(statistics.mean(error))
    lamb = lambs[np.argmin(mse)]
    knots = np.linspace(0,1,int(lamb) + 1, endpoint = False)[1:]
    adj_spl = si.splrep(x1, y1, w=w1, t = np.quantile(x1,knots))
    pred = si.splev(x1, adj_spl)
    if filtr == 'MSE':
        error = (pred - y1)**2
        mse = metrics.mean_squared_error(y_pred=pred, y_true=y1)
        sel = error<mse
        thr = mse
    elif filtr == 'WMSE':
            ### Not yet proofed, not accessible with default settings
        error = (pred - y1)**2
        mse = metrics.mean_squared_error(y_pred=pred, y_true=y1, 
                                         sample_weight=w1, multioutput = 'uniform_average')
        sel = error<mse
        thr = mse
    elif filtr == 'MAD':
        error = np.sqrt((pred - y1)**2)
        mad = np.median(error) + 2*stats.median_abs_deviation(error)
        sel = error<mad
        thr = mad
    elif filtr == 'mean':
        error = np.sqrt((pred - y1)**2)
        mad = np.mean(error)
        sel = error<mad
        thr = mad
    else: #filtr = 'PI':
        std = np.std(pred - y1)
        low_pred=pred-1.96*std
        upper_pred=pred+1.96*std
        sel = (y1>low_pred) & (y1<upper_pred)
        thr = [low_pred, upper_pred]
    
    if plot:
        y = si.splev(x1, adj_spl)
        col = np.where(sel,'black','red')
        colors = ['red', 'black']
        plt.figure(figsize = (5,3),dpi = 200)
        plt.title('Final step', fontweight='bold')
        plt.scatter(x1, y1, s = 0.1, c=col)
        plt.plot(x1,y,c = 'blue', lw = 0.8)
        plt.show()
        plt.savefig(plot_path+'Outliers_finalStep.png', dpi = 300)
    
    x1 = x1[sel]
    y1 = y1[sel]
    w1 = w1[sel]
        
    return(adj_spl, thr, x1, y1, w1)