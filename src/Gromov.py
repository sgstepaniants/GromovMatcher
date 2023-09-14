from function import *
import numpy as np
import pyreadr
import pandas as pd
from importlib import reload
import function

reload(function)

class Gromov:
    
    REALPATH= "./data/"
    RAWPATH= "./Raw_data/"
    GROMOV_RES_PATH= "./Gromov/RES/"
    GROMOV_INTERPRETATION_PATH= "./Gromov/features_link/"
    GROMOV_COUPLING_PATH= "./Gromov/coupling/"
    GROMOV_MERGE_PATH= "./Gromov/merge/"
    RHO = 5e-2
    ENT = 5e-3  
    
    def __init__(self, dataset1, dataset2):
        try:
            self.names=["feature1", "feature2"]
            self.coupling=None
            #load dataframe 1
            self.dataframe1=self.loadFile(dataset1)
            #load dataframe 2
            self.dataframe2=self.loadFile(dataset2)
            #indexes 
            self.index1= self.dataframe1.iloc[:2, :].apply(lambda x: f"{x[0]}@{x[1]}") if self.dataframe1.columns.values[0]==0 else self.dataframe1.columns.values
            self.index2= self.dataframe2.iloc[:2, :].apply(lambda x: f"{x[0]}@{x[1]}") if self.dataframe2.columns.values[0]==0 else self.dataframe2.columns.values
        except TypeError:
            print("Check your dataframes' path")
        except:
            print("please check arguments and parameters")
            raise Exception("check the parameters")
            
    #load file in init   
    def loadFile(self, db, index_col=0):
        ext=db.split(".")[-1]
        if(ext=="npy"):
            return pd.DataFrame(np.load(db, allow_pickle = True))
        elif (ext=="rds"): 
            return pyreadr.read_r(db)[None]
        elif ext=="csv":
            return pd.read_csv(db, index_col=index_col)
        else:
            raise Exception("file must be csv, npy or rds")
    
    
    def run_gromov(self, mgap = 0.01, rtfiltr = 'MAD', K_outliers = 2, names=['Study 1', 'Study2'], normalized= True, verbose=True, plot_list= ['Outliers','RT drift'], save_couple=False):
        '''
            normalized => boolean: normalize your data or not 
            save_couple => boolean: save couples as csv
            names => list[2 String(s)]  the names of differents study (used for the saving name too)
            verbose => boolean : show the step of the OT algo
        '''
        self.names=names
        #saving path
        save= "_".join(names).replace(" ","_")
        dt1= self.dataframe1.values.astype("float64")
        dt2= self.dataframe2.values.astype("float64")
        dt1[2:,:]= applyLog(dt1[2:,:])
        dt2[2:,:]= applyLog(dt2[2:,:])
        if normalized:
            dt1[2:,:]= simple_scale(dt1[2:,:])
            dt2[2:,:]= simple_scale(dt2[2:,:])
        self.coupling = testTLB(Data1=dt1, Data2=dt2, mgap = mgap, RT_filter = rtfiltr, K_outliers = K_outliers, 
                   verbose = verbose, plot_list =plot_list, rho = self.RHO, ent = self.ENT,
                   plot_path = self.GROMOV_RES_PATH+save+'/', plot_labels = names)
        self.coupling= pd.DataFrame(self.coupling)
        if save_couple:
            self.coupling.to_csv(self.GROMOV_COUPLING_PATH+save+".csv")
        return (self.coupling) 
    
    
    
    def link_feature(self, one_to_one = True, save_name=None):
        #check if coupling is set
        if self.coupling is None:
            print("please run the coupling first")
            return  
        #name dataset (with row of base1, col of base2)
        df= denomination(self.coupling, self.index1, self.index2, one_to_one)
        result=[]
        cols= df.columns
        #pi maximum
        max_pi= np.max(np.max(self.coupling))
        #for each columns
        for c in cols:
            #get id of non nul index per columns
            ids=[]
            #get the columns mz and rt
            mz2,rt2=c.split("@")
            #get the non null index related 
            frame= df.loc[df[c]!=0]
            if frame.shape[0]!=0.0:
                index= frame.index
                #if one_to_one is True just keep the id of the biggest coupling coef
                if one_to_one:
                    id=np.argmax(df.loc[index,c])
                    ids.append(id)
                #else keep all couples
                else:
                    ids.extend([i for i in range(len(index))])
                #create the row for each couple
                for val in ids:
                    ind= index[val]
                    P=df.loc[ind, c]
                    #normalisation of each coef
                    P= P/max_pi
                    #get the index mz and rt
                    mz1,rt1= ind.split("@")
                    #and then add related feature names and their relation value
                    result.append([ind, mz1, rt1, c, mz2, rt2, P])
                #loop again on empty ids
                ids=[]
        #transform into dataframe
        dataframe= pd.DataFrame(result, columns=[self.names[0], "mz1", "RT1", self.names[1], "mz2", "RT2", "coupling coefficient"])
        if save_name is not None:
            save_name= save_name.split(".")[0]
            dataframe.to_csv(self.GROMOV_INTERPRETATION_PATH+save_name+".csv")
        return dataframe

    
    
    def merge_base(self, dataset1=None, dataset2=None, one_to_one = True, save_name=None, keep=True, drop_duplicates=True, i=0):
        #features' link
        result= self.link_feature(one_to_one, save_name)
        if result is None:
            return 
        #get features' names
        ft1, ft2 = result[self.names[i%2]], result[self.names[(i+1)%2]]
        #nettoyage
        dataset1= self.loadFile(dataset1, None)
        dataset2= self.loadFile(dataset2, None)
        dataset1, dataset2= drop_specific_columns(dataset1), drop_specific_columns(dataset2)
        #get shape
        i1, j1= dataset1.shape
        i2, j2= dataset2.shape
        col= np.max([j1, j2])
        idx= i2 + i1
        columns= list(dataset1.columns)
        col= len(columns)
        na= np.full((idx, col), np.nan)
        db= pd.DataFrame(na, columns=columns)
        db.loc[:i1-1, dataset1.columns]= dataset1
        db.loc[i1:, ft1]= dataset2.loc[:, ft2].values
        db.loc[i1:, "Samplename"]= dataset2.Samplename.values
        db["Study"]= [self.names[i%2]]*i1 + [self.names[(i+1)%2]]*i2
        if not keep:
            db.dropna(inplace=True, axis=1)
        if drop_duplicates:
            db= db.drop_duplicates(subset=[x for x in db.columns if x!="Samplename"], keep='first')
        if save_name is not None:
            save_name= save_name.split(".")[0]
            db.to_csv(self.GROMOV_MERGE_PATH+save_name+".csv")
        return db
    
    
    def merge2(self, dataset1=None, dataset2=None, one_to_one = True, save_name=None, keep=True, drop_duplicates=True, i=0):
        #load datas
        dataset1= self.loadFile(dataset1, None)
        dataset2= self.loadFile(dataset2, None)
        #features' link
        result= self.link_feature(one_to_one, save_name)
        if result is None:
            return 
        #get features' by pair 
        ft= result[[self.names[i%2], self.names[(i+1)%2]]]
        cols= to_dict(ft.values)
        dt2= dataset2.rename(columns=cols)
        db= pd.concat([dataset1, dt2], ignore_index=True)
        #get shape
        i1, j1= dataset1.shape
        i2, j2= dt2.shape
        db["Study"]= [self.names[i%2]]*i1 + [self.names[(i+1)%2]]*i2
        if not keep:
            db.dropna(inplace=True, axis=1)
        if drop_duplicates:
            db= db.drop_duplicates(subset=[x for x in db.columns if x!="Samplename"], keep='first')
        if save_name is not None:
            save_name= save_name.split(".")[0]
            db.to_csv(self.GROMOV_MERGE_PATH+save_name+".csv")
        return db
    
    
    
        
        
        
        
        
        
        
        
    
    
        
    
    
        
        
    
        
    
    
        
        