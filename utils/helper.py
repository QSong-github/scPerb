import random
import os
import numpy as np
import torch

import scanpy as sc
import pandas as pd
from scipy import stats
import scipy.stats as st
import seaborn as sns
import matplotlib as mpl 
import matplotlib.pyplot as plt
import os

import scanpy as sc


def make_average(df):
    dfT = df.T
    dfT['Index'] = dfT.index
    new_df = dfT.reset_index().pivot_table(columns=['index'])
    return new_df

def keep_first(df):
    dfT = df.T
    dfT = dfT.loc[~dfT.index.duplicated(keep='first')].T
    return dfT

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cal_ssim(im1,im2,M):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    
    return ssim

# TODO remove zeros

def scale_max(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content/content.max()
        result = pd.concat([result, content],axis=1)
    return result

def scale_z_score(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = st.zscore(content)
        content = pd.DataFrame(content,columns=[label])
        result = pd.concat([result, content],axis=1)
    return result


def scale_plus(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content/content.sum()
        result = pd.concat([result,content],axis=1)
    return result

def logNorm(df):
    df = np.log1p(df)
    df = st.zscore(df)
    return df
    
class CalculateMeteics:
    def __init__(self, raw_count_file, impute_count_file, prefix, metric, name, read_file=False):

        self.impute_count_file = impute_count_file
        if read_file:
            self.raw_count = pd.read_csv(raw_count_file, header=0, index_col=0).T
        self.raw_count = raw_count_file
        self.raw_count.columns = [x.upper() for x in self.raw_count.columns]
        # self.raw_count = self.make_average(self.raw_count)
        self.raw_count = self.raw_count.T
        self.raw_count = self.raw_count.loc[~self.raw_count.index.duplicated(keep='first')].T
        self.raw_count = self.raw_count.fillna(1e-20)
        # idx = self.raw_count > 0
        
        if read_file:
            self.impute_count = pd.read_csv(impute_count_file, header = 0, index_col = 0)
        self.impute_count = impute_count_file
        self.impute_count.columns = [x.upper() for x in self.impute_count.columns]
        # self.impute_count = self.make_average(self.impute_count)
        self.impute_count = self.impute_count.T
        self.impute_count = self.impute_count.loc[~self.impute_count.index.duplicated(keep='first')].T
        self.impute_count = self.impute_count.fillna(1e-20)
        # self.impute_count[self.raw_count==0] = 0.0

        self.prefix = prefix
        self.metric = metric
        self.name = name

        print(self.raw_count.shape, self.impute_count.shape)

    def SSIM(self, raw, impute, scale = 'scale_max'):
        if scale == 'scale_max':
            raw = scale_max(raw)
            impute = scale_max(impute)
        else:
            print ('Please note you do not scale data by scale max')
        # print(raw.shape, impute.shape)
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    ssim = 0
                else:
                    raw_col =  raw.loc[:,label]
                    impute_col = impute.loc[:,label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    M = [raw_col.max(),impute_col.max()][raw_col.max()>impute_col.max()]
                    raw_col_2 = np.array(raw_col)
                    raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0],1)
                    impute_col_2 = np.array(impute_col)
                    impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0],1)
                    ssim = cal_ssim(raw_col_2,impute_col_2,M)
                
                ssim_df = pd.DataFrame(ssim, index=["SSIM"],columns=[label])
                result = pd.concat([result, ssim_df],axis=1)
        else:
            print("columns error")
        return result
            
    def PCC(self, raw, impute, scale = None):
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    pearsonr = 0
                else:
                    raw_col =  raw.loc[:,label]
                    impute_col = impute.loc[:,label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    try: 
                        pearsonr, _ = st.pearsonr(raw_col,impute_col)
                    except:
                        pearsonr = 0
                pearson_df = pd.DataFrame(pearsonr, index=["PCC"],columns=[label])
                result = pd.concat([result, pearson_df],axis=1)
        else:
            print("columns error")
        return result
    
    def JS(self, raw, impute, scale = 'scale_plus'):
        if scale == 'scale_plus':
            raw = scale_plus(raw)
            impute = scale_plus(impute)
        else:
            print ('Please note you do not scale data by plus')    
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    JS = 1
                else:
                    raw_col =  raw.loc[:,label]
                    impute_col = impute.loc[:,label]
                    raw_col = raw_col.fillna(1e-20)
                    impute_col = impute_col.fillna(1e-20)
                    M = (raw_col.to_numpy() + impute_col.to_numpy())/2
                    # print(raw_col.shape, M.shape)
                    # print(impute_col.shape, M.shape)
                    try:
                        JS = 0.5*st.entropy(raw_col,M)+0.5*st.entropy(impute_col,M)
                    except:
                        JS = 1 
                    
                JS_df = pd.DataFrame(JS, index=["JS"],columns=[label])
                result = pd.concat([result, JS_df],axis=1)
        else:
            print("columns error")
        return result
    
    def RMSE(self, raw, impute, scale = 'zscore'):
        if scale == 'zscore':
            raw = scale_z_score(raw)
            impute = scale_z_score(impute)
        else:
            print ('Please note you do not scale data by zscore')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    RMSE = 1.5   
                else:
                    raw_col =  raw.loc[:,label]
                    impute_col = impute.loc[:,label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())

                RMSE_df = pd.DataFrame(RMSE, index=["RMSE"],columns=[label])
                result = pd.concat([result, RMSE_df],axis=1)
        else:
            print("columns error")
        return result       
        
    def compute_all(self):
        raw = self.raw_count
        impute = self.impute_count

        # print(impute.index)
        # print(raw.index)

        prefix = self.prefix
        print('SSIM')
        SSIM = self.SSIM(raw,impute)
        print('PCC')
        Pearson = self.PCC(raw, impute)
        print('JS')
        JS = self.JS(raw, impute)
        print('RMSE')
        RMSE = self.RMSE(raw, impute)
        
        result_all = pd.concat([Pearson, SSIM, RMSE, JS],axis=0)
        save_path = os.path.join(prefix, self.name+"_Metrics.txt")
        # print(save_path)
        result_all.T.to_csv(save_path, sep='\t', header = 1, index = 1)
        self.accuracy = result_all
        return result_all