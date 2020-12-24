'''
Project: deep-sted-autoalign
Created on: Thursday, 31st July 2020 10:47:12 am
--------
@author: hmcgovern
'''

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
from sklearn.metrics import mean_squared_error


def clean_data(path, files, drop = []):
    """" 
         """
    data_clean = []
    data = []
    for file in files:
        with open(path + file, 'r') as f:
            data = json.load(f)
        
        data_clean = {}
        data_clean["gt"] = data["gt"][1::2]
        data_clean["preds"] = data["preds"][1::2]
        data_clean["corr"] = data["corr"]
        data_clean["init_corr"] = data["init_corr"]
        data_clean["gt_zern"] =  data["gt"][0::2]
        data_clean["preds_zern"] = data["preds"][0::2]
        
        with open(path+'clean_'+file, 'w') as f:
            json.dump(data_clean, f, indent = 4)
        


def concat_data(path, files, drop = []):
    """" reads in all provided files and concatenates into one data frame.
         Drops columns specified by inde, eg for manually removing defocussed
         data.
         Use for data from a single experimental run; data from multiple runs
         is best combined from curated data frames (to not mess up indices)
         Parameters: drop: list of indices to drop
                     paths: list of paths to read from.
         """
    df = pd.DataFrame()
    for f in files:
        p = path + f
        df = pd.concat([df, pd.read_json(p)], sort = False)
    df = df.reset_index()
    df = df.drop(drop, axis = 0)
    return df, files[0][:-5]


def plot_data(df, name):
    
    cm_data = np.loadtxt("vik.txt")
    berlin_map = lsc.from_list("berlin", cm_data)

    fig, axes = plt.subplots(nrows = 2, ncols = 2)
    plt.suptitle(name)
    
    
    # plot correlation coefficient after corrections
    #axes[0,0].plot(df["corr"], linestyle = '', marker = '+')
    axes[0,0].scatter(df.index, df["corr"], marker = '+')
    axes[0,0].set_xlabel('Trial')
    axes[0,0].set_ylabel('Correlation Coefficient after correction')
    
    
    #plot correlation coefficient before and after and a line between the two
    improv = df['init_corr'] - df['corr']
    improv = np.uint8((improv - np.min(improv)) / (np.max(improv) - np.min(improv))*256)
    
    axes[0,1].scatter(df.index, df["init_corr"], marker = '+', label = 'initial correlation')
    axes[0,1].scatter(df.index, df["corr"], marker = '+', label = 'final correlation')
    axes[0,1].legend()
    for ii in df.index:
        if df["init_corr"][ii] < df["corr"][ii]:
            axes[0,1].plot([ii, ii], [df["init_corr"][ii], df["corr"][ii]], 
                           color = 'b')
                       #color = berlin_map(np.uint8((improv[ii]/2+1/2)*256)))
        else:
            axes[0,1].plot([ii, ii], [df["init_corr"][ii], df["corr"][ii]], 
                   color = 'r')
    axes[0,1].set_xlabel('Trial')
    axes[0,1].set_ylabel('Improvement of correlation coefficient')
    
    
    df_sorted = df.sort_values(by=['init_corr']).reset_index()
    improv = df_sorted['init_corr'] - df_sorted['corr']
    improv = np.uint8((improv - np.min(improv)) / (np.max(improv) - np.min(improv))*256)
    
    #axes[1,1].scatter(df.index, df_sorted['init_corr'], marker = '+')
    #axes[1,1].scatter(df.index, df_sorted['corr'], marker = '+')
    for ii in range(df.index[-1]+1): 
        axes[1,1].plot([ii, ii], [df_sorted["init_corr"][ii], df_sorted["corr"][ii]], 
                       color = berlin_map(improv[ii]), marker = '+')
                       #color = berlin_map(np.uint8((improv[ii]/2+1/2)*256)))
    axes[1,1].set_xlabel('Trial')
    axes[1,1].set_ylabel('Improvement of correlation coefficient')
    
    
    df_preds = pd.DataFrame(np.abs(np.subtract(df["preds"].to_list(), df["gt"].to_list())), \
        columns = ["45 Astig", "90 Astig", "90 Trefoil", "90 Coma", "0 Coma", "45 Trefoil", "45 Quad", "45 Astig 2", "Spherical", "90 Astig 2", "90 Quad"])
    df_preds.boxplot(ax = axes[1,0])
    axes[1,0].set_ylabel("Difference btw predicted and gt")
    axes[1,0].set_xlabel("Zernike Mode")

    return df_sorted, improv
    


    # df3.plot.scatter(x=df3.index(), y='corr')
#         # # plt.show()
#         # print(df.loc[:, 'gt'].head().to_numpy())
#         # mean_squared_error(df['gt'].to_numpy(), df['preds'].to_numpy())
#         # print(df.head())
#         # plt.figure(2)
#         # plt.plot(df['gt'][0])
#         # plt.plot(df['preds'][0])
#         # plt.ylim(-1,1)
#         # plt.show()

#         # plt.figure(3)
#         # plt.plot(df['gt'][9])
#         # plt.plot(df['preds'][9])
#         # plt.ylim(-1,1)
#         # plt.show()
    plt.show()
        



#if __name__=="__main__":
drop = []
#maybes: 20, 81, 82, 148, 162, 177

path = '/Users/wjahr/Seafile/Synch/Share/Hope/Data_automated/20201012_Autoalign/20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64/'
files = ['20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_640.txt',
         '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_6423.txt',
         '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_6448.txt',
         '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64123.txt']
#path = '/Users/wjahr/Seafile/Synch/Share/Hope/Data_automated/20201124_Autoalign/20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_64/'
#files = ['clean_20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_640.txt']

#clean_data(path, files, drop)
df, model = concat_data(path, files, drop)
df_sorted, bla = plot_data(df, model)








