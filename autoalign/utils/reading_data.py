'''
Project: deep-sted-autoalign
Created on: Thursday, 31st July 2020 10:47:12 am
--------
@author: hmcgovern
'''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
#from sklearn.metrics import mean_squared_error
from skimage import filters
from skimage.measure import regionprops
from pyoformats import read

def clean_stats(path, files):
    """" 
         """
    import json
    
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


def read_stats(path, files):
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
    return df, files[0][:-5]

        
def read_msr(fname, series): 
    data_xy = np.squeeze(read.image_5d(path + fname, series=series[0]))[2:,2:]
    data_xz = np.squeeze(read.image_5d(path + fname, series=series[1]))[2:,2:]
    data_yz = np.squeeze(read.image_5d(path + fname, series=series[2]))[2:,2:]
    data = [data_xy, data_xz, data_yz]
    return data


def calc_CoM(img):
    threshold_value = filters.threshold_otsu(img)
    labeled_foreground = (img > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, img)
    center_of_mass = properties[0].centroid
    b = center_of_mass[0]
    a = center_of_mass[1]
    return b,a


def get_CoMs(img):
    _, x_shape, y_shape = np.shape(img)
    ####### xy ########
    b, a = calc_CoM(img[0])
    dx_xy = ((x_shape-1)/2-a)#*1e-9*px_size  # convert to m
    dy_xy = ((y_shape-1)/2-b)#*1e-9*px_size  # convert to m

    ####### xz ########
    b, a = calc_CoM(img[1])
    dx_xz = ((x_shape-1)/2-a)#*1e-9*px_size  # convert to m
    dz_xz = ((y_shape-1)/2-b)#*1e-9*px_size  # convert to m
    
    ######## yz #########
    b, a = calc_CoM(img[2])
    dy_yz = ((x_shape-1)/2-a)#*1e-9*px_size  # convert to m
    dz_yz = ((y_shape-1)/2-b)#*1e-9*px_size  # convert to m
    
    dx = np.average([dx_xy, dx_xz])
    dy = np.average([dy_xy, dy_yz])
    dz = np.average([dz_xz, dz_yz])
    
    return [dx, dy, dz]



def plot_data(df, model):
    
    cm_data = np.loadtxt("vik.txt")
    berlin_map = lsc.from_list("berlin", cm_data)

    fig, axes = plt.subplots(nrows = 2, ncols = 2)
    plt.suptitle(model)
    
    
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
    ax = axes[0,1].twinx()
    
    for ii in df.index:
        if df["init_corr"][ii] < df["corr"][ii]:
            axes[0,1].plot([ii, ii], [df["init_corr"][ii], df["corr"][ii]], 
                           color = 'b')
                       #color = berlin_map(np.uint8((improv[ii]/2+1/2)*256)))
            ax.plot(ii, np.sqrt(np.sum(np.asarray(df["CoM_correct"][ii])**2)), marker = 'x', color = "tab:blue")
        else:
            axes[0,1].plot([ii, ii], [df["init_corr"][ii], df["corr"][ii]], 
                   color = 'r')
            ax.plot(ii, np.sqrt(np.sum(np.asarray(df["CoM_correct"][ii])**2)), marker = 'x', color = "tab:orange")
        ax.set_ylim([0, 40])
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
    
    ## plotting only cell
    fig2, axes = plt.subplots(3, sharex = True)
    ax = axes[0].twinx()
    for ii in range(df.index[-1]+1): 
        axes[0].plot([ii, ii], [df_sorted["init_corr"][ii], df_sorted["corr"][ii]], 
                      color = berlin_map(improv[ii]), marker = '+')
        ax.plot(ii, np.sqrt(np.sum(np.asarray(df_sorted["CoM_aberr"][ii])**2)), marker = 'x', color = "tab:green")
        ax.set_ylim([0, 40])
        axes[1].plot(ii, np.sqrt(np.sum(np.asarray(df_sorted["CoM_aberr"][ii])**2)), marker = 'x', color = berlin_map(improv[ii]))
        if ii != 0:
            axes[2].plot(ii, df_sorted["CoM_aberr"][ii][0], marker = '+', color = "tab:blue")
            axes[2].plot(ii, df_sorted["CoM_aberr"][ii][1], marker = '+', color = "tab:orange")
            axes[2].plot(ii, df_sorted["CoM_aberr"][ii][2], marker = '+', color = "tab:green")
        else:
            axes[2].plot(ii, df_sorted["CoM_aberr"][ii][0], marker = '+', color = "tab:blue", label = "CoM xy")
            axes[2].plot(ii, df_sorted["CoM_aberr"][ii][1], marker = '+', color = "tab:orange", label = "CoM xz")
            axes[2].plot(ii, df_sorted["CoM_aberr"][ii][2], marker = '+', color = "tab:green", label = "CoM yz")
    axes[2].legend()
    
    fig3, axes = plt.subplots(3, sharex = True)
    ax = axes[0].twinx()
    for ii in range(df.index[-1]+1): 
        axes[0].plot([ii, ii], [df_sorted["init_corr"][ii], df_sorted["corr"][ii]], 
                      color = berlin_map(improv[ii]), marker = '+')
        ax.plot(ii, np.sqrt(np.sum(np.asarray(df_sorted["CoM_correct"][ii])**2)), marker = 'x', color = berlin_map(improv[ii]))
        ax.set_ylim([0, 40])
        axes[1].plot(ii, np.sqrt(np.sum(np.asarray(df_sorted["CoM_correct"][ii])**2)), marker = 'x', color = "tab:green")
        if ii != 0:
            axes[2].plot(ii, df_sorted["CoM_correct"][ii][0], marker = '+', color = "tab:blue")
            axes[2].plot(ii, df_sorted["CoM_correct"][ii][1], marker = '+', color = "tab:orange")
            axes[2].plot(ii, df_sorted["CoM_correct"][ii][2], marker = '+', color = "tab:green")
        else:
            axes[2].plot(ii, df_sorted["CoM_correct"][ii][0], marker = '+', color = "tab:blue", label = "CoM xy")
            axes[2].plot(ii, df_sorted["CoM_correct"][ii][1], marker = '+', color = "tab:orange", label = "CoM xz")
            axes[2].plot(ii, df_sorted["CoM_correct"][ii][2], marker = '+', color = "tab:green", label = "CoM yz")
    axes[2].legend()
    
    plt.show()

    return fig, axes
        



#if __name__=="__main__":
drop = []
#maybes: 20, 81, 82, 148, 162, 177

path = '/Users/wjahr/Seafile/Synch/Share/Hope/Data_automated/20201012_Autoalign/20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64/'
files = ['20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_640.txt',
         '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_6423.txt',
         '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_6448.txt',
         '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64123.txt']


#clean_stats(path, files)
df, model = read_stats(path, files)

imgs_aberr = []
CoMs_aberr = []
imgs_correct = []
CoMs_correct = []

for ii in df.index:
    img_aberr = read_msr(str(ii)+"_aberrated.msr", [2,5,8])
    CoM_aberr = get_CoMs(img_aberr)
    imgs_aberr.append(img_aberr)
    CoMs_aberr.append(CoM_aberr)
    
    img_correct = read_msr(str(ii)+"_corrected.msr", [2,5,8])
    CoM_correct = get_CoMs(img_correct)
    imgs_correct.append(img_correct)
    CoMs_correct.append(CoM_correct)
    

df["img_aberr"] = imgs_aberr
df["CoM_aberr"] = CoMs_aberr
df["img_correct"] = imgs_correct
df["CoM_correct"] = CoMs_correct


df = df.drop(drop, axis = 0)
#path = '/Users/wjahr/Seafile/Synch/Share/Hope/Data_automated/20201124_Autoalign/20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_64/'
#files = ['clean_20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_640.txt']


fig, axes = plot_data(df, model)







