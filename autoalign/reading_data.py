'''
Project: deep-sted-autoalign
Created on: Thursday, 31st July 2020 10:47:12 am
--------
@author: hmcgovern
'''
#import slm_control.Pattern_Calculator as pc
import utils.my_classes as my_classes
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
from matplotlib.patches import Circle
#from sklearn.metrics import mean_squared_error
from skimage import filters
from skimage.measure import regionprops
#from pyoformats import read

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def clean_stats(path, files):
    """"
         """
    import json

    data_clean = {}
    data = {}
    for file in files:
        with open(path + file, 'r') as f:
            data = json.load(f)

        # for offset only model, 20201124, NEEDS TO BE UPDATED
        data_clean = {}
        data_clean["gt_off"] = data["gt"][1::2]
        data_clean["preds_off"] = data["preds"][1::2]
        data_clean["corr"] = data["corr"]
        data_clean["init_corr"] = data["init_corr"]
        data_clean["gt_zern"] =  data["gt"][0::2]
        data_clean["preds_zern"] = data["preds"][0::2]

        #for 11dim model, 20201012
        # data_clean["gt_off"] = [[0,0] for z in range(np.shape(data["gt"])[0])]
        # data_clean["gt_preds"] = data_clean["gt_off"]
        # data_clean["gt_zern"] = data["gt"]
        # data_clean["preds_zern"] = data["preds"]
        # data_clean["corr"] = data["corr"]
        # data_clean["init_corr"] = data["init_corr"]


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
        print(p)
        df = pd.concat([df, pd.read_json(p)], sort = False)
    df = df.reset_index()
    return df, files[0][:-5]


def read_msr(fname, series):
    data_xy = np.squeeze(read.image_5d(path + fname, series=series[0]))[2:,2:]
    data_xz = np.squeeze(read.image_5d(path + fname, series=series[1]))[2:,2:]
    data_yz = np.squeeze(read.image_5d(path + fname, series=series[2]))[2:,2:]
    data = [data_xy, data_xz, data_yz]
    return data

def get_CoMs(img):
    if len(img) == 3:
        _, x_shape, y_shape = np.shape(img)

        ####### xy ########
        b, a = get_CoM(img[0])
        dx_xy = ((x_shape-1)/2-a)
        dy_xy = ((y_shape-1)/2-b)

        ####### xz ########
        b, a = get_CoM(img[1])
        dx_xz = ((x_shape-1)/2-a)
        dz_xz = ((y_shape-1)/2-b)

        ######## yz #########
        b, a = get_CoM(img[2])
        dy_yz = ((x_shape-1)/2-a)
        dz_yz = ((y_shape-1)/2-b)

        d_xyz = np.asarray([np.average([dx_xy, dx_xz]),
                            np.average([dy_xy, dy_yz]),
                            np.average([dz_xz, dz_yz])])
        #d_xyz = np.asarray([dx_xy, dy_xy, np.average([dz_xz, dz_yz])])
    elif len(img) == 2:
        x_shape, y_shape = np.shape(img)
        b, a = get_CoM(img)
        d_xyz = np.asarray[((x_shape-1)/2-a),
                           ((y_shape-1)/2-b),
                           0]
    return d_xyz


def get_CoM(img):
    #TODO: return offsets in nm instead of absolute positons in px
    #    dx = (x_shape-1)/2-a
    #    dy = (y_shape-1)/2-b
    # then rewrite calc_tip_tilt, calc_defocus and automate accordingly

    threshold_value = filters.threshold_otsu(img)
    labeled_foreground = (img > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, img)
    center_of_mass = properties[0].centroid
    b = center_of_mass[0]
    a = center_of_mass[1]
    return b,a


def plot_data(df, model):

    cm = plt.get_cmap("coolwarm")

    ##########################################################################
    #            plot correlation coefficient after corrections              #
    ##########################################################################
    fig, axes = plt.subplots(nrows = 1, ncols = 1)
    #plt.suptitle(model)

    axes.scatter(df.index, df["corr"], marker = '+')
    axes.set_xlabel('Trial')
    axes.set_ylabel('Correlation Coefficient after correction')
    fig.savefig("corr_coeff.pdf", transparent = True)


    ##########################################################################
    #      plot improvement of correlation coefficient after corrections     #
    #              plot residiual centering error for comparison             #
    ##########################################################################
    fig, axes = plt.subplots(3, sharex = True)
    improv = df['corr'] - df['init_corr']
    improv = 1 - ((improv / (np.abs(np.max(improv) + np.max(improv)))) + 0.5)
    axes[0].scatter(df.index, df["init_corr"], marker = '+', label = 'initial correlation')
    axes[0].scatter(df.index, df["corr"], marker = '+', label = 'final correlation')
    axes[0].legend()

    for ii in df.index:
        if df["init_corr"][ii] < df["corr"][ii]:
            axes[0].plot([ii, ii], [df["init_corr"][ii], df["corr"][ii]],
                           color = 'b')
            axes[1].plot(ii, np.sqrt(np.sum(np.asarray(df["CoM_correct"][ii])**2)), marker = 'x', color = "tab:blue")
        else:
            axes[0].plot([ii, ii], [df["init_corr"][ii], df["corr"][ii]],
                   color = 'r')
            axes[1].plot(ii, np.sqrt(np.sum(np.asarray(df["CoM_correct"][ii])**2)), marker = 'x', color = "tab:orange")
        if ii != 0:
            axes[2].plot(ii, df["CoM_aberr"][ii][0], marker = '+', color = "tab:blue")
            axes[2].plot(ii, df["CoM_aberr"][ii][1], marker = '+', color = "tab:orange")
            axes[2].plot(ii, df["CoM_aberr"][ii][2], marker = '+', color = "tab:green")
        else:
            axes[2].plot(ii, df["CoM_aberr"][ii][0], marker = '+', color = "tab:blue", label = "CoM xy")
            axes[2].plot(ii, df["CoM_aberr"][ii][1], marker = '+', color = "tab:orange", label = "CoM xz")
            axes[2].plot(ii, df["CoM_aberr"][ii][2], marker = '+', color = "tab:green", label = "CoM yz")

    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Improvement of correlation coefficient')
    axes[1].set_ylabel('residual centering error')
    axes[2].set_ylabel('residual centering error')
    axes[2].legend()
    fig.savefig("corr_coeff_improv_unsorted.pdf", transparent = True)

    df_sorted = df.sort_values(by=['init_corr']).reset_index()
    improv = df_sorted['corr'] - df_sorted['init_corr']
    improv = 1 - ((improv / (np.abs(np.max(improv) + np.max(improv)))) + 0.5)


    ##########################################################################
    #      plot improvement of correlation coefficient after corrections     #
    #              plot residiual centering error for comparison             #
    #           for data sorted wrt initial correlation coefficient          #
    ##########################################################################
    fig, axes = plt.subplots(3, sharex = True)
    for ii in range(df.index[-1]+1):
        axes[0].plot([ii, ii], [df_sorted["init_corr"][ii], df_sorted["corr"][ii]],
                      color = cm(improv[ii]), marker = '+')
        axes[1].plot(ii, np.sqrt(np.sum(np.asarray(df_sorted["CoM_aberr"][ii])**2)), marker = 'x', color = cm(improv[ii]))
        if ii != 0:
            axes[2].plot(ii, df_sorted["CoM_aberr"][ii][0], marker = '+', color = "tab:blue")
            axes[2].plot(ii, df_sorted["CoM_aberr"][ii][1], marker = '+', color = "tab:orange")
            axes[2].plot(ii, df_sorted["CoM_aberr"][ii][2], marker = '+', color = "tab:green")
        else:
            axes[2].plot(ii, df_sorted["CoM_aberr"][ii][0], marker = '+', color = "tab:blue", label = "CoM xy")
            axes[2].plot(ii, df_sorted["CoM_aberr"][ii][1], marker = '+', color = "tab:orange", label = "CoM xz")
            axes[2].plot(ii, df_sorted["CoM_aberr"][ii][2], marker = '+', color = "tab:green", label = "CoM yz")
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Improv CC')
    axes[1].set_ylabel('centering before')
    axes[2].set_ylabel('centering before')
    axes[2].legend()
    fig.savefig("corr_coeff_improv_sorted_before.pdf", transparent = True)

    fig, axes = plt.subplots(3, sharex = True)
    for ii in range(df.index[-1]+1):
        axes[0].plot([ii, ii], [df_sorted["init_corr"][ii], df_sorted["corr"][ii]],
                      color = cm(improv[ii]), marker = '+')
        axes[1].plot(ii, np.sqrt(np.sum(np.asarray(df_sorted["CoM_correct"][ii])**2)), marker = 'x', color = cm(improv[ii]))
        if ii != 0:
            axes[2].plot(ii, df_sorted["CoM_correct"][ii][0], marker = '+', color = "tab:blue")
            axes[2].plot(ii, df_sorted["CoM_correct"][ii][1], marker = '+', color = "tab:orange")
            axes[2].plot(ii, df_sorted["CoM_correct"][ii][2], marker = '+', color = "tab:green")
        else:
            axes[2].plot(ii, df_sorted["CoM_correct"][ii][0], marker = '+', color = "tab:blue", label = "CoM xy")
            axes[2].plot(ii, df_sorted["CoM_correct"][ii][1], marker = '+', color = "tab:orange", label = "CoM xz")
            axes[2].plot(ii, df_sorted["CoM_correct"][ii][2], marker = '+', color = "tab:green", label = "CoM yz")
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Improv CC')
    axes[1].set_ylabel('centering after')
    axes[2].set_ylabel('centering after')
    axes[2].legend()
    fig.savefig("corr_coeff_improv_sorted_after.pdf", transparent = True)

    plt.show()


    fig, axes = plt.subplots()
    p = df["preds_zern"].to_list()
    gt = df["gt_zern"].to_list()
    df_preds = pd.DataFrame(np.abs(np.subtract(p, gt)),
                            columns = ["45 Astig", "90 Astig", "90 Trefoil",
                                       "90 Coma", "0 Coma", "45 Trefoil",
                                       "45 Quad", "45 Astig 2", "Spherical",
                                       "90 Astig 2", "90 Quad"])
    for c_name in df_preds.columns:
        c = np.abs(df_preds[c_name])
        print(c.mean())
        #mrse
        #c.mse()
    df_preds.boxplot(ax = axes, rot = 90)
    axes.set_ylabel("Difference btw predicted and gt")
    axes.set_xlabel("Zernike Mode")
    fig.savefig("boxplot_labels.pdf", transparent = True)

    return fig, axes




#if __name__=="__main__":
drop = []
#maybes: 20, 81, 82, 148, 162, 177



##############################################################################
#                    list of files for different runs                        #
##############################################################################
# path = '/Users/wjahr/Seafile/Synch/Share/Hope/Data_automated/20201012_Autoalign/20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64/'
# files = ['20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_640.txt',
#          '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_6423.txt',
#          '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_6448.txt',
#          '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64123.txt']
# clean_stats(path, files)

# path = '/Users/wjahr/Seafile/Synch/Share/Hope/Data_automated/20201124_Autoalign/20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_64/'
# files = ['20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_640.txt',
#          '20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_6432.txt',
#          '20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_6479.txt']
# clean_stats(path, files)

# path = '/Users/wjahr/Seafile/Synch/Share/Hope/Data_automated/20210104_Autoalign/20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64/'
# files = ['20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_640.txt']#,
#          # '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_6438.txt',
#          # '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_6479.txt',
#          # '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64177.txt',
#          # '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64245.txt']
# #clean_stats(path, files)

# df, model = read_stats(path, files)

# imgs_aberr = []
# CoMs_aberr = []
# imgs_correct = []
# CoMs_correct = []
# phase_aberr = []
# phase_corr = []


# orders = [[1,-1],[1,1],[2,0],
#           [2,-2],[2,2],[3,-3],[3,-1],[3,1],[3,3],
#           [4,-4],[4,-2],[4,0],[4,2],[4,4]]

# size = [64,64]

# circ = create_circular_mask(size[0], size[1])

# for ii in range(3):#df.index:
#     img_aberr = read_msr(str(ii)+"_aberrated.msr", [2,5,8])
#     CoM_aberr = get_CoMs(img_aberr)
#     imgs_aberr.append(img_aberr)
#     CoMs_aberr.append(CoM_aberr)

#     img_correct = read_msr(str(ii)+"_corrected.msr", [2,5,8])
#     CoM_correct = get_CoMs(img_correct)
#     imgs_correct.append(img_correct)
#     CoMs_correct.append(CoM_correct)

#     ph_aberr = (pc.zern_sum(size, df["gt_zern"][ii], orders[3:]))
#     ph_corr = (pc.zern_sum(size, df["preds_zern"][ii], orders[3:]))
#     phase_aberr.append(ph_aberr)
#     phase_corr.append(ph_corr)

#     ph_aberr[~circ] = np.nan
#     ph_corr[~circ] = np.nan
#     fig = plt.figure()
#     minmax = [np.min(img_correct[0]), np.max(img_correct[0])]
#     plt.subplot(331); plt.axis('off')
#     plt.imshow(img_aberr[0], clim = minmax, cmap = 'inferno')
#     plt.subplot(332); plt.axis('off')
#     plt.imshow(img_aberr[1], clim = minmax, cmap = 'inferno')
#     plt.subplot(333); plt.axis('off')
#     plt.imshow(img_aberr[2], clim = minmax, cmap = 'inferno')
#     plt.subplot(334); plt.axis('off')
#     plt.imshow(img_correct[0], clim = minmax, cmap = 'inferno')
#     plt.subplot(335); plt.axis('off')
#     plt.imshow(img_correct[1], clim = minmax, cmap = 'inferno')
#     plt.subplot(336); plt.axis('off')
#     plt.imshow(img_correct[2], clim = minmax, cmap = 'inferno')
#     minmax = ([np.min([np.nanmin(ph_aberr), np.nanmin(ph_corr)]),
#                np.max([np.nanmax(ph_aberr), np.nanmax(ph_corr)])])
#     mm_center = np.max(np.abs(minmax))
#     print(minmax)
#     plt.subplot(337); plt.axis('off')
#     plt.imshow(ph_aberr, clim = minmax, cmap = 'RdBu')
#     plt.subplot(338); plt.axis('off')
#     plt.imshow(ph_corr, clim = minmax, cmap = 'RdBu')
#     plt.subplot(339); plt.axis('off')
#     plt.imshow((ph_corr - ph_aberr), clim = minmax, cmap = 'RdBu')
#     fig.savefig(str(ii) + "_thumbnail.png")


# df["img_aberr"] = imgs_aberr
# df["CoM_aberr"] = CoMs_aberr
# df["img_correct"] = imgs_correct
# df["CoM_correct"] = CoMs_correct
# df["phase_aberr"] = phase_aberr
# df["phase_corr"] = phase_corr


# df = df.drop(drop, axis = 0)
# #path = '/Users/wjahr/Seafile/Synch/Share/Hope/Data_automated/20201124_Autoalign/20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_64/'
# #files = ['clean_20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_640.txt']


# fig, axes = plot_data(df, model)




data_path_11 = 'autoalign/datasets/20.08.03_1D_centered_18k_norm_dist.hdf5'
data_path_13 = 'autoalign/datasets/20.10.22_1D_centered_18k_norm_dist_offset_no_noise.hdf5'

dataset = my_classes.PSFDataset(hdf5_path = data_path_11, mode = 'val') # get validation data set

sample = dataset.getitem()