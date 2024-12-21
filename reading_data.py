'''
Project: deep-sted-autoalign
Created on: Thursday, 31st July 2020 10:47:12 am
--------
@author: hmcgovern
'''
import os
import glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap as lsc
#from sklearn.metrics import mean_squared_error
from skimage import filters
from skimage.measure import regionprops
from pyoformats import read
import tifffile

import slm_control.Pattern_Calculator as pc
import autoalign.utils.vector_diffraction as vd

params_sim = {
    "optical_params_sted": {
        "n": 1.518, 
        "NA": 1.4, 
        "f": 1.8, 
        "transmittance": 0.74, 
        "lambda": 775, 
        "P_laser": 0.25, 
        "rep_rate": 40000000.0, 
        "pulse_length": 7e-10, 
        "obj_ba": 5.04,
        "px_size": 10, 
        "offset": [0, 0]}, 
    "optical_params_gauss": {
        "n": 1.518, 
        "NA": 1.4, 
        "f": 1.8, 
        "transmittance": 0.84, 
        "lambda": 640, 
        "P_laser": 0.000125, 
        "rep_rate": 40000000.0, 
        "pulse_length": 1e-10, 
        "obj_ba": 5.04, 
        "px_size": 10,
        "offset": [0, 0]},
    "numerical_params": {
        "out_scrn_size" : 1,
        "z_extent" : 1,
        "out_res" : 64, 
        "inp_res" : 64,
        "orders" : [[1,-1],[1,1],[2,0],[2,-2],[2,2],
                    [3,-3],[3,-1],[3,1],[3,3],
                    [4,-4],[4,-2],[4,0],[4,2],[4,4]]}
            }

def find_txt(path):
    files = []
    for file in glob.glob(path + '*.txt'):
        files.append(file)
    return files

def get_CoMs(img):
    
    if len(img.shape) == 3:
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
    elif len(img.shape) == 2:
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

def calc_groundtruth(scale_size = 1.1):

    opt_props = params_sim["optical_params_sted"]
    num_props = params_sim["numerical_params"]
    # left handed circular
    polarization = [1.0/np.sqrt(2), 1.0/np.sqrt(2)*1j, 0]
    num_props["out_res"] = np.uint8(scale_size * num_props["out_res"])
    num_props["out_scrn_size"] = scale_size * num_props["out_scrn_size"]
    num_props["z_extent"] = scale_size * num_props["z_extent"]

    lp_scale_sted = vd.calc_lp(opt_props["P_laser"], 
                               opt_props["rep_rate"], 
                               opt_props["pulse_length"])
    size = np.asarray([num_props["inp_res"], num_props["inp_res"]])
    vortex = pc.crop(pc.create_donut(2*size, 0, 1, radscale = 2), size, [0,0])
    zerns = pc.crop(pc.zern_sum(2*size, np.zeros(11), num_props["orders"][3::], radscale = 2), size, [0,0])
    phasemask = pc.add_images([vortex, zerns])
    amp = np.ones_like(phasemask)
    
#        def correct_aberrations(size, ratios, orders, off = [0,0], radscale = 1):
    [xy, xz, yz, xyz] = vd.vector_diffraction(
        opt_props, num_props, 
        polarization, phasemask, amp, lp_scale_sted, plane = 'all', 
        offset=opt_props['offset'])
    
    #groundtruth = np.uint8((xyz - np.min(xyz)) / (np.max(xyz) - np.min(xyz)) * 255)
    return [xy, xz, yz]

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
        # data_clean = {}
        # data_clean["gt_off"] = data["gt"][1::2]
        # data_clean["preds_off"] = data["preds"][1::2]
        # data_clean["corr"] = data["corr"]
        # data_clean["init_corr"] = data["init_corr"]
        # data_clean["gt_zern"] =  data["gt"][0::2]
        # data_clean["preds_zern"] = data["preds"][0::2]
        
        #for 11dim model, 20201012
        data_clean["gt_off"] = [[0,0] for z in range(np.shape(data["gt"])[0])]
        data_clean["gt_preds"] = data_clean["gt_off"]
        data_clean["gt_zern"] = data["gt"]
        data_clean["preds_zern"] = data["preds"]
        data_clean["corr"] = data["corr"]
        data_clean["init_corr"] = data["init_corr"]

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
        #p = path + f
        #print(p)
        df = pd.concat([df, pd.read_json(f)], sort = False)
    df = df.reset_index()
    return df, files[0][:-5]

def read_msr(path, fname, series): 
    data_xy = np.squeeze(read.image_5d(path + fname, series=series[0]))
    data_xz = np.squeeze(read.image_5d(path + fname, series=series[1]))
    data_yz = np.squeeze(read.image_5d(path + fname, series=series[2]))
    data = [data_xy, data_xz, data_yz]
    return data


def read_img2df(df, path, file = 'msr'):
    imgs_aberr = []
    imgs_correct = []
    CoMs_aberr = []
    CoMs_correct = []
    phase_aberr = []
    phase_correct = []
    phase_rms = []
    
    orders = [[1,-1],[1,1],[2,0],
              [2,-2],[2,2],[3,-3],[3,-1],[3,1],[3,3],
              [4,-4],[4,-2],[4,0],[4,2],[4,4]]
    size = [64,64]
    circ = create_circular_mask(size[0], size[1])
    df.drop(df.tail(1).index, 
        inplace = True)
    for ii in df.index:
        # read and append aberrated images and CoMs
        if file == 'msr':
            img_aberr = np.asarray(read_msr(path, str(ii) + "_aberrated.msr", 
                                            [2,5,8]))[:, 1:-1, 1:-1]
            img_correct = np.asarray(read_msr(path, str(ii) + "_corrected.msr", 
                                              [2,5,8]))[:, 1:-1, 1:-1]
        elif file == 'tif':
            img_aberr = np.asarray(tifffile.imread(path + str(ii) + "_aberrated.tif"))
            img_correct = np.asarray(tifffile.imread(path + str(ii) + "_corrected.tif"))
        CoM_aberr = get_CoMs(img_aberr)
        imgs_aberr.append(img_aberr)
        CoMs_aberr.append(CoM_aberr)
        
        # read and append corrected images and CoMs
        CoM_correct = get_CoMs(img_correct)
        imgs_correct.append(img_correct)
        CoMs_correct.append(CoM_correct)
        
        # read Zernikes, calculate and append aberrated phasemasks
        ph_aberr = (pc.zern_sum(size, df["gt_zern"][ii], orders[3:]))
        ph_corr = (pc.zern_sum(size, df["preds_zern"][ii], orders[3:]))
        ph_aberr[~circ] = np.nan
        ph_corr[~circ] = np.nan
        
        ph_diff = ph_aberr - ph_corr
        phase_aberr.append(ph_aberr)
        phase_correct.append(ph_corr)
        phase_rms.append(np.sqrt(np.nanmean(ph_diff**2) - (np.nanmean(ph_diff))**2))
        
    # write everything into df
    df["img_aberr"] = imgs_aberr
    df["CoM_aberr"] = CoMs_aberr
    df["img_correct"] = imgs_correct
    df["CoM_correct"] = CoMs_correct
    df["phase_aberr"] = phase_aberr
    df["phase_corr"] = phase_correct
    df["phase_rms"] = phase_rms
    
    return df

def replot_psf_phase(df, path, size = [64,64]):
    """ Plot aberrated and corrected images as well as phasemasks from the
        dataframe.
        """
    
    for ii in df.index:
        # shorthand for df entries
        img_correct = df.img_correct[ii]
        img_aberr = df.img_aberr[ii]
        ph_aberr = df.phase_aberr[ii]
        ph_corr = df.phase_corr[ii]
        
        fig = plt.figure()
        minmax = [np.min(img_correct[0]), np.max(img_correct[0])]
        plt.subplot(331); plt.axis('off')
        plt.imshow(img_aberr[0], clim = minmax, cmap = 'inferno')
        plt.subplot(332); plt.axis('off')
        plt.imshow(img_aberr[1], clim = minmax, cmap = 'inferno')
        plt.subplot(333); plt.axis('off')
        plt.imshow(img_aberr[2], clim = minmax, cmap = 'inferno')
        plt.subplot(334); plt.axis('off')
        plt.imshow(img_correct[0], clim = minmax, cmap = 'inferno')
        plt.subplot(335); plt.axis('off')
        plt.imshow(img_correct[1], clim = minmax, cmap = 'inferno')
        plt.subplot(336); plt.axis('off')
        plt.imshow(img_correct[2], clim = minmax, cmap = 'inferno')
        
        minmax = ([np.min([np.nanmin(ph_aberr), np.nanmin(ph_corr)]),
                    np.max([np.nanmax(ph_aberr), np.nanmax(ph_corr)])])
        mm_center = [-np.max(np.abs(minmax)), np.max(np.abs(minmax))]
        
        plt.subplot(337); plt.axis('off')
        plt.imshow(ph_aberr, clim = mm_center, cmap = 'RdBu')
        plt.subplot(338); plt.axis('off')
        plt.imshow(ph_corr, clim = mm_center, cmap = 'RdBu')
        plt.subplot(339); plt.axis('off')
        plt.imshow((ph_corr - ph_aberr)/mm_center[1], clim = [-1,1], cmap = 'RdBu')
        fig.savefig(path + str(ii) + "_thumbnail.png")
        
        

def plot_corrcoeff(df, path):
    ##########################################################################
    #            plot correlation coefficient after corrections              #
    ##########################################################################
    fig, axes = plt.subplots()
    
    axes.scatter(df.index, df["corr"], marker = '+', color = 'tab:blue')
    ax = axes.twinx()
    ax.scatter(df.index, df["phase_rms"], marker = '+', color = 'tab:orange')
    axes.set_xlabel('Trial')
    axes.set_ylabel('Correlation Coefficient')
    ax.set_ylabel('wavefront RMS')
    fig.savefig(path + "corr_coeff.pdf", transparent = True)
    return fig, axes


def plot_sorted(df, path):
    cm = plt.get_cmap("coolwarm")
    df_sorted = df.sort_values(by=['init_corr']).reset_index()
    improv = df_sorted['corr'] - df_sorted['init_corr']
    improv = 1 - ((improv / (np.abs(np.max(improv) + np.max(improv)))) + 0.5)
    
    fig, axes = plt.subplots(3, sharex = True)
    for ii in range(df.index[-1]+1): 
        axes[0].plot([ii, ii], [df_sorted["init_corr"][ii], df_sorted["corr"][ii]], 
                      color = cm(improv[ii]), marker = '+')
        axes[1].plot(ii, np.sqrt(np.sum(np.asarray(df_sorted["CoM_correct"][ii])**2)), marker = 'x', color = cm(improv[ii]))
        axes[2].plot(ii, df.phase_rms[ii], marker = '+', color = cm(improv[ii]))
    axes[2].set_xlabel('Trial')
    axes[0].set_ylabel('Improv CC')
    axes[1].set_ylabel('centering after')
    axes[2].set_ylabel('phase rms')
    fig.savefig(path + "improv_centering_phaserms.pdf", transparent = True)


def plot_sorted_allCoM(df, path):
    ##########################################################################
    #      plot improvement of correlation coefficient after corrections     #
    #              plot residiual centering error for comparison             #
    #           for data sorted wrt initial correlation coefficient          #
    ##########################################################################
    
    cm = plt.get_cmap("coolwarm")
    df_sorted = df.sort_values(by=['init_corr']).reset_index()
    improv = df_sorted['corr'] - df_sorted['init_corr']
    improv = 1 - ((improv / (np.abs(np.max(improv) + np.max(improv)))) + 0.5)

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
    axes[2].set_xlabel('Trial')
    axes[0].set_ylabel('Corr Coeff diff')
    axes[1].set_ylabel('CoM error before')
    axes[2].set_ylabel('CoM error')
    axes[2].legend()
    fig.savefig(path + "corr_coeff_improv_sorted_before.pdf", transparent = True)
    
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
    axes[2].set_xlabel('Trial')
    axes[0].set_ylabel('Corr Coeff diff')
    axes[1].set_ylabel('CoM error')
    axes[2].set_ylabel('CoM error')
    axes[2].legend()
    fig.savefig(path + "corr_coeff_improv_sorted_after.pdf", transparent = True)
        
def plot_stats(df, path):
    ##########################################################################
    #      plot box plots with improvements for each zernike polynomial      #
    ##########################################################################
    
    fig, axes = plt.subplots(nrows = 1, ncols = 2, gridspec_kw={'width_ratios': [11, 3]})
    df_preds = pd.DataFrame(np.abs(np.subtract( df.preds_zern.to_list(), df.gt_zern.to_list())),
                            columns = ["45 Astig", "90 Astig", "90 Trefoil", 
                                       "90 Coma", "0 Coma", "45 Trefoil", 
                                       "45 Quad", "45 Astig 2", "Spherical", 
                                       "90 Astig 2", "90 Quad"])
    df_off = pd.DataFrame(np.abs(np.subtract(df.preds_off.to_list(), df.gt_off.to_list())),
                          columns = ["off x", "off y"])
    df_off["phase rms"] = df.phase_rms
    c = 'k'
    df_preds.boxplot(ax = axes[0], rot = 90, 
                     color=dict(boxes=c, whiskers=c, medians=c, caps=c),
                     flierprops=dict(markeredgecolor=c)
                     )
    axes[0].set_ylabel("Difference btw predicted and gt")
    axes[0].set_xlabel("Zernike Mode")
    
    df_off.boxplot(ax = axes[1], rot = 90, 
                     color=dict(boxes=c, whiskers=c, medians=c, caps=c),
                     flierprops=dict(markeredgecolor=c)
                     )
    axes[1].set_ylabel("residual relative offset")
    plt.subplots_adjust(bottom=0.25)

    fig.savefig(path + "boxplot_labels.pdf", transparent = True)

#if __name__=="__main__":
drop = []
#maybes: 20, 81, 82, 148, 162, 177



##############################################################################
#                    list of files for different runs                        #
##############################################################################
# path = '/Users/wjahr/Seafile/Synch/Share/Hope/Data_automated/20201012_Autoalign/20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64/'
# files = ['clean_20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_640.txt',
#           'clean_20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_6423.txt',
#           'clean_20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_6448.txt',
#           'clean_20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64123.txt']
# clean_stats(path, files)


# path = '/Users/wjahr/Seafile/Synch/Share/Hope/Data_automated/20201124_Autoalign/20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_64/'
# files = ['20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_640.txt',
#          '20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_6432.txt',
#          '20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_6479.txt']
# clean_stats(path, files)

#path = 'E:/Data_eval/20210104_Autoalign/20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64/'
#path = "E:/Data_eval/20210106_Autoalign/20.10.22_3D_centered_18k_norm_dist_offset_no_noise_eps_15_lr_0.001_bs_64/"

path = 'E:/Data_eval/20210215_Autoalign/20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64/'
#path = 'E:/Data_eval/20210215_Autoalign/20.10.22_3D_centered_18k_norm_dist_offset_no_noise_eps_15_lr_0.001_bs_64/'

files = find_txt(path)
df, model = read_stats(path, files)

read_img2df(df, path, file = 'tif')

path = path + '/eval/'
try:
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(path):
        os.mkdir(path)
except:
    print("couldn't create directory!")
    
plot_stats(df, path)
replot_psf_phase(df, path)
plot_corrcoeff(df, path)
plot_sorted(df, path)
plot_sorted_allCoM(df, path)
plot_stats(df, path)


# df = df.drop(drop, axis = 0)


# with plt.rc_context({'axes.edgecolor':'white', 
#                      'xtick.color':'white', 'ytick.color':'white', 
#                      'axes.labelcolor': 'white',
#                      'figure.facecolor':'white'}):
#     fig, axes = plot_data(df, model)

gt = calc_groundtruth(scale_size = 1.0)
# print("gt shape ", np.shape(gt))




##############################################################################
#           for analyzing validation datasets, not done yet                  #
##############################################################################

# data_path11 = 'autoalign/datasets/20.08.03_1D_centered_18k_norm_dist.hdf5'
# data_path13 = 'autoalign/datasets/20.10.22_3D_centered_18k_norm_dist_offset_no_noise.hdf5'

# dataset = helpers.PSFDataset(hdf5_path = data_path13, mode = 'val')

# sample = []
# for ii in range(dataset.__len__()):
#     sample.append(dataset.__getitem__(idx = ii))
