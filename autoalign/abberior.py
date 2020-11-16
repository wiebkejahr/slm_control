'''
Project: deep-sted-autoalign
Created on: Tuesday, 7th January 2020 10:29:32 am
--------
@author: hmcgovern
'''


"""This file acquires a donut-esque image from the open Imspector window, gets its
predicted aberration coefficient weights by feeding it through a trained model, and
saves the important ones in a dictionary, which is passed to the GUI"""

import torch
import numpy as np
import argparse as ap
from skimage.transform import resize, rotate
import matplotlib.pyplot as plt
try:
    import specpy as sp
except:
    print("Specpy not installed!")
    pass
import sys
import time
sys.path.insert(1, 'autoalign/')
sys.path.insert(1, 'parameters/')
import utils.helpers as helpers
import utils.my_models as my_models
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import shift
import torchvision.models as models


def correct_tip_tilt():
    """" Acquires xy image in Imspector, calculates the degree of
         tiptilt by fitting and averaging the CoM in both."""
    
    im = sp.Imspector()
    msr = im.active_measurement()

    try:
        image_xy = msr.stack('ExpControl Ch1 {1}').data() # converts it to a numpy array
    except:
        print("Cannot find 'ExpControl Ch1 {1}' window")
        exit()
    return helpers.calc_tip_tilt(image_xy)

def correct_defocus():
    """" Acquires xz and yz images in Imspector, calculates the degree of
        defocus by fitting and averaging the CoM in both."""
        
    im = sp.Imspector()
    msr = im.active_measurement()

    try:
        image_xz = msr.stack('ExpControl Ch1 {13}').data()
    except:
        print("Cannot find 'ExpControl Ch1 {13}' window")
        exit()
    try:
        image_yz = msr.stack('ExpControl Ch1 {15}').data()
    except:
        print("Cannot find 'ExpControl Ch1 {15}' window")
    return helpers.calc_defocus(image_xz, image_yz)


def get_image(multi=False):
    """" Acquires xy, xz and yz images in Imspector, returns some stats 
        and the active configurations"""
        
    im = sp.Imspector()
    msr = im.active_measurement()
    configuration = msr.active_configuration()

    # acquires xy view
    # configuration names and measurements names are hard coded
        
    x = im.measurement(im.measurement_names()[0])
    im.activate(x)
    try:
        x.activate(x.configuration('xy2d'))
        im.start(x)
        time.sleep(3)
        im.pause(x)
        image_xy = x.stack('ExpControl Ch1 {1}').data()
        stats = [np.max(image_xy), np.min(image_xy), np.std(image_xy)]
        # takes off black edge, resizes to (64, 64) and standardizes
        image_xy = helpers.preprocess(image_xy)
    except:
        print("cannot find xy2d config or 'ExpControl Ch1 {1}' window")
        exit()
    time.sleep(0.5)

    # initialize empty arrays for the other two views
    #image_xz = np.zeros_like(image_xy)
    #image_yz = np.zeros_like(image_xy)    

    if multi:
        # acquires the other two vies and stacks them
        try:
            x.activate(x.configuration('xz2d'))
            im.start(x)
            time.sleep(3)
            im.pause(x)
            image_xz = x.stack('ExpControl Ch1 {13}').data()
            # takes off black edge, resizes to (64, 64) and standardizes
            image_xz = helpers.preprocess(image_xz)
        except:
            print("cannot find xz2d config or 'ExpControl Ch1 {13}' window")
            exit()
        time.sleep(0.5)

        try:
            x.activate(x.configuration('yz2d'))
            im.start(x)
            time.sleep(3)
            im.pause(x)
            image_yz = x.stack('ExpControl Ch1 {15}').data()
            # takes off black edge, resizes to (64, 64) and standardizes
            image_yz = helpers.preprocess(image_yz)
        except:
            print("cannot find yz2d config or 'ExpControl Ch1 {15}' window")  
            exit()
        time.sleep(0.5)
        image = np.stack((image_xy, image_xz, image_yz), axis=0)
    else:
        image = image_xy
    
    # else:
    #     # if not multi: grabs the latest image from Imspector without acquiring
    #     # for xy models: assumption that acquisition is running constantly
    #     # grabs measurment setup, stats etc
    #     try:
    #         image_xy = msr.stack('ExpControl Ch1 {1}').data() # converts it to a numpy array
    #         stats = [np.max(image_xy), np.min(image_xy), np.std(image_xy)]
    #         # takes off black edge, resizes to (64, 64) and standardizes
    #         image = helpers.preprocess(image_xy)
    #     except:
    #         print("Cannot find 'ExpControl Ch1 {1}' window")
    #         exit()

    return image, configuration, msr, stats


def abberior_predict(model_store_path, image, offset=False, multi=False, ii=1):
    
    best_coeffs = []
    best_corr = 0
    for _ in range(ii):

        if multi:
            if offset:
                model = my_models.MultiOffsetNet13()
            else:
                model = my_models.MultiNet11()
        else:
            if offset:
                model = my_models.OffsetNet13()
            else:
                model = my_models.Net11()    

        #NOTE: temporary!
        model = my_models.OffsetNet2()
     
        # gets preds
        checkpoint = torch.load(model_store_path)
        model.load_state_dict(state_dict=checkpoint['model_state_dict'])
        #model = torch.load(model_store_path)

        # Test the model
        model.eval()

        with torch.no_grad():
            # adds 3rd color channel dim and batch dim
            if multi:   
                input_image = torch.from_numpy(image).unsqueeze(0)
            else:
                # NOTE: THIS IS ONLY FOR 1D
                input_image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    
            outputs = model(input_image.float())
            coeffs = outputs.numpy().squeeze()

        
        #NOTE: temporary!
        zern = np.asarray([0.0]*11)
        offset_label = coeffs
        # return coeffs
        # if offset:
        #     zern = coeffs[:-2]
        #     offset_label = coeffs[-2:]
        # else:
        #     zern = coeffs
        #     offset_label = [0,0]
        reconstructed = helpers.get_sted_psf(coeffs=zern, offset_label=offset_label, multi=multi, defocus=False)
        corr = helpers.corr_coeff(image, reconstructed)
        if corr > best_corr:
            best_corr = corr
            best_coeffs = coeffs

    # if offset:
    #     zern = best_coeffs[:-2]
    #     offset_label = best_coeffs[-2:]
    # else:
    #     zern = best_coeffs 
    #     offset_label = [0,0]   
   
    

    return zern, offset_label







if __name__ == "__main__":
    abberior_multi('models/20.05.18_scaling_fix_eps_15_lr_0.001_bs_64_2.pth')
