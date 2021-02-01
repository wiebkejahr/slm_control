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
#import utils.my_models as my_models
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

def get_config():
    im = sp.Imspector()
    msr_names = im.measurement_names()
    msr = im.active_measurement()
    config = msr.active_configuration()
    return im, msr_names, msr, config

def grab_image(msr, window = 'ExpControl Ch1 {1}'):
    # if not multi: grabs the latest image from Imspector without acquiring
    # for xy models: assumption that acquisition is running constantly
    # grabs measurment setup, stats etc
    #im = sp.Imspector()
    #msr = im.active_measurement()
    try:
        img = msr.stack(window).data() # converts it to a numpy array
        stats = [np.max(img), np.min(img), np.std(img)]
        # takes off black edge, resizes to (64, 64) and standardizes
        img_p = helpers.preprocess(img)
    except:
        print("Cannot find ", window, " window")
    #         exit()
    return img_p, stats


def acquire_image(im, multi=False):
    """" Acquires xy, xz and yz images in Imspector, returns some stats 
        and the active configurations"""
        
    #im = sp.Imspector()
    #msr = im.active_measurement()
    #configuration = msr.active_configuration()
    
    #im, msr_names, msr, configuration = get_config()
    
    # acquires xy view
    # configuration names and measurements names are hard coded
        
    x = im.measurement(im.measurement_names()[0])
    im.activate(x)
    try:
        x.activate(x.configuration('xy2d'))
        im.start(x)
        time.sleep(3)
        im.pause(x)
        #image_xy = x.stack('ExpControl Ch1 {1}').data()
        image_xy, stats = grab_image(x, window = 'ExpControl Ch1 {1}')
        # takes off black edge, resizes to (64, 64) and standardizes
        image_xy = helpers.preprocess(image_xy)
    except:
        print("cannot find xy2d config or 'ExpControl Ch1 {1}' window")
        exit()
    time.sleep(0.5)  

    if multi:
        # acquires the other two vies and stacks them
        try:
            x.activate(x.configuration('xz2d'))
            im.start(x)
            time.sleep(3)
            im.pause(x)
            image_xz, _ = grab_image(x, window = 'ExpControl Ch1 {13}')
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
            image_yz, _ = grab_image(x, window = 'ExpControl Ch1 {15}')
            # takes off black edge, resizes to (64, 64) and standardizes
            image_yz = helpers.preprocess(image_yz)
        except:
            print("cannot find yz2d config or 'ExpControl Ch1 {15}' window")  
            exit()
        time.sleep(0.5)
        image = np.stack((image_xy, image_xz, image_yz), axis=0)
    else:
        image = image_xy
    
    return image, stats



    
    

def abberior_predict(model_store_path, image, offset=False, multi=False, zern=True, ii=1):
    
    best_coeffs = []
    best_corr = 0
    for _ in range(ii):

        # if multi:
        #     if offset:
        #         model = helpers.MultiOffsetNet13()
        #     else:
        #         model = helpers.MultiNet11()
        # else:
        #     if offset:
        #         model = helpers.OffsetNet13()
        #     else:
        #         model = helpers.Net11()    

        # NOTE: OR this all comes from params file
        if multi:
            in_dim = 3
        else:
            in_dim = 1

        out_dim = 0
        if zern: out_dim += 11
        if offset: out_dim += 2
        print('in dim', in_dim, 'out dim', out_dim)

        import json
        with open(path, 'r') as f:
                data = json.load(f)
                
                data_clean = {}
                data_clean["gt"] = data["gt"][1::2]
                data_clean["preds"] = data["preds"][1::2]
                data_clean["corr"] = data["corr"]
                data_clean["init_corr"] = data["init_corr"]

        model = helpers.TheUltimateModel(input_dim=in_dim, output_dim=out_dim)
     
        # gets preds
        print("predict: ", model_store_path)
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
