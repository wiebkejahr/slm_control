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
sys.path.insert(1, 'autoalign/')
sys.path.insert(1, 'parameters/')
import utils.helpers as helpers
import utils.my_models as my_models
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import shift


def test(model, input_image, model_store_path):
    # load the model weights from training
    checkpoint = torch.load(model_store_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # mean = 0.1083
    # std = 0.2225

    # Test the model
    model.eval()

    with torch.no_grad():
        # adds 3rd color channel dim and batch dim
        # NOTE: THIS IS ONLY FOR 1D
        image = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0)
        # NOTE: THIS IS ONLY FOR 3D
        # print(np.max(input_image), np.min(input_image))
        # image = torch.from_numpy(input_image).unsqueeze(0)

        outputs = model(image)
        coeffs = outputs.numpy().squeeze()
        # correlation = helpers.corr_coeff(helpers.get_sted_psf(coeffs=labels.numpy().squeeze(), multi=False, \
        #     corrections=[helpers.create_phase(coeffs=(-1.)*coeffs)]))

    return coeffs

def correct_tip_tilt():

    # acquire the image from Imspector
    # NOTE: from Imspector, must run Tools > Run Server for this to work
    im = sp.Imspector()

    # get active measurement
    msr = im.active_measurement()
    try:
        image_xy = msr.stack('ExpControl Ch1 {1}').data() # converts it to a numpy array
    except:
        print("Cannot find 'ExpControl Ch1 {1}' window")
        exit()
    # image = helpers.preprocess(image_xy) # (64,64), values (-.5, 4)
    return helpers.calc_tip_tilt(image_xy)

def correct_defocus():
    # acquire the image from Imspector
    # NOTE: from Imspector, must run Tools > Run Server for this to work
    #x = [0.0, 0.025, 0.05, 0.1, 0.0, -0.025, -0.05, -0.1, 0.0]
    #y = [51.0953751869621, 51.38127570663089, 51.47044657188718, 52.00566108408576, 51.68171180185857, 52.28224133443203, 52.18034362082555, 52.02026189327759, 52.9484754882291 ]
    # x = [0.1,0.0,-0.1]
    # y = [51.03816390888528, 50.59826488069055, 49.71942127100279]
    # helpers.fit(x, y)
    im = sp.Imspector()

    # get active measurement
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
    # image = helpers.preprocess(image_xy) # (64,64), values (-.5, 4)
    return helpers.calc_defocus(image_xz, image_yz)


def get_image(xy=True):
    # create Imspector object
    im = sp.Imspector()
    # get active measurement
    msr = im.active_measurement()
    try:
        image_xy = msr.stack('ExpControl Ch1 {1}').data() # converts it to a numpy array
    except:
        print("Cannot find 'ExpControl Ch1 {1}' window")
        exit()
    try:
        image_xz = msr.stack('ExpControl Ch1 {13}').data()
    except:
        print("Cannot find 'ExpControl Ch1 {13}' window")
        exit()
    try:
        image_yz = msr.stack('ExpControl Ch1 {15}').data()
    except:
        print("Cannot find 'ExpControl Ch1 {15}' window")
        exit()
    # takes off black edge, resizes to (64, 64) and standardizes
    image_xy = helpers.preprocess(image_xy) 
    
    image_xz = helpers.preprocess(image_xz)

    image_yz = helpers.preprocess(image_yz)

    # ##################
    image = np.stack((image_xy,image_xz, image_yz), axis=0)

    # NOTE: this is a hack to make it 1D for now
    if xy:
        image = image_xy
    return image

def abberior_multi(model_store_path, image):
    
    # creates an instance of CNN
    # model = my_models.MultiNetCentered()
    model = my_models.NetCentered()

    # # acquire the image from Imspector
    # # NOTE: from Imspector, must run Tools > Run Server for this to work
    # im = sp.Imspector()

    # # print Imspector host and version
    # # print('Connected to Imspector {} on {}'.format(im.version(), im.host()))

    # # get active measurement
    # msr = im.active_measurement()
    # try:
    #     image_xy = msr.stack('ExpControl Ch1 {1}').data() # converts it to a numpy array
    # except:
    #     print("Cannot find 'ExpControl Ch1 {1}' window")
    #     exit()
    # try:
    #     image_xz = msr.stack('ExpControl Ch1 {13}').data()
    # except:
    #     print("Cannot find 'ExpControl Ch1 {13}' window")
    #     exit()
    # try:
    #     image_yz = msr.stack('ExpControl Ch1 {15}').data()
    # except:
    #     print("Cannot find 'ExpControl Ch1 {15}' window")
    #     exit()

    # # takes off black edge, resizes to (64, 64) and standardizes
    # image_xy = helpers.preprocess(image_xy) 
    
    # image_xz = helpers.preprocess(image_xz)

    # image_yz = helpers.preprocess(image_yz)

    # # ##################
    # image = np.stack((image_xy,image_xz, image_yz), axis=0)

    # # NOTE: this is a hack to make it 1D for now
    # image = image_xy

    # gets preds
    coeffs = test(model, image, model_store_path)
    # 
    # # ITERATIVE LOOP #
    # so_far = -1
    # while correlation > so_far:
    #     # while it is not optimized, if it comes accross a higher correlation, work down and switch out preds
    #     new_coeffs = model(image).numpy().squeeze()
    #     new_corrected = helpers.get_sted_psf(coeffs=labels.numpy().squeeze(), multi=False, \
    #         corrections=[helpers.create_phase(coeffs=(-1.)*new_coeffs)])
    #     new_correlation = helpers.corr_coeff(new_corrected)
    #     if new_correlation > correlation:
    #         so_far = correlation
    #         correlation = new_correlation
    #         print('new corr: {}, old corr: {}'.format(correlation, so_far))
    #         coeffs = new_coeffs
    #     else:
    #         print('final correlation: {}'.format(correlation))
    #         break



    reconstructed = helpers.get_sted_psf(coeffs=coeffs, multi=False, defocus=False)
    # helpers.plot_xsection_abber(image)
    # old = 0
    # new = 0
    # while new >= old:
    #     # gets preds
    #     coeffs = test(model, image, model_store_path)
    #     reconstructed = helpers.get_sted_psf(coeffs=coeffs, multi=False, defocus=False)
    #     # plots it
    #     plt.figure(1)
    #     plt.imshow(image_xy, cmap='hot')
    #     plt.figure(2)
    #     plt.imshow(reconstructed, cmap='hot')
    #     plt.show()
        
    #     # reassign lower threshold to be current correlation
    #     old = new
    #     # get new correlation coeff
    #     new = helpers.corr_coeff(reconstructed)
    #     print(new)

    
    fig = helpers.plot_xsection_abber(image, reconstructed)
    plt.show()
    

    return coeffs



    # print(coeffs)
    # a dictionary of correction terms to be passed to SLM control
    # corrections = {
    #         "sphere": [
    #             coeffs[9],
    #             0.0
    #         ],
    #         "astig": [
    #             coeffs[2], #used to be neg
    #             coeffs[0]
    #         ],
    #         "coma": [
    #             coeffs[5],
    #             coeffs[4] #used to be neg
    #         ],
    #         "trefoil": [
    #             coeffs[6],
    #             coeffs[3]
    #         ]
    #     }

    # return corrections




if __name__ == "__main__":
    abberior_multi('models/20.05.18_scaling_fix_eps_15_lr_0.001_bs_64_2.pth')
