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


def get_image(multi=False, config = False):
    # create Imspector object
    im = sp.Imspector()
    # get active measurement
    msr = im.active_measurement()
    configuration = msr.active_configuration()

    try:
        # im.pause(msr)
        # time.sleep(0.5)
        # im.start(msr)
        image_xy = msr.stack('ExpControl Ch1 {1}').data() # converts it to a numpy array
        stats = [np.max(image_xy), np.min(image_xy), np.std(image_xy)]
        image = helpers.preprocess(image_xy)
    except:
        print("Cannot find 'ExpControl Ch1 {1}' window")
        exit()
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

    
    #image_xz = im.measurement('ExpControl Ch1 {13}')
    #image_yz = im.measurement('ExpControl Ch1 {15}')
    if multi:
        x = im.measurement(im.measurement_names()[0])
        im.activate(x)
        #########
        x.activate(x.configuration('xy2d'))
        im.start(x)
        time.sleep(3)
        im.pause(x)
        image_xy = x.stack('ExpControl Ch1 {1}').data()
        stats = [np.max(image_xy), np.min(image_xy), np.std(image_xy)]
        image_xy = helpers.preprocess(image_xy) # takes off black edge, resizes to (64, 64) and standardizes
        time.sleep(0.5)

        x.activate(x.configuration('xz2d'))
        im.start(x)
        time.sleep(3)
        im.pause(x)
        image_xz = x.stack('ExpControl Ch1 {13}').data()
        image_xz = helpers.preprocess(image_xz)
        time.sleep(0.5)

        x.activate(x.configuration('yz2d'))
        im.start(x)
        time.sleep(3)
        im.pause(x)
        image_yz = x.stack('ExpControl Ch1 {15}').data()
        image_yz = helpers.preprocess(image_yz)

        #TODO: stacking only works if all images have the same dimensions!
        image = np.stack((image_xy,image_xz, image_yz), axis=0)
        time.sleep(0.5)
        # print(image.shape)
        #helpers.plot_xsection(image)
        #plt.show()

    
    # NOTE: this is a hack to make it 1D for now
    # if xy:
    #     image = image_xy
    
    if config:
        return image, configuration, msr, stats
    else:
        return image

def abberior_test(model_store_path, image, offset=False, multi=False, i=0):
    
    # creates an instance of CNN
    # model = my_models.MultiNetCentered()
    # model = my_models.NetCentered()
    # model = my_models.MultiNetCat()
    
    
    # model= my_models.MultiNet11()
    # print(model)
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
    best_coeffs = []
    best_corr = 0
    for _ in range(5):

        if multi:
            if offset:
                model = my_models.MultiOffsetNet14()
            else:
                model = my_models.MultiNet11()
        else:
            if offset:
                model = my_models.OffsetNet13()
            else:
                model = my_models.Net11()    
        
        # overriding model
        model = models.alexnet(pretrained=False, num_classes=11)
        # gets preds
        
        checkpoint = torch.load(model_store_path)
        # print(checkpoint['model_state_dict'])
        # exit()

        model.load_state_dict(state_dict=checkpoint['model_state_dict'])

        # mean = 0.1083
        # std = 0.2225

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
            # image = image.numpy()
            # correlation = helpers.corr_coeff(helpers.get_sted_psf(coeffs=labels.numpy().squeeze(), multi=False, \
            #     corrections=[helpers.create_phase(coeffs=(-1.)*coeffs)]))

        # return coeffs

        if offset:
            zern = coeffs[:-2]
            offset_label = coeffs[-2:]
        else:
            zern = coeffs
            offset_label = [0,0]
        reconstructed = helpers.get_sted_psf(coeffs=zern, offset_label=offset_label, multi=multi, defocus=False)
        # print(np.shape(image), np.shape(reconstructed))
        corr = helpers.corr_coeff(image, reconstructed)
        if corr > best_corr:
            best_corr = corr
            best_coeffs = coeffs


    if offset:
        zern = best_coeffs[:-2]
        offset_label = best_coeffs[-2:]
    else:
        zern = best_coeffs 
        offset_label = [0,0]   
    # so as to not break existing code
    # TODO; make this not hideous
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
    # plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(image, cmap='hot')
    # plt.suptitle('Iteration {}'.format(i))
    # plt.subplot(122)
    # plt.imshow(reconstructed, cmap='hot')
    # plt.show()
    
    # fig = helpers.plot_xsection_abber(image, reconstructed)
    # plt.show()
    

    return zern, offset_label







if __name__ == "__main__":
    abberior_multi('models/20.05.18_scaling_fix_eps_15_lr_0.001_bs_64_2.pth')
