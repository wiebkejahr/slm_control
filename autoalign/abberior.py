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

import autoalign.utils.helpers as helpers
import autoalign.utils.my_models as my_models


def test(model, input_image, model_store_path):
    # load the model weights from training
    checkpoint = torch.load(model_store_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    

    ideal_coeffs = np.asarray([0.0]*12)

    # donut = helpers.get_psf(ideal_coeffs)
    
    # Test the model
    model.eval()
    
    with torch.no_grad():
        # adds 3rd color channel dim and batch dim 
        # NOTE: THIS IS ONLY FOR 1D
        # image = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0)
        # NOTE: THIS IS ONLY FOR 3D
        image = torch.from_numpy(input_image).unsqueeze(0)
       
        # avg = []
        # i = 0
        # while i < 20:
            # pass it through the trained model to get the predicted coeffs
        outputs = model(image)
    coeffs = outputs.numpy().squeeze()
    #     # avg.append(coeffs)
    #     # i += 1
    # # avg = np.stack(avg)
    # # avg = np.average(avg, axis=0)
    # # coeffs = avg
    # # corrected = normalize_img(input_image) + normalize_img(get_psf(-coeffs))
    # # plt.figure()
    # # plt.imshow(corrected)
    # # plt.show()

   
    # # print("\n\n correlation coeff is: {} \n\n".format(np.corrcoef(donut.flat, corrected.flat)[0][1]))
    # # return coeffs, np.corrcoef(donut.flat, corrected.flat)[0][1], corrected
    return coeffs
    

def correct(model_store_path):
    
    # creates an instance of CNN
    model = my_models.Net()

    # acquire the image from Imspector    
    # NOTE: from Imspector, must run Tools > Run Server for this to work
    im = sp.Imspector()
    
    # print Imspector host and version
    # print('Connected to Imspector {} on {}'.format(im.version(), im.host()))

    # get active measurement 
    msr = im.active_measurement()
    image = msr.stack('ExpControl Ch1 {1}').data() # converts it to a numpy array
    
    image = preprocess(image)
    # # a little preprocessing
    # image = normalize_img(np.squeeze(image)) # normalized (200,200) array
    # image = crop_image(image, tol=0.1) # get rid of dark line on edge
    # image = normalize_img(image) # renormalize
    # image = resize(image, (64,64)) # resize
    
    # coeffs, _, image = test(model, image, model_store_path)
    coeffs = test(model, image, model_store_path)

    # a dictionary of correction terms to be passed to SLM control
    corrections = {
            "sphere": [
                coeffs[9],
                0.0
            ],
            "astig": [
                coeffs[0],
                coeffs[2]
            ],
            "coma": [
                coeffs[4],
                coeffs[5]
            ],
            "trefoil": [
                coeffs[3],
                coeffs[6]
            ]
        }

    return corrections
    

def abberior_multi(model_store_path):
    # creates an instance of CNN
    
    model = my_models.MultiNet()

    # acquire the image from Imspector    
    # NOTE: from Imspector, must run Tools > Run Server for this to work
    im = sp.Imspector()
    
    # print Imspector host and version
    # print('Connected to Imspector {} on {}'.format(im.version(), im.host()))

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

    image_xy = helpers.preprocess(image_xy)
    # plt.figure()
    # plt.imshow(image_xy)
    # plt.show()

    
    
    image_xz = helpers.preprocess(image_xz)
    #image_xz = np.fliplr(rotate(image_xz, -90))
    # plt.figure()
    # plt.imshow(image_xz, aspect="equal")
    # plt.show()
    
    image_yz = helpers.preprocess(image_yz)
    #image_yz = np.fliplr(imgage_yz)
    
    # plt.figure()
    # plt.imshow(image_yz, aspect="equal")
    # plt.show()
    # ##################
    image = np.stack((image_xy,image_xz, image_yz), axis=0)    
    #image = np.stack((np.squeeze(image_xy), np.squeeze(image_xz), np.squeeze(image_yz)), axis=0)
    fig = helpers.plot_xsection(image)
    plt.show()
    
    # exit()


    # # coeffs, _, image = test(model, image, model_store_path)
    results = test(model, image, model_store_path)
    coeffs = results
    # coeffs = results[:-2]
    # offset = results[-2:]
    reconstructed = helpers.get_sted_psf(coeffs=coeffs, multi=True)
    fig1 = helpers.plot_xsection(reconstructed)
    plt.show()
    

    # print(coeffs)
    # a dictionary of correction terms to be passed to SLM control
    corrections = {
            "sphere": [
                coeffs[9],
                0.0
            ],
            "astig": [
                -coeffs[2],
                coeffs[0]
            ],
            "coma": [
                coeffs[5],
                -coeffs[4]
            ],
            "trefoil": [
                coeffs[6],
                coeffs[3]
            ]
        }

    return corrections
