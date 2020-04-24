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
import utils.my_models as my_models
from skimage.transform import resize
import matplotlib.pyplot as plt
import specpy as sp

from utils.integration import integrate
from utils.helpers import *


def test(model, input_image, model_store_path):
    # load the model weights from training
    checkpoint = torch.load(model_store_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test the model
    model.eval()
    
    with torch.no_grad():
        # adds 3rd color channel dim and batch dim 
        image = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0)
       
        # avg = []
        # i = 0
        # while i < 20:
            # pass it through the trained model to get the predicted coeffs
        outputs = model(image)
    coeffs = outputs.numpy().squeeze()
    coeffs = -1*coeffs

    return coeffs
    

def abberior(model_store_path):
    
    # creates an instance of CNN
    model = my_models.Net()

    # acquire the image from Imspector    
    # NOTE: from Imspector, must run Tools > Run Server for this to work
    im = sp.Imspector()

    # get active measurement 
    msr = im.active_measurement()
    image = msr.stack('ExpControl Ch1 {1}').data() # converts it to a numpy array
    image = preprocess(image)
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

    # get active measurement 
    msr = im.active_measurement()
    image_xy = msr.stack('ExpControl Ch1 {1}').data() # converts it to a numpy array
    image_xy = preprocess(image_xy)

    ##### NOTE: fill this is in lab #######
    image_xz = msr.stack('ExpControl Ch1 {13}').data()
    image_xz = preprocess(image_xz)
    
    image_yz = msr.stack('ExpControl Ch1 {15}').data()
    image_yz = preprocess(image_yz)
    ###################
    
    image = np.stack((image_xy, image_xz, image_yz), axis=0)
    
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
