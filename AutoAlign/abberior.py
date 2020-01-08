'''
Project: deep-adaptive-optics
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
from cnn import Net
from skimage.transform import resize
import matplotlib.pyplot as plt
import specpy as sp


from utils.integration import integrate
from utils.helpers import *


def test(model, input_image, model_store_path):
    # load the model weights from training
    model.load_state_dict(torch.load(model_store_path))

    ideal_coeffs = np.asarray([0.0]*12)

    donut = get_psf(ideal_coeffs)
    
    # Test the model
    model.eval()
    
    with torch.no_grad():
        # mean=torch.from_numpy(np.asarray([0.1251]))
        # std=torch.from_numpy(np.asarray([0.2146]))
        # image = image*std + mean
        
         # adds 3rd color channel dim and batch dim 
        image = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0)
       
        avg = []
        i = 0
        while i < 20:
            # pass it through the trained model to get the predicted coeffs
            outputs = model(image)
            coeffs = outputs.numpy().squeeze()
            avg.append(coeffs)
            i += 1
    avg = np.stack(avg)
    avg = np.average(avg, axis=0)
    coeffs = avg
    corrected = normalize_img(input_image) + normalize_img(get_psf(-coeffs))
    plt.figure()
    plt.imshow(corrected)
    plt.show()

   
    print("\n\n correlation coeff is: {} \n\n".format(np.corrcoef(donut.flat, corrected.flat)[0][1]))
    return coeffs, np.corrcoef(donut.flat, corrected.flat)[0][1], corrected
    

def abberior():

    model_store_path="C:/Users/hmcgover/Seafile/My Library/Models/fixed_zern_old_10_epochs_SGD_lr_0.1_w_val.pth"
    
    # creates an instance of CNN
    model = Net()

    # acquire the image from Imspector    
    # NOTE: from Imspector, must run Tools > Run Server for this to work
    im = sp.Imspector()
    
    # print Imspector host and version
    # print('Connected to Imspector {} on {}'.format(im.version(), im.host()))

    # get active measurement 
    msr = im.active_measurement()
    image = msr.stack('ExpControl Ch1 {1}').data() # converts it to a numpy array
    
    # a little preprocessing
    image = normalize_img(np.squeeze(image)) # normalized (200,200) array
    image = crop_image(image, tol=0.1) # get rid of dark line on edge
    image = normalize_img(image) # renormalize
    image = resize(image, (64,64)) # resize
    
    coeffs, _, image = test(model, image, model_store_path)
    
    # if corr_coeff < 0.94:
    #     coeffs, corr_coeff, image = test(model, image, model_store_path)


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

    print(corrections)
    return corrections
    


# if __name__ == "__main__":
    
#     MODEL_STORE_PATH="/c/Users/hmcgover/Seafile/My Library/Models/fixed_zern_old_10_epochs_SGD_lr_0.1_w_val.pth"

#     main(model_store_path)