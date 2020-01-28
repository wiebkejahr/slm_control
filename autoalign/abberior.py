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
from skimage.transform import resize
import matplotlib.pyplot as plt
try:
    import specpy as sp
except:
    # raise("Specpy not installed!")
    specpy = False
    print('specpy not installed')

from autoalign.utils.integration import integrate
import autoalign.utils.helpers as helpers
import autoalign.utils.my_models as my_models


def test(model, input_image, model_store_path):
    # load the model weights from training
    checkpoint = torch.load(model_store_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    ideal_coeffs = np.asarray([0.0]*12)

    donut = helpers.get_psf(ideal_coeffs)
    
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
        # avg.append(coeffs)
        # i += 1
    # avg = np.stack(avg)
    # avg = np.average(avg, axis=0)
    # coeffs = avg
    # corrected = normalize_img(input_image) + normalize_img(get_psf(-coeffs))
    # plt.figure()
    # plt.imshow(corrected)
    # plt.show()

   
    # print("\n\n correlation coeff is: {} \n\n".format(np.corrcoef(donut.flat, corrected.flat)[0][1]))
    # return coeffs, np.corrcoef(donut.flat, corrected.flat)[0][1], corrected
    return coeffs
    

def correct(model_store_path):
    
    # creates an instance of CNN
    model = my_models.Net()

    # acquire the image from Imspector    
    # NOTE: from Imspector, must run Tools > Run Server for this to work
    if specpy:
        im = sp.Imspector()
        
        # print Imspector host and version
        # print('Connected to Imspector {} on {}'.format(im.version(), im.host()))

        # get active measurement 
        msr = im.active_measurement()
        image = msr.stack('ExpControl Ch1 {1}').data() # converts it to a numpy array
        
        # a little preprocessing
        image = helpers.normalize_img(np.squeeze(image)) # normalized (200,200) array
        image = helpers.crop_image(image, tol=0.1) # get rid of dark line on edge
        image = helpers.normalize_img(image) # renormalize
        image = helpers.resize(image, (64,64)) # resize
        
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
    else:
        pass
        


# if __name__ == "__main__":
#     parser = ap.ArgumentParser(description='Model Hyperparameters and File I/O')
#     parser.add_argument('model_store_path', type=str, help='path to model checkpoint dir')
    
#     ARGS=parser.parse_args()

#     main(ARGS)