'''
Project: deep-adaptive-optics
Created on: Wednesday, 6th November 2019 9:47:12 am
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
# import javabridge
# import bioformats
# from bioformats import log4j


from create_train_data import create_phase
from integration import integrate
from utils.helpers import *


def test(model, image, model_store_path):
    # load the model weights from training
    model.load_state_dict(torch.load(model_store_path))

    ideal_coeffs = np.asarray([0.0]*12)
    donut = get_psf(ideal_coeffs) # (64,64)
    
    # Test the model
    model.eval()
    
    with torch.no_grad():
        # mean=torch.from_numpy(np.asarray([0.1251]))
        # std=torch.from_numpy(np.asarray([0.2146]))
        # image = image*std + mean
        
         # adds 3rd color channel dim and batch dim 
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        
        # pass it through the trained model to get the predicted coeffs
        outputs = model(image)
        # coeffs = outputs.numpy().squeeze()

        # comparing a 'corrected' psf from the predicted coeffs with the ideal donut
        corrected_img = corr_coeff(outputs, image, donut=donut)
        # checks the correlation between the ideal donut PSF and that of the "corrected" image
        corrcoeff = np.corrcoef(donut.flat, corrected_img.flat)[0][1]
        # if less than 0.97, takes the corrected image as the new input to the model and repeats
        while corrcoeff < 0.93:
            print('in loop')
            new_input = torch.from_numpy(corrected_img).unsqueeze(0).unsqueeze(0)
            new_output = model(new_input)
            corrected_img = corr_coeff(new_output, corrected_img, donut=donut)
            corrcoeff = np.corrcoef(donut.flat, corrected_img.flat)[0][1]
            print(corrcoeff)
        print('convergenge reached!')
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(image.squeeze().numpy())
    ax1.title.set_text('original')
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(corrected_img)
    ax2.title.set_text('corrected')
    plt.show()

        

def corr_coeff(outputs, images, donut):
    m_vals = []
    for j in range(1):
    # for j in range(outputs.shape[0]): # for each coeff list in the batch
        pred_coeffs = outputs[j].numpy() # [32, 12]
        # pred_coeffs = outputs[j]
        if type(images[j]) != np.ndarray: # then need to cut out batch and channel dim, n
            aber_img = images[j].squeeze().numpy()
        else:
            aber_img = images
        aber_img = normalize_img(aber_img)
        #
        corrections_neg = normalize_img(get_psf(-pred_coeffs)) # NOTE: this line is the bottleneck
        corrections_pos = normalize_img(get_psf(pred_coeffs))
        #
        # NOTE: something is still funky with the signs, so check and see which correction makes it better
        
        corrected_img_neg = aber_img + corrections_neg
        corrected_img_pos = aber_img + corrections_pos
        if np.corrcoef(donut.flat, corrected_img_neg.flat)[0][1] > np.corrcoef(donut.flat, corrected_img_pos.flat)[0][1]:
            corrected_img = corrected_img_neg
        else:
            corrected_img = corrected_img_pos
        # print(np.max(corrections), np.min(corrections))
        # print(aber_img.shape)
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(donut)
        ax1.title.set_text('donut')
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(aber_img)
        ax2.title.set_text('input image')
        ax3 = fig.add_subplot(2,2,3)
        ax3.imshow(normalize_img(get_psf(pred_coeffs)))
        ax3.title.set_text('predicted psf')
        ax4 = fig.add_subplot(2,2,4)
        ax4.imshow(corrected_img)
        ax4.title.set_text('corrected image')
        plt.show()
        
        # print(np.corrcoef(donut.flat, corrected_img.flat))
            
        # corrected_img = torch.from_numpy(get_psf(-pred_label)).unsqueeze(0) # torch.Size([1, 64, 64])
    #     corrected_imgs.append(corrected_img)
    # corrected_imgs = torch.stack(corrected_imgs) # [32, 1, 64, 64]
    # print(corrected_imgs.shape)
    
    # loss = criterion(pred_imgs, images) # performing  mean squared error calculation
    return corrected_img



def main(args):
    
    
    image_dir = args.image_dir
    model_store_path = args.model_store_path
    
    # creates an instance of CNN
    model = Net()

    # load image from msr file
    javabridge.start_vm(class_path=bioformats.JARS)
    log4j.basic_config()
    image = bioformats.load_image(image_dir, series = 2, rescale=False)
    javabridge.kill_vm()

    image = normalize_img(image)
    image = crop_image(image, tol=0.1)
    image = resize(image, (64, 64))
    image = normalize_img(image)


    # gets the model predictions for the image
    
    # avg_coeffs = []
    # for i in range(100):
    test(model, image, model_store_path)
    #     avg_coeffs.append(coeffs)
    # avg = np.mean(np.array(avg_coeffs), axis=0)
    # plot_comp(image, avg)


if __name__ == "__main__":
    # NOTE: currently takes two args, image dir and model store path, it just needs to take one, the model store path
    parser = ap.ArgumentParser(description='Model Hyperparameters and File I/O')
    # parser.add_argument('image_dir', type=str, help='path to saved input image data')
    parser.add_argument('model_store_path', type=str, help='path to model checkpoint dir')
    
    ARGS=parser.parse_args()

    main(ARGS)