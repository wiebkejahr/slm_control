'''
Project: deep-adaptive-optics
Created on: Thursday, 28th November 2019 10:54:30 am
--------
@author: hmcgovern
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from Pattern_Calculator import create_zernike, create_donut, crop
from integration import integrate

def normalize_img(img):
    """Normalizes the pixel values of an image (np array) between 0.0 and 1.0"""
    return (img-np.min(img))/(np.max(img)-np.min(img))

def crop_image(img,tol=0.1):
    """Function to crop the dark line on the edge of the acquired image data.
    img is 2D image data (NOTE: it only works with 2D!)
    tol  is tolerance."""
    
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]


def create_phase(coeffs, size, rot, amp):
    """
    Creates a phase mask of all of the weighted Zernike terms (= phase masks)
    
    coeffs: list of floats of weights for corresponding zernike terms
    size: one dim of the square output array (in pixels)
    rot, amp, offset: all necessary, static values needed for the Pattern_Calculator code. 
                      Don't change them without good reason.
 
    Zernike polynomial orders = 
            1 = [[0,0],     11 = [4,-4],    21 = [5,5],
            2 = [1,-1],     12 = [4,-2],    22 = [6,-6],
            3 = [1,1],      13 = [4,0],     23 = [6,-4],
            4 = [2,-2],     14 = [4,2],     24 = [6,-2],
            5 = [2,0],      15 = [4,4],     25 = [6,0],
            6 = [2,2],      16 = [5,-5],    26 = [6,2],
            7 = [3,-3],     17 = [5,-3],    27 = [6,4],
            8 = [3,-1],     18 = [5,-1],    28 = [6,6]] 
            9 = [3,1],      19 = [5,1],
            10 = [3,3],     20 = [5,3],
    """
   # NOTE: starting with the 4th order, bc we set the first three to zero.
    orders = [[2,-2], [2,0], [2,2],
            [3,-3], [3,-1], [3,1],[3,3],
            [4,-4], [4,-2], [4,0], [4,2], [4,4]]
    # sanity checks
    assert(len(coeffs) == len(orders)) # should both be 12
    assert(isinstance(i, float) for i in coeffs)

    # this multiplies each zernike term phase mask by its corresponding weight in a time-efficient way.
    # it's convoluted, but I've checked it backwards and forwards to make sure it's correct.
    # NOTE: changed order to reflect new ordering of args """def crop(full, size, offset = [0,0]):"""
    terms = [coeff*crop(create_zernike(size*2, order), size) for coeff, order in list(zip(coeffs, orders))]  
    zern = sum(terms)
    # returns one conglomerated phase mask containing all the weighted aberrations from each zernike term.
    # zern represents the collective abberations that will be added to an ideal donut.
    return zern

def get_psf(coeffs, res=64):
    params2 = {'rot': 0, 'amp': 1, 'size': np.asarray([res, res])}
    return  integrate(create_phase(coeffs, **params2), res)

def corr_coeff(pred_coeffs, input_image, donut):
    
    if type(input_image) != np.ndarray: # then need to cut out batch and channel dim, n
        aber_img = input_image.squeeze().numpy()
    else:
        aber_img = input_image
    aber_img = normalize_img(aber_img)
    #
    corrections_neg = normalize_img(get_psf(-pred_coeffs)) # NOTE: this line is the bottleneck
    # corrections_pos = normalize_img(get_psf(pred_coeffs))
    #
    
    # NOTE: something is still funky with the signs, so check and see which correction makes it better
    corrected_img_neg = aber_img + corrections_neg
    # corrected_img_pos = aber_img + corrections_pos
    # if np.corrcoef(donut.flat, corrected_img_neg.flat)[0][1] > np.corrcoef(donut.flat, corrected_img_pos.flat)[0][1]:
    #     corrected_img = corrected_img_neg
    # else:
    #     corrected_img = corrected_img_pos
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
    ax4.imshow(corrected_img_neg)
    ax4.title.set_text('corrected image')
    plt.show()
    
        # print(np.corrcoef(donut.flat, corrected_img.flat))
            
        # corrected_img = torch.from_numpy(get_psf(-pred_label)).unsqueeze(0) # torch.Size([1, 64, 64])
    #     corrected_imgs.append(corrected_img)
    # corrected_imgs = torch.stack(corrected_imgs) # [32, 1, 64, 64]
    # print(corrected_imgs.shape)
    
    # loss = criterion(pred_imgs, images) # performing  mean squared error calculation
    return corrected_img_neg


def plot_comp(img, coeffs):

    params2 = {'rot': 0, 'amp': 1, 'size': np.asarray([64, 64])}

    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(img.squeeze())
    # horiz = np.fliplr(integrate(create_phase(coeffs, **params2), 64))
    # vert = np.flip(integrate(create_phase(coeffs, **params2), 64))
    # both = np.flip(horiz) 
    ax2 = fig.add_subplot(1,3,2)
    reg = get_psf(coeffs)
    reg = normalize_img(reg)
    # print('for reg: max {} min {}'.format(np.max(reg), np.min(reg)))
    ax2.imshow(reg)
    ax3 = fig.add_subplot(1,3,3)
    this = normalize_img(img.squeeze().numpy())
    # print('for img: max {} min {}'.format(np.max(this), np.min(this)))
    ax3.imshow(normalize_img(reg+this))
    # setting axes titles
    ax1.title.set_text('input')
    ax2.title.set_text('pred:' + ' {:.4f}'.format(mean_squared_error(img.squeeze(), reg)))
    plt.show()



