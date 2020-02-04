'''
Project: deep-adaptive-optics
Created on: Thursday, 28th November 2019 10:54:30 am
--------
@author: hmcgovern
'''
# standard imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import factorial as mfac
from torch.utils.data import Dataset, DataLoader

# local modules
from autoalign.utils.integration import *
import autoalign.utils.my_classes as my_classes

def normalize_img(img):
    """Normalizes the pixel values of an image (np array) between 0.0 and 1.0"""
    return (img-np.min(img))/(np.max(img)-np.min(img))

def crop_image(img,tol=0.1):
    """Function to crop the dark line on the edge of the acquired image data.
    img is 2D image data (NOTE: it only works with 2D!)
    tol  is tolerance."""
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def preprocess(image):
    """function for preprocessing image pulled from Abberior msr stack. Used in abberior.py"""
    # a little preprocessing
    image = normalize_img(np.squeeze(image)) # normalized (200,200) array
    image = crop_image(image, tol=0.1) # get rid of dark line on edge
    image = normalize_img(image) # renormalize
    image = resize(image, (64,64)) # resize
    return image


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

def get_psf(coeffs, res=64, multi=False):
    """Given coefficients and an optional resolution argument, returns a point spread function resulting from those coefficients.
    If multi flag is given as True, it creates an image with 3 color channels, one for each cross-section of the PSF"""
    params2 = {'rot': 0, 'amp': 1, 'size': np.asarray([res, res])}
    if multi:
        return integrate_multi(create_phase(coeffs, **params2), res)
    else:
        return integrate(create_phase(coeffs, **params2), res)


def get_stats(data_path, batch_size, mode='train'):
    """ Finding Dataset Stats for Normalization before Training."""
    dataset = my_classes.PSFDataset(hdf5_path=data_path, mode=mode, \
        transform=my_classes.ToTensor())
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    print('mean is: {}  |   std is: {}'.format(mean, std))
    return mean, std


def corr_coeff(pred_coeffs, input_image, donut):
    """Function to help visualize the corrections made by the model. To be modified to suit unit tests. It currently plots
    the original image, the ideal donut psf, the predicted psf, and the "corrected" psf. It then computes the correlation
    coefficient between the ideal donut shape and the "corrected" psf
    """
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
    """Deprecated function for visualizing predictions. Kept around for spare parts, questionable functionality
    """
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

#########################################################################################
#
#   Functions from Wiebke's Pattern Calculator code that are necessary for redundancy.
#   I want my code to run as a self-contained unit that can be plugged into the pre-
#   existing GUI, this helps me achieve that.
#
#########################################################################################


######## fn to create a single zernike phase mask #######
def create_zernike(size, order, rad=1):
    """ Calculates the Zernike polynomial of a given order, for the given 
        image size. Normalizes the polynomials in the center quadrant of the 
        image to [0,1]. """
    xcoord, ycoord = create_coords(size)
    xcoordnorm = (xcoord/np.max(np.abs([np.max(xcoord), np.min(xcoord)])))*2
    ycoordnorm = (ycoord/np.max(np.abs([np.max(ycoord), np.min(ycoord)])))*2
    
    rho, phi = cart2polar(xcoord, ycoord)
    
    # when normalizing rho: factor of two is needed because I'm creating images
    # double the size for subsequent cropping. Factor of 2 / sqrt(2) is needed 
    # because I'm working on a square whereas the polynomials are defined on a 
    # circle
    rho = rho * 2 * 2 / np.sqrt(2) / rad
    # print(size, np.min(xcoordnorm), np.max(xcoordnorm), np.min(ycoordnorm), np.max(ycoordnorm),
    #       np.min(rho), np.max(rho), np.min(phi), np.max(phi))
    
    if order[1] < 0:
        zernike = zernike_coeff(rho, np.abs(order)) * np.sin(np.abs(order[1]) * phi)
    elif order[1] >= 0:
        zernike = zernike_coeff(rho, order) * np.cos(order[1] * phi)
    
    #mask = (rho <= 1)
    #zernike = zernike * mask
    return zernike

######## and its helper fns ##################
def create_coords(size, off = [0,0]):
    """ Returns the cartesian coordinates of each pixel in a matrix from the 
        inputs size and offset (defining the center position). ATTENTION:
        array of cartesian coordinates is created at 2x the size needed. Will 
        be cropped later in the workflow for easy offsetting. """
    x = np.arange((-size[0]/2 + off[0]), (size[0]-size[0]/2 + off[0]))
    y = np.arange((-size[1]/2 + off[1]), (size[1]-size[1]/2 + off[1]))    
    xcoords = np.multiply.outer(np.ones(size[0]), y)
    ycoords = np.multiply.outer(x, np.ones(size[1]))    
    return xcoords, ycoords

def cart2polar(x, y):
    """ Returns normalized polar coordinates for cartesian inputs x and y. """
    z = x + 1j * y
    return (np.abs(z)/np.max(np.abs(z)), np.angle(z))

def zernike_coeff(rho, order):
    """ Calculates the Zernike coefficient for a given order and radius rho. """
    coeff = 0
    nn = order[0]
    mm = np.abs(order[1])
    
    for kk in range(int((nn - mm)/2)+1):
        c = (np.power(-1, kk) * mfac(nn - kk))/ \
            (mfac(kk) * \
             mfac((nn + mm)/2 - kk) * \
             mfac((nn - mm)/2 - kk))
        r = (np.power(rho, nn - 2 * kk))
        coeff = coeff + c * r
    return coeff


########## fn to create vortex (donut phase mask) ############
def create_donut(size, rot, amp):
    """" Creates the phasemask for shaping the 2D donut with the given image 
        size, rotation and amplitude of the donut. """
    xcoord, ycoord = create_coords(size)
    dn = 0.5 / np.pi * np.mod(cart2polar(xcoord, ycoord)[1] + (rot + 180) /
                              180 * np.pi, 2*np.pi) * amp
    return dn

########## fn to crop phase mask to the right scaling ########

def crop(full, size, offset = [0,0]):
    """ Crops the full data to half the size, using the provided offset. """
    minx = int(size[0]/2 + offset[0])
    maxx = int(size[0]*3/2 + offset[0])
    miny = int(size[1]/2 + offset[1])
    maxy = int(size[1]*3/2 + offset[1])    
    cropped = full[minx:maxx, miny:maxy]
    return cropped




