'''
Project: deep-sted-autoalign
Created on: Thursday, 28th November 2019 10:54:30 am
--------
@author: hmcgovern
'''
# standard imports
import random
import json
import random
from math import factorial as mfac

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
import skimage

# local modules
from utils import my_classes as my_classes
from utils import xysted 
from utils.xysted import fluor_psf, sted_psf
from utils.vector_diffraction import vector_diffraction as vd

def normalize_img(img):
    """Normalizes the pixel values of an image (np array) between 0.0 and 1.0"""
    return (img-np.min(img))/(np.max(img)-np.min(img))

def add_noise(img):
    """Adds Poisson noise to the image using skimage's built-in method. Function normalizes image before adding noise"""
    return skimage.util.random_noise(normalize_img(img), mode='poisson', seed=None, clip=True)

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


def save_params(fname):
    """Given an output file name and a resolution which defaults to 64, this fn creates a .txt file formatted as a json, 
    containing the optical parameters for the sted and excitation beams for our Abberior system as well as numerical parameters for the 
    simulation. This output file is loaded in xysted.py when simulating a sted beam or a fluorescence psf and passed as inputs to the 
    vector_diffraction fn in vector_diffraction.py"""
    optical_params_sted = {
        "n" : 1.518,
        "NA" : 1.4,
        "f" : 1.8, # in mm
        "transmittance" : 0.74, # for olympus 100x oil, check nikon
        "lambda" : 775, # in nm
        "P_laser" : 250e-3, # in mW
        "rep_rate" : 40e6, # in MHz
        "pulse_length" : 700e-12, # in ps
        "obj_ba" : 5.04, # in mm
        "offset" : [0,0] # in mm
        }

    optical_params_gauss = {
        "n" : 1.518,
        "NA" : 1.4,
        "f" : 1.8, # in mm
        "transmittance" : 0.84, # for olympus 100x oil, check nikon
        "lambda" : 640, # in nm
        "P_laser" : 0.125e-3, # in mW
        "rep_rate" : 40e6, # in MHz
        "pulse_length" : 100e-12, # in ps
        "obj_ba" : 5.04, # in mm
        "offset" : [0,0] # in mm
        }

    data = {"optical_params_sted": optical_params_sted, "optical_params_gauss": optical_params_gauss}

    with open(fname, 'w') as outfile:
        json.dump(data, outfile)

def gen_offset():
    """A function to generate an offset [x, y] to displace the STED psf. Returns an 1d array of 2 ints"""
    x = round(random.uniform(-0.2, 0.2), 3)
    y = round(random.uniform(-0.2, 0.2), 3)
    return [x,y]


def gen_coeffs():
    """ Generates a random set of Zernike coefficients given piecewise constraints
    from Zhang et al's paper.
    1st-3rd: [0]   |  4th-6th: [+/- 1.4]  | 7th-10th: [+/- 0.8]  |  11th-15th: [+/- 0.6] 
    
    For physical intuition's sake, I'm creating a 15 dim vector, but only returning the 12 values that are non-zero.
    NOTE: this could be modified to accomodate further Zernike terms, but CNN code would have to be adjusted as well
    """
    c = np.zeros(15)
    # c[3:6] = [random.uniform(-1.4, 1.4) for i in c[3:6]]
    # c[6:10] = [random.uniform(-0.8, 0.8) for i in c[6:10]]
    # c[10:] = [random.uniform(-0.6, 0.6) for i in c[10:]]
    c = [round(random.uniform(-0.2, 0.2), 3) for i in c]
    
    return c[3:]


def create_phase(coeffs, res=64, offset=[0,0]):
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

    size=np.asarray([res+1, res+1])
    # this multiplies each zernike term phase mask by its corresponding weight in a time-efficient way.
    # it's convoluted, but I've checked it backwards and forwards to make sure it's correct.
    
    # NOTE: changed order to reflect new ordering of args """def crop(full, size, offset = [0,0]):"""
    terms = [coeff*crop(create_zernike(size*2, order), size, offset) for coeff, order in list(zip(coeffs, orders))]  
    zern = sum(terms)
    # returns one conglomerated phase mask containing all the weighted aberrations from each zernike term.
    # zern represents the collective abberations that will be added to an ideal donut.
    return zern

# TODO: make these a bit more nuanced re: offsets
def get_sted_psf(res=64, coeffs=np.asarray([0.0]*12), offset_label=[0,0], gen_coeffs=False, gen_offset=False,  multi=False):
    """Given coefficients and an optional resolution argument, returns a point spread function resulting from those coefficients.
    If multi flag is given as True, it creates an image with 3 color channels, one for each cross-section of the PSF"""

    if gen_coeffs:
        coeffs = gen_coeffs()
    
    if gen_offset:
        offset_label = gen_offset()

    aberr_phase_mask = create_phase(coeffs, res, offset_label)
    
    if multi:
        plane = 'all'
    else:
        plane = 'xy'
    img = sted_psf(aberr_phase_mask, res, offset=offset_label, plane=plane)
    return img, coeffs, offset_label

def get_fluor_psf(res=64, coeffs=np.asarray([0.0]*12), offset_label=[0,0], gen_coeffs=False, gen_offset=False, multi=False):
    
    if gen_coeffs:
        coeffs = gen_coeffs()

    if gen_offset:
        offset_label = gen_offset()

    aberr_phase_mask = create_phase(coeffs, res, offset_label)

    if multi:
        plane = 'all'
    else:
        plane = 'xy'
    
    img = fluor_psf(aberr_phase_mask, res, offset=offset_label, plane=plane)
    return img, coeffs, offset_label


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

def plot_xsection(img3d):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(img3d[0])
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(img3d[1])
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(img3d[2])
    return fig
# TODO: redirect to the original now that it's in the same repo
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
    """ Calculates the Zernike coeff for a given order and radius rho. """
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





