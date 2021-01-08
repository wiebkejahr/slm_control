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
from numpy.random import normal
from math import factorial as mfac

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize, rotate
import skimage
from sklearn.linear_model import LinearRegression
from scipy.ndimage.measurements import center_of_mass
from sklearn.linear_model import LinearRegression

from skimage import filters
from skimage.measure import regionprops

from scipy.ndimage import shift
from skimage.transform import resize
import sys, os
sys.path.insert(1, '../slm_control/')
# local modules
from slm_control import Pattern_Calculator as PC
import autoalign.utils.my_classes as my_classes
from autoalign.utils.xysted import fluor_psf, sted_psf
from autoalign.utils.vector_diffraction import vector_diffraction as vd

def normalize_img(img):
    """Normalizes the pixel values of an image (np array) between 0.0 and 1.0"""
    return (img-np.min(img))/(np.max(img)-np.min(img))

# NOTE: fn contributed by Julia Lyudchik
# TODO: tune optional argument values to match the look we're going for
def add_noise(image, bgnoise_amount=1, poiss_amount=350):
    """A fn to add background and poisson noise to an image, contributed by
    Julia Lyudchik, PhD student in the Danzl Group"""
    _, x0,y0 = image.shape # this is either (3, 64, 64) or (1, 64, 64)
    #Background noise
    Nb = np.random.normal(0, 0.001, [x0,y0])
    final_Nb = image + Nb*bgnoise_amount
    final_Nb = (final_Nb-np.amin(final_Nb))/(np.amax(final_Nb) - np.amin(final_Nb))
    #Poisson noise
    final_poiss = np.random.poisson(final_Nb / np.amax(final_Nb) * poiss_amount) / poiss_amount * np.amax(final_Nb)
    return final_poiss

def preprocess(image):
    """function for preprocessing image pulled from Abberior msr stack. Used in abberior.py"""
    # cropping one pixel all around
    image = np.squeeze(image)[1:-1, 1:-1] 
    image = resize(image, (64,64))#, preserve_range=True)
    image = (image - np.mean(image))/np.std(image) # (-.5, 4)
    return image

def get_D(a, dx, lambd=0.775, f=1.8):
    return (a*lambd*f*2) / (np.pi*dx)

def fit(x,y):
    #from sklearn.linear_model import LinearRegression
    x = np.asarray(x).reshape(-1,1)
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, model.coef_*x+model.intercept_)
    plt.show()
    # return model


def get_CoMs(img):
    
    if len(img.shape) == 3:
        _, x_shape, y_shape = np.shape(img)
        
        ####### xy ########
        b, a = get_CoM(img[0])
        dx_xy = ((x_shape-1)/2-a)
        dy_xy = ((y_shape-1)/2-b)
    
        ####### xz ########
        b, a = get_CoM(img[1])
        dx_xz = ((x_shape-1)/2-a)
        dz_xz = ((y_shape-1)/2-b)
        
        ######## yz #########
        b, a = get_CoM(img[2])
        dy_yz = ((x_shape-1)/2-a)
        dz_yz = ((y_shape-1)/2-b)
        
        d_xyz = np.asarray([np.average([dx_xy, dx_xz]), 
                            np.average([dy_xy, dy_yz]),
                            np.average([dz_xz, dz_yz])])
        #d_xyz = np.asarray([dx_xy, dy_xy, np.average([dz_xz, dz_yz])])    
    elif len(img.shape) == 2:
        x_shape, y_shape = np.shape(img)
        b, a = get_CoM(img)
        d_xyz = np.asarray[((x_shape-1)/2-a),
                           ((y_shape-1)/2-b),
                           0]
    return d_xyz


def get_CoM(img):
    #TODO: return offsets in nm instead of absolute positons in px
    #    dx = (x_shape-1)/2-a
    #    dy = (y_shape-1)/2-b
    # then rewrite calc_tip_tilt, calc_defocus and automate accordingly

    threshold_value = filters.threshold_otsu(img)
    labeled_foreground = (img > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, img)
    center_of_mass = properties[0].centroid
    b = center_of_mass[0]
    a = center_of_mass[1]
    return b,a


def calc_defocus(img_xz, img_yz, lambd=0.775, f=1.8, D=5.04, px_size=10, abberior=True):
    # TODO: read Px size from abberior
    # pass other parameters from settings files
    if abberior:
        img_xz = np.squeeze(img_xz)[1:-1, 1:-1]
        img_yz = np.squeeze(img_yz)[1:-1, 1:-1]
    
    ####### xz ########
    x_shape, y_shape = np.shape(img_xz)
    b, a = get_CoM(img_xz)
    dx_xz = (x_shape-1)/2-a
    dz_xz = (y_shape-1)/2-b
    # print(a,b,dx_xz,dz_xz)
    #d_obj = D/3/1000 # scaling
    #dz1 = -(f/d_obj)^2*8/np.sqrt(3)*lambd*dy
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(img_xz)
    # plt.scatter(a,b)

    ######## yz #########
    x_shape, y_shape = np.shape(img_yz)
    b, a = get_CoM(img_yz)
    dy_yz = (x_shape-1)/2-a
    dz_yz = (y_shape-1)/2-b
    # print(a,b,dy_yz,dz_yz)
    # plt.subplot(122)
    # plt.imshow(img_yz)
    # plt.scatter(a,b)
    # plt.show()

    dz = np.average([dz_xz, dz_yz])*px_size
    #d_obj = D/3/1000 # scaling
    
    defocus = dz/((f/D)**2 *8 * np.sqrt(3)*lambd*1e3)/2
    #TODO: freak factor of two? check derivation again
    # check aperture radius or diameter
    
    # dz = -(f/d_obj)^2*8/sqrt(3)*lambd*coeff
    # print("defocus",dz, defocus)
    return(defocus)


def calc_tip_tilt(img, lambd=0.775, f=1.8, D=5.04, px_size=10, abberior=True):
    """this fn returns the coeffs of the calculated tip/tilt
    TODO:read px_size from imspector!
    TODO:read params from params file (read D from params and divide by 3)
    """
    # During testing, D was 0.001776
    # D is potentially 7.2 instead of 5.04, need to test it out
    # testing showed D is 0.052, which is interesting as it's neither the other two
    # potentially switched bc of np vs plt coordinate system
    # assert(len(img.shape) == 2)
    # print(img.shape)
    D = D/3/1000 # scaling, =~0.00168
       
    if abberior:
        img = np.squeeze(img)[1:-1, 1:-1]
    x_shape, y_shape = np.shape(img) 
    b, a = get_CoM(img)
    # print('center of mass: {}, {}'.format(b, a))
    
    dx = (x_shape-1)/2-a
    dy = (y_shape-1)/2-b
    # print('dx: {}   dy: {}'.format(dx, dy))
    xtilt = (np.pi*dx*px_size)/(lambd*f)*D/2
    ytilt = -(np.pi*dy*px_size)/(lambd*f)*D/2

    return [xtilt, ytilt]


def center(img, res=[64,64]):
    """Returns the correction phasemask to counteract tiptilt present in given image"""
    if len(img.shape) == 2:
        xy = img
    elif len(img.shape) == 3:
        xy = img[0]
    else:
        print('Image is not the right shape')
        exit()
    xtilt, ytilt = calc_tip_tilt(xy, abberior=False)
    tiptilt = create_phase(coeffs=[xtilt, ytilt], size=res, num=[0,1])

#     return tiptilt

def gen_offset(obj_dia = 5.04, scale = 0.1):
    """A function to generate an offset [x, y] to displace the STED psf. 
    Returns an 1d array of 2 ints"""

    x, y = np.random.normal(0,scale*obj_dia/2, 2)
    x = np.round(x, 3)
    y = np.round(y, 3)
    return [x,y]

def gen_coeffs(num=11, scale = 0.1):
    """ Generates a random set of Zernike coefficients given piecewise constraints
    from Zhang et al's paper.
    1st-3rd: [0]   |  4th-6th: [+/- 1.4]  | 7th-10th: [+/- 0.8]  |  11th-15th: [+/- 0.6] 
    
    For physical intuition's sake, I'm creating a 15 dim vector, but only returning the 12 values that are non-zero.
    NOTE: this could be modified to accomodate further Zernike terms, but CNN code would have to be adjusted as well
    """

    c = np.zeros(num)
    # c[3:6] = [random.uniform(-1.4, 1.4) for i in c[3:6]]
    # c[6:10] = [random.uniform(-0.8, 0.8) for i in c[6:10]]
    # c[10:] = [random.uniform(-0.6, 0.6) for i in c[10:]]
    c = [round(np.random.normal(0,scale), 3) for i in c]
    # c = [round(random.uniform(-0.2, 0.2), 3) for i in c]
    return c


def corr_coeff(img1, img2=[], multi=False):
    if len(img2)==0:
        # if only only image is passed, get the ideal donut to compare against
        img2 = get_sted_psf(multi=multi)
    
    return np.corrcoef(img1.flat, img2.flat)[0,1]


def create_phase(coeffs=np.asarray([0.0]*11), num=np.arange(3, 14), size=[64, 64], radscale = 2):
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
    # default is defocus included but not tip/tilt
    orders = [[1,-1],[1,1],[2,0], # tip, tilt, defocus
            [2,-2],[2,2],
            [3,-3],[3,-1],[3,1],[3,3], 
            [4,-4],[4,-2],[4,0],[4,2],[4,4]]
    
    # sanity checks
    assert(np.size(coeffs) == len(num))
    # this multiplies each zernike term phase mask by its corresponding weight in a time-efficient way.
    # it's convoluted, but I've checked it backwards and forwards to make sure it's correct.
 
    # NOTE: this is changed so I can call any subset of the full orders with another list called num
    # uses the Pattern Calculator create_zernike fn
    terms = [PC.create_zernike(size, orders[i], amp = coeff, radscale=radscale) for coeff, i in list(zip(coeffs, num))] 

    zern = sum(terms)
    # returns one conglomerated phase mask containing all the weighted aberrations from each zernike term.
    # zern represents the collective abberations that will be added to an ideal donut.
    

    # # wait I might be dumb. I can just add this in to the xysted fn??
    # if len(corrections) > 0 :
    #     for corr in corrections:
    #         zern = np.add(zern, corr)

    return zern


def get_sted_psf(coeffs=np.asarray([0.0]*11), res=[64,64], offset_label=[0,0],  multi=False, tiptilt=None, defocus=False):
    """Given coefficients and an optional resolution argument, returns a point spread function resulting from those coefficients.
    If multi flag is given as True, it creates an image with 3 color channels, one for each cross-section of the PSF"""

    if defocus:
        nums = np.arange(2, 14)
    else:
        nums = np.arange(3, 14)
    
    zern = create_phase(coeffs=coeffs, num=nums, size=res)
    # plt.imshow(zern)
    # plt.show()
    # print(np.max(zern), np.min(zern))
    
    if multi:
        plane = 'all'
    else:
        plane = 'xy'

    img = sted_psf(zern, res, offset=offset_label, plane=plane, tiptilt=tiptilt)
    
    # scaling to 0-255
    img = normalize_img(img)*255
    img = img.astype(np.uint8)
    return img

def gen_fluor_psf(res=64, offset=False, multi=False):
    """generates a fluor psf at random"""
    coeffs = gen_coeffs()

    if offset:
        offset_label = gen_offset()
    else:
        offset_label = np.asarray([0,0])

    aberr_phase_mask = create_phase(coeffs, res, offset_label)

    if multi:
        plane = 'all'
    else:
        plane = 'xy'
    
    img = fluor_psf(aberr_phase_mask, res, offset=offset_label, plane=plane)
    
    return img, coeffs, offset_label

def get_fluor_psf(res=64, coeffs=np.asarray([0.0]*12), offset_label=[0,0], multi=False):
    """create the psf given a set of coefficients and offsets"""
 
    aberr_phase_mask = create_phase(coeffs, res, offset_label)

    if multi:
        plane = 'all'
    else:
        plane = 'xy'
    
    img = fluor_psf(aberr_phase_mask, res, offset=offset_label, plane=plane)
    return img

# modified from a stack overflow answer
def get_stats(data_path, batch_size, mode='train'):
    """ Finding Dataset Stats for Normalization before Training."""
    dataset = my_classes.PSFDataset(hdf5_path=data_path, mode=mode, transform=my_classes.ToTensor())
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for _, sample_batched in enumerate(loader):
        # print(i_batch, sample_batched['image'].size(),
        sample_batched['label'].size()
        # exit()
        #   for images, _ in loader:
        images = sample_batched['image']
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    print('mean is: {}  |   std is: {}'.format(mean, std))
    return mean, std


def plot_xsection(img3d, name=''):
    fig = plt.figure()
    # plt.colorbar()
    ax1 = fig.add_subplot(1,3,1)
    ax1.set_title('xy')
    ax1.imshow(img3d[0], cmap='hot')
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('xz')
    ax2.imshow(img3d[1], cmap='hot')
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('yz')
    ax3.imshow(img3d[2], cmap='hot')
    fig.suptitle(name, fontsize=16)
    return fig

def plot_xsection_eval(img1, img2, img3):
    # img1 = normalize_img(img1)
    # img2 = normalize_img(img2)
    # img3 = normalize_img(img3)
    
    fig = plt.figure(1)
    ax1 = fig.add_subplot(1,3,1)
    ax1.set_title('xy')
    ax1.imshow(img1[0], cmap='hot')
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('xz')
    ax2.imshow(img1[1], cmap='hot')
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('yz')
    ax3.imshow(img1[2], cmap='hot')
    fig.suptitle('GT', fontsize=16)
    # fig.suptitle('Original', fontsize=16)
    
    fig = plt.figure(2)
    ax1 = fig.add_subplot(1,3,1)
    ax1.set_title('xy')
    ax1.imshow(img2[0], cmap='hot')
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('xz')
    ax2.imshow(img2[1], cmap='hot')
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('yz')
    ax3.imshow(img2[2], cmap='hot')
    fig.suptitle('reconstructed', fontsize=16)
    # fig.suptitle('Reconstructed', fontsize=16)

    fig = plt.figure(3)
    ax1 = fig.add_subplot(1,3,1)
    ax1.set_title('xy')
    ax1.imshow(img3[0], cmap='hot')
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('xz')
    ax2.imshow(img3[1], cmap='hot')
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('yz')
    ax3.imshow(img3[2], cmap='hot')
    # fig.suptitle('Corrected', fontsize=16)
    fig.suptitle('corrected', fontsize=16)
    return fig

def plot_xsection_abber(img1, img2):
    # img1 = normalize_img(img1)
    # img2 = normalize_img(img2)
    fig = plt.figure(1)
    ax1 = fig.add_subplot(1,3,1)
    ax1.set_title('xy')
    ax1.imshow(img1[0], cmap='hot')
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('xz')
    ax2.imshow(img1[1], cmap='hot')
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('yz')
    ax3.imshow(img1[2], cmap='hot')
    fig.suptitle('Original', fontsize=16)
    
    fig = plt.figure(2)
    ax1 = fig.add_subplot(1,3,1)
    ax1.set_title('xy')
    ax1.imshow(img2[0], cmap='hot')
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('xz')
    ax2.imshow(img2[1], cmap='hot')
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('yz')
    ax3.imshow(img2[2], cmap='hot')
    fig.suptitle('Reconstructed', fontsize=16)

    return fig

if __name__ == "__main__":
    for i in range(10):
        print(gen_offset())
    
# TODO: redirect to the original now that it's in the same repo
#########################################################################################
#
#   Functions from Wiebke's Pattern Calculator code that are necessary for redundancy.
#   I want my code to run as a self-contained unit that can be plugged into the pre-
#   existing GUI, this helps me achieve that.

# TODO: change to be actual PC fns
#
#########################################################################################


# ######## fn to create a single zernike phase mask #######
# def create_zernike(size, order, rad=1):
#     """ Calculates the Zernike polynomial of a given order, for the given 
#         image size. Normalizes the polynomials in the center quadrant of the 
#         image to [0,1]. """
#     xcoord, ycoord = create_coords(size)
#     xcoordnorm = (xcoord/np.max(np.abs([np.max(xcoord), np.min(xcoord)])))*2
#     ycoordnorm = (ycoord/np.max(np.abs([np.max(ycoord), np.min(ycoord)])))*2
    
#     rho, phi = cart2polar(xcoord, ycoord)
    
#     # when normalizing rho: factor of two is needed because I'm creating images
#     # double the size for subsequent cropping. Factor of 2 / sqrt(2) is needed 
#     # because I'm working on a square whereas the polynomials are defined on a 
#     # circle
#     # NOTE: this has totally changed
#     rho = rho * 2 * 2 / np.sqrt(2) / rad
#     # print(size, np.min(xcoordnorm), np.max(xcoordnorm), np.min(ycoordnorm), np.max(ycoordnorm),
#     #       np.min(rho), np.max(rho), np.min(phi), np.max(phi))
    
#     if order[1] < 0:
#         zernike = zernike_coeff(rho, np.abs(order)) * np.sin(np.abs(order[1]) * phi)
#     elif order[1] >= 0:
#         zernike = zernike_coeff(rho, order) * np.cos(order[1] * phi)
    
#     #mask = (rho <= 1)
#     #zernike = zernike * mask
#     return zernike

# ######## and its helper fns ##################
# def create_coords(size, off = [0,0]):
#     """ Returns the cartesian coordinates of each pixel in a matrix from the 
#         inputs size and offset (defining the center position). ATTENTION:
#         array of cartesian coordinates is created at 2x the size needed. Will 
#         be cropped later in the workflow for easy offsetting. """
#     x = np.arange((-size[0]/2 + off[0]), (size[0]-size[0]/2 + off[0]))
#     y = np.arange((-size[1]/2 + off[1]), (size[1]-size[1]/2 + off[1]))    
#     xcoords = np.multiply.outer(np.ones(size[0]), y)
#     ycoords = np.multiply.outer(x, np.ones(size[1]))    
#     return xcoords, ycoords

# def cart2polar(x, y):
#     """ Returns normalized polar coordinates for cartesian inputs x and y. """
#     z = x + 1j * y
#     return (np.abs(z)/np.max(np.abs(z)), np.angle(z))

# def zernike_coeff(rho, order):
#     """ Calculates the Zernike coeff for a given order and radius rho. """
#     coeff = 0
#     nn = order[0]
#     mm = np.abs(order[1])
    
#     for kk in range(int((nn - mm)/2)+1):
#         c = (np.power(-1, kk) * mfac(nn - kk))/ \
#             (mfac(kk) * \
#              mfac((nn + mm)/2 - kk) * \
#              mfac((nn - mm)/2 - kk))
#         r = (np.power(rho, nn - 2 * kk))
#         coeff = coeff + c * r
#     return coeff


# ########## fn to create vortex (donut phase mask) ############
# def create_donut(size, rot, amp):
#     """" Creates the phasemask for shaping the 2D donut with the given image 
#         size, rotation and amplitude of the donut. """
#     xcoord, ycoord = create_coords(size)
#     dn = 0.5 / np.pi * np.mod(cart2polar(xcoord, ycoord)[1] + (rot + 180) /
#                               180 * np.pi, 2*np.pi) * amp
#     return dn

# ########## fn to crop phase mask to the right scaling ########

# def crop(full, size, offset = [0,0]):
#     """ Crops the full data to half the size, using the provided offset. """
#     minx = int(size[0]/2 + offset[0])
#     maxx = int(size[0]*3/2 + offset[0])
#     miny = int(size[1]/2 + offset[1])
#     maxy = int(size[1]*3/2 + offset[1])    
#     cropped = full[minx:maxx, miny:maxy]
#     return cropped







