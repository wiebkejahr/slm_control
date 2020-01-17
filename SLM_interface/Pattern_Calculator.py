#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 08:15:36 2018

@author: wjahr
"""

import numpy as np
from math import factorial as mfac
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs

from PIL import Image


def load_image(img_path):
    """ Loads the image from the provided img_path into a numpy array."""
    im = Image.open(img_path)
    image = np.array(im)
    return image

def save_image(img, img_path, img_name):
    """ Saves the provided matrix img as an image under the path img_path and
        the name img_name. """
    image = Image.fromarray((img*255).astype(np.uint8))
    image.save(img_path+img_name)   
        
def add_images(images):
    """ Takes and array of input images, and the image size. Then adds all
        individual images and returns the sum. """
    image = np.zeros(np.shape(images[0]))
    for img in images:
        image = image + img
    return image

def stitch_images(img_l, img_r):
    """ Stitches the provided left and right images into one image of twice the
        width."""
    stitched = np.hstack((img_l, img_r))
    return stitched

def phase_wrap(image, phase):
    """ Phase wraps the image into the available contrast by performing a
        modulo division. """
    pw = np.mod(image, phase)
    return pw

def cart2polar(x, y):
    """ Returns normalized polar coordinates for cartesian inputs x and y. """
    z = x + 1j * y
    return (np.abs(z)/np.max(np.abs(z)), np.angle(z))

def polar2cart(r, theta):
    """ Returns cartesian coordinates for r and theta polar coordinates. """
    z = r * np.exp( 1j * theta)
    return (np.real(z), np.imag(z))

def normalize_img(img):
    """ Normalizes the image data  to [0,1]."""
    img_norm = np.zeros([np.size(img, 0), np.size(img, 1)])
    if np.max(img) != 0:
        if (np.max(img) - np.min(img)) != 0:
            img_norm = 1 / (np.max(img) - np.min(img)) * (img - np.min(img))

    return img_norm

def normalize_radius(laser_radius, slm_px, slm_size):
    """ Normalizes the radius to the size of the SLM pixels and the radius of
        the laser beam. """
    #radnorm = laser_radius / slm_px #/ slm_size[0]
    radnorm = laser_radius / slm_px / slm_size[1]
    # print(radnorm)
    return radnorm

def bfp_radius(M, NA, f_TL):
    """ Takes magnification M and NA of the objective, as well as focal length
        of the tube lens, to calculate the radius of the objective's back focal
        plane. """
    return NA * f_TL / M

def crop(full, size, offset = [0,0]):
    """ Crops the full data to half the size, using the provided offset. """
    minx = int(size[0]/2 + offset[0])
    maxx = int(size[0]*3/2 + offset[0])
    miny = int(size[1]/2 + offset[1])
    maxy = int(size[1]*3/2 + offset[1])    
    cropped = full[minx:maxx, miny:maxy]
    return cropped

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


def create_gauss(size):
    return np.zeros(size)


def create_donut(size, rot, amp):
    """" Creates the phasemask for shaping the 2D donut with the given image 
        size, rotation and amplitude of the donut. """
    xcoord, ycoord = create_coords(size)
    dn = 0.5 / np.pi * np.mod(cart2polar(xcoord, ycoord)[1] + (rot + 180) /
                              180 * np.pi, 2*np.pi) * amp
    return dn


def create_bottleneck(size, radius, amp):
    """ Creates the phasemask for shaping the 3D donut with the given size,
        radius and amplitude. """
    xcoord, ycoord = create_coords(size)
    bn = (cart2polar(xcoord, ycoord)[0] <= radius) * amp
    print(radius)
    return bn


def create_segments(size, rot, amp, steps):
    """ Creates segmented phase masks depending on the number of steps:
        1 step: Half moon phase mask for "broetchenmode", hollow light sheets etc
        2 steps: 
        3 steps: Easy STED phase mask
        ..."""
    xcoord, ycoord = create_coords(size)    
    rad_cont = np.mod(cart2polar(xcoord, ycoord)[1] + (rot + 180) / 180 * np.pi,
                      2*np.pi)
    segments = amp / (steps + 1) * np.floor_divide(rad_cont, 
                                                     2 * np.pi / (steps + 1))
    return segments


def create_bivortex(size, radius, rot, amp):
    """ Creates a Bivortex: for radii smaller than the parameter radius, a 
        normal vortex with rotation rot and amplitude amp. For radii larger 
        than radius, a vortex that is rotated by 180 degrees. Will create a 
        coherent overlay of bottle and donut beams, hopefully making z-donut
        more robust to spherical aberrations as in Pereira ... Maiato et al.,
        Optics Express, 2019. """
    xcoord, ycoord = create_coords(size)
    mask = (cart2polar(xcoord, ycoord)[0] <= radius)
    bivortex = (mask * create_donut(size, rot, amp) + 
                (mask * (-1) + 1) * create_donut(size, rot + 180, amp))
    return bivortex

        
def compute_vortex(mode, size, rot, rad, amp, steps):
    """ Depending on the mode selected for the vortex, creates the phasemasks for
        - the donut ("2D STED")
        - the bottleneck beam ("3D STED")
        - returns an array of zeros ("Gauss")
        - "Segments"
        - "Bivortex"
        Input parameters are size of the image, rotation, radius and amplitude. """
    img = np.zeros(size)
    if mode == "2D STED":
        img = create_donut(size, rot, amp)
    elif mode == "3D STED":
        img = create_bottleneck(size, rad, amp)
    elif mode == "Gauss":
        img = np.zeros(size)
    elif mode == "Segments":
        img = create_segments(size, rot, amp, steps)
    elif mode == "Bivortex":
        img = create_bivortex(size, rad, rot, amp)
    elif mode == "From File":
        img = np.zeros(size)
        print(img.size())
        print("TODO From File")
    return img


def correct_flatfield(path):
    """ Loads and returns the flatfield correction image from the provided path. """
    ff = load_image(path)
    return ff

    
def correct_aberrations(size, ratios, orders, off = [0,0]):
    """ Calculates an aberration correction by summing up Zernike polynomials.
        Output with given size and offsets. Ratios is a 1D array containing the
        weights of the Zernike polynomials. Orders is a 1D of the same length 
        as ratios, each entry containing the [n, m] orders of the Zernike
        polynomial. """
    ab = np.zeros(size)
    for oo, ooo in enumerate(orders):
        zernike = create_zernike(size, off, ooo)
        ab = ab + ratios[oo]*zernike
    return ab


def blazed_grating(size, slope, slm_px):
    """ Creates the blazed grating in the provided size, with the provided slope
        in x and y. Does so by calculating a tilted surface, then normalizing
        to the slm size and pixel size such, that the provided slopes are in 
        units of 1/mm. Phasewraps to create the blazed grating from the tilted
        surface. """
    xcoord, ycoord = create_coords(size, [0,0])
    surf = (xcoord * slope[0] + ycoord * slope[1])
    
    # different cases are necessary because arctan is ugly:
    # for gratings blazed only in one direction  (i.e. one of the slopes = 0)
    # division and multiplication through/by zero need to be caught
    if slope[0] != 0 and slope[1] != 0:
        # ensure that arctan is always calculated as ratio from large/small slope
        sx = np.min(np.abs(slope))
        sy = np.max(np.abs(slope))
        surf =  1 / (surf.max() - surf.min()) * (surf - surf.min()) \
                * (np.abs(slope[0]) + np.abs(slope[1])) \
                / (np.sin(np.arctan(sy/sx))) * size[1] * slm_px
    elif slope[0] != 0 and slope[1] == 0:
        surf = 1 / (surf.max() - surf.min()) * (surf - surf.min()) \
              * (np.abs(slope[0]) + np.abs(slope[1])) \
              * size[1] * slm_px
    elif slope[1] != 0 and slope[0] == 0:
        surf = 1 / (surf.max() - surf.min()) * (surf - surf.min()) \
              * (np.abs(slope[0]) + np.abs(slope[1])) \
              * size[0] * slm_px
    return surf


# bunch of things left sitting around from testing
if __name__ == "__main__":
    
    #size = np.asarray([600, 792])
    size = np.asarray([100,100])
    path = 'patterns/'
    imgname = 'test.bmp'
    rot = 0
    amp = 1
    offset = [0,0]
    
#    rot = 0
#    amp = 1
#    offset = [0,0]
#    size = np.asarray([500,500])
#    
#    zern = create_zernike(size, [2,0])
#    plt.figure()
#    plt.imshow(zern)
#    plt.show()
    
#    mpl.rc('axes', edgecolor='white')
#    mpl.rc('xtick', color = 'white')
#    mpl.rc('ytick', color = 'white')
#    
#    mode = "Segments"
#    path = 'patterns/'
#    rot = 0
#    rad = 100
#    amp = 1
#    steps = 3
##    
#    size = np.asarray([1200, 792])
#    offset = np.asarray([0, 0])
#
#    donut = np.uint8(create_donut(size, rot, amp)*255)
#    bottle = np.uint8(create_bottleneck(size, rad, amp)*255)
#    print(np.min(donut), np.max(donut))    
#    print(np.min(bottle), np.max(bottle))
#    
#    im = Image.fromarray(donut)
#    im.save("/Users/wjahr/Seafile/Synch/2P_STED/Patterns_Python/Donut.bmp")
#    im = Image.fromarray(bottle)
#    im.save("/Users/wjahr/Seafile/Synch/2P_STED/Patterns_Python/Bottle.bmp")
    
#    f1 = plt.figure(1)
#    plt.imshow(create_donut(size, rot, amp))
    
#    f2 = plt.figure(2)
#    plt.imshow(create_bottleneck(size, rad, amp))
#    plt.subplot(222)
#    plt.imshow(create_halfmoon(size, 45, 1))
#    plt.subplot(223)
#    plt.imshow(create_halfmoon(size, 90, 1))
#    plt.subplot(224)
#    plt.imshow(create_halfmoon(size, 180, 1))
#
#
    orders = [[0,0],
              [1,-1], [1,1],
              [2,-2], [2,0], [2,2],
              [3,-3], [3,-1], [3,1],[3,3],
              [4,-4], [4,-2], [4,0], [4,2], [4,4],
              [5,-5], [5,-3], [5,-1], [5,1], [5,3], [5,5],
              [6,-6], [6,-4], [6,-2], [6,0], [6,2], [6,4], [6,6]]
       
    f2 = plt.figure(num = 3, figsize = (5,5), dpi = 100)
    
    for ii, oo in enumerate(orders):
#        plt.figure(ii)
#        zernike = create_zernike(2*size, oo)        
#        plt.imshow(crop(zernike, size, [0,0]))
#        plt.show()
        
        zernike = create_zernike(size*2, oo, .5)
        #ax = plt.subplot(gs[oo[0], oo[1] + 6])
        print(ii, oo)
#        if oo == [1, -1]:
#            zernike = np.zeros_like(zernike)
        ax = plt.subplot2grid((7,14), (oo[0], oo[1] + 6), colspan = 2)
        
        #im = ax.imshow(zernike, interpolation = 'Nearest', cmap = 'RdYlBu', clim = [-1,1])
        im = ax.imshow(crop(zernike, size, offset), interpolation = 'Nearest', cmap = 'RdYlBu', clim = [-1,1])
        circle = plt.Circle((size[0]/2, size[0]/2), size[0]/2, edgecolor = 'k', facecolor='None')
        ax.add_artist(circle)
        
        #ax.set_aspect(1.0)
        ax.set_xticks([]), ax.set_yticks([])
        cbar_ax = plt.subplot2grid((7,14), (0, 1), rowspan = 3)
        f2.colorbar(im, cax = cbar_ax)
        cbar_ax.yaxis.set_ticks_position('left')
        cbar_ax.invert_yaxis()
    
#        f3 = plt.figure(num = 3, figsize = (4,8), dpi = 100)
#        ax = f3.add_subplot(422)
#        ax.imshow(ls_norm, interpolation = 'Nearest', cmap = 'RdYlBu')
#        #ax.set_aspect(1.0)
#        ax.set_xticks([]), ax.set_yticks([])
#        ax = f3.add_subplot(421)
#        ax.imshow(rs_norm, interpolation = 'Nearest', cmap = 'RdYlBu')
#        #ax.set_aspect(1.0)
#        ax.set_xticks([]), ax.set_yticks([])
#        
#        ax = f3.add_subplot(424)
#        ax.imshow(ls_norm_crop, interpolation = 'Nearest', cmap = 'RdYlBu')
#        #ax.set_aspect(1.0)
#        ax.set_xticks([]), ax.set_yticks([])
#        ax = f3.add_subplot(423)
#        ax.imshow(rs_norm_crop, interpolation = 'Nearest', cmap = 'RdYlBu')
#        #ax.set_aspect(1.0)
#        ax.set_xticks([]), ax.set_yticks([])
#        
#        ax = f3.add_subplot(426)
#        ax.imshow(ls_ph, interpolation = 'Nearest', cmap = 'RdYlBu')
#        #ax.set_aspect(1.0)
#        ax.set_xticks([]), ax.set_yticks([])
#        ax = f3.add_subplot(425)
#        ax.imshow(rs_ph, interpolation = 'Nearest', cmap = 'RdYlBu')
#        #ax.set_aspect(1.0)
#        ax.set_xticks([]), ax.set_yticks([])
#        
#        plt.xticks(rotation = 90, color = "white")
#        plt.yticks(rotation = 90, color = "white")
#        ax = f3.add_subplot(428)
#        ax.plot(ls_add[250, :])
#        ax.plot(ls_ph[250, :])
#        ax.invert_yaxis()
#        ax.set_xticks([])
#        ylabels = ax.get_yticklabels()
#        for yy in ylabels:
#            yy.set_rotation(-90)
#    
#        ax = f3.add_subplot(427)
#        ax.plot(rs_add[250, :])
#        ax.plot(rs_ph[250,:])
#        ax.invert_yaxis()
#        ax.set_xticks([])
#        ylabels = ax.get_yticklabels()
#        for yy in ylabels:
#            yy.set_rotation(-90)
#        
#        
#    f1.savefig("/Users/wjahr/Seafile/Synch/2P_STED/Patterns_Python/Donut.tif", bbox_inches = 0, transparent = True)
#    f2.savefig("/Users/wjahr/Seafile/Synch/2P_STED/Patterns_Python/Bottle.tif", bbox_inches = 0, transparent = True)
#        f3.savefig("/Users/wjahr/Seafile/Synch/Labmeeting/Labmeeting_20181029/Adding.png", bbox_inches = 0, transparent = True)    
