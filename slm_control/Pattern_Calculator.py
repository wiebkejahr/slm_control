#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pattern_Calculator.py


"""
    Created on Tue Oct  9 08:15:36 2018
    @author: wjahr
    
    Mathematical backend for creating the different patterns and for image
    processing. Includes wrappers and convenience functions for coordinate 
    transforms, image loading/saving, cropping, summation.
    Implements:
        - mathematical description of the different vortizes to create STED 
          patterns.
        - Zernike polynomials for adaptive optics.
        - Blazed grating for holographic operation of SLM, and for aligment of 
          beam position within the microscope.
    Can be used stand-alone by calling functions from included main. Within the 
    SLM control framework, the functions are called by the various 
    pattern-creation routines.
    
    Copyright (C) 2022 Wiebke Jahr

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import numpy as np
from math import factorial as mfac
import matplotlib.pyplot as plt

from PIL import Image


def load_image(img_path):
    """ Loads the image from the provided img_path into a numpy array."""
    im = Image.open(img_path)
    image = np.array(im)
    return image


def save_image(img, img_path, img_name):
    """ Saves the provided matrix img as an image under the path img_path and
        the name img_name. """
    image = Image.fromarray((img).astype(np.uint8))
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
    z_norm = np.mean([(np.max(x) - np.min(x)), (np.max(y) - np.min(y))])
    return(np.abs(z) / z_norm, np.angle(z))


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


def normalize_radius(obj_ba, mag, slm_px, size_slm):
    """ Normalizes the radius to the size of the SLM pixels and the radius of
        the laser beam. """
    radius_slm = obj_ba / mag / slm_px / np.mean(size_slm * 2)
    return radius_slm


def get_mm2px(slm_px, M):
    """" Converts mm in objective backaperture into px on SLM.
        Needed eg when scaling offsets. 
        Inputs:
            aperture diameter in mm,
            slm_px size in mm / px, 
            M optical magnification
        objective_backaperture (in px, on SLM) = d_obj / slm_px / M
        1 mm (objective backaperture) => objective_backaperture (in px, on SLM) / d_obj
        1 mm (objective backaperture) => 1 / (slm_px * M)"""
    
    mm = 1 / (slm_px * M)
    return mm


def bfp_radius(M, NA, f_TL):
    """ Takes magnification M and NA of the objective, as well as focal length
        of the tube lens, to calculate the radius of the objective's back focal
        plane. """
    return NA * f_TL / M


def crop(full, size, offset = [0, 0]):
    """ Crops the full data to half the size, using the provided offset. 
        Convention for directions and signs agrees with Abberior. """
    minx = int((size[0] + 0) / 2 - offset[1]+0)
    maxx = int((size[0] * 3 + 0) / 2 - offset[1]+0)
    miny = int((size[1] + 0) / 2 - offset[0]+0)
    maxy = int((size[1] * 3 + 0) / 2 - offset[0]+0)
    cropped = full[minx:maxx, miny:maxy]
    return cropped


def create_coords(size, off = [0,0], res = None):
    """ Returns the cartesian coordinates of each pixel in a matrix from the 
        inputs size and offset (defining the center position). ATTENTION:
        size that's passed in needs to be two times the size needed due to 
        cropping later. Offset here will be offset of the pattern in the 
        backaperture."""
    if res == None:
        res = size
    x = np.linspace((-(size[0]/2) + off[0]), (size[0]-size[0]/2 + off[0]), res[0])
    y = np.linspace((-(size[1]/2) + off[1]), (size[1]-size[1]/2 + off[1]), res[1])
    xcoords, ycoords = np.meshgrid(y,x)
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


def create_zernike(size, order, amp = 1, radscale=1):
    """ Calculates the Zernike polynomial of a given order, for the given 
        image size. Normalizes the polynomials in the center quadrant of the 
        image to [0,1]. """
    xcoord, ycoord = create_coords(size)
    rho, phi = cart2polar(xcoord, ycoord)
    rho = rho * 4 / radscale

    if order[1] < 0:
        zernike = zernike_coeff(rho, np.abs(order)) * np.sin(np.abs(order[1]) * phi)
    elif order[1] >= 0:
        zernike = zernike_coeff(rho, order) * np.cos(order[1] * phi)

    # division by 2 would be needed to make it compatible with the amplitudes
    # at the Abberior. Do we want that? Or do we want to be consistent?
    return zernike * amp #/2


def create_rect(size, a, b, rot, amp = 1, radscale = 1):
    """ Creates a rectangle of size, with relative sides a, b, rotated by rot
        deg. """
    xcoord, ycoord = create_coords(size)
    rho, phi = cart2polar(xcoord, ycoord)
    rho = rho * 4 / radscale
    rect = rho <= np.minimum(a / np.abs(np.cos(phi - (rot + 180) / 180 * np.pi)), 
                             b / np.abs(np.sin(phi - (rot + 180) / 180 * np.pi)))
    return rect * amp


def create_ellipse(size, r_min, r_maj, rot, amp = 1, radscale = 1):
    """ Creates an ellipse of size with relative minor and major radii r_min,
        r_maj, and rotated by rot degree. """
    xcoord, ycoord = create_coords(size)
    rho, phi = cart2polar(xcoord, ycoord)
    rho = rho * 4 / radscale
    
    ecc = np.sqrt(1 - r_min ** 2 / r_maj ** 2)
    ellipse = rho < (r_min / np.sqrt( 1 - 
                    (ecc * np.cos(phi - (rot + 180) / 180 * np.pi)) ** 2 ))
    return ellipse * amp


def create_ring(size, r_inner, r_outer, amp = 1, radscale = 1):
    """ Creates a ring with inner and outer radius and amplitude 1. Adding 
        the ring to a pattern creates phase modulation. Multiplying creates 
        amplitude modulation."""
    xcoord, ycoord = create_coords(size)
    rho = cart2polar(xcoord, ycoord)[0]
    rho = rho * 4 / radscale
    ring = (rho >= r_inner) * np.ones(size) + (rho <= r_outer) * np.ones(size)
    return (ring - 1) * amp


def create_gauss(size, amp = 1, radscale = 1):
    return np.ones(size) * amp


def create_donut(size, rot, amp = 1, radscale = 1):
    """" Creates the phasemask for shaping the 2D donut with the given image 
        size, rotation and amplitude of the donut. """
    xcoord, ycoord = create_coords(size)
    #I don't think mod division is needed here; phasewrapping later solves this
    # just add all the angles
    # offset of pi is needed to fit with Abberior
    # dn = 0.5 / np.pi * np.mod(cart2polar(xcoord, ycoord)[1] + (rot + 180) /
    #                           180 * np.pi, 2*np.pi)
    dn = (cart2polar(xcoord, ycoord)[1] + rot / 180 * np.pi + np.pi) / np.pi / 2
    return dn * amp


def create_bottleneck(size, radius, amp = 1, radscale = 1):
    """ Creates the phasemask for shaping the 3D donut with the given size,
        radius and amplitude. Is essentially and alias for create_ring, with
        r_inner = 0 and amplitude scaling. """
    bn = create_ring(size, 0, radius, amp, radscale)
    return bn


def create_segments(size, rot, steps, amp = 1, radscale = 1):
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


def create_bessel(size, amp = 1, radscale = 1):
    xcoord, ycoord = create_coords(size)
    rho = cart2polar(xcoord, ycoord)[0]
    axicon = rho / np.max(rho) * amp
    return axicon


def create_bivortex(size, radius, rot, amp = 1, radscale = 1):
    """ Creates a Bivortex: for radii smaller than the parameter radius, a 
        normal vortex with rotation rot and amplitude amp. For radii larger 
        than radius, a vortex that is rotated by 180 degrees. Will create a 
        coherent overlay of bottle and donut beams, hopefully making z-donut
        more robust to spherical aberrations as in Pereira ... Maiato et al.,
        Optics Express, 2019. """
    mask = create_ring(size, 0, radius, 1, radscale)
    bivortex =  (mask * create_donut(size, rot, amp) + 
                (mask * (-1) + 1) * create_donut(size, rot + 180, amp))
    return bivortex

        
def compute_vortex(mode, size, rot, rad, steps, amp = 1, radscale = 1):
    """ Depending on the mode selected for the vortex, creates the phasemasks for
        - the donut ("2D STED")
        - the bottleneck beam ("3D STED")
        - returns an array of zeros ("Gauss")
        - "Segments"
        - "Bivortex"
        Input parameters are size of the image, rotation, radius and amplitude. 
        Cases Code Input and From File are handled by GUI."""
    img = np.zeros(size)
    if mode == "2D STED":
        img = create_donut(size, rot, amp)
    elif mode == "3D STED":
        img = create_bottleneck(size, rad, amp, radscale)
    elif mode == "Gauss":
        img = create_gauss(size, amp)
    elif mode == "Segments":
        img = create_segments(size, rot, steps, amp)
    elif mode == "Bivortex":
        img = create_bivortex(size, rad, rot, amp, radscale)
    
    return img


def correct_flatfield(path):
    """ Loads and returns the flatfield correction image from the provided path. """
    ff = load_image(path)
    return ff

    
def zern_sum(size, ratios, orders, radscale = 1):
    """ Calculates an aberration correction by summing up Zernike polynomials.
        Output with given size and offsets. Ratios is a 1D array containing the
        weights of the Zernike polynomials. Orders is a 1D of the same length 
        as ratios, each entry containing the [n, m] orders of the Zernike
        polynomial. """
    ab = np.zeros(size)
    for oo, ooo in enumerate(orders):
        zernike = create_zernike(size, ooo, amp = 1, radscale = radscale)
        ab = ab + ratios[oo]*zernike
    return ab


def blazed_grating(size, slope, slm_px):
    """ DEPRECATED:
        Blazed grating is now created using Zernike (-1,1) and (1,1).
        Creates the blazed grating in the provided size, with the provided slope
        in x and y. Does so by calculating a tilted surface, then normalizing
        to the slm size and pixel size such, that the provided slopes are in 
        units of 1/mm. Phasewraps to create the blazed grating from the tilted
        surface. """
    xcoord, ycoord = create_coords(size, [0,0])
    surf = - (xcoord * slope[0] + ycoord * slope[1])
    
    # different cases are necessary because arctan is ugly:
    # for gratings blazed only in one direction  (i.e. one of the slopes = 0)
    # division and multiplication through/by zero need to be caught
    print("blazed grating called")
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


def double_blazed_grating(size, drctn , phase = 1, amp = 1, radscale = 1):
    dbl_blz = create_zernike(size, [1, drctn], amp, radscale)    
    dbl_blz = phase_wrap(dbl_blz, phase)
    mask = dbl_blz > phase / 2
    dbl_blz[mask] = phase - dbl_blz[mask]
    
    return dbl_blz
        
        

# bunch of things left sitting around from testing
if __name__ == "__main__":
    
    size = np.asarray([200,200])
    path = 'patterns/'
    rot = 0
    radius = 1
    amp = 1
    offset = [0,0]
    phase = 1
    
    orders = [[0,0],
              [1,-1], [1,1],
              [2,-2], [2,0], [2,2],
              [3,-3], [3,-1], [3,1],[3,3],
              [4,-4], [4,-2], [4,0], [4,2], [4,4],
              [5,-5], [5,-3], [5,-1], [5,1], [5,3], [5,5],
              [6,-6], [6,-4], [6,-2], [6,0], [6,2], [6,4], [6,6]]
       
    f1 = plt.figure(num = 3, figsize = (10,10), dpi = 100)
    
    for ii, oo in enumerate(orders):
        
        zernike = create_zernike(size*2, oo, 1)
        
        print("mean zernike ", ii, oo, np.mean(zernike))
        ax = plt.subplot2grid((np.max(orders)+1,(np.max(orders)+1)*2), 
                                (oo[0], oo[1] + np.max(orders)), colspan = 2)
        
        im = ax.imshow(crop(zernike, size, offset), interpolation = 'Nearest', cmap = 'RdYlBu', clim = [-1,1])
        im.cmap.set_over('white')
        im.cmap.set_under('black')
        
        circle = plt.Circle((size[1]/2, size[0]/2), size[0]/2, lw= 0.1, edgecolor = 'k', facecolor='None')
        ax.add_artist(circle)
        circle = plt.Circle((size[1]/2, size[0]/2), size[1]/2, lw= 0.1, edgecolor = 'k', facecolor='None')
        ax.add_artist(circle)
        circle = plt.Circle((size[1]/2, size[0]/2), np.mean(size)/2, lw= 0.1, edgecolor = 'k', facecolor='None')
        ax.add_artist(circle)
        
        ax.set_xticks([]), ax.set_yticks([])
        cbar_ax = plt.subplot2grid((7,14), (0, 1), rowspan = 3)
        f1.colorbar(im, cax = cbar_ax)
        cbar_ax.yaxis.set_ticks_position('left')
        cbar_ax.invert_yaxis()
    plt.show()
