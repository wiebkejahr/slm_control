'''
Project: deep-adapative-optics/utils
--------
@author: Rishabh, hmcgovern
Rishabh's Vector Diffraction code with minimal modification by hmcgovern.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from skimage.transform import resize
from tqdm import tqdm

def integrate(rr, res):
    """
    Using some system parameters (n, f, NA, etc.) and an input phase pattern, it gives a point spread function of set resolution.
    rr: ndarray of phase pattern (for our purposes, the compilation of zernike phase masks)
    res: the (square) resolution of the resulting psf. Typically 64 (x 64)
    """
    ###### Set variables #########
    output_screen_size = 0.7 #set bounds of output aperture in micrometeres
    out_res = res #set resolution in output (output will be out_res X out_res grid)
    inp_res = res #input resolution on which integration will be done (increasing this will give more accurate results
    #                                              but program will run slower)
    z = 0 #(in mm) z=0 represents focal plane and can be changed if one wants to visualize change in z-direction
    n = 1.518 # refractive index of media between lens and output plane
    f = 1.8 # (in mm) effective focal length of the objective
    NA = 1.4 #numerical aperture of system
    #lambd = 775 # wavelength of light in nm
    lambd = 637
    #Set input polarization of light below
    px = 1.0/np.sqrt(2)
    py = 1.0/np.sqrt(2)*1j
    pz = 0

    ###### Calculations #######
    lambd = lambd*(10**(-9))*(1000) # wavelength in mm
    k = 2*np.pi/lambd  #wavevector
    input_aperture_size = np.tan(np.arcsin(NA/n))*f  #calculating the radius of input aperture from NA


    ###########   Declaration  #############
    xp = np.linspace(-output_screen_size,output_screen_size,out_res)*(10**(-3))  #x-space of output plane
    yp = np.linspace(-output_screen_size,output_screen_size,out_res)*(10**(-3))  #y-space of output plane


    x1 = np.linspace(-input_aperture_size,input_aperture_size,inp_res)
    y1 = np.linspace(-input_aperture_size,input_aperture_size,inp_res)
    dx = x1[1]-x1[0]
    x,y = np.meshgrid(x1,y1)   #input meshgrid
    #declare a few required functions for easy access
    r = np.sqrt(x**2+y**2)
    theta = np.arcsin(r/np.sqrt(r**2 + f**2))
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    phi = np.arctan2(y,x)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    ######### Use the space below to define a phase pattern or amplitude pattern for input #####
    
    # NOTE: phase = phi gives perfect donut
    phase = phi + (rr-0.5)*2*np.pi

    amp = np.ones_like(r)

    #############################################################################################
    #
    #
    #                      Integration processing begins
    #
    #
    ############################################################################################

    aperture = np.zeros_like(x)
    aperture[r<input_aperture_size] = 1
    aperture = aperture*amp  #Putting a hard circular aperture of NA keeping the amplitude changes

    ex = np.zeros([len(xp),len(yp)])*0j
    ey = np.zeros([len(xp),len(yp)])*0j
    ez = np.zeros([len(xp),len(yp)])*0j
    # calculating the required functions that need to be integrated out
    temp_ex = aperture*sin_theta*np.sqrt(cos_theta)*((1+(cos_theta-1)*cos_phi**2)*px + ((cos_theta-1)*cos_phi*sin_phi)*py + (-sin_theta*cos_phi)*pz)
    temp_ey = aperture*sin_theta*np.sqrt(cos_theta)*(((cos_theta-1)*cos_phi*sin_phi)*px + (1+(cos_theta-1)*sin_phi**2)*py + (-sin_theta*sin_phi)*pz)
    temp_ez = aperture*sin_theta*np.sqrt(cos_theta)*((sin_theta*cos_phi)*px + (sin_theta*sin_phi)*py + (cos_theta)*pz)
    # since only the integral kernel (the exponential part) changes as the function of output aperture variables
    # only that is calculated inside the loop for performance gains

    i = 0
    j = 0
    #r[np.where(r==0)] = np.nan
    for xi in xp:
        j = 0
        for yi in yp:
            r2 = np.sqrt(xi**2 + yi**2)
            phi2 = np.arctan2(yi,xi)
            temp_exp = np.exp(1j*(phase))
            temp_exp = np.exp(1j*(phase + k*n*r2*sin_theta*np.cos(phi-phi2) + k*n*z*cos_theta)) #need to calculate the integral kernel only once
            jacobian = cos_theta**2/f/r
            jacobian[len(x1)//2, len(y1)//2] = 1 #jacobian becomes infinity for r=0 so, it is manually edited out
            temp = temp_ex*temp_exp*jacobian
            ex[i,j] = np.sum(np.sum(temp))*dx*dx
            temp = temp_ey*temp_exp*jacobian
            ey[i,j] = np.sum(np.sum(temp))*dx*dx
            temp = temp_ez*temp_exp*jacobian
            ez[i,j] = np.sum(np.sum(temp))*dx*dx

            j = j + 1
 
        i = i + 1


    e = np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2)
    e = np.abs(e)**2
    e = e/np.amax(e)


    return e

def integrate_multi(rr, res):
    ###### Set variables #########
    output_screen_size = 0.7 # (in micrometeres) set bounds of output aperture
    z_extent = 0.7 # (in micrometeres) extent in z-direction for XZ and YZ plane results
    out_res = res #set resolution in output (output will be out_res X out_res grid)
    inp_res = res #input resolution on which integration will be done (increasing this will give more accurate results
    #                                              but program will run slower)
    n = 1.518 # refractive index of media between lens and output plane
    f = 1.8 # (in mm) effective focal length of the objective
    NA = 1.4 #numerical aperture of system
    lambd = 775 # wavelength of light in nm

    #Set input polarization of light below
    px = 1.0/np.sqrt(2)
    py = 1.0/np.sqrt(2)*1j
    pz = 0

    ###### Calculations #######
    lambd = lambd*(10**(-9))*(1000) # wavelength in mm
    k = 2*np.pi/lambd  #wavevector
    input_aperture_size = np.tan(np.arcsin(NA/n))*f  #calculating the radius of input aperture from NA

    ###########   Declaration  #############
    xp = np.linspace(-output_screen_size,output_screen_size,out_res)*(10**(-3))  #x-space of output plane
    yp = np.linspace(-output_screen_size,output_screen_size,out_res)*(10**(-3))  #y-space of output plane
    zp = np.linspace(-z_extent,z_extent,out_res)*(10**(-3))  #y-space of output plane


    x1 = np.linspace(-input_aperture_size,input_aperture_size,inp_res)
    y1 = np.linspace(-input_aperture_size,input_aperture_size,inp_res)
    dx = x1[1]-x1[0]
    x,y = np.meshgrid(x1,y1)   #input meshgrid
    #declare a few required functions for easy access
    r = np.sqrt(x**2+y**2)
    theta = np.arcsin(r/np.sqrt(r**2 + f**2))
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    phi = np.arctan2(y,x)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    ######### Use the space below to define a phase pattern or amplitude pattern for input #####

    # rr = np.zeros_like(r)
    # rr[r<2.5] = 1

    # NOTE: phase = phi gives perfect donut
    phase = phi + (rr-0.5)*2*np.pi
    amp = np.ones_like(r)

    #############################################################################################
    #
    #
    #                      Integration processing begins
    #
    #
    ############################################################################################

    aperture = np.zeros_like(x)
    aperture[r<input_aperture_size] = 1
    aperture = aperture*amp  #Putting a hard circular aperture of NA keeping the amplitude changes


    ex = np.zeros([len(xp),len(yp)])*0j
    ey = np.zeros([len(xp),len(yp)])*0j
    ez = np.zeros([len(xp),len(yp)])*0j
    # calculating the required functions that need to be integrated out
    temp_ex = aperture*sin_theta*np.sqrt(cos_theta)*((1+(cos_theta-1)*cos_phi**2)*px + ((cos_theta-1)*cos_phi*sin_phi)*py + (-sin_theta*cos_phi)*pz)
    temp_ey = aperture*sin_theta*np.sqrt(cos_theta)*(((cos_theta-1)*cos_phi*sin_phi)*px + (1+(cos_theta-1)*sin_phi**2)*py + (-sin_theta*sin_phi)*pz)
    temp_ez = aperture*sin_theta*np.sqrt(cos_theta)*((sin_theta*cos_phi)*px + (sin_theta*sin_phi)*py + (cos_theta)*pz)
    # since only the integral kernel (the exponential part) changes as the function of output aperture variables
    # only that is calculated inside the loop for performance gains

    # print ("Calculating XY plane intensity...")

    z = 0
    i = 0
    j = 0
    for xi in xp:
        j = 0
        for yi in yp:
            r2 = np.sqrt(xi**2 + yi**2)
            phi2 = np.arctan2(yi,xi)
            temp_exp = np.exp(1j*(phase + k*n*r2*sin_theta*np.cos(phi-phi2) + k*n*z*cos_theta))
            jacobian = cos_theta**2/f/r
            jacobian[len(x1)//2, len(y1)//2] = 1
            temp = temp_ex*temp_exp*jacobian
            ex[i,j] = np.sum(np.sum(temp))*dx*dx
            temp = temp_ey*temp_exp*jacobian
            ey[i,j] = np.sum(np.sum(temp))*dx*dx
            temp = temp_ez*temp_exp*jacobian
            ez[i,j] = np.sum(np.sum(temp))*dx*dx

            j = j + 1
        # if (i%10 == 0):
        #     print (str(i)+"% completed.")
        i = i + 1

    e = np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2)
    e = np.abs(e)**2
    e = e/np.amax(e)

    e_xy = e

    # For x-z plane
    # print ("Calculating XZ plane intensity...")
    i = 0
    j = 0
    yi = 0
    for xi in xp:
        j = 0
        for z in zp:
            r2 = np.sqrt(xi**2 + yi**2)
            phi2 = np.arctan2(yi,xi)
            temp_exp = np.exp(1j*(phase + k*n*r2*sin_theta*np.cos(phi-phi2) + k*n*z*cos_theta))
            jacobian = cos_theta**2/f/r
            jacobian[len(x1)//2, len(y1)//2] = 1
            temp = temp_ex*temp_exp*jacobian
            ex[i,j] = np.sum(np.sum(temp))*dx*dx
            temp = temp_ey*temp_exp*jacobian
            ey[i,j] = np.sum(np.sum(temp))*dx*dx
            temp = temp_ez*temp_exp*jacobian
            ez[i,j] = np.sum(np.sum(temp))*dx*dx

            j = j + 1
        # if (i%10 == 0):
        #     print (str(i)+"% completed.")
        i = i + 1

    e = np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2)
    e = np.abs(e)**2
    e = e/np.amax(e)

    e_xz = e

    # For x-y plane
    # print ("Calculating YZ plane intensity...")
    i = 0
    j = 0
    xi = 0
    for z in zp:
        j = 0
        for yi in yp:
            r2 = np.sqrt(xi**2 + yi**2)
            phi2 = np.arctan2(yi,xi)
            temp_exp = np.exp(1j*(phase + k*n*r2*sin_theta*np.cos(phi-phi2) + k*n*z*cos_theta))
            jacobian = cos_theta**2/f/r
            jacobian[len(x1)//2, len(y1)//2] = 1
            temp = temp_ex*temp_exp*jacobian
            ex[i,j] = np.sum(np.sum(temp))*dx*dx
            temp = temp_ey*temp_exp*jacobian
            ey[i,j] = np.sum(np.sum(temp))*dx*dx
            temp = temp_ez*temp_exp*jacobian
            ez[i,j] = np.sum(np.sum(temp))*dx*dx

            j = j + 1
        # if (i%10 == 0):
        #     print (str(i)+"% completed.")
        i = i + 1

    e = np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2)
    e = np.abs(e)**2
    e = e/np.amax(e)

    e_yz = e

    return np.stack((e_xy, e_xz, e_yz)) # (3, 64, 64)


