import numpy as np
import matplotlib.pyplot as plt
import json

# local modules
import Pattern_Calculator as PC
from autoalign.utils import vector_diffraction as vd
import autoalgin.utils.vector_diffraction as vd

def stim_em(exc, sted, isat):
    #ln(2) is needed because I_sat is "half life", not lifetime
    exponent = - np.log(2) * sted / isat
    depleted = exc * np.exp(exponent)
    return depleted


def sted_psf(rr, res=64, offset=[0,0], plane='xy'):
    
    # with open('utils/params.txt') as json_file:
    with open('parameters/laser_params.txt') as json_file:
        data = json.load(json_file)
        # params is a text file containing a dict of dicts
        optical_params_sted = data["optical_params_sted"]
        optical_params_gauss = data["optical_params_gauss"]
    
    ############# ABBERIOR #############
    
    # calculate laser power dependent scaling factor
    lp_scale_gauss = vd.calc_lp(optical_params_gauss["P_laser"], 
                                optical_params_gauss["rep_rate"], 
                                optical_params_gauss["pulse_length"])
    lp_scale_sted = vd.calc_lp(optical_params_sted["P_laser"], 
                            optical_params_sted["rep_rate"], 
                            optical_params_sted["pulse_length"])
    #lp_scale_sted = vd.calc_lp(1,1,1)

    radius = 0.64/2
    I_sat = 11 * 1e6 * 1e-2 # in MW / cm^2, convert to W/mm^2


    ##########################################################################
    #
    # Polarization
    #
    ##########################################################################

    # right handed circular
    #polarization = [1.0/np.sqrt(2)*1j, 1.0/np.sqrt(2), 0]

    # left handed circular
    polarization = [1.0/np.sqrt(2), 1.0/np.sqrt(2)*1j, 0]

    # linear x
    #polarization = [1.0, 0, 0]

    #linear y
    #polarization = [0, 1.0, 0]

    numerical_params = {
    "out_scrn_size" : 1,
    "z_extent" : 1.0,
    "out_res" : res, 
    "inp_res" : res+1 
    }

    ##########################################################################
    #
    # phase pattern and amplitude pattern for STED beam
    #
    ##########################################################################

    # calculate radius and angles needed to define phasemasks
    [r, phi] = vd.calc_rphi(optical_params_sted["obj_ba"]/2,
                            numerical_params["inp_res"], optical_params_sted["offset"])


    #rr = np.zeros_like(r)
    #rr[r<2.5] = 1

    # gauss
    #phasemask = np.zeros_like(r)

    # xy donut
    phasemask = PC.create_donut(size=numerical_params["output_res"], rot=0)
    # phasemask = phi + np.pi*(rr-0.5)
    # phasemask = phi

    # z donut
    #phasemask = (r < 0.64 / 2 / 0.9 * input_aperture_size) * np.pi

    # bivortex
    #phasemask = phi + np.pi*rr

    # phase step
    #phasemask = ((phi < -np.pi/2) + (phi > np.pi/2)) * np.pi
    #phasemask = ((phi < 0) + (phi > np.pi)) * np.pi

    #amp = np.ones_like(r) * efield
    amplitude = np.ones_like(r)

    # intensity distribution depletion (STED)
    [sted_xy, sted_xz, sted_yz, sted_xyz] = vd.vector_diffraction(
        optical_params_sted,numerical_params, polarization,phasemask, 
        amplitude, lp_scale_sted, plane=plane, offset=offset)

    if plane == 'xy':
        return sted_xy
    elif plane == 'all':
        return np.stack((sted_xy, sted_xz, sted_yz), axis=0)

    else:
        raise("Plane argument not valid. Must be one of: ['xy', 'all']")
 


def fluor_psf(rr, res=64, offset=[0,0], plane='xy'):

    # with open('utils/params.txt') as json_file:
    with open('parameters/laser_params.txt') as json_file:
        data = json.load(json_file)
        # params is a text file containing a dict of dicts
        optical_params_sted = data["optical_params_sted"]
        optical_params_gauss = data["optical_params_gauss"]
    


    # ############# ABBERIOR #############
    
    # calculate laser power dependent scaling factor
    lp_scale_gauss = vd.calc_lp(optical_params_gauss["P_laser"], 
                                optical_params_gauss["rep_rate"], 
                                optical_params_gauss["pulse_length"])
    lp_scale_sted = vd.calc_lp(optical_params_sted["P_laser"], 
                            optical_params_sted["rep_rate"], 
                            optical_params_sted["pulse_length"])
    #lp_scale_sted = vd.calc_lp(1,1,1)

    radius = 0.64/2
    I_sat = 11 * 1e6 * 1e-2 # in MW / cm^2, convert to W/mm^2


    ##########################################################################
    #
    # Polarization
    #
    ##########################################################################

    # right handed circular
    #polarization = [1.0/np.sqrt(2)*1j, 1.0/np.sqrt(2), 0]

    # left handed circular
    polarization = [1.0/np.sqrt(2), 1.0/np.sqrt(2)*1j, 0]

    # linear x
    #polarization = [1.0, 0, 0]

    #linear y
    #polarization = [0, 1.0, 0]
    
    numerical_params = {
    "out_scrn_size" : 0.05,
    "z_extent" : 1.0,
    "out_res" : res, #150,
    "inp_res" : res+1 #151
    }

 
    ##########################################################################
    #
    # phase pattern and amplitude pattern for Gaussian beam
    #
    ##########################################################################

    # calculate radius and angles needed to define phasemasks
    [r, phi] = vd.calc_rphi(optical_params_gauss["obj_ba"]/2,
                            numerical_params["inp_res"], optical_params_gauss["offset"])


    #rr = np.zeros_like(r)
    #rr[r<2.5] = 1

    # gauss
    phasemask = np.zeros_like(r)
    

    # phasemask = rr
    # xy donut
    #phasemask = phi

    # z donut
    #phasemask = (r < 0.64 / 2 / 0.9 * input_aperture_size) * np.pi

    # bivortex
    #phasemask = phi + np.pi*rr

    # phase step
    #phasemask = ((phi < -np.pi/2) + (phi > np.pi/2)) * np.pi
    #phasemask = ((phi < 0) + (phi > np.pi)) * np.pi

    #amp = np.ones_like(r) * efield
    amplitude = np.ones_like(r)

    

    # intensity distribution excitation (gauss)
    [gauss_xy, gauss_xz, gauss_yz, gauss_xyz] = vd.vector_diffraction(optical_params_gauss,numerical_params, 
                                                                    polarization,phasemask, amplitude, 
                                                                    lp_scale_gauss, plane = plane) # not specifying offset so it defaults to [0,0]
    gauss_xy = gauss_xy * optical_params_gauss["transmittance"]
    gauss_xz = gauss_xz * optical_params_gauss["transmittance"] # when plane is 'xy' this should just be 0
    gauss_yz = gauss_yz * optical_params_gauss["transmittance"] # when plane is 'xy' this should just be 0
    # gauss_xyz = gauss_xyz * optical_params_gauss["transmittance"]

    phasemask = PC.create_donut(numerical_params["output_res"], rot=0)
    # phasemask = phi + np.pi*(rr-0.5)

    # intensity distribution depletion (STED)
    [sted_xy, sted_xz, sted_yz, sted_xyz] = vd.vector_diffraction(
        optical_params_sted,numerical_params, polarization,phasemask, 
        amplitude, lp_scale_sted, plane = plane, offset=offset) 
    # sted_xy = sted_psf(rr, res)
    sted_xy = sted_xy * optical_params_sted["transmittance"]
    sted_xz = sted_xz * optical_params_sted["transmittance"] # when plane is 'xy' this should just be 0
    sted_yz = sted_yz * optical_params_sted["transmittance"] # when plane is 'xy' this should just be 0
    # sted_xyz = sted_xyz * optical_params_sted["transmittance"]


    # calculate depleted PSF
    em_psf_xy = stim_em(gauss_xy, sted_xy, I_sat)
    em_psf_xz = stim_em(gauss_xz, sted_xz, I_sat)
    em_psf_yz = stim_em(gauss_yz, sted_yz, I_sat)

    if plane == 'xy':
        return em_psf_xy
    elif plane == 'all':
        return np.stack((em_psf_xy, em_psf_xz, em_psf_yz), axis=0)





    






