import numpy as np
import matplotlib.pyplot as plt


def gauss(x, x0, y0, sigma):
    p = [x0, y0, sigma]
    return p[1] * np.exp( - ((x - p[0]) ** 2 / (2 * p[2] ** 2)))


def calc_scale(P_in, obj_ba, f, lambd, n, rep, tau):
    """"Calculates the scaling factor needed to make the vector diffraction 
        code energy conserving. Inputs: 
            P_in - Laser power going into the backaperture, 
            obj_ba - diameter of the backaperture,
            f - focal length of the objective, 
            lambd - wavelength of the laser
            n - refractive index, 
            rep - rep rate 
            tau - pulse length of the laser."""
            
    I_in = P_in / np.pi / obj_ba ** 2 # input intensity
    repxtau = rep * tau # ratio pulsed laser is turned on
    A = np.pi * f / lambd * n # scaling factor from Richards and Wolf 1959
    return np.sqrt(I_in / repxtau) * A


def calc_lp(P_in, rep, tau):
    """"Calculates the scaling factor needed to make the vector diffraction 
        code energy conserving. Inputs: 
            P_in - Laser power going into the backaperture, 
            obj_ba - diameter of the backaperture,
            f - focal length of the objective, 
            lambd - wavelength of the laser
            n - refractive index, 
            rep - rep rate 
            tau - pulse length of the laser."""
    
    return np.sqrt(P_in / (rep * tau))


def calc_rphi(aperture_size, res, off):
    x,y = np.meshgrid(np.linspace(-aperture_size, aperture_size, res),
                      np.linspace(-aperture_size, aperture_size, res))   #input meshgrid
    r = np.sqrt((x - off[0]) ** 2 + (y - off[1]) ** 2)
    phi = np.arctan2(y,x)
    return r, phi


def vector_diffraction(p_opt, p_num, polarization, phase, ampfn, LP = 1, plane = "all", offset=[0,0]):
     
    ##########################################################################
    #
    # read parameters from the dicts
    #
    ##########################################################################
    
    n = p_opt["n"]
    f = p_opt["f"]
    lambd = p_opt["lambda"]
    obj_ba = p_opt["obj_ba"]
    #offset = p_opt["offset"] # commented out so that I can generate a random offset in training
    
    output_screen_size = p_num["out_scrn_size"]
    z_extent = p_num["z_extent"]
    out_res = p_num["out_res"]
    inp_res = p_num["inp_res"]

    lambd = lambd * 1e-9 * 1e3 # wavelength in mm
    k = 2 * np.pi / lambd  #wavevector in 1/mm
    
    ##########################################################################
    #
    # calculate coordinates for integration, define variables and shorthand
    #
    ##########################################################################

    # xyz - space of output plane in mm
    xp = np.linspace(-output_screen_size, output_screen_size, out_res) * 1e-3
    yp = np.linspace(-output_screen_size, output_screen_size, out_res) * 1e-3
    zp = np.linspace(-z_extent, z_extent, out_res) * 1e-3
    
    # xyz - space of input plane, grid and px size in mm
    x1 = np.linspace(-obj_ba/2, obj_ba/2, inp_res)
    y1 = np.linspace(-obj_ba/2, obj_ba/2, inp_res)
    dx = x1[1] - x1[0]
    x,y = np.meshgrid(x1, y1)
    
    #declare a few required functions for easy access
    [r, phi] = calc_rphi(obj_ba/2, inp_res, offset)
    theta = np.arcsin(r / np.sqrt(r ** 2 + f ** 2))
    #theta = np.arccos(f / np.sqrt(r ** 2 + f ** 2))
    # commented line is version from report; two calculations are identical
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
        
    aperture = np.zeros_like(r)
    aperture[r < obj_ba / 2] = 1
    aperture = aperture * np.sqrt(ampfn)  
    #Putting a hard circular aperture of NA keeping the amplitude changes
    # absolute scaling is done via C constant
    
    
    ##########################################################################
    #
    # Integration processing begins
    #
    ##########################################################################
        
    jacobian = cos_theta ** 2 / f / r
    jacobian[len(x1) // 2, len(y1) // 2] = 1
    
    # when calculating e field (calc_lp routine), factor ce0/2 can be dropped
    # (it would be factored in again when calculating intensity:
    # e_xy = 2/ce0 * e ** 2); also dropped there
    
    C = np.sqrt(np.pi) * f * n / lambd / obj_ba # geometry dependent scaling factor
    H = LP * C * aperture * sin_theta * np.sqrt(cos_theta) * \
        jacobian * (dx ** 2)  * np.exp(1j * (phase))
    

    # polarization conversion matrix from Richardson / Wolf 1959
    pol_conv_matrix = np.asarray([[     1 + (cos_theta - 1) * cos_phi ** 2, 
                                       (cos_theta - 1) * cos_phi * sin_phi,
                                       -sin_theta * cos_phi],
                                  [    (cos_theta - 1) * cos_phi * sin_phi,
                                       1 + (cos_theta - 1) * sin_phi ** 2,
                                       -sin_theta * sin_phi],
                                  [     sin_theta * cos_phi,
                                        sin_theta * sin_phi,
                                        cos_theta]])
    
    # multiply polarization conversion matrix with polarization vector    
    temp_ex = H * (
        pol_conv_matrix[0,0] * polarization[0] +
        pol_conv_matrix[0,1] * polarization[1] + 
        pol_conv_matrix[0,2] * polarization[2] )
        
    temp_ey = H * (
        pol_conv_matrix[1,0] * polarization[0] +
        pol_conv_matrix[1,1] * polarization[1] + 
        pol_conv_matrix[1,2] * polarization[2] )
        
    temp_ez = H * (
        pol_conv_matrix[2,0] * polarization[0] +
        pol_conv_matrix[2,1] * polarization[1] + 
        pol_conv_matrix[2,2] * polarization[2] )      
        
    # since only the integral kernel (the exponential part) changes as the 
    # function of output aperture variables only that is calculated inside 
    # the loop for performance gains
    
    e_xy = np.zeros([3, len(xp), len(yp)]) * 0j
    if plane == "all" or plane == "xy":
        # print ("Calculating XY plane intensity...")
        e = e_xy
        zi = 0
        for [xxi, xi] in enumerate(xp):
            for [yyi, yi] in enumerate(yp):
                
                ri = np.sqrt(xi ** 2 + yi ** 2)
                phii = np.arctan2(yi, xi)
                
                temp_exp = np.exp(1j * k * n * (ri * sin_theta * np.cos(phi - phii) + 
                                                zi * cos_theta))
                
                e[:, xxi, yyi] = [(np.sum(temp_ex * temp_exp)), 
                                  (np.sum(temp_ey * temp_exp)), 
                                  (np.sum(temp_ez * temp_exp))]
        e_xy = np.sum(np.abs(e ** 2), axis = 0)

        
    
    e_xz = np.zeros([3, len(xp), len(zp)]) * 0j
    if plane == "all" or plane == "xz":
        #print ("Calculating XZ plane intensity...")
        e = e_xz
        yi = 0
        for [xxi, xi] in enumerate(xp):
            for [zzi, zi] in enumerate(zp):
                
                ri = np.sqrt(xi ** 2 + yi ** 2)
                phii = np.arctan2(yi, xi)
                
                temp_exp = np.exp(1j * k * n * (ri * sin_theta * np.cos(phi - phii) + 
                                                zi * cos_theta))
        
                e[:, xxi, zzi] = [(np.sum(temp_ex * temp_exp)), 
                                  (np.sum(temp_ey * temp_exp)), 
                                  (np.sum(temp_ez * temp_exp))]
        e_xz = np.sum(np.abs(e ** 2), axis = 0)

    
    

    e_yz = np.zeros([3, len(zp), len(yp)]) * 0j
    if plane == "all" or plane == "yz":    
        #print ("Calculating YZ plane intensity...")
        e = e_yz
        xi = 0
        for [zzi, zi] in enumerate(zp):
            for [yyi, yi] in enumerate(yp):
                
                ri = np.sqrt(xi ** 2 + yi ** 2)
                phii = np.arctan2(yi, xi)
                
                temp_exp = np.exp(1j * k * n * (ri * sin_theta * np.cos(phi - phii) + 
                                                zi * cos_theta))
                
                e[:, zzi, yyi] = [(np.sum(temp_ex * temp_exp)), 
                                  (np.sum(temp_ey * temp_exp)), 
                                  (np.sum(temp_ez * temp_exp))]
        e_yz = np.sum(np.abs(e ** 2), axis = 0)


    e_xyz = np.zeros([3, len(zp), len(yp), len(xp)]) * 0j
    if plane == "3D":
        # For 3D plane
        print ("Calculating 3D intensity...")
        for [zzi, zi] in enumerate(zp):
            for [yyi, yi] in enumerate(yp):
                for[xxi, xi] in enumerate(xp):
                
                    ri = np.sqrt(xi ** 2 + yi ** 2)
                    phii = np.arctan2(yi, xi)
                    
                    temp_exp = np.exp(1j * k * n * (ri * sin_theta * np.cos(phi - phii) + 
                                                    zi * cos_theta))
        
                    e[:, zzi, yyi, xxi] = [(np.sum(temp_ex * temp_exp)), 
                                            (np.sum(temp_ey * temp_exp)), 
                                            (np.sum(temp_ez * temp_exp))]      
        
        e_xyz = np.sum(np.abs(e**2), axis = 0)

    return(e_xy, e_xz, e_yz, e_xyz)

