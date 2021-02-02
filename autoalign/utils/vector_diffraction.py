import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../slm_control/')
from slm_control import Pattern_Calculator as PC

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
            P_in - Laser power going into the backaperture in W, 
            rep - rep rate in Hz
            tau - pulse length of the laser in s."""
    
    return np.sqrt(P_in / (rep * tau))


def stim_em(exc, sted, isat):
    """Calculates the effective PSF created using stimulated emission. Input
        excitation laser intensity, STED laser intensity, saturations intensity
        of the used dye. From Harke / Hell et al 2008"""
    #ln(2) is needed because I_sat is "half life", not lifetime
    exponent = - np.log(2) * sted / isat
    depleted = exc * np.exp(exponent)
    return depleted


# def calc_rphi(aperture_size, res, off):
#     x,y = np.meshgrid(np.linspace(-aperture_size, aperture_size, res),
#                       np.linspace(-aperture_size, aperture_size, res))   #input meshgrid
#     z = x + 1j * y
#     return np.abs(z), np.angle(z)


def calc_rphi(aperture_size, res, off):
    x,y = np.meshgrid(np.linspace(-aperture_size, aperture_size, res),
                      np.linspace(-aperture_size, aperture_size, res))   #input meshgrid
    r = np.sqrt((x - off[0]) ** 2 + (y - off[1]) ** 2)
    phi = np.arctan2(y,x)
    return r, phi

def loop_indices(xi, yi, zi, c, sin_theta, cos_theta, phi):
    ri = np.sqrt(xi ** 2 + yi ** 2)
    phii = np.arctan2(yi, xi)
    temp_exp = np.exp(c * (ri * sin_theta * np.cos(phi - phii) + zi * cos_theta))
    return ri, phii, temp_exp


def vector_diffraction(p_opt, p_num, polarization, phase, ampfn, LP = 1, plane = "all", offset=[0,0]):
    """ Calculates the focused PSF created by a given the wavefront in the 
        objective's backaperture according to vector diffraction threory 
        (Richards / Wolff 1959).
        
        Inputs:
        
        optical parameters as dictionary:
        p_opt["n"]: refractive index
        p_opt["f"]: focal length lens in mm
        p_opt["lambda"]: wavelength in nm
        p_opt["obj_ba"]: diameter objective backaperture in mm
        p_opt["offset"]: offset of the pattern in the backaperture, in mm
        
        numerical parameters as dictionary:
        p_num["out_scrn_size"]: desired size of focal plane in um
        p_num["z_extent"]: length of focal volume in um
        p_num["out_res"]: # of pixels for the output volume
        p_num["inp_res"]: # of pixels to create the input phase masks
        
        polarisation as keyword (todo)
        
        phase mask, amplitude mask
        
        Optional:
        peak laser power in W (use calc_lp function for pulsed lasers)
        plane(s) you want simulated
        plane = "all": calculate orthoplanes (xy, xz, yz).
        plane = "xy", "xz" or "yz": calculate one of the orthoplanes
        plane = "3d": calculate full volume. Slow! """
        
        
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
    # inp_res = np.shape(phasemask)[0]

    lambd = lambd * 1e-9 * 1e3 # wavelength in mm
    k = 2 * np.pi / lambd  #wavevector in 1/mm
    

    phase = np.rot90(phase, k=-1) * 2 * np.pi
    ampfn = np.rot90(ampfn, k=-1)
    
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
    jacobian[inp_res // 2, inp_res // 2] = 1
    
    # when calculating e field (calc_lp routine), factor ce0/2 can be dropped
    # (it would be factored in again when calculating intensity:
    # e_xy = 2/ce0 * e ** 2); also dropped there
    
    C = np.sqrt(np.pi) * f * n / lambd / obj_ba # geometry dependent scaling factor
    H = LP * C * aperture * sin_theta * np.sqrt(cos_theta) * \
        jacobian * (dx ** 2)  * np.exp(1j * (phase))
    

    # polarization conversion matrix from Richard / Wolf 1959
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
    
    e_xy = np.zeros([3, len(yp), len(xp)]) * 0j    
    e_xz = np.zeros([3, len(zp), len(xp)]) * 0j
    e_yz = np.zeros([3, len(zp), len(yp)]) * 0j
    e_xyz = np.zeros([3, len(zp), len(yp), len(xp)]) * 0j

    if plane == "all" or plane == "xy":
        # print ("Calculating XY plane intensity...")
        e = e_xy
        zi = 0
        for [yyi, yi] in enumerate(yp):
            for [xxi, xi] in enumerate(xp):                
                ri, phii, temp_exp = loop_indices(xi, yi, zi, 1j*k*n, sin_theta, cos_theta, phi)
                
                e[:, yyi, xxi] = [(np.sum(temp_ex * temp_exp)), 
                                  (np.sum(temp_ey * temp_exp)), 
                                  (np.sum(temp_ez * temp_exp))]
        e_xy = np.sum(np.abs(e ** 2), axis = 0)

        
    
    if plane == "all" or plane == "xz":
        #print ("Calculating XZ plane intensity...")
        e = e_xz
        yi = 0

        for [zzi, zi] in enumerate(zp):
            for [xxi, xi] in enumerate(xp):
                ri, phii, temp_exp = loop_indices(xi, yi, zi, 1j*k*n, sin_theta, cos_theta, phi)
        
                e[:, zzi, xxi] = [(np.sum(temp_ex * temp_exp)), 
                                  (np.sum(temp_ey * temp_exp)), 
                                  (np.sum(temp_ez * temp_exp))]
        e_xz = np.sum(np.abs(e ** 2), axis = 0)

    
    
    if plane == "all" or plane == "yz":    
        #print ("Calculating YZ plane intensity...")
        e = e_yz
        xi = 0
        for [zzi, zi] in enumerate(zp):
            for [yyi, yi] in enumerate(yp):
                ri, phii, temp_exp = loop_indices(xi, yi, zi, 1j*k*n, sin_theta, cos_theta, phi)
                
                e[:, zzi, yyi] = [(np.sum(temp_ex * temp_exp)), 
                                  (np.sum(temp_ey * temp_exp)), 
                                  (np.sum(temp_ez * temp_exp))]
        e_yz = np.sum(np.abs(e ** 2), axis = 0)


    if plane == "3D":
        # For 3D plane
        #print ("Calculating 3D intensity...")
        e = e_xyz
        for [zzi, zi] in enumerate(zp):
            for [yyi, yi] in enumerate(yp):
                for[xxi, xi] in enumerate(xp):                
                    ri, phii, temp_exp = loop_indices(xi, yi, zi, 1j*k*n, 
                                                      sin_theta, cos_theta, phi)
        
                    e[:, zzi, yyi, xxi] = [(np.sum(temp_ex * temp_exp)), 
                                           (np.sum(temp_ey * temp_exp)), 
                                           (np.sum(temp_ez * temp_exp))]      
        
        e_xyz = np.sum(np.abs(e**2), axis = 0)

    return(e_xy, e_xz, e_yz, e_xyz)


if __name__ == "__main__":
    
    ##########################################################################
    #
    # define parameters of the system
    #
    ##########################################################################

    optical_params = {
        "n" : 1.518,
        "NA" : 1.4,
        "f" : 1.8, # in mm
        "transmittance" : 0.74, # for olympus 100x oil, check nikon
        "lambda" : 775, # in nm
        "P_laser" : 0.125e-3, # in mW
        "rep_rate" : 40e6, # in MHz
        "pulse_length" : 700e-12, # in ps
        "obj_ba" : 5.04, # in mm
        "offset" : [0,0] # in mm
        }
    
    numerical_params = {
        "out_scrn_size" : 1, #um
        "z_extent" : 1,
        "out_res" : 64,
        "inp_res" : 64
        }

    polarization = [1.0/np.sqrt(2), 1.0/np.sqrt(2)*1j, 0]    
    
    # calculate laser power dependent scaling factor
    #lp_scale = calc_lp(optical_params["P_laser"], optical_params["rep_rate"],
    #                   optical_params["pulse_length"])
    
    ##########################################################################
    #
    # phase pattern and amplitude pattern for input
    #
    ##########################################################################

    [r, phi] = calc_rphi(optical_params["obj_ba"]/2, 
                         numerical_params["inp_res"], optical_params["offset"])
    size = np.asarray([numerical_params["inp_res"], numerical_params["inp_res"]])
    
    # xy donut
    phasemask = PC.create_donut(2*size, 0, 1)
    coma = PC.create_zernike(2*size, [3,1], 1)
    astig = PC.create_zernike(2*size, [2,-2], 1)
    sphere = PC.create_zernike(2*size, [4,0], 1)
    phasemask = PC.crop(phasemask, size)
    #phasemask = PC.crop(phasemask + coma, size)

    amplitude = PC.crop(PC.create_gauss(2*size, 1), size)
    
    [e_xy, e_xz, e_yz, e_xyz] = vector_diffraction(optical_params, 
                                                   numerical_params, 
                                                   polarization,
                                                   phasemask, amplitude, 
                                                   plane = 'all')
    
    
    ##########################################################################
    #
    #  Plotting of the results
    #
    ##########################################################################
    # x-space of output plane in um
    xp = np.linspace(-numerical_params["out_scrn_size"], 
                     numerical_params["out_scrn_size"], 
                     numerical_params["out_res"])
    # y-space of output plane in um
    yp = np.linspace(-numerical_params["out_scrn_size"], 
                     numerical_params["out_scrn_size"], 
                     numerical_params["out_res"])
    #z-space of output plane in um
    zp = np.linspace(-numerical_params["z_extent"], 
                     numerical_params["z_extent"], 
                     numerical_params["out_res"])
    midplane = numerical_params["out_res"]//2
    
    
    fig = plt.figure()
    plt.subplot(131)
    plt.imshow(e_xy)
    plt.subplot(132)
    plt.imshow(e_xz)
    plt.subplot(133)
    plt.imshow(e_yz)
    
    # fig, axes = plt.subplots(nrows=3, ncols=3)
    # mngr = plt.get_current_fig_manager()
    # mngr.window.setGeometry(100,100,1200,800)
    
    # axes[1, 1].imshow(e_xy, aspect='equal', 
    #                   extent=[np.amin(xp), np.amax(xp), np.amin(yp), np.amax(yp)])
    # axes[1, 1].set_xlabel('(in $\mu$m)')
    # axes[1, 1].set_ylabel('(in $\mu$m)')
    # #axes[1, 1].set_title('XY plane')
    
    # axes[1, 2].imshow(e_xz, aspect='equal', 
    #                   extent=[np.amin(zp), np.amax(zp), np.amin(xp), np.amax(xp)])
    # axes[1, 2].set_xlabel('(in $\mu$m)')
    # axes[1, 2].set_ylabel('(in $\mu$m)')
    # #axes[1, 2].set_title('XZ plane')
    
    # axes[2, 1].imshow(e_yz, aspect='equal', 
    #                   extent=[np.amin(yp), np.amax(yp), np.amin(zp), np.amax(zp)])
    # axes[2, 1].set_xlabel('(in $\mu$m)')
    # axes[2, 1].set_ylabel('(in $\mu$m)')
    # #axes[2, 1].set_title('YZ plane')
    
    
    # p0 = [0.0, 1e8, 0.100] # x0, amp, sigma
    
    # axes[0, 1].plot(xp, e_xy[midplane, :])
    # fit, tmp = curve_fit(gauss, xp, e_xy[midplane,:], p0 = p0)
    # print('sigma ', fit[2] * 1e3, ' (nm) \n1/e^2 width ', 4 * fit[2] * 1e3, ' (nm) \nFWHM ', 
    #       2 * 1e3 * np.sqrt(2 * np.log(2)) * fit[2], 
    #       ' (nm) \namp ', fit[1], ' (MW/cm^2)\nx0 ', fit[0]  * 1e3, ' (um)')
    # axes[0, 1].plot(xp, gauss(xp, fit[0], fit[1], fit[2]))
    # axes[0, 1].set_xlabel('x (in $\mu$m)')
    # axes[0, 1].set_ylabel('Intensity along x axis')
    # #axes[0, 1].set_title('for XY plane')
    
    
    # axes[1, 0].plot(yp, e_xy[:, midplane])
    # fit, tmp = curve_fit(gauss, yp, e_xy[:,midplane], p0 = p0)
    # print('sigma ', fit[2], ' (nm) \n1/e^2 width ', 4 * fit[2] * 1e3, ' (nm) \nFWHM ', 
    #       2 * 1e3 * np.sqrt(2 * np.log(2)) * fit[2], 
    #       ' (nm) \namp ', fit[1], ' (W/mm^2)\nx0 ', fit[0]  * 1e3, ' (um)')
    # axes[1, 0].plot(yp, gauss(yp, fit[0], fit[1], fit[2]))
    # axes[1, 0].set_xlabel('y (in $\mu$m)')
    # axes[1, 0].set_ylabel('Intensity along y axis')
    # #axes[1, 0].set_title('for XY plane')
    
    # axes[0, 2].plot(zp, e_xz[midplane, :])
    # fit, tmp = curve_fit(gauss, zp, e_xz[midplane, :], p0 = p0)
    # print('sigma ', fit[2], ' (nm) \n1/e^2 width ', 4 * fit[2] * 1e3, ' (nm) \nFWHM ', 
    #       2 * 1e3 * np.sqrt(2 * np.log(2)) * fit[2], 
    #       ' (nm) \namp ', fit[1], ' (W/mm^2)\nx0 ', fit[0]  * 1e3, ' (um)')
    # axes[0, 2].plot(zp, gauss(zp, fit[0], fit[1], fit[2]))
    # axes[0, 2].set_xlabel('y (in $\mu$m)')
    # axes[0, 2].set_ylabel('Intensity along optical axis')
    # #axes[0, 2].set_title('for XY plane')
    
    
    # axes[2, 0].plot(zp, e_yz[:, midplane])
    # fit, tmp = curve_fit(gauss, zp, e_yz[:, midplane], p0 = p0)
    # print('sigma ', fit[2], ' (nm) \n1/e^2 width ', 4 * fit[2] * 1e3, ' (nm) \nFWHM ', 
    #       2 * 1e3 * np.sqrt(2 * np.log(2)) * fit[2], 
    #       ' (nm) \namp ', fit[1], ' (W/mm^2)\nx0 ', fit[0]  * 1e3, ' (um)')
    # axes[2, 0].plot(zp, gauss(zp, fit[0], fit[1], fit[2]))
    # axes[2, 0].set_xlabel('y (in $\mu$m)')
    # axes[2, 0].set_ylabel('Intensity along optical axis')
    # #axes[2, 0].set_title('for XY plane')



    # #axes[0, 0].remove()  # don't display empty ax
    # axes[2, 2].remove()  # don't display empty ax
    
    # axes[0,0].imshow(phasemask * amplitude, aspect = 'equal', cmap = 'RdYlBu')
    # plt.xticks([]); plt.yticks([])
    # #fig.tight_layout()
    

    # ## some evaluation / sanity checking
    
    # thr = 0.135
    # print("2D mean max min ", np.mean(e_xy)*1e-7, np.max(e_xy)*1e-7, np.min(e_xy)*1e-7)
    # # calculate avg only over part of the PSF higher than 1/e^2
    # print("2D masked", np.sum(e_xy*(e_xy > thr * np.max(e_xy))) / 
    #       np.sum(e_xy > thr * np.max(e_xy)) * 1e-7)
    
    # print("1D mean max min", np.mean(e_xy[:,midplane])*1e-7, 
    #       np.max(e_xy[:,midplane])*1e-7, np.min(e_xy[:,midplane])*1e-7)
    # print("1D masked", np.sum(e_xy[:,midplane]*(e_xy[:,midplane] > thr * np.max(e_xy[:,midplane]))) / 
    #       np.sum(e_xy[:,midplane] > thr * np.max(e_xy[:,midplane])) * 1e-7)
    
    # print("power ", np.max(e_xy) * 2 * np.pi * (fit[2] * 1e-3) ** 2, 
    #       np.sum(e_xy)*((xp[1]-xp[0])*1e-3)**2)
        
        
    
    # fig.savefig("test.pdf", bbox_inches = 0, transparent=True)
        
        
