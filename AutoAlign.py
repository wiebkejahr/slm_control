#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:30:44 2022

@author: wiebke
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import json


import slm_control.Pattern_Calulator as pcalc
import microscope
import inits
import autoalign.utils.helpers as helpers


def correct_tiptiltdefoc(img, p):
    c = self.p.simulation["optical_params_sted"]
    mag = self.p.general["slm_mag"]
    size = 2 * np.asarray(self.p.general["size_slm"])   
    off = [self.img_l.off.xgui.value(), self.img_l.off.ygui.value()]
    
    d_xyz = helpers.get_CoMs(img) # in px!
    
    h = (np.pi * c["px_size"]) / (c["lambda"] * c["f"]) * c["obj_ba"] / mag / 2
    xtilt =  h * d_xyz[0]
    ytilt = -h * d_xyz[1]
    #calculate tip/tilt as zernike polynomials [1,1] and [-1,1]
    full = pcalc.zern_sum(size, [xtilt, ytilt], [[1,-1],[1,1]], radscale = 2)
    tiptilt_correct = pcalc.crop(full, size//2, offset = off)
    self.phase_tiptilt = self.phase_tiptilt + tiptilt_correct  
    
    h = c["px_size"]/((c["f"]/c["obj_ba"])**2 *8 * np.sqrt(3)*c["lambda"])/2 
    defocus = h * d_xyz[2]
    # calculate defocus as zernike polynomial [2,0]
    full = pcalc.zern_sum(size, [defocus], [[2,0]], radscale = 2)
    defoc_correct = pcalc.crop(full, size//2, offset = off)
    self.phase_defocus = self.phase_defocus + defoc_correct
    self.recalc_images()
  

def corrective_loop(p, groundtruth, scope, image=None, aberrs = np.zeros(11), offset=False, multi=False, ortho_sec=False, i=1):
    """ Passes trained model and acquired image to microscope.predict to 
        estimate zernike weights and offsets required to correct 
        aberrations. Calculates new SLM pattern to acquire new image and 
        calculates correlation coefficients. """
    
    size = 2 * np.asarray(p.general["size_slm"])
    scale = 2 * pcalc.get_mm2px(p.general["slm_px"], p.general["slm_mag"])
    
    orders = p.simulation["numerical_params"]["orders"]
    print("model into predict: ", p.general["autodl_model_path"])
    delta_zern, delta_off = microscope.predict(p.general["autodl_model_path"], 
                                               p.model_def,
                                               groundtruth[0], image, ii=i)

    delta_off = delta_off * scale
    #if abs(delta_off[0]) > 32:
    #    delta_off = 0
    #elif abs(delta_off[1]) > 32:
    #    delta_off = 0

    off = [self.img_l.off.xgui.value() + delta_off[1],
           self.img_l.off.ygui.value() - delta_off[0]]
    self.img_l.off.xgui.setValue(np.round(off[0]))
    self.img_l.off.ygui.setValue(np.round(off[1]))
    
    new_aberrs = aberrs - delta_zern
    full = pcalc.zern_sum(size, new_aberrs, orders[3::], np.sqrt(2)*self.slm_radius)
    self.phase_zern = pcalc.crop(full, size//2, offset = off)
    self.recalc_images()
    img, stats = scope.acquire_image(multi=multi, mask_offset = off, aberrs = new_aberrs)
    self.correct_tiptiltdefoc(img)
        
    new_img, stats = scope.acquire_image(multi=multi, mask_offset = off, aberrs = new_aberrs)
    correlation = helpers.corr_coeff(new_img, multi=multi)
    return delta_zern, delta_off, new_img, correlation

def auto_align(self, so_far = -1, best_of = 5, multi = True, offset = True):
    """This function calls abberior from AutoAlign module, passes the resulting dictionary
    through a constructor for a param object
    so_far: correlation required to stop optimizing; -1 means it only executes once"""

    size = 2 * np.asarray(self.p.general["size_slm"])
    orders = self.p.simulation["numerical_params"]["orders"]
    scope = microscope.Microscope(self.p.simulation)
    #imspector, msr_names, active_msr, conf = scope.get_config()
    # center the image before starting
    self.correct_tiptilt(scope)
    if multi:
        self.correct_defocus(scope)
        
    corr = 0
    i = 0
    new_aberrs = np.zeros(11)
    old_aberrs = new_aberrs
    while corr >= so_far:
        image = scope.acquire_image(multi=multi, mask_offset = [0,0], aberrs = new_aberrs)[0]                                              
        delta_zern, delta_off, image, new_corr = self.corrective_loop(scope, image, aberrs = new_aberrs, 
                                               offset=offset, multi=multi, i=best_of)
        if new_corr > corr:
            print('iteration: ', i, 'new corr: {}, old corr: {}'.format(new_corr, corr))
            corr = new_corr
            old_aberrs = new_aberrs
            new_aberrs = old_aberrs - delta_zern
            i = i + 1
        else:
            print('final correlation: {}'.format(corr))
            # REMOVING the last phase corrections from the SLM
            off = [self.img_l.off.xgui.value() - delta_off[1],
                   self.img_l.off.ygui.value() + delta_off[0]]
            full = pcalc.zern_sum(size, old_aberrs, orders[3::], np.sqrt(2)*self.slm_radius)
            self.phase_zern = pcalc.crop(full, size//2, offset = off)
            i -= 1
            break
    self.recalc_images()


def automate(p):
    #TODO: get from main
    groundtruth = None
    current_objective = p.general["objective"]
    
    slm_radius = inits.calc_slmradius(p,
        p.objectives[current_objective]["backaperture"],
        p.general["slm_mag"])
    num_its = 2
    px_size = 10*1e-9
    i_start = 155
    best_of = 5
    size = 2 * np.asarray(p.general["size_slm"])
    orders = p.simulation["numerical_params"]["orders"]
    scale = 2 * pcalc.get_mm2px(p.general["slm_px"], p.general["slm_mag"])
    
    #TODO: implement offsets here!
    plane = [0,0,0]
    
    # 0. creates data structure for statistics
    statistics = {'gt_off': [], 'preds_off': [], 
                  'gt_zern': [], 'preds_zern': [], 
                  'init_corr': [],'corr': [],
                  'CoM_correct': [], 'CoM_aberr': []}
    # for model name: drop everything from model path, drop extension
    mdl_name = p.general["autodl_model_path"].split("/")[-1][:-4]
    path = p.general["data_path"] + mdl_name
    p.load_model_def('', 'model_params.json', mdl_name)
    
    # TODO: clean up usage of these flags later
    multi = p.model_def['multi_flag']
    ortho_sec = p.model_def['orthosec_flag']
    offset = p.model_def['offset_flag']
    zern_flag = p.model_def['zern_flag']
    
    print("save path: ", path, "\n used model: ", mdl_name)
    try:
        if not os.path.isdir(p.general["data_path"]):
            os.mkdir(p.general["data_path"])
        if not os.path.isdir(path):
            os.mkdir(path)
    except:
        print("couldn't create directory!")
    
    scope = microscope.get_scope(p.general, p.simulation)
    if groundtruth == None:
        virtual_scope = microscope.Microscope(p.simulation)
        groundtruth = virtual_scope.calc_data()
        print("calc gt ", np.shape(groundtruth), np.shape(groundtruth[0]), np.shape(groundtruth[1]))
        #self.groundtruth = virtual_scope.calc_groundtruth(1.1)
    
    #imspector, msr_names, active_msr, conf = scope.get_config()
    xyz_init = scope.get_stage_offsets()
    for ii in range(num_its):
        # 1. zeroes SLM
        self.reload_params(self.param_path)
        # 2. get image from microscope and center
        img, stats = scope.acquire_image(multi=ortho_sec, mask_offset = [0,0], aberrs = np.zeros(11))        
        scope.center_stage(img, xyz_init, px_size, mode = 'fine')
        
        # 3. dials in random aberrations and sends them to SLM and SLM GUI
        #TODO: don't hardcode this anymore depending on model used
        #WHOLE BLOCK; BOTH for aberrs as well as off_aberr
        if zern_flag:
            aberrs = helpers.gen_coeffs(11)
        else:
            aberrs = [0 for c in range(11)]
        if offset:
            ba = p.objectives[current_objective]["backaperture"]
            off_aberr = [np.round(scale*x) for x in helpers.gen_offset(ba, 0.1)]
        else:
            off_aberr = [0,0]
        
        # calculate new offsets and write to GUI, recalc SLM image
        off = [self.img_l.off.xgui.value() - off_aberr[1],
               self.img_l.off.ygui.value() + off_aberr[0]]
        self.img_l.off.xgui.setValue(off[0])
        self.img_l.off.ygui.setValue(off[1])
        #TODO: sanity check that offsets are within boundaries
        full = pcalc.zern_sum(size, aberrs, orders[3::], np.sqrt(2)*self.slm_radius)
        self.phase_zern = pcalc.crop(full, size//2, offset = off)
        
        self.recalc_images()
        
        # 4. Acquire image, center once more using tip tilt and defocus corrections
        # save image and write correction coefficients to file
        img, stats = scope.acquire_image(multi=ortho_sec, mask_offset = off_aberr, aberrs = aberrs)
        correct_tiptiltdefoc(img)
            
        #TODO: change scope.acquire_image to return always array, then always use img[0]
        img_aberr, stats = scope.acquire_image(multi=multi, mask_offset = off_aberr, aberrs = aberrs)
        statistics['gt_zern'].append(aberrs)
        statistics['gt_off'].append(off_aberr)
        statistics['init_corr'].append(helpers.corr_coeff(img_aberr, multi=multi))
        statistics['CoM_aberr'].append(helpers.get_CoMs(img_aberr).tolist())
        scope.save_img(path + '/' + str(ii+i_start) + "_aberrated")
        
        # 5. single pass correction
        delta_zern, delta_off, img_corr, corr = corrective_loop(p, scope, img_aberr, aberrs, offset=offset, multi=multi, i = best_of)
        
        print('preds ', delta_zern, delta_off)
        statistics['preds_zern'].append(delta_zern.tolist())
        statistics['preds_off'].append(delta_off.tolist())
        statistics['corr'].append(corr)
        statistics['CoM_correct'].append(helpers.get_CoMs(img_corr).tolist())
        scope.save_img(path + '/' + str(ii+i_start) + "_corrected")
        with open(path + '/' + mdl_name +str(i_start)+'.txt', 'w') as file:
            json.dump(statistics, file)

        # use matplotlib to plot and save data
        fig = plt.figure()
        minmax = [np.min(img_corr[0]), np.max(img_corr[0])]
        if ortho_sec and multi:
            plt.subplot(231); plt.axis('off')
            plt.imshow(img_aberr[0], clim = minmax, cmap = 'inferno')
            plt.subplot(232); plt.axis('off')
            plt.imshow(img_aberr[1], clim = minmax, cmap = 'inferno')
            plt.subplot(233); plt.axis('off')
            plt.imshow(img_aberr[2], clim = minmax, cmap = 'inferno')
            plt.subplot(234); plt.axis('off')
            plt.imshow(img_corr[0], clim = minmax, cmap = 'inferno')
            plt.subplot(235); plt.axis('off')
            plt.imshow(img_corr[1], clim = minmax, cmap = 'inferno')
            plt.subplot(236); plt.axis('off')
            plt.imshow(img_corr[2], clim = minmax, cmap = 'inferno')
        elif ortho_sec and not multi:
            plt.subplot(121); plt.axis('off')
            plt.imshow(img_aberr, clim = minmax, cmap = 'inferno')
            plt.subplot(122); plt.axis('off')
            plt.imshow(img_corr[0], clim = minmax, cmap = 'inferno')
        fig.savefig(path + '/' + str(ii+i_start) + "_thumbnail.pdf")
        #TODO add missing logic blocks

    print('DONE with automated loop!', '\n', 
          'Initial correlation: ', statistics['init_corr'], '\n', 
          'final correlation: ', statistics['corr'])