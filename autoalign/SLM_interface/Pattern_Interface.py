#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 21:18:41 2018

@author: wjahr
"""

import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets

import numpy as np

import Pattern_Calculator as pcalc
import Sub_Pattern as spat
import Patterns_Zernike as patzern

class Half_Pattern(QtWidgets.QWidget):
    """ Contains the image data to create one half of the pattern on the SLM:
        offset, grid, defocus, vortex and aberrations. Flatfield correction is
        only in the Main Program. For fast offset calculation, images are 
        created at twice the size needed and then cropped differently when the 
        offset is changed. """
    def __init__(self, params, parent = None):
        super(Half_Pattern, self).__init__(parent)
        self.size = np.asarray(params.general["size_slm"])
        self.full = np.zeros(self.size * 2)
        self.data = np.zeros(self.size)
        
    def call_daddy(self, p): 
        """ Connects the instance of this class to the calling function p, in this
            case usually img_l and img_r from the Main class. Needed to tell
            apart the instances and eg to update the correct side of the image. """
        self.daddy = p

    def set_name (self, name):
        self.name = name
    
    def get_name(self):
        return self.name
        
    def create_gui(self, p_gen, p_spec):
        """ Places the GUI elements of the Subpatterns and sets default behavior
            for the spinboxes. Parameters p_gen: General parameters, p_spec: 
            parameter list for either left or right side of the image, depending
            on the instance. """
        
        self.offset = p_spec["off"]
        
        controls = QtWidgets.QGridLayout()
        self.off = spat.Off_Pattern(p_gen)
        self.off.call_daddy(self)
        size = np.asarray(p_gen.general["size_full"])
        controls.addLayout(self.off.create_gui(p_spec["off"], 
                                               [[0, 1, -size[0]/2, size[0]/2],
                                                [0, 1, -size[1]/2, size[1]/2]]),
                            0,0,2,2)
        
        self.gr = spat.Sub_Pattern_Grid(p_gen)
        self.gr.call_daddy(self)
        controls.addLayout(self.gr.create_gui(p_spec["sl"], 
                                              [[2, 0.1, -20, 20], [2, 0.1, -20, 20]]), 2,0,2,2)
        self.gr.compute_pattern(update = False)
        
        self.defoc = spat.Sub_Pattern_Defoc(p_gen)
        self.defoc.call_daddy(self)
        controls.addLayout(self.defoc.create_gui(p_spec["defoc"], 
                                                 [3, 0.1, -10, 10]), 4,0,1,2)
        self.defoc.compute_pattern(update = False)
        
        self.vort = spat.Sub_Pattern_Vortex(p_gen)
        self.vort.call_daddy(self)
        controls.addLayout(self.vort.create_gui(p_gen.general["modes"], 
                                                [p_spec["mode"], p_spec["radius"], 
                                                 p_spec["phase"], p_spec["rot"],
                                                 p_spec["steps"]]), 5,0,5,2)
        self.vort.compute_pattern(update = False)
        
        self.aberr = patzern.Aberr_Pattern(p_gen) # TODO: this should be split later into the left/right
        self.aberr.call_daddy(self)
        controls.addLayout(self.aberr.create_gui(p_gen, p_spec), 10, 0, 4, 2)
        
        self.aberr.update(update = False)

        #self.aberr = self.daddy.img_aberr
        #self.aberr.call_daddy(self)
        
        self.update(update = False)

        controls.setContentsMargins(0,0,0,0) 
        
        return controls
    
    def crop(self, update = True):
        """ Crops the full data to half the size, using the provided offset.
            setting update = False prevents all the update() function of this
            class being called. Update = True eg when the offset parameters
            are changed, but false when the data is re-cropped after changing
            any of the other parameters. """
        cropped = pcalc.crop(self.full, self.size, self.offset)
        if update:
            self.update()
        return cropped
    
    def update(self, update = True):
        """ Recalculates the image by adding the vortex, defocus and aberration
            data, cropping to the provided offset and adding the grating. Then
            Calls the update function of the function owning this instance (in 
            this case the main). """
        self.full = pcalc.add_images([self.vort.data, self.defoc.data, self.aberr.data])
        self.data = self.crop(update = False)
        self.data = pcalc.add_images([self.data, self.gr.data])
        
        if update:
            self.daddy.combine_images()
            self.daddy.update_display()
