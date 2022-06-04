#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pattern_Interface.py


"""
    Created on Mon Oct 15 21:18:41 2018
    @author: wjahr
    
    This is the highest level control for phasemask calculation, and contains 
    the class for the two "half-patterns". When the SLM is used in double pass 
    configuration, the code reates GUI for the two halfs of the pattern. It 
    calls updates of the various sub-components when needed, adds all layers
    and crops the images to account for the offset (i.e. different center 
    positions of the vortex).
    
        
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


import PyQt5.QtWidgets as QtWidgets
import numpy as np

import slm_control.Pattern_Calculator as pcalc
import slm_control.Sub_Pattern as spat
import slm_control.Patterns_Zernike as patzern


class Half_Pattern(QtWidgets.QWidget):
    """ Contains the image data to create one half of the pattern on the SLM:
        offset, grid, defocus, vortex and aberrations. Flatfield correction is
        only in the Main Program. For fast offset calculation, images are 
        created at twice the size needed and then cropped differently when the 
        offset is changed. """


    def __init__(self, params, size, parent = None):
        super(Half_Pattern, self).__init__(parent)
        self.size = size
        self.full = np.zeros(self.size * 2)
        self.data = np.zeros(self.size)
        self.blockupdating = False
        

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
        
        
        # create controls for the "basic" parameters creating the pattern
        self.offset = np.asarray(p_spec["off"])
        
        controls = QtWidgets.QGridLayout()
        self.off = spat.Off_Pattern(p_gen, self.size)
        self.off.call_daddy(self)
        offlim = [self.size[0], self.size[1]*2]
        controls.addLayout(self.off.create_gui(
                            p_spec["off"], 
                            [[0, 1, -offlim[1]/4, offlim[1]/4],
                             [0, 1, -offlim[0]/2, offlim[0]/2]]),
                            0,0,2,2)
        self.gr = spat.Sub_Pattern_Grid(p_gen, self.size)
        self.gr.call_daddy(self)
        controls.addLayout(self.gr.create_gui(p_spec["sl"], 
                                              [[2, 0.1, -20, 20], [2, 0.1, -20, 20]]), 2,0,2,2)
        self.gr.compute_pattern(update = False)
        
        self.defoc = spat.Sub_Pattern_Defoc(p_gen, self.size)
        self.defoc.call_daddy(self)
        controls.addLayout(self.defoc.create_gui(p_spec["defoc"], 
                                                 [3, 0.1, -10, 10]), 4,0,1,2)
        self.defoc.compute_pattern(update = False)
        
        self.vort = spat.Sub_Pattern_Vortex(p_gen, self.size)
        self.vort.call_daddy(self)
        controls.addLayout(self.vort.create_gui(p_gen.general["modes"], 
                                                [p_spec["mode"], p_spec["radius"], 
                                                 p_spec["phase"], p_spec["rot"],
                                                 p_spec["steps"]]), 5,0,5,2)
        self.vort.compute_pattern(update = False)
        
        
        # controls for the Zernike modes are created by the Patterns_Zernike
        # class and added in one go
        self.aberr = patzern.Aberr_Pattern(p_gen, self.size)
        self.aberr.call_daddy(self)
        controls.addLayout(self.aberr.create_gui(p_gen, p_spec), 10, 0, 4, 2)
        
        self.aberr.update(update = False)
        
        self.update(update = False)

        controls.setContentsMargins(0,0,0,0) 
        
        return controls
    
    
    def update_guivalues(self, p_gen, p_spec):
        """ This function is called eg when new parameters are loaded from
            file. Global flag is set to prevent recalculation and redrawing.
            Updating of SLM patterns is called manually in the end. """
        self.blockupdating = True
        
        self.off.xgui.setValue(p_spec["off"][0])
        self.off.ygui.setValue(p_spec["off"][1])
        self.gr.xgui.setValue(p_spec["sl"][0])
        self.gr.ygui.setValue(p_spec["sl"][1]) 
        self.vort.rotgui.setValue(p_spec["rot"])
        self.vort.radgui.setValue(p_spec["radius"])
        self.vort.phasegui.setValue(p_spec["phase"])
        self.vort.stepgui.setValue(p_spec["steps"])
        self.defoc.defocgui.setValue(p_spec["defoc"])
        self.vort.modegui.setCurrentText(p_spec["mode"])
        self.aberr.astig.xgui.setValue(p_spec["astig"][0])
        self.aberr.astig.ygui.setValue(p_spec["astig"][1])
        self.aberr.coma.xgui.setValue(p_spec["coma"][0])
        self.aberr.coma.ygui.setValue(p_spec["coma"][1])
        self.aberr.sphere.xgui.setValue(p_spec["sphere"][0])
        self.aberr.sphere.ygui.setValue(p_spec["sphere"][1])
        self.aberr.trefoil.xgui.setValue(p_spec["trefoil"][0])
        self.aberr.trefoil.ygui.setValue(p_spec["trefoil"][1])
        
        self.blockupdating = False
        
    
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
    

    def update(self, update = True, completely = False):
        """ Recalculates the image by adding the vortex, defocus and aberration
            data, cropping to the provided offset and adding the grating. Then
            Calls the update function of the function owning this instance (in 
            this case the main). """
            
        if completely:
            self.aberr.astig.compute_pattern(update = False)
            self.aberr.coma.compute_pattern(update = False)
            self.aberr.sphere.compute_pattern(update = False)
            self.aberr.trefoil.compute_pattern(update = False)
            self.aberr.update()
            
            self.gr.compute_pattern(update = False)
            self.vort.compute_pattern(update = False)
            self.defoc.compute_pattern(update = False)
            self.off.compute_pattern(update = False)
        
        self.full = pcalc.add_images([self.gr.data, self.vort.data, self.defoc.data, self.aberr.data])
        self.data = self.crop(update = False)
        
        if update:
            self.daddy.combine_and_update()
