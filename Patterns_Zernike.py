#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:20:49 2018

@author: wjahr
"""

import PyQt5.QtWidgets as QtWidgets
import numpy as np

import Pattern_Calculator as pcalc
from Sub_Pattern import Sub_Pattern

class Sub_Pattern_Zernike(Sub_Pattern):
    """ Subpattern containing the image data for aberrations. GUI contains the 
        weights, usually of the two orthogonal Zernike modes (primary and 
        secondary for spherical). Recalculates the aberration and calls an 
        update to the Aberr_Pattern instance to recalculate aberrations. """
    def __init__(self, params, order, parent = None):
        super(Sub_Pattern, self).__init__(parent)
        self.size = np.asarray(params.general["size_slm"]) * 2
        self.order = order
#        self.radnorm = pcalc.normalize_radius(params.general["laser_radius"], 
#                                      params.general["slm_px"],
#                                      params.general["size_slm"])
        self.data = np.zeros(self.size)
        
    def set_name(self, name):
        self.name = name
        
    def get_name(self):
        return self.name
        
    def compute_pattern(self, update = True):
        # somewhat ugly solution for using identical correction on both sides of the SLM
        # whenever one side is changed, just writes the values into the gui controls
        # of the other side, then updates the images accordingly
        if self.daddy.daddy.blockupdating == False:
            if self.daddy.daddy.daddy.sngl_corr_state.checkState():
                if self.daddy.daddy.get_name() == "img_l":
                    thisside = self.daddy.daddy.daddy.img_l.aberr
                    otherside = self.daddy.daddy.daddy.img_r.aberr
                    
                    if self.get_name() == "astig":
                        otherside.astig.xgui.setValue(thisside.astig.xgui.value())
                        otherside.astig.ygui.setValue(thisside.astig.ygui.value())
                    elif self.get_name() == "coma":
                        otherside.coma.xgui.setValue(thisside.coma.xgui.value())
                        otherside.coma.ygui.setValue(thisside.coma.ygui.value())
                    elif self.get_name() == "sphere":
                        otherside.sphere.xgui.setValue(thisside.sphere.xgui.value())
                        otherside.sphere.ygui.setValue(thisside.sphere.ygui.value())
                    elif self.get_name() == "trefoil":
                        otherside.trefoil.xgui.setValue(thisside.trefoil.xgui.value())
                        otherside.trefoil.ygui.setValue(thisside.trefoil.ygui.value())
    
                elif self.daddy.daddy.get_name() == "img_r":
                    thisside = self.daddy.daddy.daddy.img_r.aberr
                    otherside = self.daddy.daddy.daddy.img_l.aberr
                    
                    if self.get_name() == "astig":
                        otherside.astig.xgui.setValue(thisside.astig.xgui.value())
                        otherside.astig.ygui.setValue(thisside.astig.ygui.value())
                    elif self.get_name() == "coma":
                        otherside.coma.xgui.setValue(thisside.coma.xgui.value())
                        otherside.coma.ygui.setValue(thisside.coma.ygui.value())
                    elif self.get_name() == "sphere":
                        otherside.sphere.xgui.setValue(thisside.sphere.xgui.value())
                        otherside.sphere.ygui.setValue(thisside.sphere.ygui.value())
                    elif self.get_name() == "trefoil":
                        otherside.trefoil.xgui.setValue(thisside.trefoil.xgui.value())
                        otherside.trefoil.ygui.setValue(thisside.trefoil.ygui.value())
    
            data_a = pcalc.create_zernike(self.size, self.order[0], self.daddy.daddy.daddy.slm_radius)
            data_b = pcalc.create_zernike(self.size, self.order[1], self.daddy.daddy.daddy.slm_radius)
            self.data = self.xgui.value() * data_a + self.ygui.value() * data_b
            
            if update:
                self.daddy.update()
        return self.data
    
    
class Aberr_Pattern(QtWidgets.QWidget):
    """ Subpattern containing the aberration data. Creates instances of 
        Sub_Pattern_Zernike with different Zernike orders to contain the 
        different modes of aberration. Recalculates overall aberration when 
        weight of one is changed by adding the different Zernike modes weighted 
        according to the parameters. Calls an update to the Half Pattern to 
        recalculate the whole image data. """
    def __init__(self, params, parent = None):
        super(Aberr_Pattern, self).__init__(parent)
        self.size = np.asarray(params.general["size_slm"]) * 2
        self.data = np.zeros(self.size)

    def call_daddy(self, p):        
        self.daddy = p
    
    def create_gui(self, p, p_abberation):
        """ Zernike mode (2, -2) for obligue astigmatism,
            Zernike mode (2, 2) for vertical astigmatism.
            Zernike mode (3, -1) for vertical coma,
            Zernike mode (3, 1) for horizontal coma.
            Zernike mode (4, 0) for primary spherical,
            Zernike mode (6, 0) for secondary spherical.
            Zernike mode (3, -3) for obligue trefoil,
            Zernike mode (3, 3) for vertical trefoil. """
        
        defset = [[3, 0.1, -10, 10], [3, 0.1, -10, 10]]         
        
        gui = QtWidgets.QGridLayout()
        self.astig = Sub_Pattern_Zernike(p, [[2,2], [2,-2]])
        gui.addLayout(self.astig.create_gui(p_abberation["astig"], defset, layout = 'h'), 0,0,1,2)
        self.astig.call_daddy(self)
        self.astig.set_name("astig")
        self.astig.compute_pattern(update = False)
        
        self.coma = Sub_Pattern_Zernike(p, [[3,1], [3,-1]])
        gui.addLayout(self.coma.create_gui(p_abberation["coma"], defset, layout = 'h'), 1,0,1,2)  
        self.coma.call_daddy(self)
        self.coma.set_name("coma")
        self.coma.compute_pattern(update = False)
        
        self.sphere = Sub_Pattern_Zernike(p, [[4,0], [6,0]])
        gui.addLayout(self.sphere.create_gui(p_abberation["sphere"], defset, layout = 'h'), 2,0,1,2)
        self.sphere.call_daddy(self)
        self.sphere.set_name("sphere")
        self.sphere.compute_pattern(update = False)
        
        self.trefoil = Sub_Pattern_Zernike(p, [[3,3], [3,-3]])
        gui.addLayout(self.trefoil.create_gui(p_abberation["trefoil"], defset, layout = 'h'), 3,0,1,2)
        self.trefoil.call_daddy(self)
        self.trefoil.set_name("trefoil")
        self.trefoil.compute_pattern(update = False)
        
        
        gui.setContentsMargins(0,0,0,0)
        return gui
        
    def update(self, update = True):
        self.data = pcalc.add_images([self.astig.data, self.coma.data, self.sphere.data, self.trefoil.data])
        if update:
            self.daddy.update()
        return self.data