#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pattern_Zernike.py


"""
    Created on Wed Oct 17 16:20:49 2018
    @author: wjahr
    
    Contains all the "adaptive" optics to correct for aberrations, as
    parametrized by Zernike Polynomials. Exceptions are orders [1,+/-1] and 
    [2,0]: as they only move the beam without changing the beam's shape, these 
    orders are treated in the Sub_Pattern class.
    Weights of the common Zernike polynomials are read from the GUI, the 
    phasemask for each polynomial is computed and all images are added.
    
    
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
from slm_control.Sub_Pattern import Sub_Pattern


class Sub_Pattern_Zernike(Sub_Pattern):
    """ Subpattern containing the image data for aberrations. GUI contains the 
        weights, usually of the two orthogonal Zernike modes (primary and 
        secondary for spherical). Recalculates the aberration and calls an 
        update to the Aberr_Pattern instance to recalculate aberrations. """


    def __init__(self, params, size, name, parent = None):
        super(Sub_Pattern, self).__init__(parent)
        self.size = size * 2
        self.data = np.zeros(self.size)
        self.coeff = [0,0]
        
        self.set_name(name)        
        

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
                # if in single correction, check on which side a value was
                # changed and update it on the other side
                if self.daddy.daddy.get_name() == "img_l":
                    thisside = self.daddy.daddy.daddy.img_l.aberr
                    otherside = self.daddy.daddy.daddy.img_r.aberr
                elif self.daddy.daddy.get_name() == "img_r":
                    thisside = self.daddy.daddy.daddy.img_r.aberr
                    otherside = self.daddy.daddy.daddy.img_l.aberr
                
                # all gui values are updated once. This is ugly, but I'd need
                # to know the name of the aberration that was changed.
                # Is there a way to easily know which one was changed?
                # self.xgui.value() and self.ygui.value() doesn't work, bc
                # I also need the information which side I'm on, and _which_
                # one on the other side I'd have to change
                otherside.astig.xgui.setValue(thisside.astig.xgui.value())
                otherside.astig.ygui.setValue(thisside.astig.ygui.value())
                otherside.coma.xgui.setValue(thisside.coma.xgui.value())
                otherside.coma.ygui.setValue(thisside.coma.ygui.value())
                otherside.sphere.xgui.setValue(thisside.sphere.xgui.value())
                otherside.sphere.ygui.setValue(thisside.sphere.ygui.value())
                otherside.trefoil.xgui.setValue(thisside.trefoil.xgui.value())
                otherside.trefoil.ygui.setValue(thisside.trefoil.ygui.value())
            
            # after changing all values in the gui for single correction, (or 
            # not, for normal execution), update the coefficients of the zernike
            self.coeff = [self.xgui.value(), self.ygui.value()]
            
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
        

    def __init__(self, params, size, parent = None):
        super(Aberr_Pattern, self).__init__(parent)
        self.size = size * 2
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
        self.astig = Sub_Pattern_Zernike(p, self.size, "astig")
        gui.addLayout(self.astig.create_gui(p_abberation["astig"], 
                                            defset, layout = 'h'), 0,0,1,2)
        self.astig.call_daddy(self)
        self.astig.compute_pattern(update = False)
        
        self.coma = Sub_Pattern_Zernike(p, self.size, "coma")
        gui.addLayout(self.coma.create_gui(p_abberation["coma"], 
                                           defset, layout = 'h'), 1,0,1,2)  
        self.coma.call_daddy(self)
        self.coma.compute_pattern(update = False)
        
        self.sphere = Sub_Pattern_Zernike(p, self.size, "sphere")
        gui.addLayout(self.sphere.create_gui(p_abberation["sphere"], 
                                             defset, layout = 'h'), 2,0,1,2)
        self.sphere.call_daddy(self)
        self.sphere.compute_pattern(update = False)
        
        self.trefoil = Sub_Pattern_Zernike(p, self.size, "trefoil")
        gui.addLayout(self.trefoil.create_gui(p_abberation["trefoil"], 
                                              defset, layout = 'h'), 3,0,1,2)
        self.trefoil.call_daddy(self)
        self.trefoil.compute_pattern(update = False)
        
        gui.setContentsMargins(0,0,0,0)
        return gui
        

    def update(self, update = True):
        z = self.daddy.daddy.zernikes_normalized
        self.data = pcalc.add_images([z["astigx"]   * self.astig.coeff[0],
                                      z["astigy"]   * self.astig.coeff[1],
                                      z["comax"]    * self.coma.coeff[0],
                                      z["comay"]    * self.coma.coeff[1],
                                      z["trefoilx"] * self.trefoil.coeff[0],
                                      z["trefoily"] * self.trefoil.coeff[1],
                                      z["sphere1"]  * self.sphere.coeff[0],
                                      z["sphere2"]  * self.sphere.coeff[1]
                                     ])
        if update:
            self.daddy.update()
        return self.data
