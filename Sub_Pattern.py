#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:17:18 2018

@author: wjahr
"""
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets

import numpy as np

import Pattern_Calculator as pcalc


class Sub_Pattern(QtWidgets.QWidget):
    """ Parent Widget for all the subpattern widgets. Contains GUI, the image 
        data and a compute_pattern function that's executed upon changes in 
        any of the GUI elements. """
    def __init__(self, params, parent = None):
        super(Sub_Pattern, self).__init__(parent)
        #super().__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.size = np.asarray(params.general["size_slm"]) * 2
        self.data = np.zeros(self.size)
    
    def call_daddy(self, p):
        """ Connects the instance of this class to the calling function p, in this
            case usually the Half Pattern. Needed to tell apart the instances 
            and eg to update the correct side of the image. """
        self.daddy = p
        
    def create_gui(self, defval, setup, layout = 'v'):
        """ Default Gui: Two Spin boxes beneath each other. If layout = 'h', two
            spin boxes side by side. Overwritten by some of the subpattern 
            classes that need different layouts. """
        if layout == 'v':
            gui = QtWidgets.QVBoxLayout()
        elif layout == 'h':
            gui = QtWidgets.QHBoxLayout()
        self.xgui = self.double_spin(defval[0], setup[0], gui)
        self.ygui = self.double_spin(defval[1], setup[1], gui) 
        return gui
    
    def compute_pattern(self):
        """ Dummy function for updating the pattern data. Is overwritten by the
            Subclasses. """
        if self.daddy.blockupdating == False:
            self.daddy.update()
        return self.data
        
    def double_spin(self, defval, setup, main_layout):
        """ Creates a spin box with double precision. Inputs: default value, 
            setup [precision, step, min, max], main layout to put in. """     
        item = QtWidgets.QDoubleSpinBox()
        item.setDecimals(setup[0])
        item.setSingleStep(setup[1])
        item.setMinimum(setup[2])
        item.setMaximum(setup[3])
        item.setValue(defval)
        item.setMaximumSize(80,50)
        item.valueChanged.connect(lambda: self.compute_pattern())
        main_layout.addWidget(item)
        return item
    
    

class Off_Pattern(Sub_Pattern):
    """ Subpattern containg the GUI for setting the offsets. Calls the crop 
        function of the calling instance whenever the offset values are changed. """
    def __init__(self, params, parent = None):
        super(Sub_Pattern, self).__init__(parent)        
        #I don't get why, but it seems initializing the variables in the subclass
        # seems necessary
        # if only initialized in the super class, they are created as function,
        # not as variable        
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.size = params.general["size_slm"]
        self.value = [0,0]

    def compute_pattern(self, update = True):
        if self.daddy.blockupdating == False:
            self.value = [self.xgui.value(), self.ygui.value()]
            self.daddy.offset = self.value
            self.daddy.crop(self.value)
        return self.value
    
    
class Sub_Pattern_Grid(Sub_Pattern):
    """ Subpattern containing the image data of the blazed gratings. Recalculates 
        the grating, and calls an update to the Half Pattern to recalculate the
        whole image data. """
    def __init__(self, params, parent = None):
        super(Sub_Pattern, self).__init__(parent)        
        #I don't get why, but it seems initializing the variables in the subclass
        # seems necessary
        # if only initialized in the super class, they are created as function
        # not as variable        
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.size = np.asarray(params.general["size_slm"])
        self.slm_px = params.general["slm_px"]
        self.data = np.zeros(self.size)
        
    def compute_pattern(self, update = True):
        if self.daddy.blockupdating == False:
            slope = [self.xgui.value(), self.ygui.value()]
            
            z = self.daddy.daddy.zernikes_normalized
            self.data = pcalc.add_images([z["tiptiltx"] * slope[0],
                                          z["tiptilty"] * slope[1]])

            if update:
                self.daddy.update()
        return self.data
    

class Sub_Pattern_Vortex(Sub_Pattern):
    """ Subpattern containing the image data of the vortex. GUI contains a 
        ComboBox to select the donut mode ("2D STED", "3D STED", "Gauss", "Halfmoon", Bivortex) and 
        three boxes for radius, phase and rotation. Recalculates the vortices
        based on the parameters and calls an update to the Half Pattern to 
        recalculate the whole image data. """
    def __init__(self, params, parent = None):
        super(Sub_Pattern, self).__init__(parent) 
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.size = np.asarray(params.general["size_slm"]) * 2
        self.data = np.zeros(self.size)
        self.path = params.general["path"]
        
    def create_gui(self, modes, defval):
        gui = QtWidgets.QVBoxLayout()
        self.modegui = QtWidgets.QComboBox(self)
        self.modegui.setMaximumSize(80, 50)

        gui.addWidget(self.modegui)
        for mm in modes:
            self.modegui.addItem(mm)
            
        self.modegui.setCurrentText(defval[0])
        self.radgui = self.double_spin(defval[1], [3, 0.1, 0, 5], gui)
        self.phasegui = self.double_spin(defval[2], [3, 0.1, -5, 5], gui)
        self.rotgui = self.double_spin(defval[3], [2, 10, 0, 360], gui)
        self.stepgui = self.double_spin(defval[4], [0, 1, 0, 360], gui)
        self.modegui.activated.connect(lambda: self.compute_pattern())
        gui.setContentsMargins(0,0,0,0)
        
        return gui
        
        
    def compute_pattern(self, update = True):
        if self.daddy.blockupdating == False:
            mode = self.modegui.currentText()
            rad = self.radgui.value()
            phase = self.phasegui.value()
            rot = self.rotgui.value()
            steps = self.stepgui.value()
            slm_scale = self.daddy.daddy.slm_radius
            
            self.data = pcalc.compute_vortex(mode, self.size, rot, rad, steps, phase, slm_scale)
            
            if update:
                self.daddy.update()
        return self.data


class Sub_Pattern_Defoc(Sub_Pattern):
    """ Subpattern containing the image data for defocus. GUI contains only the
        defocus parameter. Recalculates the defocus and calls an update to the 
        Half Pattern to recalculate the whole image data. """
    def __init__(self, params, parent = None):
        super(Sub_Pattern, self).__init__(parent)
        self.size = np.asarray(params.general["size_slm"]) * 2
        self.data = np.zeros(self.size)
        
    def create_gui(self, defval, setup):
        gui = QtWidgets.QVBoxLayout()
        self.defocgui = self.double_spin(defval, setup, gui)
        
        gui.setContentsMargins(0,0,0,0)
        return gui
    
    def compute_pattern(self, update = True):
        """ Zernike mode (2,0) """
        if self.daddy.blockupdating == False:
            self.data = self.daddy.daddy.zernikes_normalized["defocus"] * self.defocgui.value()
            
            if update:
                self.daddy.update()
        return self.data
