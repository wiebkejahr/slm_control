#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Sub_Pattern.py


"""
    Created on Wed Oct 17 16:17:18 2018
    @author: wjahr
    
    Contains all the "non-adaptive" optics to create and align a basic vortex. 
    Subclasses create the GUI and compute:
        Off_Pattern: offset, controls central position of the vortex to align 
            it with the laser beam
        Sub_Pattern_Grid: blazed grating for holographic operation of SLM. Is 
            also used to align the lateral (x,y) beam positions within the 
            microscope. Calculated as Zernike polynomials [1,1] and [1,-1].
        Sub_Pattern_Vortex: Evaluates the drop down selection menue and c
            alculates the vortex to be imprinted on the beam. Several common 
            options are hard-coded for convenience:
                Gauss: flat pattern, no vortex is imprinted, Gaussian focus.
                2D: helical vortex for classical 2D "donut" STED beam
                3D: tophat vortex for bottle beam, 3D or z-STED beam
                Segments: useful for alignment:
                    1 Segment is a central line for laser focus "cut" in center
                    3 Segments cut laser focus in 4 spots, like "easy STED"
                    large # of Segments: approaches helical vortex pattern
                Bivortex: overlay of 2D and 3D patterns, like in "coherent 
                    hybrid STED" manuscript.
            Additionally, the vortex can be created via on-the-fly code input
            or loaded as an image.
        Sub_Pattern_Defoc: Defocus pattern, aligns the axial (z) beam position
             within the microscope. Calculated as Zernike polynomial [2,0].
    
    
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


import os
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets

import numpy as np
import slm_control.Pattern_Calculator as pcalc
import slm_control.syntax


class Sub_Pattern(QtWidgets.QWidget):
    """ Parent Widget for all the subpattern widgets. Contains GUI, the image 
        data and a compute_pattern function that's executed upon changes in 
        any of the GUI elements. """


    def __init__(self, params, size, parent = None):
        super(Sub_Pattern, self).__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.size = size * 2
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


    def __init__(self, params, size, parent = None):
        super(Sub_Pattern, self).__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.size = size
        self.value = [0,0]


    def compute_pattern(self, update = True):
        if self.daddy.blockupdating == False:
            self.value = np.asarray([self.xgui.value(), self.ygui.value()])
            self.daddy.offset = self.value
            if self.daddy.daddy.dbl_pass_state.checkState() and self.daddy.daddy.flt_fld_state.checkState():
                self.daddy.daddy.load_flat_field(self.daddy.daddy.p.left["cal1"], self.daddy.daddy.p.right["cal1"])
            self.daddy.crop()
        return self.value
    
    
class Sub_Pattern_Grid(Sub_Pattern):
    """ Subpattern containing the image data of the blazed gratings. 
        Recalculates the grating, and calls an update to the Half Pattern to 
        recalculate the whole image data. """
    def __init__(self, params, size, parent = None):
        super(Sub_Pattern, self).__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.size = size
        self.slm_px = params.general["slm_px"]
        self.data = np.zeros(self.size)
        
        
    def compute_pattern(self, update = True):
        if self.daddy.blockupdating == False:
            slope = [-self.xgui.value(), -self.ygui.value()]
            z = self.daddy.daddy.zernikes_normalized
            self.data = pcalc.add_images([z["tiptiltx"] * slope[0],
                                          z["tiptilty"] * slope[1]])
            if update:
                self.daddy.update()
        return self.data
    

class Sub_Pattern_Vortex(Sub_Pattern):
    """ Subpattern containing the image data of the vortex. GUI contains a 
        ComboBox to select the donut mode ("2D STED", "3D STED", "Gauss", 
        "Halfmoon", Bivortex) and four boxes for radius, phase rotation and 
        number of steps. Recalculates the vortices based on the parameters 
        and calls an update to the Half Pattern to recalculate the whole 
        image data. """


    def __init__(self, params, size, parent = None):
        super(Sub_Pattern, self).__init__(parent) 
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.size = size * 2
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
                        
            # execute all 'standard' cases via Pattern Calculator
            if mode != "Code Input" and mode != "From File":
                self.data = pcalc.compute_vortex(mode, self.size, rot, rad, 
                                                 steps, phase, slm_scale)
            
            # handle the two cases that need additional GUI separately
            elif mode == "Code Input":
                self.create_text_box(self.size, rot, rad, steps, phase, slm_scale)

            elif mode == "From File":
                fname = self.openFileDialog('../patterns/')
                if fname:
                    newvortex = pcalc.load_image(fname)
                    if np.all(newvortex.shape) == np.all(self.size):
                        if newvortex.dtype == 'uint8':
                            self.data = newvortex/(np.power(2, 8)-1)
                        elif newvortex.dtype == 'uint16':
                            self.data = newvortex/(np.power(2, 16)-1)
                        else:
                            print('Please select 8bit or 16bit image!')
                    else:
                        print('Please select file of the correct size: ', self.size, 'px!')
            
            if update:
                self.daddy.update()
        return self.data


    def openFileDialog(self, path):
        """ Creates a dialog to open a file. At the moement, it is only used 
            to load the image for the flat field correction. There is no 
            sanity check implemented whether the selected file is a valid image. """
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        work_dir = os.path.dirname(os.path.realpath(__file__))
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 
                        "Load image with vortex", work_dir +'/'+ path)
        if fileName:
            return fileName
        else:
            return None


    def create_text_box(self, size, rot, rad, steps, phase, slm_scale):
        """" Creates dialog window with text box; reads example text from file.
            Executes the code. Code should be written to calculate self.data
            (currently no measures to validate self.data is created, and of the
            correct format). """
        self.tdialog = QtWidgets.QDialog()
        screen = QtWidgets.QDesktopWidget().screenGeometry(0)
        self.tdialog.setGeometry(screen.left() + screen.width() / 4, 
                                 screen.top() + screen.height() / 4, 
                                 screen.width() / 2, screen.height() / 2)
        vbox = QtWidgets.QVBoxLayout()
        self.text_box = QtWidgets.QPlainTextEdit()
        highlight = slm_control.syntax.PythonHighlighter(self.text_box.document())
        textfile = open('slm_control/CodeInput.py', 'r')
        self.text_box.setPlainText(textfile.read())

        vbox.addWidget(self.text_box)
        hbox = QtWidgets.QHBoxLayout()
        self.crea_but(hbox, self._quit, "Cancel")
        self.crea_but(hbox, self.update_text, "Go")
        vbox.addLayout(hbox)
        self.tdialog.setLayout(vbox)
        self.tdialog.exec()


    def crea_but(self, box, action, name, param = None):
        button = QtWidgets.QPushButton(name, self)
        if param == None:
            button.clicked.connect(action)
        else:
            button.clicked.connect(lambda: action(param))
        button.setMaximumSize(120,50)
        box.addWidget(button)
        box.setAlignment(button, QtCore.Qt.AlignVCenter)
        box.setContentsMargins(0,0,0,0)
        return button
        
    
    def update_text(self):
        """ updates the  displayed image with the pattern created by the code
            from the text box"""
        text = self.text_box.toPlainText()
        try:
            exec(text)
        except:
            print("Invalid code:")
            import sys
            print(sys.exc_info())
        self.tdialog.accept()
        
    
    def _quit(self):
        print("Closing text input ...")
        self.tdialog.reject()


class Sub_Pattern_Defoc(Sub_Pattern):
    """ Subpattern containing the image data for defocus. GUI contains only the
        defocus parameter. Recalculates the defocus and calls an update to the 
        Half Pattern to recalculate the whole image data. """


    def __init__(self, params, size, parent = None):
        super(Sub_Pattern, self).__init__(parent)
        self.size = size * 2
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
