#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#SLM.py


"""
    Created on Mon Oct 15 21:20:23 2018
    @author: wjahr
    
    
    Handles communication with  SLM. Supports two operating modes, depending 
    on "display" parameter set in the parameters_general.txt:
        - external: displays image data fullscreen and in foreground with the 
          size specified by 'size_full' parameter on the (external) display 
          'display_num'
        - imspector: establishes communication with Abberior's Imspector 
          software, opens new window to display phase pattern within. In 
          Imspector, select 'custom' phasemask and connect image display via
          eyedropper tool. Important: Abberior adds up the displayed image and 
          all of their parameeters; may lead to unwanted results if there are 
          non-zero parameters in Imspector.
    

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


import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QPixmap, QImage
import numpy as np

class SLM_Display(QtWidgets.QWidget):
    """ Creates the Widget to display the image on the SLM. Opens on 2nd screen,
        full screen. Set screen resolution of secondary to 800 x 600 px for the
        SLM to work properly. """
    
    def __init__(self, data, mode, display, parent = None):
        super(SLM_Display, self).__init__(parent)
        self.display = mode
        
        if self.display == "external":
            self.create_main_frame(data, display)
            self.show()
            self.raise_()
            
        elif self.display == "imspector":
            self.create_imspec_display(data)
                          
 
    def create_main_frame(self, data, n_display):
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        print("Opening SLM ...")
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)        
        screen = QtWidgets.QDesktopWidget().screenGeometry(n_display)
        self.setGeometry(screen.left(), screen.top(), screen.width(), screen.height())        
        self.showFullScreen()  
            
        #vbox = QtWidgets.QVBoxLayout()
        self.img_frame = QtWidgets.QLabel(self)
        self.update_image(data)
        self.img_frame.setCursor(QtCore.Qt.BlankCursor) #vbox.addWidget(self.img_frame)
        self.img_frame.show() #self.setLayout(vbox)


    def create_imspec_display(self, data):
        # Handling of all the different things that could potentially go wrong
        # when communicating with the Abberior
        try:
            try:
                import specpy
                print("imported")
            except:
                print("specpy not installed")         
            try:
                imspec = specpy.Imspector()
            except:
                print("cannot connect to Imspector")
            try:
                self.meas = imspec.active_measurement()
            except:
                print("no active measurement")
                try:
                    self.meas = imspec.create_measurement()
                except:
                    print("no active measurement. cannot create measurement")
            self.stk = self.meas.create_stack(np.float, 
                                            [np.shape(data)[1], #792
                                             np.shape(data)[0], #600
                                             1, 1])
            self.stk.set_length(0, np.shape(data)[1]/1000)
            self.stk.set_length(1, np.shape(data)[0]/1000)

        except:
            print("""Something went wrong with Abberior. Cannot communicate.
                      Are you working using Imspector? If not, set 
                      'display_mode = "external"' in the parameters_general 
                      file. If you're using Imspector, check that it is 
                      running and a measurement is active. """)
        self.update_image(data)
    
    
    def update_image(self, data):
        """ updates the  displayed image with the one provided in img_path. """
        
        if self.display == "external":
            img = QImage(data, np.shape(data)[1], np.shape(data)[0], QImage.Format_Grayscale8)
            self.img_frame.setPixmap(QPixmap.fromImage(img))
        
        
        elif self.display == "imspector":
            #print("display imspector")
            try:
                self.stk.data()[:] = data / 255
                self.meas.update()
            except:
                print("Still cannot communicate with the Imspector software.")   

    
    def _quit(self):
        print("Closing SLM ...")
        self.close()