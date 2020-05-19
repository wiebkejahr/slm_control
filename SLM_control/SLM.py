#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 21:20:23 2018

@author: wjahr
"""

import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QPixmap, QImage, qRgb
#from PIL.ImageQt import ImageQt
import sys
import numpy as np

class SLM_Display(QtWidgets.QWidget):
    """ Creates the Widget to display the image on the SLM. Opens on 2nd screen,
        full screen. Set screen resolution of secondary to 800 x 600 px for the
        SLM to work properly. """
    
    def __init__(self, data, mode, parent = None):
        super(SLM_Display, self).__init__(parent)
        self.display = mode
        
        if self.display == "external":
            self.create_main_frame(data)
            self.show()
            self.raise_()
            
        elif self.display == "imspector":
            self.create_imspec_display(data)

                          
 
    def create_main_frame(self, data):
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        print("Opening SLM ...")
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)        
        screen1 = QtWidgets.QDesktopWidget().screenGeometry(1)
        self.setGeometry(screen1.left(), screen1.top(), screen1.width(), screen1.height())        
        self.showFullScreen()  
            
        vbox = QtWidgets.QVBoxLayout()
        self.img_frame = QtWidgets.QLabel(self)
        self.update_image(data)
        vbox.addWidget(self.img_frame)
        self.setLayout(vbox)

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