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
    
    def __init__(self, data, parent = None):
        super(SLM_Display, self).__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        print("Opening SLM ...")
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)        
        screen1 = QtWidgets.QDesktopWidget().screenGeometry(1)
        self.setGeometry(screen1.left(), screen1.top(), screen1.width(), screen1.height())        
        self.showFullScreen()        
        self.create_main_frame(data)
 
    def create_main_frame(self, data):
        vbox = QtWidgets.QVBoxLayout()
        self.img_frame = QtWidgets.QLabel(self)
        self.update_image(data)
        vbox.addWidget(self.img_frame)
        self.setLayout(vbox)
    
    def update_image(self, data):
        """ updates the  displayed image with the one provided in img_path. """
        img = QImage(data, np.shape(data)[1], np.shape(data)[0], QImage.Format_Grayscale8)
        self.img_frame.setPixmap(QPixmap.fromImage(img))

    
    def _quit(self):
        print("Closing SLM ...")
        self.close()