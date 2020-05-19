#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 21:20:23 2018

@author: wjahr
"""

import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QPixmap
        

class SLM_Display(QtWidgets.QWidget):
    """ Creates the Widget to display the image on the SLM. Opens on 2nd screen,
        full screen. Set screen resolution of secondary to 800 x 600 px for the
        SLM to work properly. """
    
    def __init__(self, img_path, parent = None):
        super(SLM_Display, self).__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        print("Opening SLM ...")
        #self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)        
        screen1 = QtWidgets.QDesktopWidget().screenGeometry(0)
        #self.setGeometry(screen1.left(), screen1.top(), screen1.width(), screen1.height())        
        self.move(screen1.width()/3, screen1.height()/3)        
        #self.showFullScreen()        
        self.create_main_frame(img_path)
 
    def create_main_frame(self, img_path):
        vbox = QtWidgets.QVBoxLayout()
        self.img = []
        self.img_frame = QtWidgets.QLabel(self)
        self.update_image(img_path)
        vbox.addWidget(self.img_frame)
        self.setLayout(vbox)
    
    def update_image(self, img_path):
        """ updates the  displayed image with the one provided in img_path. """
        self.img = QPixmap(img_path)
        self.img_frame.setPixmap(self.img)
    
    def _quit(self):
        print("Closing SLM ...")
        self.close()