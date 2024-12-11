#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py


"""
    Created on Tue Oct 16 10:15:38 2018
    @author: wjahr
    
    
    Code to create the phasemasks needed for STED microscopy and for adaptive 
    optics. Calculated phasemasks are displayed on an monitor output, to which 
    a spatial light modulator is connected; or sent to the Abberior Imspector 
    software. More detailed documentation can be found in the PDF file you 
    should have received with this code. If not, see:
    https://github.com/wiebkejahr/slm_control
    for more information and full documentation.
    This is the main class of the SLM control software. It launches and closes 
    the main GUI window and calls the Slm_Gui class to initiliaze the GUI with 
    the last saved parameters.
    
    
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


import sys
import PyQt5.QtWidgets as QtWidgets
import slm_control.Slm_Gui


class App(QtWidgets.QApplication):
    """ Creates the App containing all the widgets. Event handling to exit
        properly once the last window is closed, even when the Quit button
        was not used. """
    def __init__(self, *args):
        QtWidgets.QApplication.__init__(self, *args)
        self.main = slm_control.Slm_Gui.Main_Window(self)
        self.lastWindowClosed.connect(self.byebye)
        self.main.show()


    def byebye( self ):
        print("byebye")
        self.exit(0)
    
    
def main(args):
    global app
    app = App(args)
    app.exec_()


if __name__ == "__main__":
    """ makes sure that main is only executed if this code is the main program.
        The classes defined here are accessible from the outside as well, but
        then main isn't executed. """
    main(sys.argv)
