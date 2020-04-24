# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:22:25 2016

@author: wjahr
"""

import sys, os
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QPixmap, QImage

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import SLM_control.Pattern_Calculator as pcalc
import SLM_control.Pattern_Interface as PI
import SLM_control.Patterns_Zernike as PZ
import SLM_control.SLM_forLabmeeting as SLM
from SLM_control.Parameters import param

mpl.rc('text', usetex=False)
mpl.rc('font', family='serif')
mpl.rc('pdf', fonttype=42)


class PlotCanvas(FigureCanvas):
    """ Provides a matplotlib canvas to be embedded into the widgets. "Native"
        matplotlib.pyplot doesn't work because it interferes with the Qt5
        framework. Plot function of this class takes the data passed as an
        argument and plots via imshow(). Handy for testing things, because
        the QPixmap automatically phasewraps intensities into the space between
        [0,1], which might interfere with the phasewrapping implemented for 
        the SLM. """
        
    def __init__(self, parent=None, w=800, h=600, dpi=200):
        w = w / dpi
        h = h / dpi
        fig = Figure(figsize=(w,h), dpi=dpi)
        self.img_ax = fig.add_subplot(111)
        self.img_ax.set_xticks([]), self.img_ax.set_yticks([])
        
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
    def plot(self, data):
        self.img_ax.imshow(data, interpolation = 'nearest', cmap = 'PRGn')
        self.draw()


class Main_Window(QtWidgets.QMainWindow):
    """ Main window for SLM control. Controls to change the parameters, and all
        function calls. """

    def __init__(self, app, parent=None):
        """ Called upon start up of the class. Initializes the Gui, places all
            windows. Loads the parameters from file and initializes the 
            patterns with the parameter sets loaded from the files."""
        QtWidgets.QMainWindow.__init__(self, parent)        
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        
        self.setWindowTitle('Main Window')
        self.app = app
        self.slm = None
        
        screen0 = QtWidgets.QDesktopWidget().screenGeometry()
        #self.move(screen0.left(), screen0.top())
        self.setGeometry(screen0.left(), screen0.top(), screen0.width()/4, .9*screen0.height())
        
        self.load_params('parameters/params')
        self.init_images()
        self.create_main_frame()
        self.combine_images()
        self.update_display()
        
        self.show()
        self.raise_()

        
    def load_params(self, fname):
        """ Calls the load_file function implemented in the parameters class, 
            which loads the parameter lists from the text file. Called after
            startup of the program. """
        self.p = param()
        self.p.load_file(fname)
        
        
    def save_params(self, fname):
        """" Calls write_file implemented in parameters class to save the 
            current parameters to the file provided in fname. These are then 
            loaded as default on next startup."""
        self.p.update(self)
        self.p.write_file(fname)
        
    def init_images(self):
        """ Called upon startup of the program. Initizialises the variables
            containing the images for flatfield correction, for the left and
            right halves of the SLM and for aberration correction. """
        self.flat_field(self.p.general["cal"])
        
        self.img_l = PI.Half_Pattern(self.p)
        self.img_l.call_daddy(self)
        
        self.img_r = PI.Half_Pattern(self.p)
        self.img_r.call_daddy(self)

        self.img_aberr = PZ.Aberr_Pattern(self.p)
        
    def create_main_frame(self):
        """ Creates the UI: Buttons to load/save parameters and flatfield 
            correction. Frames to display the patterns. Creates the GUI 
            elements contained in the Half_Pattern and Aberr_Pattern classes 
            that are used to change the parameters for pattern creation. """
        
        self.main_frame = QtWidgets.QWidget()  
        
        # vertical box to place the controls above the image in
        vbox = QtWidgets.QVBoxLayout()     
        self.crea_but(vbox, self._quit, "Quit")
        
        # doesn't do anything at the moment, could be used to set another path
        # to load the images from
        #strng_path = self.labeled_qt(QtWidgets.QLineEdit, "Path for images", vbox)
        #strng_path.setText(self.p.general["path"])

        # horizontal box for buttons that go side by side        
        hbox = QtWidgets.QHBoxLayout()
        self.crea_but(hbox, self.load_params, "Load Config", "parameters/params")
        self.crea_but(hbox, self.save_params, "Save Config", "parameters/params")
        vbox.addLayout(hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.crea_but(hbox, self.open_SLMDisplay, "Initialize SLM")
        self.crea_but(hbox, self.close_SLMDisplay, "Close SLM")
        
        self.crea_but(hbox, self.openFileNameDialog, "load SLM calib", 
                      self.p.general["path"]+self.p.general["cal"])
        
        vbox.addLayout(hbox)
        
        imgbox = QtWidgets.QHBoxLayout()
#        self.image = QPixmap(self.p.general["path"]+self.p.general["last_img_nm"])
#        self.img_frame = QtWidgets.QLabel(self)
#        self.img_frame.setPixmap(self.image.scaledToWidth(self.p.general["displaywidth"]))
#        imgbox.addWidget(self.img_frame)
        imgbox.setAlignment(QtCore.Qt.AlignRight)
        
        # this creates the matplotlib canvas for testing purposes
        # uncomment if needed, also uncomment the 
        # "self.plt_frame.plot()" line in the update_display function"
        self.plt_frame = PlotCanvas(self)      
        imgbox.addWidget(self.plt_frame)

        # create the labels beneath image
        lbox_img = QtWidgets.QVBoxLayout()
        lbox_img.addWidget(QtWidgets.QLabel('Offset X')) 
        lbox_img.addWidget(QtWidgets.QLabel('Offset Y'))
        lbox_img.addWidget(QtWidgets.QLabel('Grating X'))
        lbox_img.addWidget(QtWidgets.QLabel('Grating Y'))
        lbox_img.addWidget(QtWidgets.QLabel('Defocus L/R'))
        lbox_img.addWidget(QtWidgets.QLabel('STED Mode'))
        lbox_img.addWidget(QtWidgets.QLabel('Radius'))
        lbox_img.addWidget(QtWidgets.QLabel('Phase'))
        lbox_img.addWidget(QtWidgets.QLabel('Rotation'))
        
        lbox_aber = QtWidgets.QVBoxLayout()
        lbox_aber.addWidget(QtWidgets.QLabel('Astigmatism X/Y'))
        lbox_aber.addWidget(QtWidgets.QLabel('Coma X/Y'))
        lbox_aber.addWidget(QtWidgets.QLabel('Spherical 1/2'))
        lbox_aber.addWidget(QtWidgets.QLabel('Trefoil Vert/Obl'))
        
        # create the controls provided by img_l, img_r and img_aberr
        # these are used to set the parameters to create the patterns
        c = QtWidgets.QGridLayout()
        c.addLayout(lbox_img, 0, 0, 1, 1)
        c.addLayout(self.img_l.create_gui(self.p, self.p.left), 0, 1, 1, 2)
        c.addLayout(self.img_r.create_gui(self.p, self.p.right), 0, 3, 1, 2)
        c.addLayout(lbox_aber, 1, 0, 1, 1)
        c.addLayout(self.img_aberr.create_gui(self.p), 1, 2, 1, 4)
        c.setAlignment(QtCore.Qt.AlignRight)
        
        # add all the widgets
        vbox.addLayout(imgbox)
        vbox.addLayout(c)
        self.main_frame.setLayout(vbox)       
        self.setCentralWidget(self.main_frame)
        
    def openFileNameDialog(self, path):
        """ Creates a dialog to open a file. At the moement, it is only used 
            to load the image for the flat field correction. There is no 
            sanity check implemented whether the selected file is a valid image. """
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        work_dir = os.path.dirname(os.path.realpath(__file__))
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load flat field correction", work_dir +'/'+ path)
        if fileName:
            self.flat_field(fileName)
            self.combine_images()
            self.update_display()        
    
    def flat_field(self, path):
        """ Opens the image in the parameter path and sets as new flatfield
            correction. """
        #self.p.general["cal"] = path
        #print(self.p.general["cal"])
        self.flatfield = np.asarray(pcalc.load_image(path))/255

            
    def crea_but(self, box, action, name, param = None):
        """ Creates and labels a button and connects button and action. Input: 
            Qt layout to place the button, function: action to perform, string: 
            name of the button. Returns the button. """
        button = QtWidgets.QPushButton(name, self)
        if param == None:
            button.clicked.connect(action)
        else:
            button.clicked.connect(lambda: action(param))
        box.addWidget(button)
        box.setAlignment(button, QtCore.Qt.AlignVCenter)             
        return button


    def labeled_qt(self, QtItem, label, main_layout):
        """ Adds a QtItem with a label. Inputs: function pointer to the QtItem,
            label string, main_layout to put the item in. """
        box = QtWidgets.QHBoxLayout()     
        box.addWidget(QtWidgets.QLabel(label))
        item = QtItem()
        box.addWidget(item)
        main_layout.addLayout(box)
        return item

    
    def recalc_images(self):
        """ Function to recalculate the left and right images completely. 
            Update is set to false to prevent redrawing after every step of 
            the recalculation. Image display is only updated once at the end. """
        self.img_l.update(update = False)
        self.img_r.update(update = False)
        self.combine_images()
        self.update_display()


    def combine_images(self):
        """ Stitches the images for left and right side of the SLM, adds the 
            flatfield correction and phasewraps everything. Updates
            self.img_data with the new image data. """
        self.img_data = pcalc.stitch_images(self.img_l.data, self.img_r.data)
        self.img_data = pcalc.add_images([self.flatfield, self.img_data])
        self.img_data = pcalc.phase_wrap(self.img_data, self.p.general["phasewrap"])


    def update_display(self):
        """ Updates the displayed images in the control window and (if active)
            on the SLM. To do so, rescales the image date to the SLM's required
            pitch (depends on the wavelength, and is set in the general 
            parameters). Saves the image patterns/latest.bmp and then reloads
            into the Pixmap for display. """
        self.img_data = self.img_data * self.p.general["slm_range"] / 255
        pcalc.save_image(self.img_data, self.p.general["path"], self.p.general["last_img_nm"])
        #self.image = QPixmap(self.p.general["path"]+self.p.general["last_img_nm"])
        #self.img_frame.setPixmap(self.image.scaledToWidth(self.p.general["displaywidth"]))
        if self.slm != None:
            self.slm.update_image(self.p.general["path"]+self.p.general["last_img_nm"])       
        self.plt_frame.plot(self.img_data)
 
    
    def open_SLMDisplay(self):
        """ Opens a widget fullscreen on the secondary screen that displays
            the latest image. """
        self.slm = SLM.SLM_Display(self.p.general["path"]+self.p.general["last_img_nm"])
        self.slm.show()
        self.slm.raise_()

        
    def close_SLMDisplay(self):
        """ Closes the SLM window (if it exists) and resets reference to None 
            to prevent errors when trying to close SLM again. """
        if self.slm != None:
            self.slm._quit()
            self.slm = None


    def _quit(self):
        """ Action called when the Quit button is pressed. Closes the SLM 
            control window and exits the main window. """
        #self.save_params("parameters/params")
        self.close_SLMDisplay()           
        self.close()


class App(QtWidgets.QApplication):
    """ Creates the App containing all the widgets. Event handling to exit
        properly once the last window is closed, even when the Quit button
        was not used. """
    def __init__(self, *args):
        QtWidgets.QApplication.__init__(self, *args)
        self.main = Main_Window(self)
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
