# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:22:25 2016

@author: wjahr

TODO:
+ set checkboxes according to params files upon startup
+ Load flat field correction according to checkbox
+ make sure that not only 1st aberration is loaded correctly from params
+ code sngl aberration case
- code split image
+ checkbox format when exporting to JSON
+ load image from file for vortex
+ scale according to wavelength
+ code bivortex
+ code segmented phase plate out of half moon
- save paths for corrections etc correctly when saving params (hardcoded atm)
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

import Pattern_Calculator as pcalc
import Pattern_Interface as PI
import Patterns_Zernike as PZ
import SLM

from Parameters import param

import autoalign.abberior as abberior

mpl.rc('text', usetex=False)
mpl.rc('font', family='serif')
mpl.rc('pdf', fonttype=42)

# NOTE: hardcoded for now:
# MODEL_STORE_PATH="models/04.02.20_xsection_10_epochs_Adam_lr_0.001_batchsize_64.pth"
MODEL_STORE_PATH="models/24.01.20_multi_test_20_epochs_Adam_lr_0.001_batchsize_64.pth"

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
        self.img_ax.imshow(data, interpolation = 'nearest', clim = [0,1], cmap = 'RdYlBu')#'PRGn')
        
        #TODO wj: remove circles
        # circle = plt.Circle((396/2, 600/2), 396/2, lw= 0.1, edgecolor = 'k', facecolor='None')
        # self.img_ax.add_artist(circle)
        # circle = plt.Circle((396/2, 600/2), 600/2, lw= 0.1, edgecolor = 'k', facecolor='None')
        # self.img_ax.add_artist(circle)
        # circle = plt.Circle((396/2, 600/2), np.mean([600,396])/2, lw= 0.1, edgecolor = 'k', facecolor='None')
        # self.img_ax.add_artist(circle)
        
        # circle = plt.Circle((3*396/2, 600/2), 396/2, lw= 0.1, edgecolor = 'k', facecolor='None')
        # self.img_ax.add_artist(circle)
        # circle = plt.Circle((3*396/2, 600/2), 600/2, lw= 0.1, edgecolor = 'k', facecolor='None')
        # self.img_ax.add_artist(circle)
        # circle = plt.Circle((3*396/2, 600/2), np.mean([600,396])/2, lw= 0.1, edgecolor = 'k', facecolor='None')
        # self.img_ax.add_artist(circle)
        
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
        self.setGeometry(screen0.left(), screen0.top(), screen0.width()/4, .9*screen0.height())
            
        self.load_params('parameters/params')
        self.init_images()
        self.create_main_frame()

        if self.p.general["abberior"] == 1:
            try:
                try:
                    import specpy
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
                                                [self.p.general["size_full"][1], #792
                                                 self.p.general["size_full"][0], #600
                                                 1, 1])
                self.stk.set_length(0, self.p.general["size_full"][1]/1000)
                self.stk.set_length(1, self.p.general["size_full"][0]/1000)
            except:
                print("""Something went wrong with Abberior. Cannot communicate.
                          Are you working at the Abberior? If not, set 
                          'abberior = 0' in the parameters_general file.
                          If you're at the Abberior, check that Imspector is 
                          running and a measurement is active. """)

            
        self.load_flat_field(self.p.general["cal1"])
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
        

    def reload_params(self, fname):
        """ Calls the load_file function implemented in the parameters class, 
            which loads the parameter lists from the text file. Called after
            button is clicked. """
        self.load_params(fname)
        self.img_l.update_guivalues(self.p, self.p.left)
        self.img_r.update_guivalues(self.p, self.p.right)
        
        self.objective_changed()
        #self.split_image(self.splt_img_state.checkState())
        #self.single_correction(self.sngl_corr_state.checkState())
        #self.flat_field(self.flt_fld_state.checkState(), recalc = False)
        
        self.recalc_images()
        #self.init_images()   

    # NOTE: I WROTE THIS
    def auto_align(self, model_store_path=MODEL_STORE_PATH):
        """This function calls abberior from AutoAlign module, passes the resulting dictionary
        through a constructor for a param object"""
        
        # try:
        self.zernike = abberior.abberior_multi(MODEL_STORE_PATH)
        # self.zernike = abberior.correct(MODEL_STORE_PATH) # returns a dictionary
        # print(self.zernike)
        
        # NOTE: self.p exists bc load_params() is called during creation of mainframe
        # this is setting each value to its current value plus the correction

        self.p.left["astig"] = [x+y for x,y in zip(self.p.left["astig"], self.zernike["astig"])]
        self.p.left["coma"] = [x+y for x,y in zip(self.p.left["coma"], self.zernike["coma"])]
        self.p.left["sphere"] = [x+y for x,y in zip(self.p.left["sphere"], self.zernike["sphere"])]
        self.p.left["trefoil"] = [x+y for x,y in zip(self.p.left["trefoil"], self.zernike["trefoil"])]

        self.p.right["astig"] = [x+y for x,y in zip(self.p.right["astig"], self.zernike["astig"])]
        self.p.right["coma"] = [x+y for x,y in zip(self.p.right["coma"], self.zernike["coma"])]
        self.p.right["sphere"] = [x+y for x,y in zip(self.p.right["sphere"], self.zernike["sphere"])]
        self.p.right["trefoil"] = [x+y for x,y in zip(self.p.right["trefoil"], self.zernike["trefoil"])]
        

        # this update_combvalues function has to retrieve the current GUI values and add on your dict vals
        self.img_l.update_guivalues(self.p, self.p.left)
        self.img_r.update_guivalues(self.p, self.p.right)
        # self.objective_changed()
        # self.split_image(self.splt_img_state.checkState())
        # self.single_correction(self.sngl_corr_state.checkState())
        # self.flat_field(self.flt_fld_state.checkState(), recalc = False)

        self.recalc_images()

        
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
            
        self.img_l = PI.Half_Pattern(self.p)
        self.img_l.call_daddy(self)
        self.img_l.set_name("img_l")
        
        self.img_r = PI.Half_Pattern(self.p)
        self.img_r.call_daddy(self)
        self.img_r.set_name("img_r")
        
        #self.img_aberr = PZ.Aberr_Pattern(self.p)
        
    def create_main_frame(self):
        """ Creates the UI: Buttons to load/save parameters and flatfield 
            correction. Frames to display the patterns. Creates the GUI 
            elements contained in the Half_Pattern and Aberr_Pattern classes 
            that are used to change the parameters for pattern creation. """
        
        self.main_frame = QtWidgets.QWidget()  
        # vertical box to place the controls above the image in
        vbox = QtWidgets.QVBoxLayout()     
        hbox = QtWidgets.QHBoxLayout()

        self.crea_but(hbox, self._quit, "Quit")
        
        self.rad_but = QtWidgets.QDoubleSpinBox()
        self.rad_but.setDecimals(3)
        self.rad_but.setSingleStep(0.01)
        self.rad_but.setMinimum(0.01)
        self.rad_but.setMaximum(10)
        self.rad_but.setValue(1.68)
        self.rad_but.setMaximumSize(80,50)
        self.rad_but.valueChanged.connect(lambda: self.radius_changed())
        hbox.addWidget(self.rad_but)

        
        vbox.addLayout(hbox)
        # NOTE: I WROTE THIS
        self.crea_but(hbox, self.auto_align, "Auto Align")
        
                    
        # doesn't do anything at the moment, could be used to set another path
        # to load the images from
        #strng_path = self.labeled_qt(QtWidgets.QLineEdit, "Path for images", vbox)
        #strng_path.setText(self.p.general["path"])

        # horizontal box for buttons that go side by side        
        hbox = QtWidgets.QHBoxLayout()
        self.obj_sel = QtWidgets.QComboBox(self)
        self.obj_sel.setMaximumSize(100, 50)
        hbox.addWidget(self.obj_sel)            
        for mm in self.p.objectives:
            self.obj_sel.addItem(self.p.objectives[mm]["name"])
        self.current_objective = self.p.general["objective"]
        self.obj_sel.setCurrentText(self.current_objective)
        self.obj_sel.activated.connect(lambda: self.objective_changed())
        
        self.slm_radius = pcalc.normalize_radius(self.p.objectives[self.p.general["objective"]]["backaperture"], 
                                                 self.p.general["slm_mag"], 
                                                 self.p.general["slm_px"], 
                                                 self.p.general["size_slm"])
            
        self.crea_but(hbox, self.reload_params, "Load Config", "parameters/params")
        self.crea_but(hbox, self.save_params, "Save Config", "parameters/params")
        hbox.setContentsMargins(0,0,0,0)
        vbox.addLayout(hbox)
        

        hbox = QtWidgets.QHBoxLayout()
        self.crea_but(hbox, self.open_SLMDisplay, "Initialize SLM")
        self.crea_but(hbox, self.close_SLMDisplay, "Close SLM")
        
        self.crea_but(hbox, self.openFlatFieldDialog, "load SLM calib", 
                      self.p.general["path"]+self.p.general["cal1"])        
        hbox.setContentsMargins(0,0,0,0)
        vbox.addLayout(hbox)

        hbox = QtWidgets.QHBoxLayout()
        
        self.splt_img_state = self.crea_checkbox(hbox, self.split_image, 
                                                 "Split image", 
                                                 self.p.general["split_image"])
        self.sngl_corr_state = self.crea_checkbox(hbox, self.single_correction, 
                                                  "Single correction", 
                                                  self.p.general["single_aberr"])
        self.flt_fld_state = self.crea_checkbox(hbox, self.flat_field, 
                                                "Flatfield", 
                                                self.p.general["flat_field"])
        hbox.setContentsMargins(0,0,0,0)
        vbox.addLayout(hbox)
        
        imgbox = QtWidgets.QHBoxLayout()
        #self.image = QPixmap(self.p.general["path"]+self.p.general["last_img_nm"])
        #self.img_frame = QtWidgets.QLabel(self)
        #self.img_frame.setPixmap(self.image.scaledToWidth(self.p.general["displaywidth"]))
        #imgbox.addWidget(self.img_frame)
        imgbox.setAlignment(QtCore.Qt.AlignRight)
        
        # this creates the matplotlib canvas for testing purposes
        # uncomment if needed, also uncomment the 
        #"self.plt_frame.plot()" line in the update_display function"
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
        lbox_img.addWidget(QtWidgets.QLabel('Steps'))
        
        lbox_img.addWidget(QtWidgets.QLabel('Astigmatism X/Y'))
        lbox_img.addWidget(QtWidgets.QLabel('Coma X/Y'))
        lbox_img.addWidget(QtWidgets.QLabel('Spherical 1/2'))
        lbox_img.addWidget(QtWidgets.QLabel('Trefoil Vert/Obl'))
        
        lbox_img.setContentsMargins(0,0,0,0)
        
        # create the controls provided by img_l, img_r and img_aberr
        # these are used to set the parameters to create the patterns
        c = QtWidgets.QGridLayout()
        c.addLayout(lbox_img, 0, 0, 1, 1)
        c.addLayout(self.img_l.create_gui(self.p, self.p.left), 0, 1, 1, 2)
        c.addLayout(self.img_r.create_gui(self.p, self.p.right), 0, 3, 1, 2)
        #c.addLayout(lbox_aber, 1, 0, 1, 1)
        #c.addLayout(self.img_aberr.create_gui(self.p), 1, 2, 1, 4)
        c.setAlignment(QtCore.Qt.AlignRight)
        c.setContentsMargins(0,0,0,0)
        

        # add all the widgets
        vbox.addLayout(imgbox)
        vbox.addLayout(c)
        vbox.setContentsMargins(0,0,0,0)
        self.main_frame.setLayout(vbox)       
        self.setCentralWidget(self.main_frame)
        

    def openFileDialog(self, path):
        """ Creates a dialog to open a file. At the moement, it is only used 
            to load the image for the flat field correction. There is no 
            sanity check implemented whether the selected file is a valid image. """
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        work_dir = os.path.dirname(os.path.realpath(__file__))
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load flat field correction", work_dir +'/'+ path)
        if fileName:
            return fileName
        else:
            return None

        
    def openFlatFieldDialog(self, path):
        """ Creates a dialog to open a file. At the moement, it is only used 
            to load the image for the flat field correction. There is no 
            sanity check implemented whether the selected file is a valid image. """
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        work_dir = os.path.dirname(os.path.realpath(__file__))
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load flat field correction", work_dir +'/'+ path)
        if fileName:
            self.load_flat_field(fileName)
            self.combine_images()
            self.update_display()
    
    def load_flat_field(self, path):
        """ Opens the image in the parameter path and sets as new flatfield
            correction. """
        self.flatfieldcor = np.asarray(pcalc.load_image(path))/255
        #self.flatfield = self.flatfieldcor
        self.flat_field(self.flt_fld_state.checkState())

    def flat_field(self, state, recalc = True):
        """ Opens the image in the parameter path and sets as new flatfield
            correction. """
        if state:
            self.flatfield = self.flatfieldcor
        else:
            self.flatfield = np.zeros_like(self.flatfieldcor)
        if recalc:
            self.recalc_images()
            
    def crea_but(self, box, action, name, param = None):
        """ Creates and labels a button and connects button and action. Input: 
            Qt layout to place the button, function: action to perform, string: 
            name of the button. Returns the button. """
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


    def labeled_qt(self, QtItem, label, main_layout):
        """ Adds a QtItem with a label. Inputs: function pointer to the QtItem,
            label string, main_layout to put the item in. """
        box = QtWidgets.QHBoxLayout()     
        box.addWidget(QtWidgets.QLabel(label))
        item = QtItem()
        box.addWidget(item)
        main_layout.addLayout(box)
        return item

    def crea_checkbox(self, box, action, name, state, param = None):
        """ Creates and labels a checkbox and connects button and action. Input: 
            Qt layout to place the button, function: action to perform, string: 
            name of the button. Returns the button. """
        checkbox = QtWidgets.QCheckBox(name, self)
        if param == None:
            checkbox.clicked.connect(action)
        else:
            checkbox.clicked.connect(lambda: action(param))
        box.addWidget(checkbox)
        box.setAlignment(checkbox, QtCore.Qt.AlignVCenter)   
        box.setContentsMargins(0,0,0,0)
        if state:
            checkbox.setChecked(True)
        return checkbox



    def split_image(self, state = True):
        """ Action called when the "Split image" checkbox is selected. Toggles
            between split image operation and single image operation."""
        if state:
            print(state, "TODO splitimage")
        else:
            print(state, "TODO nosplitimage")
        
    def single_correction(self, state):
        """ Action called when the "Single Correction" checkbox is selected.
            Toggles between identical correction for both halves of the sensor
            and using individidual corrections. When single correction is 
            active, the values from the left sensor half are used. """
        if state:
            self.img_r.aberr.astig.xgui.setValue(self.img_l.aberr.astig.xgui.value())
            self.img_r.aberr.astig.ygui.setValue(self.img_l.aberr.astig.ygui.value())
            self.img_r.aberr.coma.xgui.setValue(self.img_l.aberr.coma.xgui.value())
            self.img_r.aberr.coma.ygui.setValue(self.img_l.aberr.coma.ygui.value())
            self.img_r.aberr.sphere.xgui.setValue(self.img_l.aberr.sphere.xgui.value())
            self.img_r.aberr.sphere.ygui.setValue(self.img_l.aberr.sphere.ygui.value())
            self.img_r.aberr.trefoil.xgui.setValue(self.img_l.aberr.trefoil.xgui.value())
            self.img_r.aberr.trefoil.ygui.setValue(self.img_l.aberr.trefoil.ygui.value())
            
    def radius_changed(self):
        """ Action called when the users selects a different objective. 
            Calculates the diameter of the BFP; then recalculates the the
            patterns based on the selected objective. """
        self.current_objective = self.p.objectives[self.obj_sel.currentText()]
        
        self.radius_input = self.rad_but.value()
        print("radius changed")

        self.slm_radius = pcalc.normalize_radius(self.radius_input, 
                                                1, 
                                                self.p.general["slm_px"], 
                                                self.p.general["size_slm"])
        self.recalc_images()

            
    def objective_changed(self):
        """ Action called when the users selects a different objective. 
            Calculates the diameter of the BFP; then recalculates the the
            patterns based on the selected objective. """
        self.current_objective = self.p.objectives[self.obj_sel.currentText()]
        self.slm_radius = pcalc.normalize_radius(self.p.objectives[self.current_objective["name"]]["backaperture"], 
                                                 self.p.general["slm_mag"], 
                                                 self.p.general["slm_px"], 
                                                 self.p.general["size_slm"])
        self.recalc_images()
        
        
    def apply_correction(self):
        print("applying corrections")
    
        
    def recalc_images(self):
        """ Function to recalculate the left and right images completely. 
            Update is set to false to prevent redrawing after every step of 
            the recalculation. Image display is only updated once at the end. """
        self.img_l.update(update = False, completely = True)
        self.img_r.update(update = False, completely = True)
        self.combine_images()
        self.update_display()


    def combine_images(self):
        """ Stitches the images for left and right side of the SLM, adds the 
            flatfield correction and phasewraps everything. Updates
            self.img_data with the new image data. """
        self.img_data = pcalc.stitch_images(self.img_l.data, self.img_r.data)
        self.img_data = pcalc.add_images([self.flatfield, self.img_data])
        #TODO put back in
        self.img_data = pcalc.phase_wrap(self.img_data, self.p.general["phasewrap"])


    def update_display(self):
        """ Updates the displayed images in the control window and (if active)
            on the SLM. To do so, rescales the image date to the SLM's required
            pitch (depends on the wavelength, and is set in the general 
            parameters). Saves the image patterns/latest.bmp and then reloads
            into the Pixmap for display. """
        img_data_scaled = self.img_data * self.p.general["slm_range"] / 255
        pcalc.save_image(img_data_scaled, self.p.general["path"], self.p.general["last_img_nm"])
        self.image = QPixmap(self.p.general["path"]+self.p.general["last_img_nm"])
        #self.img_frame.setPixmap(self.image.scaledToWidth(self.p.general["displaywidth"]))
        
        if self.p.general["abberior"] == 1:
                try:
                    self.stk.data()[:]=img_data_scaled
                    self.meas.update()
                except:
                    print("Still cannot communicate with the Abberior.")            
        elif self.slm != None:
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
