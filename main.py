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
# standard imports
import sys, os

# third party imports 
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QPixmap, QImage
try:
    import specpy as sp
except:
    print("Specpy not installed!")
    pass

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

from PIL import Image

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import scipy
import scipy.ndimage 

# local packages
import slm_control.Pattern_Calculator as pcalc
import slm_control.Pattern_Interface as PI
import slm_control.Patterns_Zernike as PZ
import slm_control.SLM as SLM

from slm_control.Parameters import param

sys.path.insert(1, os.getcwd())
sys.path.insert(1, 'slm_control/')
sys.path.insert(1, 'autoalign/')
import autoalign.abberior as abberior   
import autoalign.utils.helpers as helpers

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
        self.img_ax.imshow(data, interpolation = 'nearest', clim = [0,1], cmap = 'RdYlBu')#'PRGn')
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
        self.setGeometry(screen0.left(), screen0.top(), 
                         screen0.width()/4, .9*screen0.height())
        
                    
        self.param_path = ['parameters/', 'params']
        self.p = param()
        self.p.load_file_general(self.param_path[0], self.param_path[1])
        self.current_objective = self.p.general["objective"]
        self.p.load_file_obj(self.param_path[0], self.current_objective, self.param_path[1])
                
        self.slm_radius = self.calc_slmradius(
            self.p.objectives[self.current_objective]["backaperture"],
            self.p.general["slm_mag"])
        
        
        self.init_data()
        
        self.show()
        self.raise_()
        
    def init_data(self):
        """ Called upon start up to initialize all the date for the first time.
            Recalled when the split_image checkbox is changed, because this
            will change the size of all the images etc. """        
        
        if self.p.general["split_image"]:
            self.img_size = np.asarray(self.p.general["size_slm"])
            self.load_flat_field(self.p.left["cal1"], self.p.right["cal1"], recalc = False)
        else:
            self.img_size = np.asarray([self.p.general["size_slm"][0], 
                                        self.p.general["size_slm"][1]*2])
            self.load_flat_field(self.p.general["cal1"], self.p.general["cal1"], recalc = False)
        
        self.init_zernikes()
        self.init_images()
        self.create_main_frame()
        self.combine_and_update()

        
    # def load_params(self, fname):
    #     """ Calls the load_file function implemented in the parameters class, 
    #         which loads the parameter lists from the text file. Called after
    #         startup of the program. """
    #     self.p = param()
    #     self.p.load_file(fname[0], self.current_objective["name"], fname[1])
        

    def reload_params(self, fname):
        """ Calls the load_file function implemented in the parameters class, 
            which loads the parameter lists from the text file. Called after
            button is clicked. """
        #self.load_params(fname)
        #self.p = param()
        self.p.load_file_obj(fname[0], self.current_objective, fname[1])
        print("params ", self.p.full["mode"])
        # print("left", self.p.left)
        # print("right", self.p.right)
        if self.p.general["split_image"]:
            self.img_l.update_guivalues(self.p, self.p.left)
            self.img_r.update_guivalues(self.p, self.p.right)
            self.zernikes_all = np.zeros_like(self.img_l.data)
            self.phase_tiptilt = np.zeros_like(self.img_l.data)
            self.phase_defocus = np.zeros_like(self.img_l.data)
        else:
            #TODO: general values, not left values
            print("loading from file")
            self.img_full.update_guivalues(self.p, self.p.full)
            self.zernikes_all = np.zeros_like(self.img_full.data)
            self.phase_tiptilt = np.zeros_like(self.img_full.data)
            self.phase_defocus = np.zeros_like(self.img_full.data)

        #self.objective_changed()
        # TODO WJ
        #self.split_image(self.splt_img_state.checkState())
        #self.single_correction(self.sngl_corr_state.checkState())
        #self.flat_field(self.flt_fld_state.checkState(), recalc = False)
        
        self.recalc_images()
        #self.init_images()   

        
    def save_params(self, fname):
        """" Calls write_file implemented in parameters class to save the 
            current parameters to the file provided in fname. These are then 
            loaded as default on next startup."""
        self.p.update(self)
        
        self.p.write_file(fname[0], self.current_objective, fname[1])
        
        
    def init_images(self):
        """ Called upon startup of the program. Initizialises the variables
            containing the left and right halves of the SLM or the full image,
            depending on the state of the "split image" boolean. """

        # setting all to None first helps when function is called later on
        # to clear all data, eg when 'split image' boolean is changed 
        # self.img_l = None
        # self.img_r = None
        # self.img_full = None
        # #self.flatfield_orig = None
        # self.zernikes_all = None
        # self.phase_tiptilt = None
        # self.phase_defocs = None
        
        if self.p.general["split_image"]:
            
            self.img_l = PI.Half_Pattern(self.p, self.img_size)
            self.img_l.call_daddy(self)
            self.img_l.set_name("img_l")
            
            self.img_r = PI.Half_Pattern(self.p, self.img_size)
            self.img_r.call_daddy(self)
            self.img_r.set_name("img_r")
    
            self.zernikes_all = np.zeros_like(self.img_l.data)
            self.phase_tiptilt = np.zeros_like(self.img_l.data)
            self.phase_defocus = np.zeros_like(self.img_l.data)
            #print("left img ", self.img_l)
            #print(self.p)
            
        else:
            self.img_full = PI.Half_Pattern(self.p, self.img_size)
            self.img_full.call_daddy(self)
            self.img_full.set_name("full")
            self.zernikes_all = np.zeros_like(self.img_full.data)
            self.phase_tiptilt = np.zeros_like(self.img_full.data)
            self.phase_defocus = np.zeros_like(self.img_full.data)
            #print("full img ", self.img_full.data)
            #print("full img ", self.img_full.radgui.value())
            #print(self.p)
        
        
        # images are created after startup
        # then the gui is created, which also links all the defoc etc parameters into the half_pattern class
        # how do I relink?!        
        #if self.p.general["split_image"]:
        #    c.addLayout(self.img_l.create_gui(self.p, self.p.left), 0, 1, 1, 2)
        #    c.addLayout(self.img_r.create_gui(self.p, self.p.right), 0, 3, 1, 2)
        #else:
        #    #TODO: change this left; requires change for all objectives
        #    c.addLayout(self.img_full.create_gui(self.p, self.p.left), 0, 1, 1, 2)
        
    def init_zernikes(self):
        """ Creates a dictionary containing all of the Zernike polynomials by
            their names. Updates in the GUI only change the weight of each 
            polynomial. Thus, polynomials do not have to be updated, just their
            weights. """
            
        # normalisation for tip / tilt is different from the other Zernikes:
        # grating periods should be in /mm and they should be independent of
        # the objective's diameter:
        # we're using the SLM off-axis to reflect the beam into the center of
        # the objective's backaperture. None of the actual optics depends on the
        # objective used.
        # I'm using the SLM radius calculation with backaperture = mag = 1
        # to determine correct size for tip/tilt. Extra factor of two is because
        # patterns are created at double size, then cropped.

        self.rtiptilt = 2 * pcalc.normalize_radius(1, 1, self.p.general["slm_px"], 
                                                    self.p.general["size_slm"])
        
        self.zernikes_normalized = {
            "tiptiltx" : pcalc.create_zernike(2 * self.img_size, [ 1,  1], 1, self.rtiptilt),
            "tiptilty" : pcalc.create_zernike(2 * self.img_size, [ 1, -1], 1, self.rtiptilt),
            "defocus"  : pcalc.create_zernike(2 * self.img_size, [ 2,  0], 1, self.slm_radius),
            "astigx"   : pcalc.create_zernike(2 * self.img_size, [ 2,  2], 1, self.slm_radius),
            "astigy"   : pcalc.create_zernike(2 * self.img_size, [ 2, -2], 1, self.slm_radius),
            "comax"    : pcalc.create_zernike(2 * self.img_size, [ 3,  1], 1, self.slm_radius),
            "comay"    : pcalc.create_zernike(2 * self.img_size, [ 3, -1], 1, self.slm_radius),
            "trefoilx" : pcalc.create_zernike(2 * self.img_size, [ 3,  3], 1, self.slm_radius),
            "trefoily" : pcalc.create_zernike(2 * self.img_size, [ 3, -3], 1, self.slm_radius),           
            "sphere1"  : pcalc.create_zernike(2 * self.img_size, [ 4,  0], 1, self.slm_radius),
            "sphere2"  : pcalc.create_zernike(2 * self.img_size, [ 6,  0], 1, self.slm_radius)
            }
    
        
    def create_main_frame(self):
        """ Creates the UI: Buttons to load/save parameters and flatfield 
            correction. Frames to display the patterns. Creates the GUI 
            elements contained in the Half_Pattern and Aberr_Pattern classes 
            that are used to change the parameters for pattern creation. """
        
        self.main_frame = QtWidgets.QWidget()  
        vbox = QtWidgets.QVBoxLayout()     

        # Quit, objective diameter and autoalign buttons
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
        self.crea_but(hbox, self.auto_align, "Auto Align")
        vbox.addLayout(hbox)
        self.crea_but(hbox, self.correct_tiptilt, "Tip/Tilt")
        self.crea_but(hbox, self.automate, "Auto-test")
                    
        # doesn't do anything at the moment, could be used to set another path
        # to load the images from
        #strng_path = self.labeled_qt(QtWidgets.QLineEdit, "Path for images", vbox)
        #strng_path.setText(self.p.general["path"])

        # controls to change objectives and to load/save calibration files         
        hbox = QtWidgets.QHBoxLayout()
        self.obj_sel = QtWidgets.QComboBox(self)
        self.obj_sel.setMaximumSize(100, 50)
        hbox.addWidget(self.obj_sel)            
        for mm in self.p.objectives:
            self.obj_sel.addItem(mm)
        self.obj_sel.setCurrentText(self.current_objective)
        self.obj_sel.activated.connect(lambda: self.objective_changed())
        self.crea_but(hbox, self.reload_params, "Load Config", self.param_path)
        self.crea_but(hbox, self.save_params, "Save Config", self.param_path)
        hbox.setContentsMargins(0,0,0,0)
        vbox.addLayout(hbox)
        
        # controls for basic SLM operation: opening/closing the display,
        # changing calibration
        hbox = QtWidgets.QHBoxLayout()
        self.crea_but(hbox, self.open_SLMDisplay, "Initialize SLM")
        self.crea_but(hbox, self.close_SLMDisplay, "Close SLM")
        
        # TODO: currently does not work; need to figure out a way to implement
        # two paths, one for left, one for right side
        #self.crea_but(hbox, self.openFlatFieldDialog, "load SLM calib", 
        #              self.p.general["path"]+self.p.general["cal1"])        
        hbox.setContentsMargins(0,0,0,0)
        vbox.addLayout(hbox)

        # checkboxes for the different modes of operation: split image
        # (currently not uptdating life), flatfield correction
        # and single correction and cross correction for double pass geometry
        # (as on the Abberior))
        hbox = QtWidgets.QHBoxLayout()
        self.splt_img_state = self.crea_checkbox(hbox, self.split_image, 
                        "Split image", self.p.general["split_image"])
        self.sngl_corr_state = self.crea_checkbox(hbox, self.single_correction, 
                        "Single correction", self.p.general["single_aberr"])
        self.dbl_pass_state = self.crea_checkbox(hbox, self.double_pass,
                        "Double pass", self.p.general["double_pass"])
        self.flt_fld_state = self.crea_checkbox(hbox, self.flat_field, 
                        "Flatfield", self.p.general["flat_field"])
        hbox.setContentsMargins(0,0,0,0)
        vbox.addLayout(hbox)
        
        # creates the Widget to display the image that's being sent to the SLM
        imgbox = QtWidgets.QHBoxLayout()
        imgbox.setAlignment(QtCore.Qt.AlignRight)
        self.plt_frame = PlotCanvas(self)      
        imgbox.addWidget(self.plt_frame)

        
        # create the labels beneath image. Numeric controls are added in the
        # respective subfunctions.
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
        if self.p.general["split_image"]:
            c.addLayout(self.img_l.create_gui(self.p, self.p.left), 0, 1, 1, 2)
            c.addLayout(self.img_r.create_gui(self.p, self.p.right), 0, 3, 1, 2)
        else:
            c.addLayout(self.img_full.create_gui(self.p, self.p.full), 0, 1, 1, 2)
            
        c.setAlignment(QtCore.Qt.AlignRight)
        c.setContentsMargins(0,0,0,0)
        
        # add all the widgets
        vbox.addLayout(imgbox)
        vbox.addLayout(c)
        vbox.setContentsMargins(0,0,0,0)
        self.main_frame.setLayout(vbox)       
        self.setCentralWidget(self.main_frame)
        
        
    def correct_defocus(self):
        self.defocus = abberior.correct_defocus()#(const=1/6.59371319)
        
        
        size = 2 * np.asarray(self.p.general["size_slm"])   
        off = [self.img_l.off.xgui.value(), self.img_l.off.ygui.value()]  
        
        defoc_correct = pcalc.crop(helpers.create_phase([self.defocus], 
                                                        num=[2], 
                                                        size = size,
                                                        radscale = self.slm_radius),
                                          size/2, offset = off)
        #TODO: Why is added defocus positive, not negative?
        # and why is radscale w/o sqrt(2)?
        self.phase_defocus = self.phase_defocus + defoc_correct
        self.recalc_images()


    def correct_tiptilt(self):
        self.tiptilt = abberior.correct_tip_tilt()
        size = np.asarray(self.p.general["size_slm"])
        tiptilt_correct = helpers.create_phase(coeffs=self.tiptilt, num=[0,1], 
                                               size = size, radscale = 2*self.rtiptilt)
        
        self.phase_tiptilt = self.phase_tiptilt + tiptilt_correct
        self.recalc_images()
    
    
    def corrective_loop(self, image=None, offset=False, multi=False,  ortho_sec=False, i=1):
        """ Passes trained model and acquired image to abberior_predict to 
            estimate zernike weights and offsets required to correct 
            aberrations. Calculates new SLM pattern to acquire new image and 
            calculates correlation coefficients. """
        
        size = 2 * np.asarray(self.p.general["size_slm"])
        #TODO: code this properly
        scale = 26.6*2
        
        self.zernike, self.offset = abberior.abberior_predict(self.p.general["autodl_model_path"], 
                                                           image, offset=offset, multi=multi, ii=i)
        
        off = [self.img_l.off.xgui.value() + self.offset[1]*scale,
               self.img_l.off.ygui.value() - self.offset[0]*scale]
        self.img_l.off.xgui.setValue(np.round(off[0]))
        self.img_l.off.ygui.setValue(np.round(off[1]))
            
        zern_correct = pcalc.crop(helpers.create_phase(self.zernike, 
                                                       num=np.arange(3, 14), 
                                                       size = size, 
                                                       radscale = np.sqrt(2)*self.slm_radius), 
                                  size/2, offset = off)
        
        #self.zernikes_all = self.zernikes_all - zern_correct
        #TODO: needs to be changed when using offset + zernike model
        #self.zernikes_all = self.zernikes_all - phasemask_aberrs
        #HOTFIX for offset only
        self.zernikes_all = zern_correct                    

        self.recalc_images()
        self.correct_tiptilt()
        if ortho_sec:
            self.correct_defocus()
            
        new_img = abberior.get_image(multi=multi)
        correlation = np.round(helpers.corr_coeff(new_img, multi=multi), 2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        print('correlation coeff is: {}'.format(correlation))
        
        return self.zernike, self.offset*scale, new_img, correlation


    def auto_align(self, so_far = -1, best_of = 5, multi = True, offset = True):
        """This function calls abberior from AutoAlign module, passes the resulting dictionary
        through a constructor for a param object
        so_far: correlation required to stop optimizing; -1 means it only executes once"""

        size = 2 * np.asarray(self.p.general["size_slm"])
        
        # center the image before starting
        self.correct_tiptilt()
        if multi:
            self.correct_defocus()
            
        #preds = np.zeros(11) 
        corr = 0
        i = 0
        while corr >= so_far:
            image = abberior.get_image(multi=multi)                                                   
            preds, off_pred, image, new_corr = self.corrective_loop(image, offset=offset, multi=multi, i=best_of)
            if new_corr > corr:
                so_far = corr
                corr = new_corr
                print('iteration: ', i, 'new corr: {}, old corr: {}'.format(corr, so_far))
                i = i + 1
            else:
                print('final correlation: {}'.format(corr))
                # REMOVING the last phase corrections from the SLM
                off = [self.img_l.off.xgui.value(), self.img_l.off.ygui.value()]
                zern_correct = pcalc.crop(helpers.create_phase(self.zernike, 
                                                               num=np.arange(3, 14), 
                                                               size = size, 
                                                               radscale = np.sqrt(2)*self.slm_radius),
                                          size/2, offset = off)
                self.zernikes_all = self.zernikes_all + zern_correct
                i -= 1
                break
           
        self.recalc_images()
        # not needed?
        # self.correct_tiptilt()
        # self.correct_defocus()

    def automate(self):
        multi=False
        ortho_sec = True
        offset=True
        num_its=500
        px_size = 10
        i_start = 0
        best_of = 5
        print("save path: ", self.p.general["data_path"])
        print("used model: ", self.p.general["autodl_model_path"])
        # 0. creates data structure
        d = {'gt': [], 'preds': [], 'init_corr': [],'corr': []}
        
        # for model name: drop everything from model path, drop extension
        mdl_name = self.p.general["autodl_model_path"].split("/")[-1][:-4]
        path = self.p.general["data_path"] + mdl_name
        if not os.path.isdir(self.p.general["data_path"]):
            os.mkdir(self.p.general["data_path"])
            if not os.path.isdir(path):
                os.mkdir(path)
        
        # NOTE: multi is meant to be hardcoded here, we only need the xy to return the config
        img, conf, msr, stats = abberior.get_image(multi=False, config=True)
        x_init = conf.parameters('ExpControl/scan/range/x/g_off')
        y_init = conf.parameters('ExpControl/scan/range/y/g_off')
        z_init = conf.parameters('ExpControl/scan/range/z/g_off')
        for ii in range(num_its):
            # 1. zeroes SLM
            self.reload_params(self.param_path)
            # get image from Abberior
            img, conf, msr, stats = abberior.get_image(multi=ortho_sec, config=True)
        
            #TODO: Which values are good will depend on the system & acquisition parameters. Needs to be tested
            #IMPORTANT: should also stop acquisition, potentially shut down Imspector?
            if stats[2] < 25:
                print("Interrupted because no signal. Stats: ", stats)
                break

            # 2.a fits CoM, for multi model: average from the two views
            if ortho_sec:
                _, x_shape, y_shape = np.shape(img)
                ####### xy ########
                b, a = helpers.get_CoM(img[0])
                dx_xy = ((x_shape-1)/2-a)*1e-9*px_size  # convert to m
                dy_xy = ((y_shape-1)/2-b)*1e-9*px_size  # convert to m

                ####### xz ########
                b, a = helpers.get_CoM(img[1])
                dx_xz = ((x_shape-1)/2-a)*1e-9*px_size  # convert to m
                dz_xz = ((y_shape-1)/2-b)*1e-9*px_size  # convert to m
                
                ######## yz #########
                b, a = helpers.get_CoM(img[2])
                dy_yz = ((x_shape-1)/2-a)*1e-9*px_size  # convert to m
                dz_yz = ((y_shape-1)/2-b)*1e-9*px_size  # convert to m
                
                dx = np.average([dx_xy, dx_xz])
                dy = np.average([dy_xy, dy_yz])
                dz = np.average([dz_xz, dz_yz])
                
            # 2.b fits CoM, for single model
            else:
                x_shape, y_shape = np.shape(img)
                b, a = helpers.get_CoM(img)
                dx = ((x_shape-1)/2-a)*1e-9*px_size  # convert to m
                dy = ((y_shape-1)/2-b)*1e-9*px_size
                dz = 0
                
            # if CoM is more than 200 nm from center, skip and try againg
            lim = 160e-9
            if np.abs(dx) >= lim or np.abs(dy) >= lim or np.abs(dz)>=lim:
                print('skipped', dx, dy, dz)
                # fine, using galvo
                conf.set_parameters('ExpControl/scan/range/x/g_off', x_init)
                conf.set_parameters('ExpControl/scan/range/y/g_off', y_init)
                conf.set_parameters('ExpControl/scan/range/z/g_off', z_init)
                # coarse, using stage
                # conf.set_parameters('ExpControl/scan/range/offsets/coarse/x/g_off', x_init)
                # conf.set_parameters('ExpControl/scan/range/offsets/coarse/y/g_off', x_init)
                # conf.set_parameters('ExpControl/scan/range/offsets/coarse/z/g_off', x_init)
                continue

            # 3. centers using ImSpector
            # coarse: center using the stage. keep for reference
            #xo = conf.parameters('ExpControl/scan/range/offsets/coarse/x/g_off')
            #yo = conf.parameters('ExpControl/scan/range/offsets/coarse/y/g_off')
            #zo = conf.parameters('ExpControl/scan/range/offsets/coarse/z/g_off')
            # fine: center using the galvos.
            # get positions from Imspector
            xo = conf.parameters('ExpControl/scan/range/x/g_off')
            yo = conf.parameters('ExpControl/scan/range/y/g_off')
            zo = conf.parameters('ExpControl/scan/range/z/g_off')
            
            # calculate new positions
            xPos = xo - dx
            yPos = yo - dy
            zPos = zo - dz
            
            # if overall, drift has been more then 800 um, reset.
            # TODO: test again if this works
            # if np.abs(xPos) >= 800e-6 or np.abs(yPos) >= 800e-6:
            #     print('skipped', xPos, yPos)
            #     # fine
            #     conf.set_parameters('ExpControl/scan/range/x/g_off', x_init)
            #     conf.set_parameters('ExpControl/scan/range/y/g_off', y_init)
            #     conf.set_parameters('ExpControl/scan/range/z/g_off', z_init)
            #     continue

            
            # write new position values
            conf.set_parameters('ExpControl/scan/range/x/g_off', xPos)
            conf.set_parameters('ExpControl/scan/range/y/g_off', yPos)
            conf.set_parameters('ExpControl/scan/range/z/g_off', zPos)

            
            # 4. dials in random aberrations and sends them to SLM
            #aberrs = helpers.gen_coeffs(11)
            aberrs = [0 for c in range(11)]
            #TODO: pass in obj_dia and weight as arguments
            scale = 26.6*2
            off_aberr = [np.round(scale*x) for x in helpers.gen_offset()]
            #np.round(np.asarray(helpers.gen_offset()) * scale)
            size = 2 * np.asarray(self.p.general["size_slm"])
            #off = [self.img_l.off.xgui.value(), self.img_l.off.ygui.value()]
            off = [self.img_l.off.xgui.value() - off_aberr[1],
                   self.img_l.off.ygui.value() + off_aberr[0]]
            
            self.img_l.off.xgui.setValue(off[0])
            self.img_l.off.ygui.setValue(off[1])
            #TODO: create random offset and add to values
            #TODO: retest offset + zernike model!
            #TODO: sanity check that offsets are within boundaries
            phasemask_aberrs = pcalc.crop(helpers.create_phase(aberrs, 
                                                       num=np.arange(3, 14), 
                                                       size = size, 
                                                       radscale = np.sqrt(2)*self.slm_radius), 
                                          size/2, offset = off)
            
            #TODO: needs to be changed when using offset + zernike model
            #self.zernikes_all = self.zernikes_all - phasemask_aberrs
            #HOTFIX for offset only
            self.zernikes_all = phasemask_aberrs
            self.recalc_images()
            #TODO append both aberrs and off_aberr
            d['gt'].append(aberrs)
            d['gt'].append(off_aberr)
            
            # 5. Get image, center once more using tip tilt and defocus corrections
            # save image and write correction coefficients to file
            img = abberior.get_image(multi=ortho_sec)
            self.correct_tiptilt()
            if ortho_sec:
                self.correct_defocus()
            img = abberior.get_image(multi=multi)
            name = path + '/' + str(ii+i_start) + "_aberrated.msr"
            msr.save_as(name)
            d['init_corr'].append(helpers.corr_coeff(img, multi=multi))
            
            # 6. single pass correction
            self.zernike, off_pred, _, corr = self.corrective_loop(img, offset=offset, multi=multi, i = best_of)
            
            d['preds'].append(self.zernike.tolist())
            d['preds'].append(off_pred.tolist())
            d['corr'].append(corr)
            name = path + '/' + str(ii+i_start) + "_corrected.msr"
            msr.save_as(name)
            with open(path + '/' + mdl_name +str(i_start)+'.txt', 'w') as file:
                json.dump(d, file)


            # use matplotlib to plot and save data
            if ortho_sec and multi:
                minmax = [img.min(), img.max()]
                fig = plt.figure()
                plt.subplot(231); plt.axis('off')
                plt.imshow(img[0], clim = minmax, cmap = 'inferno')
                plt.subplot(232); plt.axis('off')
                plt.imshow(img[1], clim = minmax, cmap = 'inferno')
                plt.subplot(233); plt.axis('off')
                plt.imshow(img[1], clim = minmax, cmap = 'inferno')
                img = abberior.get_image(multi = multi)
                plt.subplot(234); plt.axis('off')
                plt.imshow(img[0], clim = minmax, cmap = 'inferno')
                plt.subplot(235); plt.axis('off')
                plt.imshow(img[1], clim = minmax, cmap = 'inferno')
                plt.subplot(236); plt.axis('off')
                plt.imshow(img[2], clim = minmax, cmap = 'inferno')
                fig.savefig(path + '/' + str(ii+i_start) + "_thumbnail.png")
            elif ortho_sec and not multi:
                minmax = [img.min(), img.max()]
                fig = plt.figure()
                plt.subplot(121); plt.axis('off')
                plt.imshow(img, clim = minmax, cmap = 'inferno')
                
                img = abberior.get_image(multi = ortho_sec)
                plt.subplot(122); plt.axis('off')
                plt.imshow(img[0], clim = minmax, cmap = 'inferno')
                fig.savefig(path + '/' + str(ii+i_start) + "_thumbnail.png")
            #TODO add missing logic blocks

            # d['offset'].append(self.offset.tolist())
        print('DONE with automated loop!', '\n', 'Initial correlation: ', d['init_corr'], '\n', 'final correlation: ', d['corr'])


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


    def openFileDialog(self, path):
        """ Creates a dialog to open a file. At the moement, it is only used 
            to load the image for the flat field correction. There is no 
            sanity check implemented whether the selected file is a valid image. """
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        work_dir = os.path.dirname(os.path.realpath(__file__))
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 
                        "Load flat field correction", work_dir +'/'+ path)
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
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 
                        "Load flat field correction", work_dir +'/'+ path)
        # TODO: needs to be implemented to work for two paths, one for left, 
        # one for right side
        print("Currently not implemented. Please add paths in the files for \
              left and right side parameters as 'cal1'.")
        #if fileName:
        #    self.load_flat_field(fileName)
        #    self.combine_and_update()

    
    def load_flat_field(self, path_l, path_r, recalc = True):
        """ Opens the images in the parameter paths, combines two halves to 
            one image and sets as new flatfield correction. """
     
        s = np.asarray(self.p.general["size_slm"])    
        lhalf = pcalc.crop(np.asarray(pcalc.load_image(path_l))/255, 
                           s, [ s[1] // 2, s[0] // 2])
        rhalf = pcalc.crop(np.asarray(pcalc.load_image(path_r))/255, 
                           s, [-(s[1] // 2), s[0] // 2])
        
        # check whethere double pass is activated and cross correction as on 
        # Abberior should be applied: det offsets to [0,0] for not activated        
        if self.p.general["double_pass"]:            
            ff_l_patched = np.zeros([2 * s[0], 2 * s[1]])
            ff_l_patched[s[0] // 2 : 3 * s[0] // 2, s[1] // 2 : 3 * s[1] // 2] = lhalf
            ff_r_patched = np.zeros([2 * s[0], 2 * s[1]])
            ff_r_patched[s[0] // 2 : 3 * s[0] // 2, s[1] // 2 : 3 * s[1] // 2] = rhalf
            off = self.img_r.offset - self.img_l.offset
            lhalf = lhalf + pcalc.crop(ff_r_patched, s, -off)
            rhalf = rhalf + pcalc.crop(ff_l_patched, s,  off)

        self.flatfieldcor = [lhalf, rhalf]
        self.flat_field(self.p.general["flat_field"], recalc)


    def flat_field(self, state, recalc = True):
        """ Opens the image in the parameter path and sets as new flatfield
            correction. """
        self.p.general["flat_field"] = int(state)
        if state:
            self.flatfield = self.flatfieldcor
        else:
            self.flatfield = [np.zeros_like(self.flatfieldcor[0]), 
                              np.zeros_like(self.flatfieldcor[1])]
        if recalc:
            self.combine_and_update()


    def split_image(self, state = True):
        """ Action called when the "Split image" checkbox is selected. Toggles
            between split image operation and single image operation."""
        self.p.general["split_image"] = int(state)
        
        if state:
            #self.dbl_pass_state.setChecked(False)
            print(state, "Currently not implemented. Please restart code and \
                          set split image flag in parameters file.")
        else:
            self.img_full = None
            print(state, "Currently not implemented. Please restart code and \
                          set split image flag in parameters file.")
        self.init_data()
        self.init_images()
        
        
    def single_correction(self, state):
        """ Action called when the "Single Correction" checkbox is selected.
            Toggles between identical correction for both halves of the sensor
            and using individidual corrections. When single correction is 
            active, the values from the left sensor half are used. """
        self.p.general["single_aberr"] = int(state)
        if state:
            self.img_r.aberr.astig.xgui.setValue(self.img_l.aberr.astig.xgui.value())
            self.img_r.aberr.astig.ygui.setValue(self.img_l.aberr.astig.ygui.value())
            self.img_r.aberr.coma.xgui.setValue(self.img_l.aberr.coma.xgui.value())
            self.img_r.aberr.coma.ygui.setValue(self.img_l.aberr.coma.ygui.value())
            self.img_r.aberr.sphere.xgui.setValue(self.img_l.aberr.sphere.xgui.value())
            self.img_r.aberr.sphere.ygui.setValue(self.img_l.aberr.sphere.ygui.value())
            self.img_r.aberr.trefoil.xgui.setValue(self.img_l.aberr.trefoil.xgui.value())
            self.img_r.aberr.trefoil.ygui.setValue(self.img_l.aberr.trefoil.ygui.value())
            
    def double_pass(self, state):
        """ Activates the double pass geometry cross correction as on Abberior.
            The laser beam hits the SLM twice, once it's modulated, once not 
            (because it has the wrong polarization). To still correct for SLM
            curvature during the unmodulated reflection, the flatfield pattern
            from the first impact needs to be shifted by the offset and added 
            to the flatfield correction of the second impact. """
        self.p.general["double_pass"] = int(state)
        if self.p.general["flat_field"]:#flt_fld_state.checkState():
            print("calling flatfield")
            if self.p.general["split_image"]:
                self.load_flat_field(self.p.left["cal1"], self.p.right["cal1"])
            else:
                self.load_flat_field(self.p.full["cal1"], self.p.full["cal1"])
        
    
    def calc_slmradius(self, backaperture, mag):
        """ Calculates correct scaling factor for SLM based on objective
            backaperture, optical magnification of the beampath, SLM pixel
            size and size of the SLM. Required values are directly taken from
            the parameters files. """
            
        rad = pcalc.normalize_radius(backaperture, mag, 
                    self.p.general["slm_px"], self.p.general["size_slm"])
        return rad
    
    
    def radius_changed(self):
        """ Radius of pattern on SLM can be hardcoded instead of calculating
            from the objectives backaperture and optical magnification. """
        #self.current_objective = self.p.objectives[self.obj_sel.currentText()]
        self.radius_input = self.rad_but.value()
        print("radius changed")
        self.slm_radius = self.calc_slmradius(self.radius_input, 1)
        self.init_zernikes()
        #pcalc.normalize_radius(self.radius_input, 1, 
        #                self.p.general["slm_px"], self.p.general["size_slm"])
        self.recalc_images()

            
    def objective_changed(self):
        """ Action called when the users selects a different objective. 
            Calculates the diameter of the BFP; then recalculates the the
            patterns based on the selected objective. """
            
        self.current_objective = self.obj_sel.currentText()#["name"]
        self.reload_params(self.param_path)
        self.slm_radius = self.calc_slmradius(
            self.p.objectives[self.current_objective]["backaperture"],
            self.p.general["slm_mag"])
        self.init_zernikes()
        self.recalc_images()
        
    def apply_correction(self):
        print("applying corrections")
    
        
    def recalc_images(self):
        """ Function to recalculate the left and right images completely. 
            Update is set to false to prevent redrawing after every step of 
            the recalculation. Image display is only updated once at the end. """
        if self.p.general["split_image"]:
            self.img_l.update(update = False, completely = True)
            self.img_r.update(update = False, completely = True)
        else:
            self.img_full.update(update = False, completely = True)
        self.combine_and_update()

        
    def combine_and_update(self):
        """ Stitches the images for left and right side of the SLM, adds the 
            flatfield correction and phasewraps everything. Updates
            self.img_data with the new image data.
            Updates the displayed images in the control window and (if active)
            on the SLM. To do so, rescales the image date to the SLM's required
            pitch (depends on the wavelength, and is set in the general 
            parameters). Saves the image patterns/latest.bmp and then reloads
            into the Pixmap for display. """
        
        if self.p.general["split_image"]:
            l = pcalc.phase_wrap(pcalc.add_images([self.img_l.data, 
                                                   self.flatfield[0],
                                                   self.zernikes_all, 
                                                   self.phase_tiptilt,
                                                   self.phase_defocus]), 
                                 self.p.left["phasewrap"])
            r = pcalc.phase_wrap(pcalc.add_images([self.img_r.data,
                                                   self.flatfield[1],
                                                   self.zernikes_all, 
                                                   self.phase_tiptilt,
                                                   self.phase_defocus]), 
                                 self.p.right["phasewrap"])
            # this is hack:
            # for preview display, do not scale with SLM range
            # for SLM display, do scale ... scales may differ left / right
            self.img_data = pcalc.stitch_images(l * self.p.left["slm_range"],
                                                r * self.p.right["slm_range"])
            
            
            self.plt_frame.plot(pcalc.stitch_images(l, r))
                                
        else:            
            self.img_data = pcalc.phase_wrap(pcalc.add_images([self.img_full.data, 
                                                               pcalc.stitch_images(self.flatfield[0], self.flatfield[1]), 
                                                               self.zernikes_all, 
                                                               self.phase_tiptilt,
                                                               self.phase_defocus]), 
                                             self.p.general["phasewrap"])
            self.plt_frame.plot(self.img_data)
            self.img_data = self.img_data * self.p.general["slm_range"]
            
        if self.slm != None:
            self.slm.update_image(np.uint8(self.img_data))

 
    
    def open_SLMDisplay(self):
        """ Opens a widget fullscreen on the secondary screen that displays
            the latest image. """
        self.slm = SLM.SLM_Display(np.uint8(self.img_data), self.p.general["display_mode"])

        
    def close_SLMDisplay(self):
        """ Closes the SLM window (if it exists) and resets reference to None 
            to prevent errors when trying to close SLM again. """
        if self.slm != None:
            self.slm._quit()
            self.slm = None


    def _quit(self):
        """ Action called when the Quit button is pressed. Closes the SLM 
            control window and exits the main window. """
        pcalc.save_image(self.img_data, self.p.general["path"], 
                         self.p.general["last_img_nm"])
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
