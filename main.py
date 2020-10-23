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


# TODO: add drop down window to select model w/ which to autoalign
# NOTE: hardcoded for now:
# MODEL_STORE_PATH="autoalign/models/08.01.20_corrected_pattern_calc_w_val_200_epochs_Adam_lr_0.001_batchsize_64_custom_loss.pth"
# MODEL_STORE_PATH="autoalign/models/4.27.20_3d_offset_sted_20k_eps_15_lr_0.001_bs_64.pth"
# MODEL_STORE_PATH="autoalign/models/20.02.12_xsection_20k_15_epochs_Adam_lr_0.001_batchsize_64.pth"
# MODEL_STORE_PATH="autoalign/models/20.05.04_larger_weight_range_20k_eps_15_lr_0.001_bs_64.pth"
# MODEL_STORE_PATH="autoalign/models/20.05.04_noise_20k_local_eps_15_lr_0.001_bs_64.pth"
# MODEL_STORE_PATH="autoalign/models/20.05.18_scaling_fix_eps_15_lr_0.001_bs_64_2.pth"
# MODEL_STORE_PATH="autoalign/models/20.05.27_shift_invariant_15k_eps_15_lr_0.001_bs_64_2.pth"
# MODEL_STORE_PATH="autoalign/models/20.16.06_1D_20k_eps_15_lr_0.001_bs_64.pth"
# MODEL_STORE_PATH="autoalign/models/20.16.06_1D_20k_eps_15_lr_0.001_bs_64_standardized_not_norm.pth"
# MODEL_STORE_PATH="autoalign/models/20.05.18_scaling_fix_eps_15_lr_0.001_bs_64_standardized.pth"
# MODEL_STORE_PATH="autoalign/models/20.06.22_no_defocus_multi_20k_eps_15_lr_0.001_bs_64.pth"
# MODEL_STORE_PATH="autoalign/models/20.06.22_no_defocus_multi_20k_eps_15_lr_0.001_bs_64_precentered.pth"
# MODEL_STORE_PATH="autoalign/models/20.07.12_no_defocus_1D_centered_20k_eps_15_lr_0.001_bs_64.pth"
# MODEL_STORE_PATH="autoalign/models/20.06.22_no_defocus_multi_20k_eps_5_lr_0.001_bs_64_concat.pth"
# to try on 20.07.23: two noise models and an offset model
# MODEL_STORE_PATH="autoalign/models/20.07.12_no_defocus_1D_centered_20k_eps_15_lr_0.001_bs_64_noise_poiss1000.pth"
# MODEL_STORE_PATH="autoalign/models/20.07.12_no_defocus_1D_centered_20k_eps_15_lr_0.001_bs_64_noise.pth"
# MODEL_STORE_PATH='autoalign/models/20.07.12_no_defocus_1D_centered_20k_eps_15_lr_0.001_bs_64_noise_bg2poiss350.pth'
# MODEL_STORE_PATH="autoalign/models/20.07.22_1D_offset_15k_eps_15_lr_0.001_bs_64_offset.pth"
# MODEL_STORE_PATH="autoalign/models/20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_64.pth"
# MODEL_STORE_PATH="autoalign/models/20.07.23_1D_offset_only_2k_eps_15_lr_0.001_bs_64_noise_bg2poiss500.pth"
MODEL_STORE_PATH="autoalign/models/20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64.pth"
# MODEL_STORE_PATH="autoalign/models/20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64_noise_bg2poiss500.pth"
# MODEL_STORE_PATH="autoalign/models/20.07.26_1D_centered_offset_18k_eps_15_lr_0.001_bs_64_noise_bg2poiss350.pth"
# MODEL_STORE_PATH="autoalign/models/20.08.03_1D_centered_18k_norm_dist_eps_15_lr_0.001_bs_64.pth"
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
            self.img_full.update_guivalues(self.p, self.p.left)
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
            
        else:
            self.img_full = PI.Half_Pattern(self.p, self.img_size)
            self.img_full.call_daddy(self)
            self.img_full.set_name("full")
            self.zernikes_all = np.zeros_like(self.img_full.data)
            self.phase_tiptilt = np.zeros_like(self.img_full.data)
            self.phase_defocus = np.zeros_like(self.img_full.data)

        
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

        #size = 2 * np.asarray(self.p.general["size_slm"])
        #size = 2 * self.img_size
        
        self.rtiptilt = 2 * pcalc.normalize_radius(1, 1, self.p.general["slm_px"], 
                                              self.img_size)#self.p.general["size_slm"])
        
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
        # NOTE: I WROTE THIS
        self.crea_but(hbox, self.auto_align, "Auto Align")

        vbox.addLayout(hbox)
        # NOTE: I WROTE THIS
        self.crea_but(hbox, self.correct_tiptilt, "Tip/Tilt")
        #self.crea_but(hbox, self.correct_defocus, "Defocus")
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

        # checkboxes for the different modes of operation: split image (currently
        # full display operation is not supported), flatfield correction
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

        # scale_img = QtWidgets.QVBoxLayout()
        
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
            #TODO: change this left; requires change for all objectives
            c.addLayout(self.img_full.create_gui(self.p, self.p.left), 0, 1, 1, 2)
            
        c.setAlignment(QtCore.Qt.AlignRight)
        c.setContentsMargins(0,0,0,0)
        
        # add all the widgets
        vbox.addLayout(imgbox)
        vbox.addLayout(c)
        vbox.setContentsMargins(0,0,0,0)
        self.main_frame.setLayout(vbox)       
        self.setCentralWidget(self.main_frame)
        
        
    # NOTE: I wrote these fns
    def correct_defocus(self):
        self.defocus = abberior.correct_defocus()#(const=1/6.59371319)
        # self.phase_defocus = self.phase_defocus + pcalc.crop((1.)*helpers.create_phase_defocus(self.defocus, res1=1200, res2=792, radscale = self.slm_radius), [600, 396], offset = [self.img_l.off.xgui.value(), self.img_l.off.ygui.value()])
        self.phase_defocus = self.phase_defocus + pcalc.crop((1.)*helpers.create_phase(coeffs=[self.defocus], num=[2], res1=1200, res2=792, radscale = self.slm_radius), [600, 396], offset = [self.img_l.off.xgui.value(), self.img_l.off.ygui.value()])
        self.recalc_images()


    def correct_tiptilt(self):
        self.tiptilt = abberior.correct_tip_tilt()
        # self.phase_tiptilt = self.phase_tiptilt + (1.)*helpers.create_phase_tip_tilt(self.tiptilt, res1=600, res2=396, radscale = 2*self.rtiptilt)
        # self.phase_tiptilt = self.phase_tiptilt + (1.)*helpers.create_phase(coeffs=self.tiptilt, num=[0,1], res1=600, res2=396, radscale = 2*self.rtiptilt)
        self.phase_tiptilt = self.phase_tiptilt + (1.)*helpers.create_phase(coeffs=self.tiptilt, num=[0,1], res1=600, res2=396, radscale = 2*self.rtiptilt)
        self.recalc_images()
    
    
    def corrective_loop(self, model_store_path=MODEL_STORE_PATH, image=None, offset=False, multi=False,  i=0):
        size = 2 * np.asarray(self.p.general["size_slm"])
        #TODO: code this properly
        scale = 26.6*2
        
        self.zernike, self.offset = abberior.abberior_test(MODEL_STORE_PATH, image, offset=offset, multi=multi, i=i)
        
        # TODO: why is there a factor of sqrt(2) in the radscale?! added June 19th
        # chaning the scale factor did not improve it
        self.img_l.off.xgui.setValue(self.img_l.off.xgui.value()+self.offset[1]*scale)
        self.img_l.off.ygui.setValue(self.img_l.off.ygui.value()-self.offset[0]*scale)
        # print(self.offset, self.img_l.off.xgui.value(), self.img_l.off.ygui.value())

        self.zernikes_all = self.zernikes_all + pcalc.crop((-1.)*helpers.create_phase(self.zernike, num=np.arange(3, 14), res1=size[0], res2=size[1], 
                radscale = np.sqrt(2)*self.slm_radius), size/2, offset = [self.img_l.off.xgui.value(), self.img_l.off.ygui.value()])
        # self.zernikes_all = self.zernikes_all + pcalc.crop((-1.)*helpers.create_phase(self.zernike, num=np.arange(3, 14), res1=size[0], res2=size[1], 
        #         radscale = np.sqrt(2)*self.slm_radius), size/2, offset = [self.img_l.off.xgui.value(), self.img_l.off.ygui.value()])
        self.recalc_images()
    
        self.correct_tiptilt()
        if multi:
            self.correct_defocus()
        new_img = abberior.get_image(multi=multi)
        correlation = np.round(helpers.corr_coeff(new_img, multi=multi), 2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        print('correlation coeff is: {}'.format(correlation))
        
        return self.zernike, new_img, correlation


    def auto_align(self, model_store_path=MODEL_STORE_PATH):
        """This function calls abberior from AutoAlign module, passes the resulting dictionary
        through a constructor for a param object"""
        # self.correct_tiptilt()
        # image = abberior.get_image()
        # _, new_img, corr = self.corrective_loop(MODEL_STORE_PATH, image)
        # ITERATIVE LOOP #
        multi=False
        offset=False
        size = 2 * np.asarray(self.p.general["size_slm"])
        
        self.correct_tiptilt()
        if multi:
            self.correct_defocus()
        so_far = -1
        corr = 0                                                             
        preds = np.zeros(11)
        i = 1
        while corr >= so_far:
            image = abberior.get_image(multi=multi)                                                   
            preds, image, new_corr = self.corrective_loop(MODEL_STORE_PATH, image, offset=offset, multi=multi, i=i)
            if new_corr > corr:
                so_far = corr
                corr = new_corr
                print('iteration: ', i, 'new corr: {}, old corr: {}'.format(corr, so_far))
                i = i + 1
            else:
                print('final correlation: {}'.format(corr))
                # REMOVING the last phase corrections from the SLM
                # TODO: why is there a factor of sqrt(2) in the radscale?! added June 19th
                # chaning the scale factor did not improve it
                self.zernikes_all = self.zernikes_all - pcalc.crop((-1.)*helpers.create_phase(self.zernike, num=np.arange(3, 14), res1=size[0], res2=size[1], 
                    radscale = np.sqrt(2)*self.slm_radius), size/2, offset = [self.img_l.off.xgui.value(), self.img_l.off.ygui.value()])
                i -= 1
                break
            # if i >= 0:
            #     break
           
        self.recalc_images()
        # not needed?
        # self.correct_tiptilt()
        # self.correct_defocus()


    def automate(self, model_store_path=MODEL_STORE_PATH, num_its=3):
        multi=True
        offset=False
        px_size = 10
        i_start = 0
        # 0. creates data structure
        d = {'gt': [], 'preds': [], 'init_corr': [],'corr': []} 
        path = 'D:/Data/20200804_Wiebke_Hope_Autoalign/test_centering/'
        # NOTE: multi is meant to be hardcoded here, we only need the xy to return the config
        img, conf, msr, stats = abberior.get_image(multi=False, config=True)
        x_init = conf.parameters('ExpControl/scan/range/x/g_off')
        y_init = conf.parameters('ExpControl/scan/range/y/g_off')
        z_init = conf.parameters('ExpControl/scan/range/z/g_off')

        for ii in range(num_its):
            # 1. zeroes SLM
            self.reload_params(self.param_path)
            # get image from Abberior
            img, conf, msr, stats = abberior.get_image(multi=multi, config=True)

            #TODO: Which values are good will depend on the system & acquisition parameters. Needs to be tested
            #IMPORTANT: should also stop acquisition, potentially shut down Imspector?
            if stats[2] < 25:
                print("Interrupted because no signal. Stats: ", stats)
                break

            # 2. fits CoM
            if multi:
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

                print(dx_xy*1e9, dx_xz*1e9, dy_xy*1e9, dy_yz*1e9, dz_xz*1e9, dz_yz*1e9)
                
                dx = np.average([dx_xy, dx_xz])
                dy = np.average([dy_xy, dy_yz])
                dz = np.average([dz_xz, dz_yz])


                #d_obj = D/3/1000 # scaling
                # print(dz, f, D, lambd)
            else:
                x_shape, y_shape = np.shape(img)
                b, a = helpers.get_CoM(img)
                dx = ((x_shape-1)/2-a)*1e-9*px_size  # convert to m
                dy = ((y_shape-1)/2-b)*1e-9*px_size
                dz = 0
            print("delta positions: ", dx*1e9, dy*1e9, dz*1e9)
            lim = 160e-9
            if np.abs(dx) >= lim or np.abs(dy) >= lim or np.abs(dz)>=lim:
                print('skipped', dx, dy, dz)
                conf.set_parameters('ExpControl/scan/range/x/g_off', x_init)
                conf.set_parameters('ExpControl/scan/range/y/g_off', y_init)
                conf.set_parameters('ExpControl/scan/range/z/g_off', z_init)
                continue

            # 3. centers using ImSpector
            #coarse
            #xo = conf.parameters('ExpControl/scan/range/offsets/coarse/x/g_off')
            #yo = conf.parameters('ExpControl/scan/range/offsets/coarse/y/g_off')
            #zo = conf.parameters('ExpControl/scan/range/offsets/coarse/z/g_off')
            #fine
            xo = conf.parameters('ExpControl/scan/range/x/g_off')
            yo = conf.parameters('ExpControl/scan/range/y/g_off')
            zo = conf.parameters('ExpControl/scan/range/z/g_off')
            print("positions: ", xo, yo, zo)
            
            xPos = xo - dx
            yPos = yo - dy
            zPos = zo - dz
            print("positions: ", xPos, yPos, zPos)

            # if np.abs(xPos) >= 800e-6 or np.abs(yPos) >= 800e-6:
            #     print('skipped', xPos, yPos)
            #     conf.set_parameters('ExpControl/scan/range/x/g_off', x_init)
            #     conf.set_parameters('ExpControl/scan/range/y/g_off', y_init)
            #     conf.set_parameters('ExpControl/scan/range/z/g_off', z_init)
            #     continue
            #coarse
            #conf.set_parameters('ExpControl/scan/range/offsets/coarse/x/g_off', xPos)
            #conf.set_parameters('ExpControl/scan/range/offsets/coarse/y/g_off', yPos)
            #fine
            #TODO:  as always, not entirely sure where the random factor of two comes from
            conf.set_parameters('ExpControl/scan/range/x/g_off', xPos)
            conf.set_parameters('ExpControl/scan/range/y/g_off', yPos)
            conf.set_parameters('ExpControl/scan/range/z/g_off', zPos)
            #conf.set_parameters('ExpControl/scan/range/offsets/coarse/z/g_off', zPos)
            
            # 4. dials in random aberrations and sends them to SLM
            aberrs = helpers.gen_coeffs(11)

            size = 2 * np.asarray(self.p.general["size_slm"])
            self.zernikes_all = self.zernikes_all - pcalc.crop(helpers.create_phase(aberrs, num=np.arange(3, 14), res1=size[0], res2=size[1], 
                        radscale = np.sqrt(2)*self.slm_radius), size/2, offset = [self.img_l.off.xgui.value(), self.img_l.off.ygui.value()])
            
            d['gt'].append(aberrs)
            
            # 5. Center using tip tilt and defocus correction
            self.correct_tiptilt()
            if multi:
                self.correct_defocus()
            img = abberior.get_image(multi=multi)
            #TODO get location of python script instead of hardcoding path
            name = path + str(ii+i_start) + "_aberrated.msr"
            msr.save_as(name)

            d['init_corr'].append(helpers.corr_coeff(img, multi=multi))
            # 6. single pass
            self.zernike, _, corr = self.corrective_loop(model_store_path=model_store_path, offset=offset, multi=multi, image=img)
            d['preds'].append(self.zernike.tolist())
            d['corr'].append(corr)
            name = path + str(ii+i_start) + "_corrected.msr"
            msr.save_as(name)
            with open(path +'test_centering_' +str(i_start)+'.txt', 'w') as file:
                json.dump(d, file)
            # d['offset'].append(self.offset.tolist())
        print('DONE with automated loop!', '\n', 'Initial correlation: ', d['init_corr'], '\n', 'final correlation: ', d['corr'])

        # NOTE: need to know from the model itself which model to use, maybe some kind of json like for
        # the obejctives, but for now, can change manually 
        # size = 2 * np.asarray(self.p.general["size_slm"])
        #######################################
        # self.zernike = abberior.abberior_multi(MODEL_STORE_PATH, image)
        # self.zernikes_all = self.zernikes_all + pcalc.crop((-1.)*helpers.create_phase(self.zernike, num=np.arange(3, 14), res1=size[0], res2=size[1], 
        #         radscale = np.sqrt(2)*self.slm_radius), size/2, offset = [self.img_l.off.xgui.value(), self.img_l.off.ygui.value()])
        # self.recalc_images()
        #######################################

    # NOTE: end of my fns


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
            print(state, "TODO splitimage")
        else:
            self.img_full = None
            print(state, "TODO nosplitimage")
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
                self.load_flat_field(self.p.general["cal1"], self.p.general["cal1"])
        
    
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
