#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Parameters.py


"""
    Created on Tue Oct 16 11:13:09 2018
    @author: wjahr
    
    Class to handle creation of text files to save and load parameters. Can be 
    used standalone: when executed as main, all parameter_xyz.txt files are 
    recreated according to the values hardcoded in this file.
    
    
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


import json
import os

class param():
    """ Class containing the dictionary for the parameters to be saved.
        Implements a reader/writer to import/export to json. Before exporting,
        the parameters for left/right images and aberrations can be updated
        according to the parameters provided in the user input. """


    def __init__(self):
        self.init_defaults()
        
    
    def init_defaults(self):
        self.general = {
                    "scope_mode"   : "simulated", #simulated, abberior, homebuilt
                    "display_mode" : "external", # "imspector", "othersuite"
                    "display_num"  : 1,
                    "data_source"  : "imspector", # "path", "othersuite"
                    "size_slm_mm"  : [7.5, 10],
                    "slm_px"       : 0.0125,
                    "size_full"    : [600, 792],
                    "size_slm"     : [600, 396],
                    "displaywidth" : 300,
                    "slm_mag"      : 3,
                    "laser_radius" : 1.2,
                    "objective"    : "100xOil_NA_140",
                    "path"         : "patterns/",
                    "cal0"         : "patterns/Black_Full.bmp",
                    "last_img_nm"  : "latest.bmp",
                    "modes"        : ["Gauss", "2D STED", "3D STED", 
                                      "Segments", "Bivortex", "Code Input",
                                      "From File"],
                    "split_image"  : 1,
                    "flat_field"   : 0,
                    "single_aberr" : 0,
                    "double_pass"  : 0,
                    "slm_range"    : 255,
                    "phasewrap"    : 1,
                    "cal1"         : "patterns/CAL_LSH0801768_780nm.bmp",
                    "autodl_model_path" : "autoalign/models/20.10.22_3D_centered_18k_norm_dist_offset_no_noise_eps_15_lr_0.001_bs_64.pth",
                    "data_path"    : "D:/Data/20201110_Autoalign/"
                    }
       
        self.left = {
                    "sl"        : [0,0],
                    "off"       : [0,0],
                    "rot"       : 0,
                    "radius"    : 1.0,
                    "phase"     : 1,
                    "steps"     : 1,
                    "defoc"     : 0,
                    "mode"      : "2D STED",
                    "astig"     : [0,0],
                    "coma"      : [0,0],
                    "sphere"    : [0,0],
                    "trefoil"   : [0,0],
                    "slm_range" : 255,
                    "phasewrap" : 1,
                    "cal1"      : "patterns/CAL_LSH0801768_780nm.bmp",
                    }

        self.right = {
                    "sl"        : [0,0],
                    "off"       : [0,0],
                    "rot"       : 0,
                    "radius"    : 0.64,
                    "phase"     : 0.5,
                    "steps"     : 1,
                    "defoc"     : 0,
                    "mode"      : "3D STED",
                    "astig"     : [0,0],
                    "coma"      : [0,0],
                    "sphere"    : [0,0],
                    "trefoil"   : [0,0],
                    "slm_range" : 255,
                    "phasewrap" : 1,
                    "cal1"      : "patterns/CAL_LSH0801768_780nm.bmp",
                    }
        
        self.full = {
                    "sl"        : [0,0],
                    "off"       : [0,0],
                    "rot"       : 0,
                    "radius"    : 0.64,
                    "phase"     : 0.5,
                    "steps"     : 1,
                    "defoc"     : 0,
                    "mode"      : "3D STED",
                    "astig"     : [0,0],
                    "coma"      : [0,0],
                    "sphere"    : [0,0],
                    "trefoil"   : [0,0],
                    "slm_range" : 255,
                    "phasewrap" : 1,
                    "cal1"      : "patterns/CAL_LSH0801768_780nm.bmp",
                    }
        
        self.objectives = {
                        "100xOil_NA_140": {
                                            "name": "100xOil_NA_140",
                                            "mag": 100,
                                            "NA": 1.4,
                                            "f_tl": 200,
                                            "immersion": "Oil",
                                            "backaperture": 5.04
                                            },
                        "100xSil_NA_135": {
                                            "name": "100xSil_NA_135",
                                            "mag": 100,
                                            "NA": 1.35,
                                            "f_tl": 200,
                                            "immersion": "Sil",
                                            "backaperture": 4.86
                                            },
                        "60xWat_NA_120": {
                                            "name": "60xWat_NA_120",
                                            "mag": 60,
                                            "NA": 1.2,
                                            "f_tl": 200,
                                            "immersion": "Wat",
                                            "backaperture": 7.2
                                            },
                        "20xAir_NA_070": {
                                            "name": "20xAir_NA_070",
                                            "mag": 20,
                                            "NA": 0.7,
                                            "f_tl": 200,
                                            "immersion": "Air",
                                            "backaperture": 12.6
                                            },
                        }
        
        self.simulation = {"optical_params_sted": {
                                                    "n": 1.518, 
                                                    "NA": 1.4, 
                                                    "f": 1.8, 
                                                    "transmittance": 0.74, 
                                                    "lambda": 775, 
                                                    "P_laser": 0.25, 
                                                    "rep_rate": 40000000.0, 
                                                    "pulse_length": 7e-10, 
                                                    "obj_ba": 5.04,
                                                    "px_size": 10, 
                                                    "offset": [0, 0]
                                                    }, 
                            "optical_params_gauss": {
                                                    "n": 1.518, 
                                                    "NA": 1.4, 
                                                    "f": 1.8, 
                                                    "transmittance": 0.84, 
                                                    "lambda": 640, 
                                                    "P_laser": 0.000125, 
                                                    "rep_rate": 40000000.0, 
                                                    "pulse_length": 1e-10, 
                                                    "obj_ba": 5.04, 
                                                    "px_size": 10,
                                                    "offset": [0, 0]
                                                    }, 
                            "numerical_params" :    {
                                                    "out_scrn_size" : 1,
                                                    "z_extent" : 1,
                                                    "out_res" : 64, 
                                                    "inp_res" : 64,
                                                    "orders" : [[1,-1],[1,1],[2,0],
                                                                [2,-2],[2,2],
                                                                [3,-3],[3,-1],[3,1],[3,3], 
                                                                [4,-4],[4,-2],[4,0],[4,2],[4,4]]
                                                    }
                            }
        
    
    def update(self, daddy):
        """ Parameter p: self of the calling function. Needed to have access to
            parameters provided via the GUI. Updates the values in the 
            dictionary with the values from the GUI. """
        
        self.general = {
                    "scope_mode"   : daddy.p.general["scope_mode"],
                    "display_mode" : daddy.p.general["display_mode"],
                    "data_source"  : daddy.p.general["data_source"],
                    "display_num"  : daddy.p.general["display_num"],
                    "size_slm_mm"  : daddy.p.general["size_slm_mm"],
                    "slm_px"       : daddy.p.general["slm_px"],
                    "size_full"    : daddy.p.general["size_full"],
                    "size_slm"     : daddy.p.general["size_slm"],
                    "displaywidth" : daddy.p.general["displaywidth"],
                    "slm_mag"      : daddy.p.general["slm_mag"], 
                    "laser_radius" : daddy.p.general["laser_radius"],
                    "objective"    : daddy.current_objective,
                    "path"         : daddy.p.general["path"],
                    "cal0"         : daddy.p.general["cal0"],
                    "last_img_nm"  : daddy.p.general["last_img_nm"],
                    "modes"        : daddy.p.general["modes"],
                    "split_image"  : daddy.p.general["split_image"],#splt_img,
                    "flat_field"   : daddy.p.general["flat_field"],#flt_fld,
                    "single_aberr" : daddy.p.general["single_aberr"],#sngl_corr,
                    "double_pass"  : daddy.p.general["double_pass"],#dbl_ps,
                    "slm_range"    : daddy.p.general["slm_range"],
                    "phasewrap"    : daddy.p.general["phasewrap"],
                    "cal1"         : daddy.p.general["cal1"],
                    "autodl_model_path" : daddy.p.general["autodl_model_path"],
                    "data_path"    : daddy.p.general["data_path"]
                    }

        
        if daddy.p.general["split_image"]:
            d = daddy.img_l
            self.left = {
                        "sl"        : [d.gr.xgui.value(), d.gr.ygui.value()],
                        "off"       : [d.off.xgui.value(), d.off.ygui.value()],
                        "rot"       : d.vort.rotgui.value(),
                        "radius"    : d.vort.radgui.value(),
                        "phase"     : d.vort.phasegui.value(),
                        "steps"     : d.vort.stepgui.value(),
                        "defoc"     : d.defoc.defocgui.value(),
                        "mode"      : d.vort.modegui.currentText(),
                        "astig"     : [d.aberr.astig.xgui.value(), 
                                       d.aberr.astig.ygui.value()],
                        "coma"      : [d.aberr.coma.xgui.value(), 
                                       d.aberr.coma.ygui.value()],
                        "sphere"    : [d.aberr.sphere.xgui.value(), 
                                       d.aberr.sphere.ygui.value()],
                        "trefoil"   : [d.aberr.trefoil.xgui.value(), 
                                       d.aberr.trefoil.ygui.value()],
                        "slm_range" : daddy.p.left["slm_range"],
                        "phasewrap" : daddy.p.left["phasewrap"],
                        "cal1"      : daddy.p.left["cal1"],
                        }
            d = daddy.img_r
            self.right = {
                        "sl"        : [d.gr.xgui.value(), d.gr.ygui.value()],
                        "off"       : [d.off.xgui.value(), d.off.ygui.value()],
                        "rot"       : d.vort.rotgui.value(),
                        "radius"    : d.vort.radgui.value(),
                        "phase"     : d.vort.phasegui.value(),
                        "steps"     : d.vort.stepgui.value(),
                        "defoc"     : d.defoc.defocgui.value(),
                        "mode"      : d.vort.modegui.currentText(),
                        "astig"     : [d.aberr.astig.xgui.value(), 
                                       d.aberr.astig.ygui.value()],
                        "coma"      : [d.aberr.coma.xgui.value(), 
                                       d.aberr.coma.ygui.value()],
                        "sphere"    : [d.aberr.sphere.xgui.value(), 
                                       d.aberr.sphere.ygui.value()],
                        "trefoil"   : [d.aberr.trefoil.xgui.value(), 
                                     d.aberr.trefoil.ygui.value()],
                        "slm_range" : daddy.p.right["slm_range"],
                        "phasewrap" : daddy.p.right["phasewrap"],
                        "cal1"      : daddy.p.right["cal1"],
                        }
        else:
            d = daddy.img_full
            self.full = {
                        "sl"        : [d.gr.xgui.value(), d.gr.ygui.value()],
                        "off"       : [d.off.xgui.value(), d.off.ygui.value()],
                        "rot"       : d.vort.rotgui.value(),
                        "radius"    : d.vort.radgui.value(),
                        "phase"     : d.vort.phasegui.value(),
                        "steps"     : d.vort.stepgui.value(),
                        "defoc"     : d.defoc.defocgui.value(),
                        "mode"      : d.vort.modegui.currentText(),
                        "astig"     : [d.aberr.astig.xgui.value(), 
                                       d.aberr.astig.ygui.value()],
                        "coma"      : [d.aberr.coma.xgui.value(), 
                                       d.aberr.coma.ygui.value()],
                        "sphere"    : [d.aberr.sphere.xgui.value(), 
                                       d.aberr.sphere.ygui.value()],
                        "trefoil"   : [d.aberr.trefoil.xgui.value(), 
                                     d.aberr.trefoil.ygui.value()],
                        "slm_range" : daddy.p.general["slm_range"],
                        "phasewrap" : daddy.p.general["phasewrap"],
                        "cal1"      : daddy.p.general["cal1"],
                        }


    def write_file(self, path, obj_path, name_base):
        print("Writing parameters to: ", path + obj_path + '_' + name_base)

        if not os.path.exists(os.path.dirname(path + obj_path)):
            try:
                os.makedirs(os.path.dirname(path + obj_path))
            except:
                print("cannot create directory")
        
        with open(path + name_base + "_general.txt", 'w') as f:
            json.dump(self.general, f , indent = 4)        
        with open(path + name_base + "_objectives.txt", 'w') as f:
            json.dump(self.objectives, f, indent =4)
        with open(path + obj_path + '_' + name_base + "_left.txt", 'w') as f:    
            json.dump(self.left, f, indent = 4)
        with open(path + obj_path + '_' + name_base + "_right.txt", 'w') as f:  
            json.dump(self.right, f, indent = 4)
        with open(path + obj_path + '_' + name_base + "_full.txt", 'w') as f:
            json.dump(self.full, f, indent =4)
                
  
    def load_file_general(self, path, name_base):
        with open(path + name_base + "_general.txt", 'r') as f:
            self.general = json.load(f)        
        with open(path + name_base + "_objectives.txt", 'r') as f:
            self.objectives =json.load(f)
            
            
    def load_file_obj(self, path, obj_path, name_base):
        print("Loading parameters from: ", path + obj_path + '_' + name_base)
        with open(path + obj_path + '_' + name_base + "_left.txt", 'r') as f:    
            self.left = json.load(f)
        with open(path + obj_path + '_' + name_base + "_right.txt", 'r') as f:  
            self.right = json.load(f)
        with open(path + obj_path + '_' + name_base + "_full.txt", 'r') as f:
            self.full = json.load(f)
        return 

    
    def load_file_sim(self, path, name_base):
        with open(path + name_base + "_simulation.txt", 'r') as f:
            self.simulation = json.load(f)

            
    def load_file(self, path, obj_path, name_base):
        print("Loading parameters from: ", path + obj_path + '_' + name_base)
        with open(path + name_base + "_general.txt", 'r') as f:
            self.general = json.load(f)        
        with open(path + name_base + "_objectives.txt", 'r') as f:
            self.objectives =json.load(f)
        with open(path + obj_path + '_' + name_base + "_left.txt", 'r') as f:    
            self.left = json.load(f)
        with open(path + obj_path + '_' + name_base + "_right.txt", 'r') as f:  
            self.right = json.load(f)
        with open(path + obj_path + '_' + name_base + "_right.txt", 'r') as f:
            self.full = json.load(f)


    def load_model_def(self, path, fname, model_name):
        with open(path + fname, 'r') as f:
            models = json.load(f)
            self.model_def = models[model_name]


    def get(self, param):
        return param

    
    def set_(self, param, value):
        self.param = value

        
    def mm2px(self, mm):
        """ Can be  used to convert mm to px. Not needed at the moment. To use,
            comment 1st line and uncomment 2nd."""
        return mm



if __name__ == "__main__":
    """ Run this main to reset the parameter files to the defaults hardcoded in
        here. """
    p = param()
    p.init_defaults()
    path = ["../parameters/", "params"]
    objectives = ["100xOil_NA_140", "100xSil_NA_135", "60xWat_NA_120",
                  "20xAir_NA_070"]
    
    for o in objectives:
        p.write_file(path[0], o , path[1])
        