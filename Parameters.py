#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:13:09 2018

@author: wjahr
"""

import json
import os

class param():
    """ Class containing the dictionary for the parameters to be saved.
        Implements a reader/writer to import/export to json. Before exporting,
        the parameters for left/right images and aberrations can be updated
        according to the parameters provided in the user input.
        TODO: update general parameters (according to user input) if needed. """
    def __init__(self):
        self.init_defaults()
        
    
    def init_defaults(self):
        self.general = {
                    "abberior"     : 0,
                    "size_slm_mm"  : [7.5, 10],
                    "slm_px"       : 0.0125,
                    "size_full"    : [600, 792],
                    "size_slm"     : [600, 396],
                    "displaywidth" : 300,
                    "slm_mag"      :  3,
                    "laser_radius" : 1.2,
                    "objective"    : "100xOil_NA_140",
                    "path"         : 'patterns/',                    
                    "cal0"         : "patterns/Black_Full.bmp",
                    "last_img_nm"  : 'latest.bmp',
                    "modes"        : ["Gauss", "2D STED", "3D STED", 
                                      "Segments", "Bivortex", "Code Input",
                                      "From File"],
                    "split_image"  : 1,
                    "flat_field"   : 0,
                    "single_aberr" : 0,
                    "double_pass"  : 0,
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
        
#        self.aberrations = {
#                            "astig" : [0,0],
#                            "coma" : [0,0],
#                            "sphere" : [0,0],
#                            "trefoil" : [0,0],
#                            }
    
    def update(self, daddy):
        """ Parameter p: self of the calling function. Needed to have access to
            parameters provided via the GUI. Updates the values in the 
            dictionary with the values from the GUI. """
        splt_img = 0
        if daddy.splt_img_state.checkState():
            splt_img = 1
        flt_fld = 0
        if daddy.flt_fld_state.checkState():
            flt_fld = 1
        sngl_corr = 0
        if daddy.sngl_corr_state.checkState():
            sngl_corr = 1
        dbl_ps = 0
        if daddy.dbl_pass_state.checkState():
            dbl_ps = 1
        abberior = 0
        if daddy.p.general["abberior"] == 1:
            abberior = 1
        
        self.general = {
                    "abberior"     : abberior,
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
                    "split_image"  : splt_img,
                    "flat_field"   : flt_fld,
                    "single_aberr" : sngl_corr,
                    "double_pass"  : dbl_ps,
                    }
#        self.slm_radius = pcalc.normalize_radius(self.p.objectives[self.current_objective["name"]]["backaperture"], 

        
        
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
#        d = daddy.img_aberr
#        self.aberrations = {
#                            "astig" : [d.astig.xgui.value(), d.astig.ygui.value()],
#                            "coma" : [d.coma.xgui.value(), d.coma.ygui.value()],
#                            "sphere" : [d.sphere.xgui.value(), d.sphere.ygui.value()],
#                            "trefoil" : [d.trefoil.xgui.value(), d.trefoil.ygui.value()],
#                            }


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
        return 
            
          
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


    def get(self, param):
        return param
    
    def set_(self, param, value):
        self.param = value
        
    def mm2px(self, mm):
        """ Can be  used to convert mm to px. Not needed at the moment. To use,
            comment 1st line and uncomment 2nd."""
        return mm
        #return mm/self.general["slm_px"]
        



if __name__ == "__main__":
    """ Run this main to reset the parameter files to the defaults hardcoded in
        here. """
    p = param()
    p.init_defaults()
    path = ["parameters/", "params"]
    objectives = ["100xOil_NA_140", "100xSil_NA_135", "60xWat_NA_120",
                  "20xAir_NA_070"]
    
    for o in objectives:
        p.write_file(path[0], o , path[1])
        
        
        