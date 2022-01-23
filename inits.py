#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:50:12 2022

@author: wiebke
"""


import slm_control.Pattern_Calculator as pcalc

def calc_slmradius(p, backaperture, mag):
    """ Calculates correct scaling factor for SLM based on objective
        backaperture, optical magnification of the beampath, SLM pixel
        size and size of the SLM. Required values are directly taken from
        the parameters files. """
        
    rad = pcalc.normalize_radius(backaperture, mag, 
                p.general["slm_px"], p.general["size_slm"])
    return rad