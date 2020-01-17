# -*- coding: utf-8 -*-

"""

Created on Thu Aug 29 09:39:28 2019



@author: jlyudchi

"""

import javabridge

import bioformats
from bioformats import log4j
import matplotlib.pyplot as plt

javabridge.start_vm(class_path=bioformats.JARS)
log4j.basic_config()

#image path

# input_image = "C:/20190528_2_DG_01.msr"
input_image = "Z:/Measurements/Abberior/Alignment_WJ_PV_NAD/20191104_Routine_PV/20191104_0102_100xSil_AuSTED_before.msr"

#load specified channel (series):
image = bioformats.load_image(input_image, series = 2)
javabridge.kill_vm()
# image = bioformats.load_image(input_image)
plt.imshow(image, cmap='hot')
plt.show()
# print(image.getSeriesCount())

#load metadata

o = bioformats.OMEXML(bioformats.get_omexml_metadata(input_image))
# x_in_um = o.image(1).Pixels.PhysicalSizeX
# print(x_in_um)
'''
Example how to address metadata:

x_in_um = o.image(5).Pixels.PhysicalSizeX

y_in_um = o.image(5).Pixels.PhysicalSizeY

'''

