
'''
Project: deep-sted-autoalign
Created on: Wednesday, 6th November 2019 9:47:12 am
--------
@author: hmcgovern
'''
# third party
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
from tqdm import tqdm
import json
import h5py
import skimage
from skimage.transform import resize, AffineTransform, warp, rotate
# from scipy.ndimage.measurements import center_of_mass
# from scipy.ndimage import shift

# local packages
from utils.helpers import *

###############################################################################    
    
def main(args):
    """Using input arugments and some constants, this makes training, validation, and test sets for 
    training and evaluating a model. 90/10 train/validation split, and additional parameter for 
    number of test images. 
    
    The images (aberrated donuts) are inputs to the model 
    The labels are the coefficient lists used to make the aberrated donuts (ground truth for model)
    
    Everything goes into one .hdf5 file"""
    
    # some constants
    num_data_pts = args.num_points
    hdf5_path = args.data_dir
    res = args.resolution
    if_zern = args.zern
    if_offset = args.offset
    if_multi = args.multi

    # create partition: 90/10 train/val split
    train_num = int(0.9*num_data_pts)
    val_num = num_data_pts - train_num 
    test_num = args.test_num
    
    print('number of training examples: {}'.format(train_num))
    print('number of validation examples: {}'.format(val_num))
    print('number of test images: {}'.format(test_num))

    # if the flag for multi-channel is there, give it 3 color channels
    if if_multi:
        channel_num = 3
    else:
        channel_num = 1

    # shift_num = 3
    # NOTE: used to be train_num, channel_num, res, res
    train_shape = (train_num, res, res, channel_num)
    val_shape = (val_num, res, res, channel_num)
    test_shape = (test_num, res, res, channel_num)

    # open a hdf5 file and create arrays
    hdf5_file = h5py.File(hdf5_path, mode='w-')

    # create the image arrays
    hdf5_file.create_dataset("train_img", train_shape, np.float32)
    hdf5_file.create_dataset("val_img", val_shape, np.float32)
    hdf5_file.create_dataset("test_img", test_shape, np.float32)
    
    train_labels = []
    val_labels = []
    test_labels = []

    # label_dim will be 0,2,11,or 13 depending on combo of features
    label_dim = 0
    if if_zern: label_dim += 11
    if if_offset: label_dim += 2
    print('label dimention is: {}'.format(label_dim))
 

    for i in tqdm(range(train_num)):
        if args.mode == 'sted':
            # if flags for zern and offset given, will generate labels
            # otherwise they are zeros
            if if_zern: 
                zern_label = gen_coeffs()
            else:
                zern_label = np.asarray([0.0]*11)
            if if_offset: 
                offset_label = gen_offset()
            else:
                offset_label = np.asarray([0.0]*2)

            img = get_sted_psf(coeffs=zern_label, res=[res,res], offset_label=offset_label, multi=if_multi)
            # calculates the tip and tilt present in the image and corrects it
            tiptilt = center(img, res=[res, res])
            img = get_sted_psf(coeffs=zern_label, res=[res, res], multi=args.multi, offset_label=offset_label, tiptilt=tiptilt)

        # NOTE: get_fluor_psf not up to date yet
        elif args.mode == 'fluor':
            img, zern_label, offset_label = gen_fluor_psf(res, offset=args.offset, multi=args.multi)
        
        # save the label and image
        # always save a 13 dim label, can always truncate if you need
        train_labels.append(np.append(zern_label, offset_label))
        
        # adding extra dimension res x res x 1
        img = np.expand_dims(img, axis=2)
        hdf5_file["train_img"][i, ...] = img


    # create the label array
    hdf5_file.create_dataset("train_labels", (train_num, label_dim), np.float32)
    hdf5_file["train_labels"][...] = train_labels
    print('Training examples completed.')
    
    for i in tqdm(range(val_num)):
        if args.mode == 'sted':
            if if_zern: 
                zern_label = gen_coeffs()
            else:
                zern_label = np.asarray([0.0]*11)
            if if_offset: 
                offset_label = gen_offset()
            else:
                offset_label = np.asarray([0.0]*2)

            img = get_sted_psf(coeffs=zern_label, res=[res,res], offset_label=offset_label, multi=if_multi)
            tiptilt = center(img, res=[res, res])
            img = get_sted_psf(coeffs=zern_label,res = [res, res],multi=args.multi, offset_label=offset_label, tiptilt=tiptilt)

        elif args.mode == 'fluor':
            img, zern_label, offset_label = gen_fluor_psf(res, offset=args.offset, multi=args.multi)
        
        # save the label and image
        val_labels.append(np.append(zern_label, offset_label))

        img = np.expand_dims(img, axis=2)
        hdf5_file["val_img"][i, ...] = img


    # create the label array
    hdf5_file.create_dataset("val_labels", (val_num, label_dim), np.float32)
    hdf5_file["val_labels"][...] = val_labels
    print('Validation examples completed.')
    
    for i in tqdm(range(test_num)):
        
        if args.mode == 'sted':
            if if_zern: 
                zern_label = gen_coeffs()
            else:
                zern_label = np.asarray([0.0]*11)
            if if_offset: 
                offset_label = gen_offset()
            else:
                offset_label = np.asarray([0.0]*2)

            img = get_sted_psf(coeffs=zern_label, res=[res,res], offset_label=offset_label, multi=if_multi)
            tiptilt = center(img, res=[res, res])
            img = get_sted_psf(coeffs=zern_label, res=[res, res], multi=args.multi, offset_label=offset_label, tiptilt=tiptilt)
        
        elif args.mode == 'fluor':
            img, zern_label, offset_label = gen_fluor_psf(res, offset=args.offset, multi=args.multi)
        
        # save the label and image
        test_labels.append(np.append(zern_label, offset_label))
        
        img = np.expand_dims(img, axis=2)
        hdf5_file["test_img"][i, ...] = img

    # create the label array
    hdf5_file.create_dataset("test_labels", (test_num, label_dim), np.float32)
    hdf5_file["test_labels"][...] = test_labels
    print('Test images completed.')

if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Dataset Parameters")
    parser.add_argument('num_points', type=int,\
        help='number of data points for the dataset, will be split 90/10 training/validation')
    parser.add_argument('test_num', type=int, default=20, \
        help='number of points for the test set')
    parser.add_argument('data_dir', \
        help='path to where you want the dataset stored.')
    parser.add_argument('-r','--resolution', type=int, default=64, \
        help='resolution of training example image. Default is 64 x 64')
    parser.add_argument('--multi', type=int, default=0, \
        help='whether or not to use cross-sections')  
    parser.add_argument('--offset', type=int, default=0, \
        help='whether or not to incorporate offset')
    parser.add_argument('--zern', type=int, default=1, \
        help='whether or not to include optical aberrations')  
    parser.add_argument('--mode', type=str, choices=['fluor', 'sted', 'z-sted'],\
        help='which mode of data to create') 
    ARGS = parser.parse_args()
    

    main(ARGS)


    

