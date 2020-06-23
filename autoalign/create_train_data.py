
'''
Project: deep-sted-autoalign
Created on: Wednesday, 6th November 2019 9:47:12 am
--------
@author: hmcgovern
'''
# third party
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
from tqdm import tqdm
import json
import h5py
import skimage
from skimage.transform import resize, AffineTransform, warp, rotate
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import shift

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

    # create partition: 90/10 train/val split
    train_num = int(0.9*num_data_pts)
    val_num = num_data_pts - train_num 
    test_num = args.test_num
    
    # print('number of training examples: {}'.format(train_num))
    # print('number of validation examples: {}'.format(val_num))
    # print('number of test images: {}'.format(test_num))

    # if the flag for multi-channel is there, give it 3 color channels
    if args.multi:
        channel_num = 3
    else:
        channel_num = 1

    # shift_num = 3
    train_shape = (train_num, channel_num, res, res)
    val_shape = (val_num, channel_num, res, res)
    test_shape = (test_num, channel_num, res, res)

    # open a hdf5 file and create arrays
    hdf5_file = h5py.File(hdf5_path, mode='w-')

    # create the image arrays
    hdf5_file.create_dataset("train_img", train_shape, np.float32)
    hdf5_file.create_dataset("val_img", val_shape, np.float32)
    hdf5_file.create_dataset("test_img", test_shape, np.float32)
    
    train_labels = []
    val_labels = []
    test_labels = []

    if args.offset:
        label_dim = 14
    else:
        label_dim = 12
    #NOTE: overriding
    label_dim = 11



    for i in tqdm(range(train_num)):
        if args.mode == 'sted':
            img, zern_label, offset_label = gen_sted_psf(multi=args.multi, defocus=False)
            # img = get_sted_psf_tip_tilt(coeffs=[1,1])
            # print(zern_label)
            # plt.imshow(img, cmap='hot')
            # a, b = center_of_mass(img)
            # plt.scatter(b,a, color='b')
            # plt.show()
            # exit()
            # # coeffs = np.asarray([0.0]*14)
            # # coeffs[0] = 0.2
            # # coeffs[1] = 0.1
            # # coeffs[2] = 0.2
            # # coeffs[4] = 0.1
            # # img = get_sted_psf_tip_tilt(coeffs=coeffs)
            # # img = center(img)
            # # coeffs = np.asarray([0.0]*12)
            # # img = get_sted_psf(coeffs=coeffs)
            # # a,b = center_of_mass(img)
            # # print(a,b)
            # # print(31.5-a, 31.5-b)

            # # D = get_D(a=coeffs[1], dx=abs(31.5-a))
            # # print(D)
            # # shift_img = correct_tip_tilt(img) # 0.0001
            # print(center_of_mass(img))
            # new = correct_tip_tilt(img)
            
        elif args.mode == 'fluor':
            img, zern_label, offset_label = gen_fluor_psf(res, offset=args.offset, multi=args.multi)
        # save the label and image
        if args.offset:
            train_labels.append(zern_label+offset_label)
        else:
            train_labels.append(zern_label)

        hdf5_file["train_img"][i, ...] = img[None]

    # hdf5_file["train_img"][...] = np.expand_dims(train_data, axis=1)
    # # exit()
    # create the label array
    hdf5_file.create_dataset("train_labels", (train_num, label_dim), np.float32)
    hdf5_file["train_labels"][...] = train_labels
    print('Training examples completed.')
    
    for i in tqdm(range(val_num)):
        if args.mode == 'sted':
            img, zern_label, offset_label = gen_sted_psf(multi=args.multi, defocus=False)

        elif args.mode == 'fluor':
            img, zern_label, offset_label = gen_fluor_psf(res, offset=args.offset, multi=args.multi)
        
        # save the label and image
        if args.offset:
            val_labels.append(zern_label+offset_label)
        else:
            val_labels.append(zern_label)

        hdf5_file["val_img"][i, ...] = img[None]

    # hdf5_file["val_img"][...] = np.expand_dims(val_data, axis=1)
    # create the label array
    hdf5_file.create_dataset("val_labels", (val_num, label_dim), np.float32)
    hdf5_file["val_labels"][...] = val_labels
    print('Validation examples completed.')
    
    for i in tqdm(range(test_num)):
        
        if args.mode == 'sted':
            img, zern_label, offset_label = gen_sted_psf(multi=args.multi, defocus=False)

        elif args.mode == 'fluor':
            img, zern_label, offset_label = gen_fluor_psf(res, offset=args.offset, multi=args.multi)
        
        # save the label and image
        if args.offset:
            test_labels.append(zern_label+offset_label)
        else:
            test_labels.append(zern_label)
            
  
    
        hdf5_file["test_img"][i, ...] = img[None]

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
    parser.add_argument('--multi', action='store_true', \
        help='whether or not to use cross-sections')  
    parser.add_argument('--offset', action='store_true', \
        help='whether or not to incorporate offset')  
    parser.add_argument('--mode', type=str, choices=['fluor', 'sted', 'z-sted'],\
        help='which mode of data to create') 
    ARGS = parser.parse_args()
    

    main(ARGS)


    

