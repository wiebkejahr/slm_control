'''
Project: deep-adaptive-optics
Created on: Wednesday, 6th November 2019 9:47:12 am
--------
@author: hmcgovern
'''

import random
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
from tqdm import tqdm

import h5py
from skimage.transform import resize

from utils.integration import integrate
from utils.helpers import get_psf as get_psf

###############################################################################    
    
def gen_coeffs():
    """ Generates a random set of Zernike coefficients given piecewise constraints
    from Zhang et al's paper.
    1st-3rd: [0]   |  4th-6th: [+/- 1.4]  | 7th-10th: [+/- 0.8]  |  11th-15th: [+/- 0.6] 
    
    For physical intuition's sake, I'm creating a 15 dim vector, but only returning the 12 values that are non-zero.
    NOTE: this could be modified to accomodate further Zernike terms, but CNN code would have to be adjusted as well
    """
    c = np.zeros(15)
    c[3:6] = [random.uniform(-1.4, 1.4) for i in c[3:6]]
    c[6:10] = [random.uniform(-0.8, 0.8) for i in c[6:10]]
    c[10:] = [random.uniform(-0.6, 0.6) for i in c[10:]]
    
    return c[3:]



def main(args):
    """Using input arugments and some constants, this makes training, validation, and test sets for 
    training and evaluating a model. 90/10 train/validation split, and additional parameter for 
    number of test images. 
    
    The images (aberrated donuts) are inputs to the model 
    The labels are the coefficient lists used to make the aberrated donuts (ground truth for model)
    
    Everything goes into one .hdf5 file"""
    
    # some constants
    num_data_pts = args.num_points
    hdf5_path = args.dataset_dir
    res = args.resolution
    label_dim = 12

    # create partition: 90/10 train/val split
    train_num = int(0.9*num_data_pts)
    val_num = num_data_pts - train_num 
    test_num = args.test_num
    
    print('number of training examples: {}'.format(train_num))
    print('number of validation examples: {}'.format(val_num))
    print('number of test images: {}'.format(test_num))

    train_shape = (train_num, 1, res, res)
    val_shape = (val_num, 1, res, res)
    test_shape = (test_num, 1, res, res)
    
    # open a hdf5 file and create arrays
    hdf5_file = h5py.File(hdf5_path, mode='w-')

    # create the image arrays
    hdf5_file.create_dataset("train_img", train_shape, np.float32)
    hdf5_file.create_dataset("val_img", val_shape, np.float32)
    hdf5_file.create_dataset("test_img", test_shape, np.float32)
    
    train_labels = []
    val_labels = []
    test_labels = []
    
    #create train set
    for i in tqdm(range(train_num)):
        # generate coefficient list
        label = gen_coeffs()
        # create 64 x 64 image from coefficient list
        img = get_psf(label, res)
        # save the label and image
        train_labels.append(label)
        hdf5_file["train_img"][i, ...] = img[None]
    print('Training examples completed.')
    
    for i in tqdm(range(val_num)):
        label = gen_coeffs()
        # create 64 x 64 image from coefficient list
        img = get_psf(label, res)
        # save the label and image
        val_labels.append(label)
        hdf5_file["val_img"][i, ...] = img[None]
    print('Validation examples completed.')
    
    for i in tqdm(range(test_num)):
        label = gen_coeffs()
        # create 64 x 64 image from coefficient list
        img = get_psf(label, res)
        # save the label and image
        test_labels.append(label)
        hdf5_file["test_img"][i, ...] = img[None]
    print('Test images completed.')
    
    # create the label arrays
    hdf5_file.create_dataset("train_labels", (train_num, label_dim), np.float32)
    hdf5_file["train_labels"][...] = train_labels
    hdf5_file.create_dataset("val_labels", (val_num, label_dim), np.float32)
    hdf5_file["val_labels"][...] = val_labels
    hdf5_file.create_dataset("test_labels", (test_num, label_dim), np.float32)
    hdf5_file["test_labels"][...] = test_labels


if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Dataset Parameters")
    parser.add_argument('num_points', type=int,\
        help='number of data points for the dataset, will be split 90/10 training/validation')
    parser.add_argument('dataset_dir', \
        help='path to where you want the dataset stored. Name will be automatically generated\
            based on the number of points.')
    parser.add_argument('resolution', type=int, default=64, \
        help='resolution of training example psf image. Default is 64 x 64')
    parser.add_argument('test_num', type=int, default=2000, \
        help='number of points to use in test set')
    ARGS = parser.parse_args()
    
    main(ARGS)


    

