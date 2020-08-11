
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
# from scipy.ndimage.measurements import center_of_mass
# from scipy.ndimage import shift

# local packages
from helpers import *
import pickle

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
    path = args.data_dir
    res = 64
    # res = args.resolution

    # create partition: 90/10 train/val split
    train_num = int(0.9*num_data_pts)
    val_num = num_data_pts - train_num 
    test_num = args.test_num
    

    train_images=[]
    train_masks=[]
    for i in tqdm(range(train_num)):
        img, zern = helpers.gen_sted_psf_phase(res=res)
        train_images.append([img, img, img])

        train_masks.append([zern, zern, zern])
    print('Training examples completed.')
    print(np.asarray(train_images).shape)
    exit()
    val_images=[]
    val_masks=[]
    for i in tqdm(range(val_num)):
        img, zern = helpers.gen_sted_psf_phase(res=res)
        val_images.append([img, img, img])
        val_masks.append([zern, zern, zern])
    print('Validation examples completed.')
    
    test_images=[]
    test_masks=[]
    for i in tqdm(range(test_num)):
        img, zern = helpersgen_sted_psf_phase(res=res)
        test_images.append([img, img, img])
        test_masks.append([zern, zern, zern])
    print('Test examples completed.')

    
    # with open(path+'/trainRGB_5.pkl', 'wb') as f:
    #     pickle.dump({'images': np.asarray(train_images), 'masks': np.asarray(train_masks)}, f)
    # with open(path+'/valRGB_5.pkl', 'wb') as f:
    #     pickle.dump({'images': np.asarray(val_images), 'masks': np.asarray(val_masks)}, f)
    # with open(path+'/testRGB_5.pkl', 'wb') as f:
    #     pickle.dump({'images': np.asarray(test_images), 'masks': np.asarray(test_masks)}, f)

if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Dataset Parameters")
    parser.add_argument('num_points', type=int,\
        help='number of data points for the dataset, will be split 90/10 training/validation')
    parser.add_argument('test_num', type=int, default=20, \
        help='number of points for the test set')
    parser.add_argument('data_dir', \
        help='path to where you want the dataset stored.')

    ARGS = parser.parse_args()
    
    main(ARGS)


    

