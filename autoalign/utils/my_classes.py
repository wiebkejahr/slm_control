'''
Project: deep-sted-autoalign
Created on: Wednesday, 6th November 2019 9:47:12 am
--------
@author: hmcgovern
'''


from random import shuffle
from math import ceil
import matplotlib.pyplot as plt
import h5py
import numpy as np
from torch.utils import data
import torch
from torchvision import transforms


class PSFDataset(data.Dataset):
    """ Point Spread Function h5py Dataset. """

    def __init__(self, hdf5_path, mode, transform=None):
        """
        Args:
            hdf5_path (str): Path to the hdf5 file 
        """
        # Creates an h5py object from the given path
        self.file = h5py.File(hdf5_path, "r")
        self.transform = transform

        # if training, loads the training and validation images and labels
        if mode =='train':
            self.images = self.file['train_img']
            self.labels = self.file['train_labels']
        elif mode == 'val':
            self.images = self.file['val_img']
            self.labels = self.file['val_labels']
        # if testing, loads the test images and labels
        elif mode == 'test':
            self.images = self.file['test_img']
            self.labels = self.file['test_labels']
        
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        sample = {'image': self.images[idx], 'label': self.labels[idx]}
        
        if self.transform:
            sample = self.transform(sample)

        return (sample['image'], sample['label'])

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # NOTE: really not sure about these axes, 
        # I'm only using 1 color channel, so it's usually non-existent
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

class Normalize(object):
    """Given a mean and std with constructor call, it normalizes the input. 
    Mean and std must be calculatedfirst."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        for channel in range(image.size(0)):
            image[channel] = (image[channel] - self.mean[channel])/ self.std[channel]

        return {'image': image,
                'label': label}


   

