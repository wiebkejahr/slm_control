'''
Project: deep-adaptive-optics
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
        #print('type of transformed image is {}'.format(type(sample['image'])))
        #print('type of transformed label is {}'.format(type(sample['label'])))
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

class MyNormalize(object):
    """Given a mean and std with constructor call, it normalizes the input. 
    Mean and std must be calculatedfirst."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # for the 20k datapoint set
        # mean=torch.from_numpy(np.asarray([0.1251]))
        # std=torch.from_numpy(np.asarray([0.2146]))

        # for dataset_500.hdf5
        # mean=torch.from_numpy(np.asarray([0.1229]))
        # std=torch.from_numpy(np.asarray([0.2120]))

        # for some other dataset
        #mean=torch.from_numpy(np.asarray([0.9780]))
        #std=torch.from_numpy(np.asarray([0.3355]))
        # print(self.mean)
        # print(self.std)
        # mean = torch.from_numpy(self.mean)
        # std = torch.from_numpy(self.std)

        return {'image': (image - self.mean) / self.std,
                'label': label}


   

