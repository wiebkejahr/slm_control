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

# import utils.helpers as helpers


class PSFDataset(data.Dataset):
    """ Point Spread Function h5py Dataset. """

    def __init__(self, hdf5_path, mode, transform=None, modify=False, offset=False, noise=True, bgnoise=2, poiss=350, center=False):
        """
        Args:
            hdf5_path (str): Path to the hdf5 file 
        """
        # Creates an h5py object from the given path
        self.file = h5py.File(hdf5_path, "r")
        self.transform = transform
        self.mod = Modify()

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

        # NOTE: MODIFICATION NEEDS TO GO HERE!!!
        if modify:
            # print(self.images.shape) # (18000, 3, 64, 64)
            # print(self.labels.shape) # (18000, 11)
            # exit()
            self.new_images = np.zeros_like(self.images)
            self.new_labels = np.zeros_like(self.labels)
            for i in range(len(self.images)): # to 1800
                print(i)
                # print(helpers.get_CoM(self.images[i][0]))
                # helpers.plot_xsection(self.images[i])
                # plt.show()
                sample = self.mod({'image': self.images[i], 'label': self.labels[i]})
                self.new_images[i] = sample['image']
                # helpers.plot_xsection(self.new_images[i])
                # plt.show()
                # print(helpers.get_CoM(self.new_images[i][0]))
                # print('\n')
                self.new_labels[i] = sample['label']
                
            self.images = self.new_images
            self.labels = self.new_labels

    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        sample = {'image': self.images[idx], 'label': self.labels[idx]}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
        # return {sample['image'], sample['label']}



class Modify(object):

    def __init__(self, offset=False, noise=True, bgnoise=1, poiss=350, center=False):
        self.offset=offset
        self.center=center
        self.noise=noise
        self.bgnoise=bgnoise
        self.poiss=poiss
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # print(image.shape)
        if self.offset:
            off = helpers.gen_offset()
        else:
            off = [0,0]

        if self.noise:
            image = helpers.add_noise(image, bgnoise_amount=self.bgnoise, poiss_amount=self.poiss)
        
        if self.center:
            # print(image[0].shape)
            tiptilt = helpers.center(image[0].squeeze(), label)
        else:
            tiptilt = []
        
        # calculcating new psf with offset label and tiptilt correction
        # new_img = helpers.get_sted_psf(coeffs=label, offset_label=off,\
        #          multi=True, corrections=tiptilt)

        # adding noise back on top
        new_img = helpers.add_noise(image, bgnoise_amount=self.bgnoise, poiss_amount=self.poiss)
        
        if self.offset:
            label = np.append(label, off)
        
        return {'image': new_img,
                'label': label}
        

class Offset(object):
    """Given a synthetic data point, it modifies both the label 
    and the image to reflect a shift in phasemask."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        off = helpers.gen_offset()
        print(off)
        image2 = helpers.get_sted_psf(coeffs=label, offset_label=off, multi=True)
        label2 = np.append(label, off)
        return {'image': image2,
                'label': label2}


class Center(object):
    """Given a synthetic data point, it corrects for the observed 
    tip & tilt and passes back a corrected image."""
    def __init__(self, offset=False):
        self.offset = offset
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # logic here is needed to just use the zernike coefficients
        # instead of coeffs + offset if an offset was added before
        # calling centering. Maybe want something more sophisticated
        # later
        if self.offset:
            image = helpers.center(image[0], label[:-2])
        else:
            image = helpers.center(image[0], label)
    
        return {'image': image,
                'label': label}

class Make1D(object):
    """Given a 3D dataset, it returns the same label with the xy cross section only."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
    
        return {'image': image[0],
                'label': label}


class Noise(object):
    """Given a bgnoise and poisson_noise with constructor call, it adds noise to the input."""
    def __init__(self, bgnoise=1, poiss=1000):
        self.bgnoise = bgnoise
        self.poiss = poiss

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        return {'image': helpers.add_noise(image, bgnoise_amount=self.bgnoise, poiss_amount=self.poiss),
                'label': label}


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





   

