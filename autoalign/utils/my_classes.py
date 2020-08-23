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
from PIL import Image
from matplotlib import cm


# from utils import helpers
# import autoalign.utils.helpers as helpers


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

    def _get_dim(self):
        return self.images.shape[1]
    
    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image = helpers.normalize_img(image.squeeze())

        if self._get_dim() == 1:
            # this is only for 1D (64, 64)
            image = Image.fromarray(np.uint8(image*255), 'L')
        elif self._get_dim() == 3:
            # first need to rearrange (3, 64, 64) into (64, 64, 3)
            image = np.transpose(image, (1, 2, 0))
            image = Image.fromarray(np.uint8(image*255))

        if self.transform:
            image = self.transform(image)

        # return [image, label]
        return {'image': image, 'label':label}


# class Offset(object):
#     """Given a synthetic data point, it modifies both the label 
#     and the image to reflect a shift in phasemask."""

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
        
#         off = helpers.gen_offset()
#         print(off)
#         image2 = helpers.get_sted_psf(coeffs=label, offset_label=off, multi=True)
#         label2 = np.append(label, off)
#         return {'image': image2,
#                 'label': label2}


# class Center(object):
#     """Given a synthetic data point, it corrects for the observed 
#     tip & tilt and passes back a corrected image."""
#     def __init__(self, offset=False):
#         self.offset = offset
    
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         # logic here is needed to just use the zernike coefficients
#         # instead of coeffs + offset if an offset was added before
#         # calling centering. Maybe want something more sophisticated
#         # later
#         if self.offset:
#             image = helpers.center(image[0], label[:-2])
#         else:
#             image = helpers.center(image[0], label)
    
#         return {'image': image,
#                 'label': label}

# class Make1D(object):
#     """Given a 3D dataset, it returns the same label with the xy cross section only."""
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
    
#         return {'image': image[0],
#                 'label': label}


class Noise(object):
    """Given a bgnoise and poisson_noise with constructor call, it adds noise to the input."""
    def __init__(self, bgnoise, poiss):
        self.bgnoise = bgnoise
        self.poiss = poiss

    def __call__(self, image):
        image =image.numpy()
        return torch.from_numpy(helpers.add_noise(image, bgnoise_amount=self.bgnoise, poiss_amount=self.poiss))


class Normalize(object):
    """Given a mean and std with constructor call, it normalizes the input. 
    Mean and std must be calculatedfirst."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        
        for channel in range(image.size(0)):
            image[channel] = (image[channel] - self.mean[channel])/ self.std[channel]

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        
        # NOTE: really not sure about these axes, 
        # I'm only using 1 color channel, so it's usually non-existent
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)





   

