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

#from utils import helpers
#import helpers
import autoalign.utils.helpers as helpers


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
    
        image = self.images[idx]
        image = helpers.normalize_img(image)*255
        image = np.squeeze(np.stack((image, image, image), axis=-1))

        image = Image.fromarray(image.astype(np.uint8), 'RGB')
        sample = {'image': image, 'label': self.labels[idx]}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

class UnNormalize(object):
    # unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # unorm(tensor)
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


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





class Normalize(object):
    """Given a mean and std with constructor call, it normalizes the input. 
    Mean and std must be calculatedfirst."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image): #sample):
        
        #image, label = sample['image'], sample['label']
        
        for channel in range(image.size(0)):
            image[channel] = (image[channel] - self.mean[channel])/ self.std[channel]

        return image
        # return {'image': image,
        #         'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image): #sample):
        #image, label = sample['image'], sample['label']
        # NOTE: really not sure about these axes, 
        # I'm only using 1 color channel, so it's usually non-existent
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        
        return torch.from_numpy(image)
        # return {'image': torch.from_numpy(image),
        #         'label': torch.from_numpy(label)}





   

