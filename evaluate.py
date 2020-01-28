'''
Project: deep-adaptive-optics
Created on: Wednesday, 6th November 2019 9:47:12 am
--------
@author: hmcgovern
'''
"""Module docstring goes here."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import argparse as ap
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
import matplotlib.pyplot as plt


from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import utils.my_models as my_models
import utils.my_classes as my_classes
from utils.integration import *
from utils.helpers import *

def log_images(logdir, images, coeffs):
    logdir_test = logdir + '/test'
    test_writer = SummaryWriter(log_dir=logdir_test)


    grid = torchvision.utils.make_grid(images)
    test_writer.add_image('GT Images', grid, 0)
    
    preds = get_preds(coeffs)
    preds = torch.from_numpy(preds).unsqueeze(3)

    test_writer.add_images('Predicted Images', preds, 0, dataformats='NHWC')
    
    test_writer.close()

    return None


def test(model, test_loader, logdir, model_store_path):
    
    # logdir_test = logdir + '/test'
    # test_writer = SummaryWriter(log_dir=logdir_test)
    

    ideal_coeffs = np.asarray([0.0]*12)
    donut = get_psf(ideal_coeffs) # (64,64)

    # Test the model
    model.eval()
    for i, (images, labels) in enumerate(test_loader): # i is 0 when batch_size is 1
            
    
        with torch.no_grad(): # drastically increases computation speed and reduces memory usage
            # Get model outputs (the predicted Zernike coefficients)
            outputs = model(images)
            # comparing a 'corrected' psf from the predicted coeffs with the ideal donut
            print('predicted: {}'.format(outputs[0]))
            print('ground truth: {}'.format(labels[0]))
            print('MSE is: {}'.format(mean_squared_error(labels[0], outputs[0])))
            


def main(args):
    data_path = args.test_dataset_dir
    logdir = args.logdir
    model_store_path = args.model_store_path
    model = my_models.MultiNet()


    checkpoint = torch.load(model_store_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    mean, std = get_stats(data_path, batch_size=10, mode='val')
    test_dataset = my_classes.PSFDataset(hdf5_path=data_path, mode='val', transform=transforms.Compose([
        my_classes.ToTensor(), 
        my_classes.Normalize(mean=mean, std=std)]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=10, \
        shuffle=False, num_workers=0)

    # avg_coeffs = []
    # for i in range(10):
    test(model, test_loader, logdir, model_store_path)
        # print(coeffs[0][0])
    #     avg_coeffs.append(coeffs)
    # avg = np.mean(np.array(avg_coeffs), axis=0)

    # log_images(logdir, images, avg)


if __name__ == "__main__":
    parser = ap.ArgumentParser(description='Model Hyperparameters and File I/O')
    parser.add_argument('test_dataset_dir', type=str, help='path to dataset')
    parser.add_argument('logdir', type=str, help='keyword for path to tensorboard logging info')
    parser.add_argument('model_store_path', type=str, help='path to model checkpoint dir')
    
    ARGS=parser.parse_args()

    main(ARGS)