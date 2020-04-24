'''
Project: deep-sted-autoalign
Created on: Wednesday, 6th November 2019 9:47:12 am
--------
@author: hmcgovern
'''
"""Module docstring goes here."""

# third party imports
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

# local packages
import utils.my_models as my_models
import utils.my_classes as my_classes
from utils.vector_diffraction import *
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
    # donut = get_psf(ideal_coeffs, res=64, multi=True) # (64,64)

    # Test the model
    model.eval()
    for i, (images, labels) in enumerate(test_loader): # i is 0 when batch_size is 1
    
        with torch.no_grad(): # drastically increases computation speed and reduces memory usage
            # Get model outputs (the predicted Zernike coefficients)
            to_plot = images.numpy().squeeze()
            donut = to_plot


            outputs = model(images)
            preds = outputs.numpy().squeeze()

            remaining = labels.numpy().squeeze() - preds
            print(mean_squared_error(remaining, np.asarray([0.0]*12)))
            
            corrected = get_fluor_psf(remaining) #, multi=True
            # corrected = normalize_img(donut) + normalize_img(get_psf(-1*outputs.numpy().squeeze()))

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(to_plot)
            ax2 = fig.add_subplot(1,3,2)
            ax2.imshow(get_fluor_psf(preds))
            ax3 = fig.add_subplot(1,3,3)
            ax3.imshow(corrected)
            plt.show()
            # fig = plt.figure()
            # ax1 = fig.add_subplot(2,3,1)
            # ax1.imshow(donut)
            # # ax1.title.set_text('input img')
            zern = preds[:-2]
            offsets = preds[-2:]
            # print(preds)
            # print("zern is: {}".format(zern))
            # print("offsets are: {}".format(preds[-2:]))
            remaining = labels.numpy().squeeze() - preds
            remaining_zern = remaining[:-2]
            remaining_offsets = remaining[-2:]

            #TODO: here's where you need to split them
            corrected = get_sted_psf(remaining_zern, offset=remaining_offsets) 
            # # corrected = normalize_img(donut) + normalize_img(get_psf(-1*outputs.numpy().squeeze()))

            fig = plt.figure()
            ax1 = fig.add_subplot(1,3,1)
            ax1.imshow(donut)
            ax1.title.set_text('input img')
            ax2 = fig.add_subplot(1,3,2)
            ax2.title.set_text('reconstructed')
            ax2.imshow(get_sted_psf(zern, offset=offsets))
            ax3 = fig.add_subplot(1,3,3)
            ax3.imshow(corrected)
            ax3.title.set_text('corrected')
            plt.show()
            
            # fig = plt.figure()
            # ax1 = fig.add_subplot(2,3,1)
            # ax1.imshow(donut)
            # ax1.title.set_text('input img')
            # ax2 = fig.add_subplot(2,3,2)
            # ax2.imshow(to_plot[1])
            # ax3 = fig.add_subplot(2,3,3)
            # ax3.imshow(to_plot[2])
            # ax4 = fig.add_subplot(2,3,4)
            # # ax4.title.set_text('reconstructed')
            # ax4.imshow(corrected[0])
            # ax5 = fig.add_subplot(2,3,5)
            # ax5.imshow(corrected[1])
            # ax6 = fig.add_subplot(2,3,6)
            # ax6.imshow(corrected[2])
            # # ax5.title.set_text('corrected')
            # ax4.title.set_text('reconstructed')
            # ax4.imshow(get_sted_psf(preds))
            # ax5 = fig.add_subplot(2,3,5)
            # ax5.imshow(corrected)
            # ax5.title.set_text('corrected')
        
            # plt.show()


def main(args):
    data_path = args.test_dataset_dir
    logdir = args.logdir
    model_store_path = args.model_store_path
    multi = args.multi   
    if multi:
        model = my_models.MultiNet()
    else:
        model = my_models.OffsetNet()


    # model.load_state_dict(torch.load(model_store_path))
    checkpoint = torch.load(model_store_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    mean, std = get_stats(data_path, batch_size=10, mode='val')
    test_dataset = my_classes.PSFDataset(hdf5_path=data_path, mode='val', transform=transforms.Compose([
        my_classes.ToTensor(), 
        my_classes.Normalize(mean=mean, std=std)]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, \
        shuffle=False, num_workers=0)


    test(model, test_loader, logdir, model_store_path)



if __name__ == "__main__":
    parser = ap.ArgumentParser(description=' ')
    parser.add_argument('test_dataset_dir', type=str, help='path to dataset')
    parser.add_argument('model_store_path', type=str, help='path to model checkpoint dir')
    
    parser.add_argument('--logdir', type=str, help='path to logging dir for optional tensorboard visualization')

    
    ARGS=parser.parse_args()

    main(ARGS)