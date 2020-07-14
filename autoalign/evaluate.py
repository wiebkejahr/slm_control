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
    
    # Test the model
    model.eval()
    for i, (images, labels) in enumerate(test_loader): # i is 0 when batch_size is 1
    
        with torch.no_grad(): # drastically increases computation speed and reduces memory usage
            
            # NOTE: goal here is to normalize the input image and see if the prediction goes to trash
            # plt.figure(1)
            # xy = images.numpy().squeeze()[0]
            # plt.imshow(xy)
            # com1 = get_CoM(xy)
            # plt.scatter(com1[0], com1[1], c='r')
            # # xtilt, ytilt = calc_tip_tilt(xy, abberior=False)
            # # # plt.scatter(31.5+ xtilt, 31.5+ytilt, c='r')
            # # # plt.show()
            # # tiptilt = create_phase_tip_tilt([xtilt, ytilt])
            # # # print(len(labels.numpy()))
            # # # exit()
            # # corrected = get_sted_psf(coeffs=labels.numpy().squeeze(), multi=True, tiptilt = tiptilt)
            # corrected = center(xy, labels.numpy().squeeze())
            # calc_tip_tilt(corrected[0], abberior=False)
            # plt.figure(2)
            # plt.imshow(corrected[0])
            # com2 = get_CoM(corrected[0])
            # plt.scatter(com2[0], com2[1], c='r')

            # # plot_xsection(corrected, name='new')
            # # plt.figure(4)
            # # plot_xsection(images.numpy().squeeze(), name = 'old')
            # plt.show()
            # exit()
            
            ################
            outputs = model(images)
            preds = outputs.numpy().squeeze()
            # print(labels.numpy().squeeze())
            # print(preds)
            # # exit()

            reconstructed = get_sted_psf(coeffs=preds, multi=True)
            # print(np.min(reconstructed), np.max(reconstructed))
            # print(np.mean(reconstructed), np.std(reconstructed))
            correction = create_phase(coeffs=(-1.)*preds, defocus=False)
            print(np.mean(correction), np.std(correction))
            # this is correction via phasemask
            corrected = get_sted_psf(coeffs=labels.numpy().squeeze(), multi=True, correction=correction)
            # plt.figure(1)
            # plt.imshow(correction)
            # plt.colorbar()
            # # plt.show()
            # plt.figure(2)
            # test = create_phase(coeffs=labels.numpy().squeeze(), defocus=False)
            # plt.imshow((-1.)*test)
            # print(np.mean(test), np.std(test))
            # plt.colorbar()
            # plt.show()

            # exit()
            
            # old way
            remaining = labels.numpy().squeeze() - preds
            

            corrected_old = get_sted_psf(coeffs=remaining, multi=True)

            plot_xsection_eval(images.numpy().squeeze(), reconstructed, corrected)
            
            plot_xsection(corrected_old, name='old way')
            # fig2 = plot_xsection_eval(images.numpy().squeeze(), reconstructed, corrected)
            plt.show()

            ###########
            



def main(args):
    data_path = args.test_dataset_dir
    logdir = args.logdir
    model_store_path = args.model_store_path
    # print(model_store_path) 
    # exit()
    if args.multi:
        if args.offset:
            model = my_models.MultiOffsetNet()
        else:
            model = my_models.MultiNet()
    else:
        if args.offset:
            model = my_models.OffsetNet()
        else:
            model = my_models.Net()
    
    # model = my_models.MultiNetCentered()
    model = my_models.MultiNetCat()
    # print(model)

    
    # NOTE: this part needs work. determine which model to use from loading the data and checking the shape
    # multi = args.multi   
    # if multi:
    #     model = my_models.MultiNet()
    # else:
    #     model = my_models.OffsetNet()


    checkpoint = torch.load(model_store_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    mean, std = get_stats(data_path, batch_size=10, mode='test')
    test_dataset = my_classes.PSFDataset(hdf5_path=data_path, mode='test', transform=transforms.Compose([
        my_classes.ToTensor(), 
        my_classes.Normalize(mean=mean, std=std)]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, \
        shuffle=False, num_workers=0)


    test(model, test_loader, logdir, model_store_path)



if __name__ == "__main__":
    parser = ap.ArgumentParser(description=' ')
    parser.add_argument('test_dataset_dir', type=str, help='path to dataset')
    parser.add_argument('model_store_path', type=str, help='path to model checkpoint dir')
    parser.add_argument('--multi', action='store_true', \
        help='whether or not to use cross-sections')  
    parser.add_argument('--offset', action='store_true', \
        help='whether or not to incorporate offset')
    parser.add_argument('--logdir', type=str, help='path to logging dir for optional tensorboard visualization')

    
    ARGS=parser.parse_args()

    main(ARGS)