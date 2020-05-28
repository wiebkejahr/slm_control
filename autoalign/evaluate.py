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
            # image = images.numpy().squeeze()
            print(images.numpy().shape) # (1, 3, 64, 64)
            print(np.max(images.numpy()), np.min(images.numpy())) # 5.04, -0.96
            images = torch.from_numpy(np.stack([normalize_img(i) for i in images.numpy()], axis=0))
            print(images.numpy().shape)
            print(np.max(images.numpy()), np.min(images.numpy())) # 1, 0 
            # example for syntax
            # img2 = np.stack([add_noise(i) for i in img], axis=0)
            # exit()
            # NOTE: goal here is to normalize the input image and see if the prediction goes to trash
            preds = model(images).numpy().squeeze()

            # zern = preds[:-2]
            # offset = preds[-2:]
            zern = preds
            offset=[0,0]
            reconstructed = get_sted_psf(coeffs=zern, offset_label=offset, multi=True)
            print('reconstructed')
            print(np.max(reconstructed), np.min(reconstructed))
            # print(preds)
            # print("zern is: {}".format(zern))
            # print("offsets are: {}".format(preds[-2:]))
            remaining = labels.numpy().squeeze() - preds
            # remaining_zern = remaining[:-2]
            # remaining_offsets = remaining[-2:]

            corrected = get_sted_psf(coeffs=remaining, multi=True)
            print('corrected')
            print(np.max(corrected), np.min(corrected))
            #TODO: here's where you need to split them
            #NOTE: the offset used to be a boolean
            # corrected = get_sted_psf(coeffs=remaining_zern, offset_label=remaining_offsets, multi=True) 
            # # corrected = normalize_img(donut) + normalize_img(get_psf(-1*outputs.numpy().squeeze()))

            # fig = plot_xsection(donut)
            # plt.show()
            # print(reconstructed.shape)
            # exit()
            fig2 = plot_xsection_eval(images.numpy().squeeze(), reconstructed, corrected)
            plt.show()
            


def main(args):
    data_path = args.test_dataset_dir
    logdir = args.logdir
    model_store_path = args.model_store_path
    print(model_store_path) 
    # exit()
    if args.multi:
        if args.offset:
            model = my_models.MultiOffsetNet()
        else:
            model = my_models.MultiNetLtd()
    else:
        if args.offset:
            model = my_models.OffsetNet()
        else:
            model = my_models.Net()
    print(model)
    
    # NOTE: this part needs work. determine which model to use from loading the data and checking the shape
    # multi = args.multi   
    # if multi:
    #     model = my_models.MultiNet()
    # else:
    #     model = my_models.OffsetNet()


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
    parser.add_argument('--multi', action='store_true', \
        help='whether or not to use cross-sections')  
    parser.add_argument('--offset', action='store_true', \
        help='whether or not to incorporate offset')
    parser.add_argument('--logdir', type=str, help='path to logging dir for optional tensorboard visualization')

    
    ARGS=parser.parse_args()

    main(ARGS)