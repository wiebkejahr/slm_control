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
    
    # with open('full.txt', 'r') as fname:
    #     full = [list(eval(i.rstrip())) for i in fname.readlines()]

    ideal_coeffs = np.asarray([0.0]*12)
    # donut = get_psf(ideal_coeffs, res=64, multi=True) # (64,64)

    # Test the model
    model.eval()
    for i, (images, labels) in enumerate(test_loader): # i is 0 when batch_size is 1
    
        with torch.no_grad(): # drastically increases computation speed and reduces memory usage
            
            # NOTE: goal here is to normalize the input image and see if the prediction goes to trash
            images = torch.from_numpy(np.stack([normalize_img(i) for i in images.numpy()], axis=0))

            # example for syntax
            # img2 = np.stack([add_noise(i) for i in img], axis=0)
            
            # Get model outputs (the predicted Zernike coefficients)
            outputs = model(images[:,0].unsqueeze(1), 
                            images[:,1].unsqueeze(1),
                            images[:,2].unsqueeze(1))

            preds = outputs.numpy().squeeze()

            # zern = preds[:-2]
            # offset = preds[-2:]
            # zern = preds

            offset=[0,0]
            reconstructed = get_sted_psf(coeffs=preds, multi=True)

            remaining = labels.numpy().squeeze() - preds
            # remaining_zern = remaining[:-2]
            # remaining_offsets = remaining[-2:]

            corrected = get_sted_psf(coeffs=remaining, multi=True)

            fig2 = plot_xsection_eval(images.numpy().squeeze(), reconstructed, corrected)
            plt.show()



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
            model = my_models.MultiNetShift()
    else:
        if args.offset:
            model = my_models.OffsetNet()
        else:
            model = my_models.Net()
    
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