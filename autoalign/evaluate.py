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
    
    # preds = get_preds(coeffs)
    # preds = torch.from_numpy(preds).unsqueeze(3)

    # test_writer.add_images('Predicted Images', preds, 0, dataformats='NHWC')    
    test_writer.close()

    return None


def test(model, test_loader, logdir, model_store_path, multi, offset):
    
    # logdir_test = logdir + '/test'
    # test_writer = SummaryWriter(log_dir=logdir_test)
    
    # Test the model
    model.eval()
    
    correlations = []
    MSE = []
    
    for i, sample in enumerate(test_loader): # i is 0 when batch_size is 1
        images = sample['image']
        labels = sample['label']
        
        
        #log_images(logdir, images, labels)
        # print(images)#, images, labels)
        # exit()
        
        with torch.no_grad(): # drastically increases computation speed and reduces memory usage

            
        ################
            
            def get_preds(multi, offset):
                # gets preds
                outputs = model(images)
                if offset:
                    tmp = outputs.numpy().squeeze()
                    preds = tmp[:-2]
                    offset = tmp[-2:]
                else:
                    preds = outputs.numpy().squeeze()
                    offset = [0,0]
                return preds, offset

            print(multi, offset)
            preds, offset = get_preds(multi, offset)
            print(len(preds))
            reconstructed = get_sted_psf(coeffs=preds, multi=multi, offset_label=offset)
            corrected = get_sted_psf(coeffs=preds, multi=multi, offset_label=offset, corrections=[np.asarray(create_phase(coeffs=(-1.)*preds))])
            correlation = corr_coeff(corrected, multi=multi)
            
            #exit()
            so_far = -1
            while correlation > so_far:
                # while it is not optimized, if it comes accross a higher correlation, work down and switch out preds
                new_preds, new_offset = get_preds(multi, offset)
                print(len(new_preds))
                exit()
                new_corrected = get_sted_psf(coeffs=labels.numpy().squeeze(), multi=multi, offset_label=new_offset,\
                    corrections=[create_phase(coeffs=(-1.)*new_preds)])
                new_correlation = corr_coeff(new_corrected, multi=multi)
                if new_correlation > correlation:
                    so_far = correlation
                    correlation = new_correlation
                    print('upper: {}, lower: {}'.format(correlation, so_far))
                    preds = new_preds
                else:
                    correlations.append(correlation)
                    MSE.append(mean_squared_error(labels.numpy().squeeze(), preds))
                    # print('final correlation: {}'.format(correlation))
                    break
                    
            # plt.figure()
            # plt.plot(np.arange(3, 14), labels.numpy().squeeze())
            # plt.plot(np.arange(3, 14), preds, linestyle='dashed')
            # plt.show()
            # print(mean_squared_error(labels.numpy().squeeze(), preds))
            # print('finished this example \n')
            # exit()
            
    print(len(correlations))
    print(np.mean(correlations))
    print(np.mean(MSE))


        


def main(args):
    data_path = args.test_dataset_dir
    logdir = args.logdir
    model_store_path = args.model_store_path


    # if args.multi:
    #     if args.offset:
    #         model = my_models.MultiOffsetNet13()
    #     else:
    #         model = my_models.MultiNet11()
    # else:
    #     if args.offset:
    #         model = my_models.OffsetNet()
    #     else:
    #         model = my_models.Net()
    

    # checkpoint = torch.load(model_store_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model = torch.load(model_store_path)
    
    # mean, std = get_stats(data_path, batch_size=10, mode='test')
    test_dataset = my_classes.PSFDataset(hdf5_path=data_path, mode='test', transform=transforms.Compose([
        my_classes.ToTensor()])) 
        #, my_classes.Normalize(mean=mean, std=std)]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, \
        shuffle=False, num_workers=0)

    # batch_size = matrix_loss.shape[0]
    #             # print(batch_size) # 20
                
    #             metric = matrix_loss.numpy() < 1e-4
    #             accuracy = np.sum(metric)/(batch_size*13)

    # print(next(iter(test_loader)))
    # exit()
    
    with torch.no_grad():
        model.eval()
        
        for i, sample in enumerate(test_loader): # i is 0 when batch_size is 1
            images = sample['image']
            labels = sample['label']

            outputs = model(images)

            criterion = nn.MSELoss(reduction='none')
            matrix_loss = criterion(outputs, labels) # performing  mean squared error calculation
            avg_loss_per_coeff = torch.mean(matrix_loss, dim=0)
            loss = torch.sum(avg_loss_per_coeff)

            batch_size = matrix_loss.shape[0]
            # print(batch_size) # 1
            
            metric = matrix_loss.numpy() < 1e-4
            accuracy = np.sum(metric)/(batch_size*13)
            # print(matrix_loss)
            
            print('Accuracy: {}'.format(accuracy))


    # test(model, test_loader, logdir, model_store_path, args.multi, args.offset)



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