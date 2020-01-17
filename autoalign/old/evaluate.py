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
from utils.my_classes import PSFDataset, ToTensor, MyNormalize
from cnn import Net


from utils.integration import integrate
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
    
    # load the model
    model.load_state_dict(torch.load(model_store_path))

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
            
            # aber_img, corrected_img = corr_coeff(outputs, images, donut=donut)
            # checks the correlation between the ideal donut PSF and that of the "corrected" image
            # corrcoeff = np.corrcoef(donut.flat, corrected_img.flat)[0][1]
            # if less than 0.97, takes the corrected image as the new input to the model and repeats
            # while corrcoeff < 0.93:
            #     print('in loop')
            #     new_input = torch.from_numpy(corrected_img).unsqueeze(0).unsqueeze(0)
            #     new_output = model(new_input)
            #     _, corrected_img = corr_coeff(new_output, corrected_img, donut=donut)
            #     corrcoeff = np.corrcoef(donut.flat, corrected_img.flat)[0][1]
            #     print(corrcoeff)
            # print('convergenge reached!')


def corr_coeff(outputs, images, donut):
    m_vals = []
    for j in range(1):
    # for j in range(outputs.shape[0]): # for each coeff list in the batch
        pred_coeffs = outputs[j].numpy() # [32, 12]
        # pred_coeffs = outputs[j]
        if type(images[j]) != np.ndarray: # then need to cut out batch and channel dim, n
            aber_img = images[j].squeeze().numpy()
        else:
            aber_img = images
        aber_img = normalize_img(aber_img)
        #
        corrections_neg = get_psf(-pred_coeffs) # NOTE: this line is the bottleneck
        corrections_pos = get_psf(pred_coeffs)
        #
        # NOTE: something is still funky with the signs, so check and see which correction makes it better
        
        corrected_img_neg = aber_img + corrections_neg
        corrected_img_pos = aber_img + corrections_pos
        if np.corrcoef(donut.flat, corrected_img_neg.flat)[0][1] > np.corrcoef(donut.flat, corrected_img_pos.flat)[0][1]:
            corrected_img = corrected_img_neg
        else:
            corrected_img = corrected_img_pos
        # print(np.max(corrections), np.min(corrections))
        # print(aber_img.shape)
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(donut)
        ax1.title.set_text('donut')
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(aber_img)
        ax2.title.set_text('input image')
        ax3 = fig.add_subplot(2,2,3)
        ax3.imshow(normalize_img(get_psf(pred_coeffs)))
        ax3.title.set_text('predicted psf')
        ax4 = fig.add_subplot(2,2,4)
        ax4.imshow(corrected_img)
        ax4.title.set_text('corrected image')
        plt.show()
        
        # print(np.corrcoef(donut.flat, corrected_img.flat))
            
        # corrected_img = torch.from_numpy(get_psf(-pred_label)).unsqueeze(0) # torch.Size([1, 64, 64])
    #     corrected_imgs.append(corrected_img)
    # corrected_imgs = torch.stack(corrected_imgs) # [32, 1, 64, 64]
    # print(corrected_imgs.shape)
    
    # loss = criterion(pred_imgs, images) # performing  mean squared error calculation
    return aber_img, corrected_img



def main(args):
    data_path = args.test_dataset_dir
    logdir = args.logdir
    model_store_path = args.model_store_path
    model = Net()

    # Norm = MyNormalize(mean=0.1916, std=0.2246)
    # for newest
    Norm = MyNormalize(mean=0.2357, std=0.2236)
    
    test_dataset = PSFDataset(hdf5_path=data_path, mode='val', transform=transforms.Compose([ToTensor(), Norm]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=50, \
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