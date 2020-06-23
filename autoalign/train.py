'''
Project: deep-sted-autoalign
Created on: Wednesday, 6th November 2019 9:47:12 am
--------
@author: hmcgovern
'''
"""Module docstring goes here."""
# standard imports
from datetime import datetime

# third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import argparse as ap
from scipy import signal
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

# local packages
import utils.my_classes as my_classes
import utils.helpers as helpers
import utils.my_models as my_models

def train(model, data_loaders, optimizer, num_epochs, logdir, device, model_store_path):

    # will return a [batchsize x 12] array of losses
    criterion = nn.MSELoss(reduction='none') 

    train_writer = SummaryWriter(logdir)
    
    model.train()
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        running_loss = 0.0
        val_loss = 0.0

        # TRAINING LOOP
        for i, (images, labels) in enumerate(data_loaders['train']):
            # print(i)
            # i is the number of batches. With a batch size of 32, for the 500 pt dataset, it's 13. for 20000 pt, it's 563.
            # if GPU is available, this allows the computation to happen there
            # ex = images.numpy()
            # NOTE: here's where you normalize it. 
            # print(images.numpy().shape)
            # print(np.min(images.numpy()[0]), np.max(images.numpy()[0]))
            # print(np.mean(images.numpy()[0]), np.std(images.numpy()[0]))
            # exit()
            # NOTE: this normalizes all the incoming images to be between 0 and 1
            # ideally, you have a dataset where that's already done, but this is a hack

            # images = torch.from_numpy(np.stack([helpers.normalize_img(i) for i in images.numpy()], axis=0))
            
            
            # print('min: {}      max: {}'.format(np.min(images.numpy()[0]), np.max(images.numpy()[0])))
            # print('mean: {}     std: {}'.format(np.mean(images.numpy()[0]), np.std(images.numpy()[0])))
            # exit()
            # print(images.size())
            # print(images.numpy().shape) # (64,1,64,64)
            # centers the images before training. 
            # images_np = np.stack([helpers.center(i, 64) for i in images.numpy().squeeze()], axis=0)
            # images = torch.from_numpy(images_np).unsqueeze(1)
            # print(images.size())
            # exit()
            images = images.to(device)
            labels = labels.to(device)
            
            # Run the forward pass
            outputs = model(images) # e.g. [32, 12] = [batch_size, output_dim]

            # no activation function on the final layer means that outputs is the weight of the final layer
            loss = criterion(outputs, labels) # MSE
            # sum of averages for each coeff position
            loss = torch.sum(torch.mean(loss, dim=0))
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # backward + optimize only in train
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() # loss.item() is the loss over a single batch
        

            total_step= len(data_loaders['train']) 
            update_num = 5
            if (i + 1) % update_num == 0: # will log to tensorboard after `update_num` batches, roughly
                # ...log the running loss
                # print('running train loss: {}'.format(running_loss))
                train_writer.add_scalar('training loss',
                            running_loss/update_num,
                            epoch * total_step + i)
                running_loss = 0.0

        # saving model state parameters so I can return to training later if necessary
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, model_store_path)
        
        
        # VALIDATION LOOP   
        model.eval()
        for i, (images, labels) in enumerate(data_loaders['val']):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad(): # drastically increases computation speed and reduces memory usage
                # Get model outputs (the predicted Zernike coefficients)
                # outputs = model(images)
                outputs = model(images)
                loss = criterion(outputs, labels) # performing  mean squared error calculation
                loss = torch.sum(torch.mean(loss, dim=0))

                # statistics logging
                val_loss += loss.item()
                total_step= len(data_loaders['val']) # number of batches
                update_num = 5
                if (i + 1) % update_num == 0:
                    # ...log the validation loss
                    # print('running val loss: {}'.format(val_loss))
                    train_writer.add_scalar('validation loss',
                                val_loss/update_num,
                                epoch * total_step + i)
                    val_loss = 0.0
    
        
    return model

def main(args):
    """Trains a CNN based on AlexNet on a custom dataset. 
    First it iterates through the data to find the mean and std & normalizes to that (helps training get off to a good start)
    Then it creates a random seed and sets np.random.seed and torch.manual_seed to it so that every time you train with the same 
        params, you get the exact same results.
    Next it instantiates custom PSFDataset objects (definition in utils.my_classes) and then, with those, instantiates pytorch DataLoader objects
    Finally, it checks if GPU is available, sets an optimizer, and starts the training/validation loop."""
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    data_path = args.dataset
    model_store_path = args.model_store_path
    logdir = args.logdir
    warm_start = args.warm_start
    

    mean, std = helpers.get_stats(data_path, batch_size)
    # Norm = my_classes.MyNormalize(mean=mean, std=std)

    # this is for reproducibility, it renders the model deterministic
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_dataset = my_classes.PSFDataset(hdf5_path=data_path, mode='train', transform=transforms.Compose([
        my_classes.ToTensor(), 
        my_classes.Normalize(mean=mean, std=std)]))
    val_dataset = my_classes.PSFDataset(hdf5_path=data_path, mode='val', transform=transforms.Compose([
        my_classes.ToTensor(), 
        my_classes.Normalize(mean=mean, std=std)]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, \
        shuffle=True, num_workers=0)
    # NOTE: my dumb butt had this as train_dataset for a LONG time
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, \
        shuffle=True, num_workers=0)

    data_loaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    for i, j in dataset_sizes.items():
        print(i, j)
    # exit()
    print('is CUDA available? {}'.format(torch.cuda.is_available()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ################################## running ###################################
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

    model = my_models.MultiNetCentered()

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # if a warm start was specified, load model and optimizer state parameters
    if args.warm_start:
        checkpoint = torch.load(warm_start)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    # train the model
    train(model, data_loaders, optimizer, num_epochs, logdir, device, model_store_path)


if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Model Hyperparameters and File I/O')
    parser.add_argument('lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('num_epochs', type=int, default=20, help='number of epochs to run')
    parser.add_argument('batch_size', type=int, default=64, help='batch size')
    
    parser.add_argument('dataset', type=str, help='path to dataset on which to train')
    parser.add_argument('model_store_path', type=str, help='path to where you want to save model checkpoints')
    parser.add_argument('--multi', action='store_true', \
        help='whether or not to use cross-sections')  
    parser.add_argument('--offset', action='store_true', \
        help='whether or not to incorporate offset') 
    parser.add_argument('--logdir', type=str, help='path to logging dir for optional tensorboard visualization')
    parser.add_argument('--warm_start', type=str, help='path to a previous checkpoint dir for a warm start')     
    ARGS=parser.parse_args()

    main(ARGS)
