'''
Project: deep-sted-autoalign
Created on: Wednesday, 6th November 2019 9:47:12 am
--------
@author: hmcgovern
'''
"""Module docstring goes here."""
# standard imports
from datetime import datetime
import json

# third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import argparse as ap
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

# local packages
import utils.my_classes as my_classes
import utils.helpers as helpers
import utils.my_models as my_models

def train(model, data_loaders, optimizer, num_epochs, logdir, device, model_store_path):

    # will return a [batchsize x 12] array of losses
    criterion = nn.MSELoss(reduction='none') 

    # for tensorboard logging
    train_writer = SummaryWriter(logdir)
    
    #puts model into train mode
    model.train()
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        running_loss = 0.0
        val_loss = 0.0

        # TRAINING LOOP
        for i, sample in enumerate(data_loaders['train']):
            # i is the number of batches. With a batch size of 32, for the 500 pt dataset, it's 13. for 20000 pt, it's 563.
            images = sample['image']
            labels = sample['label']
            

            # if GPU is available, this allows the computation to happen there
            images = images.to(device)
            labels = labels.to(device)

            # Run the forward pass
            outputs = model(images) # e.g. [32, 12] = [batch_size, output_dim]
            print(outputs.numpy().shape)
            exit()
            # no activation function on the final layer means that the output IS the weight of the final layer
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
            update_num = 10
            if (i + 1) % update_num == 0: # will log to tensorboard after `update_num` batches, roughly
                # ...log the running loss
                # print('running train loss: {}'.format(running_loss))
                train_writer.add_scalar('training_loss',
                            running_loss/update_num,
                            epoch * total_step + i)
                running_loss = 0.0

        # saving model state parameters so I can return to training later if necessary
        # torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict()
        #         }, model_store_path)
        torch.save(model, model_store_path) 
        
        # VALIDATION LOOP   
        model.eval()
        for i, sample in enumerate(data_loaders['val']):
            
            images = sample['image']
            labels = sample['label']

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
                update_num = 10
                if (i + 1) % update_num == 0:
                    # ...log the validation loss
                    # print('running val loss: {}'.format(val_loss))
                    train_writer.add_scalar('validation_loss',
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
    
    # this is for reproducibility, it renders the model deterministic
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # tsfms = transforms.Compose([my_classes.ToTensor(), my_classes.Normalize(mean=mean, std=std)])
    tsfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    train_dataset = my_classes.PSFDataset(hdf5_path=data_path, mode='train', transform=tsfms)
    val_dataset = my_classes.PSFDataset(hdf5_path=data_path, mode='val', transform=tsfms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, \
        shuffle=True, num_workers=0)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, \
        shuffle=True, num_workers=0)
    
    data_loaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    for i, j in dataset_sizes.items():
        print(i, j)
    print('is CUDA available? {}'.format(torch.cuda.is_available()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ################################## running ###################################
    
    if args.multi:
        in_dim = 3
    else:
        in_dim = 1

    out_dim = 0
    if args.zern: out_dim += 11
    if args.offset: out_dim += 2
    print('out_dim is: ')
    print(out_dim)


    # if args.multi:
    #     if args.offset:
    #         model = my_models.MultiOffsetNet13()
    #         name = 'MultiOffsetNet13()'
    #     else:
    #         model = my_models.MultiNet11()
    #         name = 'MultiOffsetNet11()'
    # else:
    #     if args.offset:
    #         model = my_models.OffsetNet13()
    #         name = 'OffsetNet13()'
    #     else:
    #         model = my_models.Net11()
    #         name = 'OffsetNet13()'

    model = my_models.MyNet(input_dim=in_dim, output_dim=out_dim)
    # print(summary(model, input_size=(in_dim, 64, 64), batch_size=-1))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # if a warm start was specified, load model and optimizer state parameters
    if args.warm_start:
        checkpoint = torch.load(warm_start)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    

    # grabbing just the model name with this split
    name = model_store_path.split('/')[-1]
    model_params = {'multi': args.multi,
                    'offset': args.multi,
                    'zern': args.zern,
                    'learning rate': args.lr,
                    'epochs': args.num_epochs,
                    'batch size': args.batch_size,
                    'dataset': args.dataset}

    # this works to make the json once, but will always overwrite
    with open("model_params.json", "w") as f:
        data = json.dumps({name: model_params}, indent=4)
        f.write(data)

    # train the model
    train(model, data_loaders, optimizer, num_epochs, logdir, device, model_store_path)


if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Model Hyperparameters and File I/O')
    parser.add_argument('lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('num_epochs', type=int, default=20, help='number of epochs to run')
    parser.add_argument('batch_size', type=int, default=64, help='batch size')
    
    parser.add_argument('dataset', type=str, help='path to dataset on which to train')
    parser.add_argument('model_store_path', type=str, help='path to where you want to save model checkpoints')
    parser.add_argument('--multi', type=int, default=0, \
        help='whether or not to use cross-sections')  
    parser.add_argument('--offset', type=int, default=0, \
        help='whether or not to incorporate offset')
    parser.add_argument('--zern', type=int, default=1, \
        help='whether or not to include optical aberrations') 
    parser.add_argument('--logdir', type=str, help='path to logging dir for optional tensorboard visualization')
    parser.add_argument('--warm_start', type=str, help='path to a previous checkpoint dir for a warm start')     
    ARGS=parser.parse_args()

    main(ARGS)
