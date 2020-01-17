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
import torch.optim as optim
import torchvision
import numpy as np
import argparse as ap
from scipy import signal
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import copy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import utils.my_classes as my_classes
import utils.helpers as helpers
from utils.integration import integrate


class Net(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture followed exactly from Zhang et. al, "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(8 * 8 * 64, 512)  # 64 channels, final img size 8x8
        self.fc2 = nn.Linear(512, 512)
        

        self.fc3 = nn.Linear(512, 12)

    def forward(self, x):
        
        x = x.float()
        x = F.dropout(F.max_pool2d(F.relu(self.conv1(x)), (2, 2)), p=0.1)
        x = F.dropout(F.max_pool2d(F.relu(self.conv2(x)), (2, 2)), p=0.1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        # flatten
        x = x.reshape(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.2)
        x = F.dropout(F.relu(self.fc2(x)), p=0.2)
        x = self.fc3(x)
        return x


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
            # i is the number of batches. With a batch size of 32, for the 500 pt dataset, it's 13. for 20000 pt, it's 563.
            
            # if GPU is available, this allows the computation to happen there
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
            update_num = 20
            if (i + 1) % update_num == 0: # will log to tensorboard after `update_num` batches, roughly
                # ...log the running loss
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
                outputs = model(images)
                loss = criterion(outputs, labels) # performing  mean squared error calculation
                loss = torch.sum(torch.mean(loss, dim=0))

                # statistics logging
                val_loss += loss.item()
                total_step= len(data_loaders['val']) # number of batches
                update_num = 20
                if (i + 1) % update_num == 0:
                    # ...log the validation loss
                    train_writer.add_scalar('validation loss',
                                val_loss/update_num,
                                epoch * total_step + i)
                    val_loss = 0.0
    
        
    return model


def get_stats(data_path, batch_size, mode='train'):
    """ Finding Dataset Stats for Normalization """
    dataset = my_classes.PSFDataset(hdf5_path=data_path, mode=mode, transform=my_classes.ToTensor())
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    print('mean is: {}  |   std is: {}'.format(mean, std))
    return mean, std

def main(args):
    """Trains a CNN based on AlexNet on a custom dataset. 
    First it iterates through the data to find the mean and std & normalizes to that (helps training get off to a good start)
    Then it creates a random seed and sets np.random.seed and torch.manual_seed to it so that every time you train with the same 
        params, you get the exact same results.
    Next it instantiates custom PSFDataset objects (definition in utils.my_classes) and then, with those, instantiates pytorch DataLoader objects
    Finally, it checks if GPU is available, sets an optimizer, and starts the training/validation loop."""
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    data_path = args.dataset_dir
    lr = args.lr
    logdir = args.logdir
    model_store_path = args.model_store_path
    
    if args.warm_start_path:
        warm_start_path = args.warm_start_path

    mean, std = get_stats(data_path, batch_size)
    Norm = my_classes.MyNormalize(mean=mean, std=std)

    # this is for reproducibility, it renders the model deterministic
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_dataset = my_classes.PSFDataset(hdf5_path=data_path, mode='train', transform=transforms.Compose([my_classes.ToTensor(), Norm]))
    val_dataset = my_classes.PSFDataset(hdf5_path=data_path, mode='val', transform=transforms.Compose([my_classes.ToTensor(), Norm]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, \
        shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, \
        shuffle=True, num_workers=0)



    data_loaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ################################## running ###################################
    model = Net()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # if a warm start was specified, load model and optimizer state parameters
    if args.warm_start_path:
        checkpoint = torch.load(warm_start_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    # train the model
    train(model, data_loaders, optimizer, num_epochs, logdir, device, model_store_path)


if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Model Hyperparameters and File I/O')
    parser.add_argument('num_epochs', type=int, default=2, help='number of epochs to run')
    parser.add_argument('batch_size', type=int, default=32, help='batch size')
    parser.add_argument('lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('dataset_dir', type=str, help='path to dataset')
    parser.add_argument('logdir', type=str, help='keyword for path to tensorboard logging info')
    parser.add_argument('model_store_path', type=str, help='path to model checkpoint dir')
    parser.add_argument('warm_start_path', type=str, help='path to a previous checkpoint dir for a warm start')    
    ARGS=parser.parse_args()

    main(ARGS)
