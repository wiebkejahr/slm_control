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
from torch.optim import lr_scheduler
import torchvision
import torchvision.models as models

import numpy as np
import argparse as ap
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import copy

# local packages
import utils.my_classes as my_classes
import utils.helpers as helpers
import utils.my_models as my_models
import utils.resnet as resnet

def train(model, data_loaders, optimizer, exp_lr_scheduler, criterion, num_epochs, logdir, device, model_store_path, dataset_sizes):

    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1e4
    train_writer = SummaryWriter(logdir)
    
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
        
            running_loss = 0.0
            running_corrects = 0

            # TRAINING LOOP, iterating over data
            for i, sample in enumerate(data_loaders[phase]):
                # i is the number of batches. With a batch size of 32, for the 500 pt dataset, it's 13. for 20000 pt, it's 563.
                images = sample['image']
                labels = sample['label']
                
                # if GPU is available, this allows the computation to happen there
                images = images.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # Run the forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images.float()) # e.g. [32, 12] = [batch_size, output_dim]
                    # no activation function on the final layer means that the output IS the weight of the final layer
                    loss = criterion(outputs, labels) # MSE
                    # print('loss shape: {}'.format(loss.shape)) # [64, 11]
                    
                    # # sum of averages for each coeff position
                    loss = torch.mean(loss, dim=0)
                    # print('loss shape: {}'.format(loss.shape))
                    loss = torch.sum(loss)
                    # print('loss shape: {}'.format(loss.shape)) 
                    # exit()

                    if phase == 'train':
                        # backward + optimize only in train
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()*images.size(0) # loss.item() is the loss over a single batch
                close = np.isclose(outputs.detach().numpy(), labels, rtol=0.1)
                close = np.sum(close, axis=1)
                # counts how many labels are guessed completely correctly within the tolerance
                running_corrects += np.count_nonzero(close == 11)
            if phase == 'train':
                exp_lr_scheduler.step()
                # saving model state parameters so I can return to training later if necessary
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, model_store_path)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            

            # deep copy the model
            # NOTE: OH, DUH! this makes a huge difference, it's never being updated from the initial params
            if phase == 'val' and epoch_loss < best_loss:
                print('loss decreased...updating best model weights')
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # saving model state parameters so I can return to training later if necessary
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, model_store_path)
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
    # exit()
    # Norm = my_classes.MyNormalize(mean=mean, std=std)
    # this is for reproducibility, it renders the model deterministic
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # tsfms = transforms.Compose([my_classes.Center(), my_classes.Normalize(mean=mean, std=std), my_classes.Noise(), my_classes.ToTensor()])
    # tsfms = transforms.Compose([my_classes.Noise(), my_classes.Center(), my_classes.ToTensor(), my_classes.Normalize(mean=mean, std=std)])
    
    # TODO: set image between [0,1] so you can use transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) like ImageNet
    # transforms.RandomHorizontalFlip(),
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # tsfms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), my_classes.ToTensor(), my_classes.Normalize(mean=mean, std=std), my_classes.Noise(bgnoise=2, poiss=350)])
    # tsfms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), my_classes.Normalize(mean=mean, std=std), my_classes.Noise(bgnoise=2, poiss=350)])
    
    tsfms = transforms.Compose([transforms.ToTensor(), my_classes.Normalize(mean=mean, std=std), my_classes.Noise(bgnoise=2, poiss=350)])
    
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


    # for i, sample in enumerate(data_loaders['train']):
    #     print(sample['image'].shape)
    #     plt.figure()
    #     plt.imshow(sample['image'].squeeze()[0])
    #     plt.show()
    # exit()
    ################################## running ###################################
    
    # # TODO: the above method of modifying the dataset after creation also gives a streamlined
    # # way of choosing the model based on the features. For now just override, as below
    # if args.multi:
    #     if args.offset:
    #         model = my_models.MultiOffsetNet()
    #     else:
    #         model = my_models.MultiNet11()
    # else:
    #     if args.offset:
    #         model = my_models.OffsetNet13()
    #     else:
    #         model = my_models.Net11()

    model = models.alexnet(pretrained=False, num_classes=11)
    first_conv_layer = [nn.Conv2d(1,3, kernel_size=3, stride=1, padding=1, bias=True)]
    first_conv_layer.extend(list(model.features))
    model.features = nn.Sequential(*first_conv_layer )
    # num_ftrs = model..in_features
    # model.fc = nn.Linear(num_ftrs, 11)
    # print(model)
    # exit()


    # print(model)
    # criterion = nn.MSELoss(reduction='none') 
    criterion = nn.MSELoss(reduction='none') 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # if a warm start was specified, load model and optimizer state parameters
    if args.warm_start:
        checkpoint = torch.load(warm_start)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # train the model
    model = train(model, data_loaders, optimizer, exp_lr_scheduler, criterion, num_epochs, logdir, device, model_store_path, dataset_sizes)
    # saving model state parameters so I can return to training later if necessary

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
