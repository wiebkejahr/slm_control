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
from torch.optim import lr_scheduler
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import copy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from my_classes import *
from helpers import * 
from integration import integrate


class Net(nn.Module):

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


def train(model, data_loaders, optimizer, scheduler, num_epochs, logdir, device, model_store_path):
    
    # for a validation loop, probably won't include
    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0
    #
    criterion = nn.MSELoss()    

    train_writer = SummaryWriter(logdir)
    
    model.train()
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        running_loss = 0.0
        val_loss = 0.0
        # batch_loss = 0.0
        # epoch_loss = 0.0
        

        # TRAINING LOOP
        for i, (images, labels) in enumerate(data_loaders['train']):
            # i is the number of batches. With a batch size of 32, for the 500 pt dataset, it's 13. for 20000 pt, it's 563.
            # if GPU is available, this allows the computation to happen there
            images = images.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Run the forward pass
            outputs = model(images) # [32, 12] = [batch_size, output_dim]
            loss = criterion(outputs, labels) # MSE
            # backward + optimize only in train
            loss.backward()
            optimizer.step()
            # decreases the learning rate, didn't seem to help in training
            # scheduler.step()

            # statistics
            running_loss += loss.item() # loss.item() is the loss over a single batch
            # batch_loss += loss.item() # redudantly logging for tensorboard _and_ terminal output
        

            total_step= len(data_loaders['train']) 
            update_num = 15
            if (i + 1) % update_num == 0: # will log to tensorboard after `update_num` batches, roughly
                # ...log the running loss
                train_writer.add_scalar('training loss',
                            running_loss/update_num,
                            epoch * total_step + i)
                running_loss = 0.0
        # epoch_loss += batch_loss / len(data_loaders['train']) # average loss over number of batches

        # # print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_train_loss, epoch_train_acc))
        # print('Train Loss: {:.4f}'.format(epoch_loss))


        # VALIDATION LOOP   
        model.eval()
        for i, (images, labels) in enumerate(data_loaders['val']):
    #         images = images.to(device)
    #         labels = labels.to(device)
            
    #         # zero the parameter gradients
    #         optimizer.zero_grad()

            with torch.no_grad(): # drastically increases computation speed and reduces memory usage
                # Get model outputs (the predicted Zernike coefficients)
                outputs = model(images)
                loss = criterion(outputs, labels) # performing  mean squared error calculation

                # statistics logging
                val_loss += loss.item()
                total_step= len(data_loaders['val']) # number of batches
                update_num = 15
                if (i + 1) % update_num == 0:
                    # ...log the validation loss
                    train_writer.add_scalar('validation loss',
                                val_loss/update_num,
                                epoch * total_step + i)
                    val_loss = 0.0
    #             # comparing a 'corrected' psf from the predicted coeffs with the ideal donut
    #             # if the correlation coefficient is above a certain threshold, it's counted as 'correct'
    #             m_vals.append(corr_coeff(outputs, images, donut=donut))
    #             # val_accuracy = val_corrects/outputs.shape[0]
    #     m_vals = np.asarray(m_vals)
    #     m_vals = m_vals.flatten()
    #     print(m_vals)

    #     epoch_acc = np.sum([m_vals > 0.97])/dataset_sizes['val'] # average accuracy over the validation set
    #     # considered correct if the correlative 
    #     print('accuracy: {}'.format(epoch_acc))
    #     #epoch_acc = val_corrects / total_step
    #     # # ...log the validation accuracy
    #     # train_writer.add_scalar('validation accuracy',
    #     #             epoch_acc,
    #     #             epoch * total_step + i)
    #     # running_loss = 0.0
        
    # #     epoch_val_loss = train_loss / dataset_sizes['val']
    # #     epoch_val_acc = train_corrects.double() / dataset_sizes['val']

    # #     print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_val_loss, epoch_val_acc))

    #     # deep copy the model
    #     if epoch_acc > best_acc:
    #         best_acc = epoch_acc
    #         best_model_wts = copy.deepcopy(model.state_dict())
    #         # average loss over the epoch
    #         # epoch_loss = epoch_loss / len(data_loaders[phase]) # I'm thinking this is the number of examples in the batch
    #         # check to make sure
            
    # #         # ...log the epoch loss
    # #         # train_writer.add_scalar('epoch loss',
    # #         #             epoch_loss, epoch+1)

    # print()
    
    # print('Best val Acc: {:4f}'.format(best_acc))
    
    # # load best model weights
    # model.load_state_dict(best_model_wts)
    
    torch.save(model.state_dict(), model_store_path)
    return model


def get_stats(data_path, batch_size, mode='train'):
    """ Finding Dataset Stats for Normalization """
    dataset = PSFDataset(hdf5_path=data_path, mode=mode, transform=my_classes.ToTensor())
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
    """Docstring to go here."""
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    data_path = args.dataset_dir
    lr = args.lr
    logdir = args.logdir
    model_store_path = args.model_store_path

    mean, std = get_stats(data_path, batch_size)
    Norm = MyNormalize(mean=mean, std=std)
    # TODO: write this to a text file along with the hparams to be loaded during eval

    train_dataset = PSFDataset(hdf5_path=data_path, mode='train', transform=transforms.Compose([ToTensor(), Norm]))
    val_dataset = PSFDataset(hdf5_path=data_path, mode='val', transform=transforms.Compose([ToTensor(), Norm]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, \
        shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, \
        shuffle=True, num_workers=0)



    data_loaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ################################## running ###################################
    model = Net()
    print(model)
  
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # train the model
    # model, data_loaders, optimizer, scheduler, num_epochs, logdir, device, model_store_path):
    train(model, data_loaders, optimizer, scheduler, num_epochs, logdir, device, model_store_path)
    # return Norm.mean, Norm.std


if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Model Hyperparameters and File I/O')
    parser.add_argument('num_epochs', type=int, default=2, help='number of epochs to run')
    parser.add_argument('batch_size', type=int, default=32, help='batch size')
    parser.add_argument('lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('dataset_dir', type=str, help='path to dataset')
    parser.add_argument('logdir', type=str, help='keyword for path to tensorboard logging info')
    parser.add_argument('model_store_path', type=str, help='path to model checkpoint dir')
    
    ARGS=parser.parse_args()

    main(ARGS)
