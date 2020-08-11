# adapted from https://github.com/usuyama/pytorch-unet

from helpers import generate_random_data
import h5py
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils

import torch
import torch.nn as nn
from torchvision import models

from collections import defaultdict
import torch.nn.functional as F

class PhasemaskDataset(Dataset):
    def __init__(self, path=None, transform=None):
        # self.file = h5py.File(hdf5_path, "r")
        
        # if mode =='train':
        #     self.input_images = self.file['train_img']
        #     self.target_masks = self.file['train_mask']
        # elif mode == 'val':
        #     self.input_images = self.file['val_img']
        #     self.target_masks = self.file['val_mask']
        # elif mode == 'test':
        #     self.input_images = self.file['test_img']
        #     self.target_masks = self.file['test_mask']
        
        # self.input_images, self.target_masks = generate_random_data(res=192, count=count)
        self.file = path

        with open(path, 'rb') as f:
            my_dict = pickle.load(f)
        self.input_images = my_dict['images']
        self.input_images = torch.stack([torch.Tensor(i) for i in self.input_images])

        self.target_masks = my_dict['masks']
        self.target_masks = torch.stack([torch.Tensor(i) for i in self.target_masks])
    
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]


def make_data():
    # use the same transformations for train/val in this example
    # NOTE: used to also have totensor, but I'm not using PIL images
    trans = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet

    # train_set = PhasemaskDataset(count=1, transform = trans)
    # val_set = PhasemaskDataset(count=1, transform = trans)
    # train_set = PhasemaskDataset('./test.hdf5', mode='train', transform = trans)
    # val_set = PhasemaskDataset('./test.hdf5', mode='val', transform = trans)
    train_set = PhasemaskDataset(path='./trainRGB_5.pkl', transform = trans)
    val_set = PhasemaskDataset(path='./valRGB_5.pkl', transform = trans)
    
    # print(len(train_set))
    image_datasets = {
        'train': train_set, 'val': val_set
    }

    batch_size = 1

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    return dataloaders



def see_data(dataloaders):
    
    # def reverse_transform(inp):
    #     inp = inp.numpy().transpose((1, 2, 0))
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    #     inp = std * inp + mean
    #     inp = np.clip(inp, 0, 1)
    #     inp = (inp * 255).astype(np.uint8)

    #     return inp

    # Get a batch of training data
    for i, sample in enumerate(dataloaders['train']):
        # print(i)
        plt.figure()
        plt.subplot(121)
        plt.imshow(sample[0][0].numpy()[0])
        plt.subplot(122)
        plt.imshow(sample[1][0].numpy()[0])
        plt.show()


############ model stuff ##########

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

########## now this is defining the training loop #############

# NOTE: I have no clue what's going on in this loss fn
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()




def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, scheduler, num_epochs=25):
    logdir='./autoalign/runs/best_unet_model'
    train_writer = SummaryWriter(logdir)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for i, sample in enumerate(dataloaders[phase]):
                
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # NOTE: this used to be in the first if phase == "train"
                        # but pytorch gave me a warning it should be the other way
                        scheduler.step()
                        for param_group in optimizer.param_groups:
                            print("LR", param_group['lr'])

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            
            train_writer.add_scalar('train_loss',
                            epoch_loss, i)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, './best_unet_model2.pth')
    return model

########## eval stuff #############
def test(model):

    model.eval()   # Set model to the evaluation mode

    # Create another simulation dataset for test
    trans = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    test_dataset = PhasemaskDataset('./testRGB_5.pkl', transform = trans)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Get the first batch
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Predict
    pred = model(inputs)
    # The loss functions include the sigmoid function.
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    print(pred.shape) # (1, 3, 192, 192)
    
    plt.figure()
    plt.subplot(131)
    plt.imshow(inputs.numpy()[0][0])
    plt.subplot(132)
    plt.imshow(labels.numpy()[0][0])
    plt.subplot(133)
    plt.imshow(pred[0][0])
    plt.show()
    exit()
    # Change channel-order and make 3 channels for matplot
    
    # input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

    # # Map each channel (i.e. class) to each color
    # target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
    # pred_rgb = [helper.masks_to_colorimg(x) for x in pred]

    # helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])

if __name__ == "__main__":

    ########### this is actually training ############
    import torch
    import torch.optim as optim
    from torch.optim import lr_scheduler
    import time
    import copy

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    dataloaders = make_data() 
    # see_data(dataloaders)

    

    # # freeze backbone layers
    # #for l in model.base_layers:
    # #    for param in l.parameters():
    # #        param.requires_grad = False
    # this seems to be the channel number?
    num_class = 3
    model = ResNetUNet(num_class).to(device)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    # model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=10)
    
    
    # torch.load
    test(model)