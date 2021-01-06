
'''
Project: deep-sted-autoalign
Created on: Friday, 24th January 2020 9:18:54 am
--------
@author: hmcgovern
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture followed from Zhang et. al, 
    "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
    """
    def __init__(self, output_dim=13):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(8 * 8 * 64, 512)  # 64 channels, final img size 8x8
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_dim)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = x.float()
        x = F.dropout(F.max_pool2d(self.relu(self.conv1(x)), (2, 2)), p=0.1)
        x = F.dropout(F.max_pool2d(self.relu(self.conv2(x)), (2, 2)), p=0.1)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        
        x = F.max_pool2d(self.relu(self.conv5(x)), (2, 2))
        # print(a.shape) # [batch, 64, 8, 8]
        # flatten
        x = x.reshape(x.size(0), -1)
        x = F.dropout(self.relu(self.fc1(x)), p=0.2)
        x = F.dropout(self.relu(self.fc2(x)), p=0.2)
        x = self.fc3(x)
        return x


class TheUltimateModel(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture followed from Zhang et. al, 
    "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
    """
    def __init__(self, input_dim, output_dim=13, res=64, concat=False):
        super(TheUltimateModel, self).__init__()
        self.input_dim = input_dim
        self.concat = concat
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        if not concat:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # will be concatenated after this, so 64*3
        self.conv5 = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        if not concat: 
            self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # this had to be changed to be 224/2/2/2 = 28x28 final image size
        self.fc1 = nn.Linear(res//8 * res//8 * 64, 512)
        # self.fc1 = nn.Linear(8 * 8 * 64, 512)  # 64 channels, final img size 8x8
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_dim)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, img):
        
        if self.concat:
            # redefining
            self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            # if the model is only one image
            if self.input_dim == 1:
                x = img.float()
                x = F.dropout(F.max_pool2d(self.relu(self.conv1(x)), (2, 2)), p=0.1)
                x = F.dropout(F.max_pool2d(self.relu(self.conv2(x)), (2, 2)), p=0.1)
                x = self.relu(self.conv3(x))
                x = self.relu(self.conv4(x))
                
                x = F.max_pool2d(self.relu(self.conv5(x)), (2, 2))
                # print(a.shape) # [batch, 64, 8, 8]
                # flatten
                x = x.reshape(x.size(0), -1)
                x = F.dropout(self.relu(self.fc1(x)), p=0.2)
                x = F.dropout(self.relu(self.fc2(x)), p=0.2)
                x = self.fc3(x)
                return x
            # if you're using all three orthosections
            # TODO: make this independent of the arbitrary 3, could put in a whole stack of images
            elif self.input_dim == 3:
                x = img[:, 0].unsqueeze(1) # adding dim of 0 after batch dim
                y = img[:, 1].unsqueeze(1) # [batch, 1, 64, 64]
                z = img[:, 2].unsqueeze(1)

                x = x.float()
                x = F.dropout(F.max_pool2d(F.relu(self.conv1(x)), (2, 2)), p=0.1)
                x = F.dropout(F.max_pool2d(F.relu(self.conv2(x)), (2, 2)), p=0.1)
                # [batch, 32, 16, 16]
                
                x = F.relu(self.conv3(x))
                x = F.relu(self.conv4(x))
                # [32, 64, 16, 16]

                y = y.float()
                y = F.dropout(F.max_pool2d(F.relu(self.conv1(y)), (2, 2)), p=0.1)
                y = F.dropout(F.max_pool2d(F.relu(self.conv2(y)), (2, 2)), p=0.1)
                y = F.relu(self.conv3(y))
                y = F.relu(self.conv4(y))

                z = z.float()
                z = F.dropout(F.max_pool2d(F.relu(self.conv1(z)), (2, 2)), p=0.1)
                z = F.dropout(F.max_pool2d(F.relu(self.conv2(z)), (2, 2)), p=0.1)
                z = F.relu(self.conv3(z))
                z = F.relu(self.conv4(z))

                a = torch.cat((x, y, z), dim=1) # [batch, 96, 16, 16] that's what we want okay
                # print(a.shape) # [batch, 192, 16, 16]

                a = F.max_pool2d(self.relu(self.conv5(a)), (2, 2))
                # [batch, 64, 8, 8]

                # flatten
                a = a.reshape(a.size(0), -1)
                a = F.dropout(self.relu(self.fc1(a)), p=0.2)
                a = F.dropout(self.relu(self.fc2(a)), p=0.2)
                a = self.fc3(a)
                return a
            else:
                raise("Error: wrong input dimension for model")
        else: # backwards compatability
            x = img.float()
            x = F.dropout(F.max_pool2d(self.relu(self.conv1(x)), (2, 2)), p=0.1)
            x = F.dropout(F.max_pool2d(self.relu(self.conv2(x)), (2, 2)), p=0.1)
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv4(x))
            
            x = F.max_pool2d(self.relu(self.conv5(x)), (2, 2))
            # print(a.shape) # [batch, 64, 8, 8]
            # flatten
            x = x.reshape(x.size(0), -1)
            x = F.dropout(self.relu(self.fc1(x)), p=0.2)
            x = F.dropout(self.relu(self.fc2(x)), p=0.2)
            x = self.fc3(x)
            return x

class Net12(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture followed from Zhang et. al, 
    "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
    """
    def __init__(self):
        super(Net12, self).__init__()
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

class MultiNet12(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture modified from Zhang et. al, "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
    to include a multi-channelled image
    """
    def __init__(self):
        super(MultiNet12, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2) 
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.fc1 = nn.Linear(8 * 8 * 64, 512)  # 64 channels, final img size 3x8x8
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
        # x = x.view(-1, 64*8*8)
        x = F.dropout(F.relu(self.fc1(x)), p=0.2)
        x = F.dropout(F.relu(self.fc2(x)), p=0.2)
        x = self.fc3(x)
        return x

class MultiNetCat11(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture modified from Zhang et. al, "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
    to include a multi-channelled image
    """
    def __init__(self):
        super(MultiNetCat11, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2) 
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv4 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv5 = nn.Conv2d(96, 96, 3, padding=1)

        self.fc1 = nn.Linear(8 * 8 * 96, 512)  # 64 channels, final img size 3x8x8
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 11) #NOTE; this used to be 12

    def forward(self, img):
        
        x = img[:, 0].unsqueeze(1) # adding dim of 0 after batch dim
        # print(x.shape)
        # print(img[:,1].shape)
        y = img[:, 1].unsqueeze(1)
        z = img[:, 2].unsqueeze(1)
        # print(img.shape)
        # print(x.shape)
        # print(y.shape)
        # exit()
        x = x.float()
        x = F.dropout(F.max_pool2d(F.relu(self.conv1(x)), (2, 2)), p=0.1)
        x = F.dropout(F.max_pool2d(F.relu(self.conv2(x)), (2, 2)), p=0.1)
        # print(x.size()) # [4, 32, 16, 16]
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        
        y = y.float()
        y = F.dropout(F.max_pool2d(F.relu(self.conv1(y)), (2, 2)), p=0.1)
        y = F.dropout(F.max_pool2d(F.relu(self.conv2(y)), (2, 2)), p=0.1)
        # y = F.relu(self.conv3(y))
        # y = F.relu(self.conv4(y))

        z = z.float()
        z = F.dropout(F.max_pool2d(F.relu(self.conv1(z)), (2, 2)), p=0.1)
        z = F.dropout(F.max_pool2d(F.relu(self.conv2(z)), (2, 2)), p=0.1)
        # z = F.relu(self.conv3(z))
        # z = F.relu(self.conv4(z))

        a = torch.cat((x, y, z), dim=1) # [4, 96, 16, 16] that's what we want okay
        # print(a.size()) # torch.Size([4, 24576]), 24576 = 64*64*6
        # exit()
        a = F.relu(self.conv3(a))
        a = F.relu(self.conv4(a))

        a = F.max_pool2d(F.relu(self.conv5(a)), (2, 2))
        # flatten
        a = a.reshape(a.size(0), -1)
        # x = x.view(-1, 64*8*8)
        a = F.dropout(F.relu(self.fc1(a)), p=0.2)
        a = F.dropout(F.relu(self.fc2(a)), p=0.2)
        a = self.fc3(a)
        return a


class MultiNet11(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture modified from Zhang et. al, "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
    to include a multi-channelled image
    """
    def __init__(self):
        super(MultiNet11, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2) 
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.fc1 = nn.Linear(8 * 8 * 64, 512)  # 64 channels, final img size 3x8x8
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 11)

    def forward(self, x):
        
        x = x.float()
        x = F.dropout(F.max_pool2d(F.relu(self.conv1(x)), (2, 2)), p=0.1)
        x = F.dropout(F.max_pool2d(F.relu(self.conv2(x)), (2, 2)), p=0.1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        # flatten
        x = x.reshape(x.size(0), -1)
        # x = x.view(-1, 64*8*8)
        x = F.dropout(F.relu(self.fc1(x)), p=0.2)
        x = F.dropout(F.relu(self.fc2(x)), p=0.2)
        x = self.fc3(x)
        return x

class Net11(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture modified from Zhang et. al, "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
    to include a multi-channelled image
    """
    def __init__(self):
        super(Net11, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2) 
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.fc1 = nn.Linear(8 * 8 * 64, 512)  # 64 channels, final img size 3x8x8
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 11)

    def forward(self, x):
        
        x = x.float()
        x = F.dropout(F.max_pool2d(F.relu(self.conv1(x)), (2, 2)), p=0.1)
        x = F.dropout(F.max_pool2d(F.relu(self.conv2(x)), (2, 2)), p=0.1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        # flatten
        x = x.reshape(x.size(0), -1)
        # x = x.view(-1, 64*8*8)
        x = F.dropout(F.relu(self.fc1(x)), p=0.2)
        x = F.dropout(F.relu(self.fc2(x)), p=0.2)
        x = self.fc3(x)
        return x

class OffsetNet2(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture followed exactly from Zhang et. al, "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
    """
    def __init__(self):
        super(OffsetNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(8 * 8 * 64, 512)  # 64 channels, final img size 8x8
        self.fc2 = nn.Linear(512, 512)
        

        self.fc3 = nn.Linear(512, 2)

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

class OffsetNet13(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture followed exactly from Zhang et. al, "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
    """
    def __init__(self):
        super(OffsetNet13, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(8 * 8 * 64, 512)  # 64 channels, final img size 8x8
        self.fc2 = nn.Linear(512, 512)
        

        self.fc3 = nn.Linear(512, 13)

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

class MultiOffsetNet13(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture modified from Zhang et. al, "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
    to include a multi-channelled image
    """
    def __init__(self):
        super(MultiOffsetNet13, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2) 
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.fc1 = nn.Linear(8 * 8 * 64, 512)  # 64 channels, final img size 3x8x8
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 13)

    def forward(self, x):
        
        x = x.float()
        x = F.dropout(F.max_pool2d(F.relu(self.conv1(x)), (2, 2)), p=0.1)
        x = F.dropout(F.max_pool2d(F.relu(self.conv2(x)), (2, 2)), p=0.1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        # flatten
        x = x.reshape(x.size(0), -1)
        # x = x.view(-1, 64*8*8)
        x = F.dropout(F.relu(self.fc1(x)), p=0.2)
        x = F.dropout(F.relu(self.fc2(x)), p=0.2)
        x = self.fc3(x)
        return x


