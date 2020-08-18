
'''
Project: deep-sted-autoalign
Created on: Friday, 24th January 2020 9:18:54 am
--------
@author: hmcgovern
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# class MyResNe50(models.resnet.ResNet):
#     def __init__(self, training=True):
#         super(MyResNe50, self).__init__(block=models.resnet.Bottleneck,
#                                         layers=[3, 4, 6, 3], 
#                                         groups=32, 
#                                         width_per_group=4)
#         self.fc = nn.Linear(2048, 1)

class ISOnet(nn.Module):
    def __init__(self):
        super(ISOnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=7)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))

class MyAlexNet(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(MyAlexNet, self).__init__()
        self.model = models.alexnet(pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x):


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
        super(MultiNet, self).__init__()
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

class MultiOffsetNet14(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture modified from Zhang et. al, "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
    to include a multi-channelled image
    """
    def __init__(self):
        super(MultiOffsetNet14, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2) 
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.fc1 = nn.Linear(8 * 8 * 64, 512)  # 64 channels, final img size 3x8x8
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 14)

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

