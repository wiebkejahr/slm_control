
'''
Project: deep-sted-autoalign
Created on: Friday, 24th January 2020 9:18:54 am
--------
@author: hmcgovern
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture followed from Zhang et. al, 
    "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
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

class MultiNet(nn.Module):
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

class MultiNetLtd(nn.Module):
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
        self.fc3 = nn.Linear(512, 7) # NOTE: this is the only difference

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

class OffsetNet(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture followed exactly from Zhang et. al, "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
    """
    def __init__(self):
        super(OffsetNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(8 * 8 * 64, 512)  # 64 channels, final img size 8x8
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
        x = F.dropout(F.relu(self.fc1(x)), p=0.2)
        x = F.dropout(F.relu(self.fc2(x)), p=0.2)
        x = self.fc3(x)
        return x

class MultiOffsetNet(nn.Module):
    """
    A simple CNN based on AlexNet
    Architecture modified from Zhang et. al, "Machine learning based adaptive optics for doughnut-shaped beam" (2019)
    to include a multi-channelled image
    """
    def __init__(self):
        super(MultiOffsetNet, self).__init__()
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

