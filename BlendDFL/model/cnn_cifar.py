import copy
import logging
import math
import random

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
track_running_stats=False
# class cnn_cifar10(nn.Module):
#     def __init__(self):
#         super(cnn_cifar10, self).__init__()
#         self.n_cls = 10
#         self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5)
#         self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5)
#         self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
#         self.fc1 = torch.nn.Linear(64*5*5, 384)
#         self.fc2 = torch.nn.Linear(384, 192)
#         self.fc3 = torch.nn.Linear(192, self.n_cls)

#     def forward(self, x, return_features=False):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         features = x.view(x.size(0), -1)  # Flatten after convs

#         x = F.relu(self.fc1(features))
#         x = F.relu(self.fc2(x))
#         if return_features:
#             return features, self.fc3(x)
#         else:
#             return self.fc3(x)

class cnn_cifar10(nn.Module):
    def __init__(self):
        super(cnn_cifar10, self).__init__()
        self.n_cls = 10

        # Conv layer 1: 3 → 64 channels, 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=64)  # GroupNorm instead of BatchNorm

        # Conv layer 2: 64 → 64 channels, 5x5 kernel
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=64)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        features = x.view(x.size(0), -1)

        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))

        if return_features:
            return features, self.fc3(x)
        else:
            return self.fc3(x)

class cnn_cifar100(nn.Module):
    def __init__(self):
        super(cnn_cifar100, self).__init__()
        self.n_cls = 100

        # Conv layer 1: 3 → 64 channels, 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=64)  # GroupNorm instead of BatchNorm

        # Conv layer 2: 64 → 64 channels, 5x5 kernel
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=64)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        features = x.view(x.size(0), -1)

        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))

        if return_features:
            return features, self.fc3(x)
        else:
            return self.fc3(x)


# class cnn_cifar100(nn.Module):
#     def __init__(self):
#         super(cnn_cifar100, self).__init__()
#         self.n_cls = 100
#         self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5)
#         self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5)
#         self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
#         self.fc1 = torch.nn.Linear(64*5*5, 384)
#         self.fc2 = torch.nn.Linear(384, 192)
#         self.fc3 = torch.nn.Linear(192, self.n_cls)

#     def forward(self, x, return_features=False):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64*5*5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         if return_features:
#             return x, self.fc3(x)
#         else:
#             return self.fc3(x)




# class cnn_cifar10(nn.Module):
#
#     def __init__(self):
#         super(cnn_cifar10, self).__init__()
#         self.conv1 = torch.nn.Conv2d(in_channels=3,
#                                           out_channels=64,
#                                           kernel_size=5,
#                                           stride=1,
#                                           padding=0, bias=False)
#         self.conv2 = torch.nn.Conv2d(64, 64, 5,bias=False)
#         self.pool = torch.nn.MaxPool2d(kernel_size=3,
#                                        stride=2)
#         self.fc1 = torch.nn.Linear(64 * 4 * 4, 10, bias=False)
#         # self.fc2 = torch.nn.Linear(384, 192)
#         # self.fc3 = torch.nn.Linear(192, 10)
#
#
#     def forward(self, x):
#         x = self.pool(F.relu( self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 4 * 4)
#         x = self.fc1(x)
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)
#         return x

# class Meta_net(nn.Module):
#     def __init__(self, mask):
#         super(Meta_net, self).__init__()
#         size = int(mask.flatten().shape[0])
#         # self.fc11 = nn.Linear(size, 200)
#         # self.fc12 = nn.Linear(200, 200)
#         # self.fc13 = nn.Linear(200, size)
#         size = int(mask.flatten().shape[0])
#         self.fc11 = nn.Linear(size, 50)
#         self.fc12 = nn.Linear(50, 50)
#         self.fc13 = nn.Linear(50, size)
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight.data)
#                 nn.init.constant_(m.bias.data, 0)
#
#     def forward(self, input):
#         x = F.relu(self.fc11(input.flatten()))
#         x = F.relu(self.fc12(x))
#         conv_weight = self.fc13(x).view(input.shape)
#         return conv_weight
#
#     def initialize_weights(m):
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias.data, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight.data, 1)
#             nn.init.constant_(m.bias.data, 0)
#         elif isinstance(m, nn.Linear):
#             nn.init.kaiming_uniform_(m.weight.data)
#             nn.init.constant_(m.bias.data, 0)


class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        
        # First convolutional layer: 32 channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        # Max pooling layer: 2x2 kernel, stride 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Second convolutional layer: 64 channels, 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        
        # Fully connected layer: 512 units
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        
        # Output layer: softmax
        self.fc2 = nn.Linear(512, 10)  # Assuming 10 classes for CIFAR-10
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
                         #Behnaz
        #extracted_features = x
        
        # Flatten before passing through fully connected layer
        x = x.view(-1, 64 * 5 * 5)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x



class CNN_CIFAR100(nn.Module):
    def __init__(self):
        super(CNN_CIFAR100, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # (32, 32, 32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                                 # (32, 16, 16)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # (64, 16, 16)
                                                                                          # After pool: (64, 8, 8)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # (128, 8, 8)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=100)  # CIFAR-100 has 100 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
        x = F.relu(self.conv3(x))             # Conv3 + ReLU (no pool)
        x = x.view(x.size(0), -1)             # Flatten
        x = F.relu(self.fc1(x))               # FC1 + ReLU
        x = self.fc2(x)                       # FC2 (logits)
        return x


