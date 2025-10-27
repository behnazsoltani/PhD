#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x, neighbor_train=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
         #Behnaz
        extracted_features = x
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), extracted_features

class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
        
        
import torch
import torch.nn as nn


class CNN_OriginalFedAvg(torch.nn.Module):
    """The CNN model used in the original FedAvg paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629.

    The number of parameters when `only_digits=True` is (1,663,370), which matches
    what is reported in the paper.
    When `only_digits=True`, the summary of returned model is

    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 32)        832
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0
    _________________________________________________________________
    dense (Dense)                (None, 512)               1606144
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130
    =================================================================
    Total params: 1,663,370
    Trainable params: 1,663,370
    Non-trainable params: 0

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_OriginalFedAvg, self).__init__()
        self.only_digits = only_digits
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.max_pooling(x)
                 #Behnaz
        extracted_features = x
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        #x = self.softmax(self.linear_2(x))
        return x, extracted_features


class CNN_DropOut(torch.nn.Module):
    """
    Recommended model by "Adaptive Federated Optimization" (https://arxiv.org/pdf/2003.00295.pdf)
    Used for EMNIST experiments.
    When `only_digits=True`, the summary of returned model is
    ```
    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 12, 12, 64)        0
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0
    _________________________________________________________________
    dense (Dense)                (None, 128)               1179776
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    ```
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_DropOut, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        #x = self.softmax(self.linear_2(x))
        return x
        
        

class CNN_Cifar10(nn.Module):
    def __init__(self):
        super(CNN_Cifar10, self).__init__()
        
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
        extracted_features = x
        
        # Flatten before passing through fully connected layer
        x = x.view(-1, 64 * 5 * 5)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x, extracted_features
        
class CNN_emnist(nn.Module):
    def __init__(self):
        super(CNN_emnist, self).__init__()
        
        # First convolutional layer: 32 channels, 5x5 kernel
        #self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        # Max pooling layer: 2x2 kernel, stride 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Second convolutional layer: 64 channels, 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        
        # Fully connected layer: 512 units
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        
        # Output layer: softmax
        self.fc2 = nn.Linear(512, 62)  # Assuming 10 classes for CIFAR-10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
                         #Behnaz
        extracted_features = x
        
        # Flatten before passing through fully connected layer
        x = x.view(-1, 64 * 5 * 5)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x, extracted_features

#class CNN_CIFAR100(nn.Module):
#    def __init__(self):
#        super(CNN_CIFAR100, self).__init__()
#
#        # First convolutional layer: 32 channels, 5x5 kernel
#        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
#        # Max pooling layer: 2x2 kernel, stride 2
#        self.pool = nn.MaxPool2d(2, 2)
#
#        # Second convolutional layer: 64 channels, 5x5 kernel
#        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
#
#        # Additional convolutional layers
#        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
#        self.conv4 = nn.Conv2d(128, 256, kernel_size=5)
#        
#        
#
#        # Fully connected layers
#        self.fc1 = nn.Linear(256 * 5 * 5, 512)
#        self.fc2 = nn.Linear(512, 100)  # Change this to 100 for CIFAR-100
#
#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        x = self.pool(F.relu(self.conv3(x)))
#        x = self.pool(F.relu(self.conv4(x)))
#        
#        extracted_features = x
#
#        # Flatten before passing through fully connected layer
#        x = x.view(-1, 256 * 5 * 5)
#        #x = x.view(-1, 8192)  # Use the correct size based on the actual output size
#
#
#        x = F.relu(self.fc1(x))
#        x = self.fc2(x)
#
#        return x, extracted_features


class CNN_CIFAR100(nn.Module):
    def __init__(self):
        super(CNN_CIFAR100, self).__init__()
        
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
        self.fc2 = nn.Linear(512, 100)  # Assuming 10 classes for CIFAR-10
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
                         #Behnaz
        extracted_features = x
        
        # Flatten before passing through fully connected layer
        x = x.view(-1, 64 * 5 * 5)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x, extracted_features
