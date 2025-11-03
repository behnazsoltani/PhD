import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn_mnist(nn.Module):
    def __init__(self):
        super(cnn_mnist, self).__init__()
        self.n_cls = 10
        # Adjust in_channels from 3 (CIFAR-10) to 1 (MNIST)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Update fc1 input features from 64*5*5 to 64*4*4
        self.fc1 = torch.nn.Linear(64*4*4, 384)
        self.fc2 = torch.nn.Linear(384, 192)
        self.fc3 = torch.nn.Linear(192, self.n_cls)

    def forward(self, x):
        # First Convolutional Layer
        x = self.pool(F.relu(self.conv1(x)))  # Output: [64, 12, 12]
        # Second Convolutional Layer
        x = self.pool(F.relu(self.conv2(x)))  # Output: [64, 4, 4]
        # Flatten the tensor
        x = x.view(-1, 64*4*4)                # Flatten
        # First Fully Connected Layer
        x = F.relu(self.fc1(x))               # Output: [384]
        # Second Fully Connected Layer
        x = F.relu(self.fc2(x))               # Output: [192]
        # Output Layer
        x = self.fc3(x)                       # Output: [10]
        return x
