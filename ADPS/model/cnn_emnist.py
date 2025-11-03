import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn_emnist(nn.Module):
    def __init__(self, num_classes=10):  # Default is 10 for MNIST, adjust for EMNIST subsets
        super(cnn_emnist, self).__init__()
        self.n_cls = num_classes
        # Adjust in_channels from 3 (CIFAR-10) to 1 (MNIST/EMNIST)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Update fc1 input features from 64*5*5 to 64*4*4 for 28x28 images
        self.fc1 = nn.Linear(64*4*4, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        # First Convolutional Layer
        x = self.pool(F.relu(self.conv1(x)))  # Output shape: [batch_size, 64, 12, 12]
        # Second Convolutional Layer
        x = self.pool(F.relu(self.conv2(x)))  # Output shape: [batch_size, 64, 4, 4]
        # Flatten the tensor
        x = x.view(-1, 64*4*4)                # Flatten to shape: [batch_size, 64*4*4]
        # First Fully Connected Layer
        x = F.relu(self.fc1(x))               # Output shape: [batch_size, 384]
        # Second Fully Connected Layer
        x = F.relu(self.fc2(x))               # Output shape: [batch_size, 192]
        # Output Layer
        x = self.fc3(x)                       # Output shape: [batch_size, num_classes]
        return x
