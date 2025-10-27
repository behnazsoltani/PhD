import torch.utils.data as data
from torchvision.datasets import MNIST

import numpy as np
from PIL import Image


import logging
import numpy as np
import torch




class MNIST_truncated(data.Dataset):
    def __init__(self, root, cache_data_set=None, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        """
        Args:
            root (string): Root directory of dataset where MNIST exists or will be saved to.
            cache_data_set (MNIST, optional): Cached MNIST dataset object to avoid redundant loading.
            dataidxs (list or array, optional): List of indices to subset the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise from test set.
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            download (bool, optional): If True, downloads the dataset from the internet and puts it in root directory.
        """
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        # Build the truncated dataset
        self.data, self.target = self.__build_truncated_dataset__(cache_data_set)

    def __build_truncated_dataset__(self, cache_data_set):
        """
        Builds the truncated dataset based on provided indices and caching.

        Args:
            cache_data_set (MNIST, optional): Cached MNIST dataset object.

        Returns:
            Tuple: (data, target) where data is a NumPy array of images and target is a NumPy array of labels.
        """
        if cache_data_set is None:
            # Load MNIST dataset without transformations for initial loading
            mnist_dataobj = MNIST(self.root, train=self.train, transform=None,
                                  target_transform=None, download=self.download)
        else:
            # Use the cached dataset
            mnist_dataobj = cache_data_set

        # Extract data and targets as NumPy arrays
        data = mnist_dataobj.data.numpy()
        target = mnist_dataobj.targets.numpy()

        if self.dataidxs is not None:
            # Select data and targets based on provided indices
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Retrieves the image and label at the specified index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Tuple: (image, target) where image is the transformed image and target is the label.
        """
        img, target = self.data[index], self.target[index]

        # Convert the NumPy array to a PIL Image for compatibility with torchvision transforms
        img = Image.fromarray(img, mode='L')  # 'L' mode for grayscale images

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)
