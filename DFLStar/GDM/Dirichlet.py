import os
import numpy as np
import time
import argparse
# import logging

from mpi4py import MPI
from math import ceil
from random import Random
import networkx as nx

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as IMG_models

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import ImageFolder

#from models import *
#from models import LogisticRegression

# logging.basicConfig(level=logging.INFO)

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234, isNonIID=False, alpha=0, dataset=None, dataset_name = None):
        self.data = data
        self.dataset = dataset
        if isNonIID:
            self.partitions, self.ratio = self.__getDirichletData__(data, sizes, seed, alpha, dataset_name)

        else:
            self.partitions = [] 
            self.ratio = sizes
            rng = Random() 
            rng.seed(seed) 
            data_len = len(data) 
            indexes = [x for x in range(0, data_len)] 
            rng.shuffle(indexes) 
             
     
            for frac in sizes: 
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]
                
            #print(f"IID data, ratio: {self.ratio}, partition: {self.partitions}")

        

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    def __getNonIIDdata__(self, data, sizes, seed, alpha):
        labelList = data.train_labels
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]
        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
            labelIdxDict.setdefault(label,[])
            labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]
        labelIdxPointer = [0] * labelNum
        # sizes = number of nodes
        partitions = [list() for i in range(len(sizes))]
        eachPartitionLen= int(len(labelList)/len(sizes))
        # majorLabelNumPerPartition = ceil(labelNum/len(partitions))
        majorLabelNumPerPartition = 2
        basicLabelRatio = alpha

        interval = 1
        labelPointer = 0

        #basic part
        for partPointer in range(len(partitions)):
            requiredLabelList = list()
            for _ in range(majorLabelNumPerPartition):
                requiredLabelList.append(labelPointer)
                labelPointer += interval
                if labelPointer > labelNum - 1:
                    labelPointer = interval
                    interval += 1
            for labelIdx in requiredLabelList:
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(basicLabelRatio*len(labelIdxDict[labelNameList[labelIdx]]))
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                labelIdxPointer[labelIdx] += idxIncrement

        #random part
        remainLabels = list()
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
        rng.shuffle(remainLabels)
        for partPointer in range(len(partitions)):
            idxIncrement = eachPartitionLen - len(partitions[partPointer])
            partitions[partPointer].extend(remainLabels[:idxIncrement])
            rng.shuffle(partitions[partPointer])
            remainLabels = remainLabels[idxIncrement:]

        return partitions

    def __getDirichletData__(self, data, psizes, seed, alpha, dataset_name):
        n_nets = len(psizes)
        if dataset_name == 'cifar10':
            K = 10
        elif dataset_name == 'cifar100':
            K = 100
        labelList = np.array(data.targets)
        min_size = 0
        N = len(labelList)
        np.random.seed(2020)

        net_dataidx_map = {}
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            
        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        print('Data statistics: %s' % str(net_cls_counts))

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes/np.sum(local_sizes)
        print(weights)
        #print(f"idx_batch: {idx_batch}")

        return idx_batch, weights

#def partition_dataset(rank, size, comm, args):
#    comm.Barrier()
#    print('==> load train data')
#    transform_train = transforms.Compose([
#        transforms.RandomCrop(32, padding=4),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#    ])
#    if args.dataset == 'cifar10':
#        trainset = torchvision.datasets.CIFAR10(root=args.datasetRoot, 
#                                            train=True, 
#                                            download=True, 
#                                            transform=transform_train)
#    elif args.dataset == 'cifar100':
#        trainset = torchvision.datasets.CIFAR100(root=args.datasetRoot, 
#                                                      train=True, 
#                                                      download=True, 
#                                                      transform=transform_train)
#    partition_sizes = [1.0 / size for _ in range(size)]
#    partition = DataPartitioner(trainset, partition_sizes, isNonIID=args.noniid, alpha=args.alpha)
#    ratio = partition.ratio
#    partition = partition.use(rank)
#    train_loader = torch.utils.data.DataLoader(partition, 
#                                            batch_size=args.bs, 
#                                            shuffle=True, 
#                                            pin_memory=True)
#    comm.Barrier()
#    print('==> load test data')
#    transform_test = transforms.Compose([
#        transforms.ToTensor(),
#        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#    ])
#    
#    if args.dataset == 'cifar10':
#        testset = torchvision.datasets.CIFAR10(root=args.datasetRoot, 
#                                        train=False, 
#                                        download=True, 
#                                        transform=transform_test)
#    elif args.dataset == 'cifar100':
#        testset = torchvision.datasets.CIFAR100(root=args.datasetRoot, 
#                                                     train=False, 
#                                                     download=True, 
#                                                     transform=transform_test)
#                                        
#    t1, t2 = torch.utils.data.random_split(testset, [500, 9500])
#
#    test_loader = torch.utils.data.DataLoader(t1, batch_size=64, shuffle=False) 
##    test_loader = torch.utils.data.DataLoader(testset, 
##                                            batch_size=64, 
##                                           shuffle=False, 
##                                            num_workers=size)
#                                            
#    comm.Barrier() 
#
#    # You can add more datasets here
#    return train_loader, test_loader, ratio


def partition_dataset(rank, size, comm, args, dataset_name='CIFAR10'):
    comm.Barrier()

    # Transforms for CIFAR datasets
    transform_cifar = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Transforms for MNIST dataset
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize MNIST images to 32x32 to match CIFAR
        transforms.Grayscale(num_output_channels=3),  # Convert MNIST to 3-channel images
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    # Transforms for EMNIST dataset
    transform_emnist = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize EMNIST images to 32x32 to match CIFAR
        transforms.Grayscale(num_output_channels=3),  # Convert EMNIST to 3-channel images
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]) 
    
        # Transforms for Fashion MNIST dataset
    transform_fashion_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
        # Transforms for CINIC-10 dataset
    transform_cinic = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4786, 0.4727, 0.4309), (0.2420, 0.2382, 0.2589)),
    ]) 
        

    # Load and partition the dataset
    if args.dataset == 'cifar10':
        print('==> load CIFAR10 data')
        dataset_transform = transform_cifar
        dataset_class = torchvision.datasets.CIFAR10
    elif args.dataset == 'cifar100':
        print('==> load CIFAR100 data')
        dataset_transform = transform_cifar
        dataset_class = torchvision.datasets.CIFAR100
    elif args.dataset == 'mnist':
        print('==> load MNIST data')
        dataset_transform = transform_mnist
        dataset_class = torchvision.datasets.MNIST
        
    elif args.dataset == 'emnist':
        print('==> load EMNIST data')
        dataset_transform = transform_emnist
        dataset_class = torchvision.datasets.EMNIST
        
    elif args.dataset == 'fashion_mnist':
        print('==> load Fashion MNIST data')
        dataset_transform = transform_fashion_mnist
        dataset_class = torchvision.datasets.FashionMNIST
        
    elif args.dataset == 'cinic10':
        print('==> load CINIC-10 data')
        dataset_transform = transform_cinic
        # Adjust the path to your CINIC-10 dataset location
        cinic_root = args.datasetRoot
#        if not os.path.exists(os.path.join(cinic_root, 'cinic-10')):
#            download_and_extract_archive(url='https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz', 
#                                         download_root=cinic_root, 
#                                         extract_root=cinic_root, 
#                                         remove_finished=True)
        #dataset_class = ImageFolder(root=os.path.join(cinic_root, 'cinic-10'), transform=dataset_transform)
    
        
    
    else:
        raise ValueError('Unsupported dataset')

    # Load train and test datasets
    if args.dataset == 'emnist':
        trainset = dataset_class(root=args.datasetRoot, split='byclass', train=True, download=True, transform=dataset_transform)
        testset = dataset_class(root=args.datasetRoot, split='byclass', train=False, download=True, transform=dataset_transform)
        print("size of testset of emnist:",len(testset))
    elif args.dataset == 'cinic10':

        train_root = os.path.join(args.datasetRoot, 'cinic-10', 'train')  # Adjust the path accordingly
        trainset = torchvision.datasets.ImageFolder(root=train_root, transform=dataset_transform)
        test_root = os.path.join(args.datasetRoot, 'cinic-10', 'test')  # Adjust the path accordingly
        testset = torchvision.datasets.ImageFolder(root=test_root, transform=dataset_transform)
    
       # trainset = dataset_class(root=args.datasetRoot, train=True, transform=dataset_transform)
        
    else:
        trainset = dataset_class(root=args.datasetRoot, train=True, download=True, transform=dataset_transform)
        testset = dataset_class(root=args.datasetRoot, train=False, download=True, transform=dataset_transform)

    # Partition train dataset
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(trainset, partition_sizes, isNonIID=args.noniid, alpha=args.alpha, dataset_name = args.dataset)
    ratio = partition.ratio
    partition = partition.use(rank)
    train_loader = torch.utils.data.DataLoader(partition, batch_size=args.bs, shuffle=True, pin_memory=True)

    # Prepare test data loader
    if args.dataset == 'emnist':
        t1, t2 = torch.utils.data.random_split(testset, [1000, len(testset) - 1000])
    elif args.dataset == 'cinic10':
        t1, t2 = torch.utils.data.random_split(testset, [1000, len(testset) - 1000])  # Use 1000 samples for testing
        print("the length ofa test data:", len(testset))
    elif args.dataset == 'cifar100':
        t1, t2 = torch.utils.data.random_split(testset, [1000, 9000])  # Modify split sizes as needed
    elif args.dataset == 'cifar10':
        t1, t2 = torch.utils.data.random_split(testset, [500, 9500])  # Modify split sizes as needed
    test_loader = torch.utils.data.DataLoader(t1, batch_size=64, shuffle=False)
    #if rank == 2:
    #print("testset:",t1[0][0]) 
    
  #  for batch_idx, (inputs, targets) in enumerate(test_loader):
   #     if batch_idx == 1:
   #         print(f"Inputs: {inputs[:1]}")

    comm.Barrier()

    return train_loader, test_loader, ratio
    
def worker_init_fn(rank):
    np.random.seed(2828 + rank)
    torch.manual_seed(2828 + rank)



def select_model(num_class, args):
    if args.model == 'VGG':
        model = vgg11()

    # You can add more models here
    return model

def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res 

class Meter(object):
    """ Computes and stores the average, variance, and current value """

    def __init__(self, init_dict=None, ptag='Time', stateful=False,
                 csv_format=True):
        """
        :param init_dict: Dictionary to initialize meter values
        :param ptag: Print tag used in __str__() to identify meter
        :param stateful: Whether to store value history and compute MAD
        """
        self.reset()
        self.ptag = ptag
        self.value_history = None
        self.stateful = stateful
        if self.stateful:
            self.value_history = []
        self.csv_format = csv_format
        if init_dict is not None:
            for key in init_dict:
                try:
                    # TODO: add type checking to init_dict values
                    self.__dict__[key] = init_dict[key]
                except Exception:
                    print('(Warning) Invalid key {} in init_dict'.format(key))

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.sqsum = 0
        self.mad = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sqsum += (val ** 2) * n
        if self.count > 1:
            self.std = ((self.sqsum - (self.sum ** 2) / self.count)
                        / (self.count - 1)
                        ) ** 0.5
        if self.stateful:
            self.value_history.append(val)
            mad = 0
            for v in self.value_history:
                mad += abs(v - self.avg)
            self.mad = mad / len(self.value_history)

    def __str__(self):
        if self.csv_format:
            if self.stateful:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.mad:.3f}'
                           .format(dm=self))
            else:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.std:.3f}'
                           .format(dm=self))
        else:
            if self.stateful:
                return str(self.ptag) + \
                       str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.mad:.3f})'
                           .format(dm=self))
            else:
                return str(self.ptag) + \
                       str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.std:.3f})'
                           .format(dm=self))