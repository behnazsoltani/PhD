import logging
import math
import pdb
import numpy as np
import torch
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from .datasets import MNIST_truncated  # Ensure you have a MNIST_truncated class similar to CIFAR10_truncated

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = []

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = []
        for i in range(10):
            if i in unq:
                tmp.append(unq_cnt[np.argwhere(unq == i)][0, 0])
            else:
                tmp.append(0)
        net_cls_counts.append(tmp)
    return net_cls_counts

def record_part(y_test, train_cls_counts, test_dataidxs, logger):
    test_cls_counts = []

    for net_i, dataidx in enumerate(test_dataidxs):
        unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
        tmp = []
        for i in range(10):
            if i in unq:
                tmp.append(unq_cnt[np.argwhere(unq == i)][0, 0])
            else:
                tmp.append(0)
        test_cls_counts.append(tmp)
        logger.debug('DATA Partition: Train %s; Test %s' % (str(train_cls_counts[net_i]), str(tmp)))
    return

def _data_transforms_mnist():
    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
        # Add any additional augmentations if needed, e.g., transforms.RandomRotation
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])

    return train_transform, valid_transform

def load_mnist_data(datadir):
    train_transform, test_transform = _data_transforms_mnist()
    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=train_transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    return (X_train, y_train, X_test, y_test)

def partition_data(datadir, partition, n_nets, alpha, logger):
    logger.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_mnist_data(datadir)
   
    n_train = X_train.shape[0]

    if partition == 'n_cls':
        n_client = n_nets
        n_cls = 10

        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0, size=n_client)
        clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(y_train)).astype(int)
        cls_priors = np.zeros(shape=(n_client, n_cls))
        for i in range(n_client):
            cls_priors[i][random.sample(range(n_cls), int(alpha))] = 1.0 / alpha

        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {j: [] for j in range(n_client)}

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                if cls_amount[cls_label] <= 0:
                    cls_amount[cls_label] = np.random.randint(0, len(idx_list[cls_label]))
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                break

    elif partition == 'dir':
    
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / n_nets)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]    

    elif partition == 'my_part':
        n_shards = int(alpha)
        n_client = n_nets
        n_cls = 10

        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0, size=n_client)
        clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(y_train)).astype(int)
        cls_priors = np.zeros(shape=(n_client, n_cls))

        cls_priors_tmp = np.random.dirichlet(alpha=[0.3] * n_cls, size=int(n_shards))

        for i in range(n_client):
            cls_priors[i] = cls_priors_tmp[int(i / int(n_client / n_shards))]

        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {j: [] for j in range(n_client)}

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                if cls_amount[cls_label] <= 0:
                    cls_amount[cls_label] = len(idx_list[cls_label])
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                break

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

def get_dataloader_mnist(datadir, train_bs, test_bs, dataidxs=None, test_idxs=None, cache_train_data_set=None, cache_test_data_set=None, logger=None):
    transform_train, transform_test = _data_transforms_mnist()
    
    if dataidxs is not None:
        dataidxs = np.array(dataidxs)
        logger.info("train_num{}  test_num{}".format(len(dataidxs), len(test_idxs)))
    
    train_ds = MNIST_truncated(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True, cache_data_set=cache_train_data_set)
    test_ds = MNIST_truncated(datadir, dataidxs=test_idxs, train=False, transform=transform_test, download=True, cache_data_set=cache_test_data_set)
    
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=True, drop_last=False)
    return train_dl, test_dl

def load_partition_data_mnist(data_dir, partition_method, partition_alpha, client_number, batch_size, logger):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        data_dir,
        partition_method,
        client_number,
        partition_alpha,
        logger
    )

    # Global dataloader
    train_data_global, test_data_global = get_dataloader_mnist(data_dir, batch_size, batch_size, logger=logger)

    # Local datasets
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    transform_train, transform_test = _data_transforms_mnist()
    cache_train_data_set = MNIST(data_dir, train=True, transform=transform_train, download=True)
    cache_test_data_set = MNIST(data_dir, train=False, transform=transform_test, download=True)
    idx_test = [[] for _ in range(10)]

    for label in range(10):
        idx_test[label] = np.where(y_test == label)[0]
    
    test_dataidxs = [[] for _ in range(client_number)]
    tmp_tst_num = math.ceil(len(cache_test_data_set) / client_number)
    
    for client_idx in range(client_number):
        for label in range(10):
            label_num = math.ceil(
                traindata_cls_counts[client_idx][label] / sum(traindata_cls_counts[client_idx]) * tmp_tst_num
            )
            rand_perm = np.random.permutation(len(idx_test[label]))
            if len(test_dataidxs[client_idx]) == 0:
                test_dataidxs[client_idx] = idx_test[label][rand_perm[:label_num]]
            else:
                test_dataidxs[client_idx] = np.concatenate(
                    (test_dataidxs[client_idx], idx_test[label][rand_perm[:label_num]])
                )
        
        dataidxs = net_dataidx_map[client_idx]
        train_data_local, test_data_local = get_dataloader_mnist(
            data_dir,
            batch_size,
            batch_size,
            dataidxs,
            test_dataidxs[client_idx],
            cache_train_data_set=cache_train_data_set,
            cache_test_data_set=cache_test_data_set,
            logger=logger
        )
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        logger.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    record_part(y_test, traindata_cls_counts, test_dataidxs, logger)

    return None, None, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, traindata_cls_counts
