import logging
import math
import numpy as np
import torch
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from .datasets import EMNIST_truncated  # Ensure you have an EMNIST_truncated class similar to MNIST_truncated

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = []

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = []
        for i in range(max(y_train) + 1):  # Adjust based on the number of classes in the specific split
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
        for i in range(max(y_test) + 1):  # Adjust based on the number of classes in the specific split
            if i in unq:
                tmp.append(unq_cnt[np.argwhere(unq == i)][0, 0])
            else:
                tmp.append(0)
        test_cls_counts.append(tmp)
        logger.debug('DATA Partition: Train %s; Test %s' % (str(train_cls_counts[net_i]), str(tmp)))
    return

def _data_transforms_emnist():
    EMNIST_MEAN = (0.1307,)  # Mean for grayscale images
    EMNIST_STD = (0.3081,)   # Standard deviation for grayscale images

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
    ])

    return train_transform, valid_transform

def load_emnist_data(datadir, split='balanced'):
    train_transform, test_transform = _data_transforms_emnist()
    emnist_train_ds = EMNIST_truncated(datadir, split=split, train=True, download=True, transform=train_transform)
    emnist_test_ds = EMNIST_truncated(datadir, split=split, train=False, download=True, transform=test_transform)

    X_train, y_train = emnist_train_ds.data, emnist_train_ds.target
    X_test, y_test = emnist_test_ds.data, emnist_test_ds.target

    return (X_train, y_train, X_test, y_test)

def get_dataloader_emnist(datadir, split, train_bs, test_bs, dataidxs=None, test_idxs=None, cache_train_data_set=None, cache_test_data_set=None, logger=None):
    transform_train, transform_test = _data_transforms_emnist()

    if dataidxs is not None:
        dataidxs = np.array(dataidxs)
        logger.info(f"train_num: {len(dataidxs)} test_num: {len(test_idxs)}")

    train_ds = EMNIST_truncated(datadir, split=split, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = EMNIST_truncated(datadir, split=split, dataidxs=test_idxs, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=True, drop_last=False)
    return train_dl, test_dl
    

def partition_data(datadir, partition, n_nets, alpha, logger):
    """
    Partitions the dataset into multiple clients for federated learning.

    Args:
        datadir (str): Directory where the dataset is stored.
        partition (str): Partitioning method ('homo', 'n_cls', 'dir', etc.).
        n_nets (int): Number of clients.
        alpha (float): Parameter controlling the non-IID-ness in Dirichlet partitioning.
        logger (logging.Logger): Logger object for logging partition details.

    Returns:
        Tuple: (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)
    """
    logger.info("********* Partitioning data ***************")

    # Load the dataset using the appropriate loader function
    X_train, y_train, X_test, y_test = load_emnist_data(datadir, split='balanced')  # Change split as needed

    n_train = X_train.shape[0]
    n_cls = len(np.unique(y_train))  # Number of classes in the dataset

    net_dataidx_map = {}

    if partition == "homo":
        # IID Partitioning: data is distributed uniformly across all clients
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == 'n_cls':
        # Non-IID partitioning based on number of classes per client
        n_data_per_clnt = len(y_train) / n_nets
        clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0, size=n_nets)
        clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(y_train)).astype(int)
        cls_priors = np.zeros((n_nets, n_cls))

        for i in range(n_nets):
            cls_priors[i][np.random.choice(range(n_cls), int(alpha), replace=False)] = 1.0 / alpha

        prior_cumsum = np.cumsum(cls_priors, axis=1)
        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]

        for j in range(n_nets):
            net_dataidx_map[j] = []

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_nets)
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

    elif partition == 'dir':
        # Dirichlet-based partitioning for non-IID setup
        min_size = 0
        K = n_cls
        N = y_train.shape[0]
        logger.info(f"N = {N}")

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))

                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    # Add more partitioning methods as required

    # Record the class distribution in each client dataset
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


def load_partition_data_emnist(data_dir, partition_method, partition_alpha, client_number, batch_size, logger, split = 'balanced'):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        data_dir,
        partition_method,
        client_number,
        partition_alpha,
        logger
    )

    train_data_global, test_data_global = get_dataloader_emnist(
        data_dir,
        split,
        batch_size,
        batch_size,
        logger=logger
    )

    # Local dataset creation and partitioning logic similar to MNIST
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    transform_train, transform_test = _data_transforms_emnist()
    cache_train_data_set = EMNIST(data_dir, split=split, train=True, transform=transform_train, download=True)
    cache_test_data_set = EMNIST(data_dir, split=split, train=False, transform=transform_test, download=True)
    
    idx_test = [[] for _ in range(max(y_test) + 1)]  # Adjust range based on classes

    for label in range(max(y_test) + 1):
        idx_test[label] = np.where(y_test == label)[0]
    
    test_dataidxs = [[] for _ in range(client_number)]
    tmp_tst_num = math.ceil(len(cache_test_data_set) / client_number)
    
    for client_idx in range(client_number):
        for label in range(max(y_test) + 1):
            if sum(traindata_cls_counts[client_idx]) == 0:
                label_num = 0
            else:
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
        train_data_local, test_data_local = get_dataloader_emnist(
            data_dir,
            split,
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
        logger.info(f"client_idx = {client_idx}, local_sample_number = {local_data_num}")
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    record_part(y_test, traindata_cls_counts, test_dataidxs, logger)

    return None, None, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, traindata_cls_counts
