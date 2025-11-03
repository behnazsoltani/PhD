import logging
import math
import pdb
import numpy as np
import torch
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from .datasets import CIFAR10_truncated
import torchvision.transforms.functional as F
from PIL import Image
from collections import defaultdict
import copy
from torchvision.transforms import functional as TF
import os

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = []

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = []
        for i in range(10):
            if i in unq:
                tmp.append( unq_cnt[np.argwhere(unq==i)][0,0])
            else:
                tmp.append(0)
        net_cls_counts.append (tmp)
    return net_cls_counts

def record_part(y_test, train_cls_counts,test_dataidxs, logger):
    test_cls_counts = []

    for net_i, dataidx in enumerate(test_dataidxs):
        unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
        tmp = []
        for i in range(10):
            if i in unq:
                tmp.append( unq_cnt[np.argwhere(unq==i)][0,0])
            else:
                tmp.append(0)
        test_cls_counts.append ( tmp)
    #    logger.debug('DATA Partition: Train %s; Test %s' % (str(train_cls_counts[net_i]), str(tmp) ))
    return


class RandomCropWithSeed:
    def __init__(self, size, padding=0, seed=None):
        self.size = size
        self.padding = padding
        self.seed = seed

    def __call__(self, img):
        if self.padding > 0:
            img = TF.pad(img, self.padding)
        if self.seed is not None:
            random.seed(self.seed)
        # Use get_params from the RandomCrop class
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.size)
        return TF.crop(img, i, j, h, w)

class RandomHorizontalFlipWithSeed:
    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.seed = seed

    def __call__(self, img):
        if self.seed is not None:
            random.seed(self.seed)
        if random.random() < self.p:
            return TF.hflip(img)
        return img

def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    
    #bnz
        # Define the ImageNet mean and standard deviation
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    
    generator = torch.Generator().manual_seed(42)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        #BNZ
        #lambda img: F.resize(Image.fromarray(img), [224, 224]),  # Faster resize using functional API
        transforms.RandomCrop(32, padding=4),
        #RandomCropWithSeed(32, padding=4, seed=42),
        transforms.RandomHorizontalFlip(),
      #  RandomHorizontalFlipWithSeed(seed=42),
        #bnz
#change        transforms.Resize(64),   # Resize to 224x224 for ResNet18
       
        transforms.ToTensor(),
        #bnz
        #change
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),  # Use ImageNet normalization
        #transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
 
    # train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
    #bnz
        #lambda img: F.resize(Image.fromarray(img), [224, 224]),
    #    transforms.ToPILImage(),       # Convert numpy.ndarray to PIL.Image
#change        transforms.Resize(64),   # Resize to 224x224 for ResNet18
       # transforms.Lambda(lambda img: img.copy()),  # Ensure a safe copy
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    
  


    return train_transform, valid_transform


def load_cifar10_data(datadir):
    train_transform, test_transform = _data_transforms_cifar10()
    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True,transform=copy.deepcopy(train_transform))
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True,transform=copy.deepcopy(test_transform))

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data( datadir, partition, n_nets, alpha, logger):
    logger.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
   
    n_train = X_train.shape[0]
    
    if partition == "homo":
        total_num = n_train
       # np.random.seed(42)
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
     
    elif partition == 'n_cls':
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
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            # If current node is full resample a client
            # print('Remaining Data: %d' %np.sum(clnt_data_list))
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if trn_y is out of that class
                if cls_amount[cls_label] <= 0:
                    cls_amount[cls_label] = np.random.randint(0, len(idx_list[cls_label]))
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])

                break

    elif partition == 'dir':
    
    #     min_size = 0
    #     K = 10
    #     N = y_train.shape[0]
    #     logging.info("N = " + str(N))
    #     net_dataidx_map = {}
    #   #  np.random.seed(42)
    #     while min_size < 10:
    #         idx_batch = [[] for _ in range(n_nets)]
    #         # for each class in the dataset
    #         for k in range(K):
    #             idx_k = np.where(y_train == k)[0]
                
    #             np.random.shuffle(idx_k)
    #             proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
    #             ## Balance
    #             proportions = np.array(
    #                 [
    #                     p * (len(idx_j) < N / n_nets)
    #                     for p, idx_j in zip(proportions, idx_batch)
    #                 ]
    #             )
    #             proportions = proportions / proportions.sum()
    #             proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
    #             idx_batch = [
    #                 idx_j + idx.tolist()
    #                 for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    #             ]
    #             min_size = min([len(idx_j) for idx_j in idx_batch])

    #     for j in range(n_nets):
    #         np.random.shuffle(idx_batch[j])
    #         net_dataidx_map[j] = idx_batch[j]  
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
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
    #    n_client = n_nets
    #    n_cls = 10

    #    n_data_per_clnt = len(y_train) / n_client
    #    print(f" n_client: {n_client}, n_data_per_clnt: {n_data_per_clnt}")
    #    clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0, size=n_client)
    #    clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(y_train)).astype(int)
    #    cls_priors = np.random.dirichlet(alpha=[alpha] * n_cls, size=n_client)
    #    prior_cumsum = np.cumsum(cls_priors, axis=1)

    #    idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
    #    cls_amount = [len(idx_list[i]) for i in range(n_cls)]
    #    net_dataidx_map = {}
    #    for j in range(n_client):
    #        net_dataidx_map[j] = []

   
    #    while np.sum(clnt_data_list) != 0:
    #        print("np.sum(clnt_data_list) is:", np.sum(clnt_data_list))
    #        curr_clnt = np.random.randint(n_client)
    #        # If current node is full resample a client
    #        # print('Remaining Data: %d' %np.sum(clnt_data_list))
    #        if clnt_data_list[curr_clnt] <= 0:
    #            continue
    #        clnt_data_list[curr_clnt] -= 1
    #        curr_prior = prior_cumsum[curr_clnt]
    #        while True:
    #            cls_label = np.argmax(np.random.uniform() <= curr_prior)
    #            # Redraw class label if trn_y is out of that class
    #            if cls_amount[cls_label] <= 0:
    #                continue
    #            cls_amount[cls_label] -= 1
    #            net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
    #            break



           
       

    elif partition == 'my_part':
        n_shards = int(alpha)
        n_client = n_nets
        n_cls = 10

        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0, size=n_client)
        clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(y_train)).astype(int)
        cls_priors = np.zeros(shape=(n_client, n_cls))

        # default partition method with Dirichlet=0.3
        cls_priors_tmp = np.random.dirichlet(alpha=[0.3] * n_cls, size = int(n_shards))

        for i in range(n_client):
            cls_priors[i] = cls_priors_tmp[int(i / int(n_client / n_shards))]

        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            # If current node is full resample a client
            # print('Remaining Data: %d' %np.sum(clnt_data_list))
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if trn_y is out of that class
                if cls_amount[cls_label] <= 0:
                    cls_amount[cls_label] = len(idx_list[cls_label])
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                break

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32  # Ensures each worker gets a different seed
    np.random.seed(seed)
    random.seed(seed) 
def get_dataloader_cifar10(datadir, train_bs, test_bs, dataidxs=None,test_idxs=None, cache_train_data_set=None,cache_test_data_set=None,logger=None, seed = 42, save_preprocessed=False, partition_alpha = 0, global_test = False, client_idx = None):
    transform_train, transform_test = _data_transforms_cifar10()
    #dataidxs=np.array(dataidxs)
    #BZ
    #if dataidxs != None:
    if dataidxs is not None:
        dataidxs=np.array(dataidxs)
        logger.info("train_num:{}  test_num:{}".format(len(dataidxs),len(test_idxs)))
        
        
        
           # Paths to save preprocessed datasets
    #train_save_path = os.path.join(datadir, "cifar10_train_preprocessed.pt")
    #test_save_path = os.path.join(datadir, "cifar10_test_preprocessed.pt")
    if client_idx != None:
        train_save_path = os.path.join(datadir, f"train_data_client_{partition_alpha}_{client_idx}.pt")
        test_save_path = os.path.join(datadir, f"test_data_client_{partition_alpha}_{client_idx}.pt")
    
    
    global_test_save_path = os.path.join(datadir, "cifar10_global_test_preprocessed.pt")  # New save path for global test
    global_train_save_path = os.path.join(datadir, "cifar10_global_train_preprocessed.pt")  # New save path for global test
    
    if global_test:
    
        if save_preprocessed and os.path.exists(global_train_save_path):  # Check for global test data
            print("Loading preprocessed global test dataset...")
            global_train_data = torch.load(global_train_save_path)
            if test_idxs is not None:
                global_train_data["data"] = global_train_data["data"]
                global_train_data["labels"] = global_train_data["labels"]
            
            train_ds = data.TensorDataset(global_train_data["data"], global_train_data["labels"])
        else:
            print("Processing and saving global test dataset...")
            global_train_ds = CIFAR10_truncated(datadir, dataidxs=test_idxs, train=True, transform=copy.deepcopy(transform_test), download=True, cache_data_set=cache_test_data_set)
            
            global_train_data = {"data": [], "labels": []}
            for img, label in global_train_ds:
                global_train_data["data"].append(img)
                global_train_data["labels"].append(label)
            global_train_data["data"] = torch.stack(global_train_data["data"])
            global_train_data["labels"] = torch.tensor(global_train_data["labels"])
            torch.save(global_train_data, global_train_save_path)  # Save global test data
            train_ds = data.TensorDataset(global_train_data["data"], global_train_data["labels"])
            
        # Load or preprocess global test dataset
        if save_preprocessed and os.path.exists(global_test_save_path):  # Check for global test data
            print("Loading preprocessed global test dataset...")
            global_test_data = torch.load(global_test_save_path)
            if test_idxs is not None:
                global_test_data["data"] = global_test_data["data"]
                global_test_data["labels"] = global_test_data["labels"]
            
            test_ds = data.TensorDataset(global_test_data["data"], global_test_data["labels"])
        else:
            print("Processing and saving global test dataset...")
            global_test_ds = CIFAR10_truncated(datadir, dataidxs=test_idxs, train=False, transform=copy.deepcopy(transform_test), download=True, cache_data_set=cache_test_data_set)
            
            global_test_data = {"data": [], "labels": []}
            for img, label in global_test_ds:
                global_test_data["data"].append(img)
                global_test_data["labels"].append(label)
            global_test_data["data"] = torch.stack(global_test_data["data"])
            global_test_data["labels"] = torch.tensor(global_test_data["labels"])
            torch.save(global_test_data, global_test_save_path)  # Save global test data
            test_ds = data.TensorDataset(global_test_data["data"], global_test_data["labels"])
            
    else:
    # Load or preprocess train dataset
        if save_preprocessed and os.path.exists(train_save_path):
            print(f"Saving preprocessed training data for client {client_idx}...")
            train_data = torch.load(train_save_path)
    
            if dataidxs is not None:
                train_data["data"] = train_data["data"]
                train_data["labels"] = train_data["labels"]
            train_ds = data.TensorDataset(train_data["data"], train_data["labels"])
        else:
            print(f"Processing and saving training dataset for client {client_idx}...")
            train_ds = CIFAR10_truncated(datadir, dataidxs=dataidxs, train=True, transform=copy.deepcopy(transform_train), download=True, cache_data_set=cache_train_data_set)
            
            train_data = {"data": [], "labels": []}
            for img, label in train_ds:
                train_data["data"].append(img)
                train_data["labels"].append(label)
            train_data["data"] = torch.stack(train_data["data"])
            train_data["labels"] = torch.tensor(train_data["labels"])
            torch.save(train_data, train_save_path)
            train_ds = data.TensorDataset(train_data["data"], train_data["labels"])
    
        # Load or preprocess general test dataset
        if save_preprocessed and os.path.exists(test_save_path):
            print(f"Saving preprocessed test data for client {client_idx}...")
            test_data = torch.load(test_save_path)
            if test_idxs is not None:
                test_data["data"] = test_data["data"]
                test_data["labels"] = test_data["labels"]
            
            test_ds = data.TensorDataset(test_data["data"], test_data["labels"])
        else:
            print(f"Processing and saving general test dataset for client {client_idx}...")
            test_ds = CIFAR10_truncated(datadir, dataidxs=test_idxs, train=False, transform=copy.deepcopy(transform_test), download=True, cache_data_set=cache_test_data_set)
            
            test_data = {"data": [], "labels": []}
            for img, label in test_ds:
                test_data["data"].append(img)
                test_data["labels"].append(label)
            test_data["data"] = torch.stack(test_data["data"])
            test_data["labels"] = torch.tensor(test_data["labels"])
            torch.save(test_data, test_save_path)  # Save general test data
            test_ds = data.TensorDataset(test_data["data"], test_data["labels"])
    
        
            print(f"Maximum test_idxs: {max(test_idxs)}")
            print(f"Size of test_data['data']: {test_data['data'].shape[0]}")
          
    

     
    
   # train_ds = CIFAR10_truncated(datadir, dataidxs=dataidxs, train=True, transform=copy.deepcopy(transform_train), download=True,cache_data_set=cache_train_data_set)
   # test_ds = CIFAR10_truncated(datadir, dataidxs=test_idxs, train=False, transform=copy.deepcopy(transform_test), download=True,
               #       cache_data_set=cache_test_data_set)
                      
    # Create separate generators for train and test DataLoaders
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)  # Set seed for reproducibility

    test_generator = torch.Generator()
    test_generator.manual_seed(seed)  # Set seed for reproducibility
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False, num_workers=0, pin_memory=True, generator=train_generator)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=True, generator = test_generator)
  #  indices = [i for i, _ in enumerate(test_dl.dataset)]
  #  logger.info("Test dataset indices: {}".format(indices))
 
#    print("Before iterating over test_data:")
#    print(next(iter(train_dl))[1][:5])  # Print first 5 labels
#  
#    for batch_idx, (x, target) in enumerate(test_dl):
#        break
#    
#    print("After iterating over test_data:")
#    print(next(iter(train_dl))[1][:5])  # Print first 5 labels again
#    
#    input()

    return train_dl, test_dl

def load_partition_data_cifar10( data_dir, partition_method, partition_alpha, client_number, batch_size, logger, seed):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha, logger)
                                                                                             
    
    
                                                                                             
    #BZ 
    ######### global data #############################################   
    #train_data_global, test_data_global = get_dataloader_cifar10(data_dir, batch_size, batch_size, logger=logger)
    # Define the number of test samples and target test size
    reduced_test_size = 1000  # Total samples for the reduced test set
    num_classes = 10          # CIFAR-10 has 10 classes
    
    # Load the CIFAR-10 test set to get labels
    full_test_dataset = CIFAR10(root=data_dir, train=False, download=True)
    all_test_indices = np.arange(len(full_test_dataset))
    all_test_labels = np.array(full_test_dataset.targets)
    
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in zip(all_test_indices, all_test_labels):
        class_indices[label].append(idx)
    
    # Determine number of samples per class for balanced sampling
    samples_per_class = reduced_test_size // num_classes
    
    # Perform stratified sampling (equal samples per class)
    selected_test_indices = []
   # np.random.seed(42)  # Set seed for reproducibility
    for label in range(num_classes):
        selected_test_indices.extend(
            np.random.choice(class_indices[label], size=samples_per_class, replace=False)
        )
        
    logger.info(f"Test indices (run 1): {selected_test_indices}")
    
    # Call the function with the IID subset of indices
    train_data_global, test_data_global = get_dataloader_cifar10(
        datadir=data_dir,
        train_bs=batch_size,
        test_bs=batch_size,
        test_idxs=selected_test_indices,  # Pass the IID subset
        #test_idxs=None,
        logger=logger,
        seed = seed,
        partition_alpha = partition_alpha,
        global_test = True
    
    )
    
    # Get the labels from test_data_global
    all_labels = []
    for _, labels in test_data_global:  # Iterate through test_data_global
        all_labels.extend(labels.numpy())  # Collect labels from each batch
    
    # Count the number of samples per class
    class_counts = {label: all_labels.count(label) for label in range(num_classes)}
    
    # Log the class distribution
    logger.info("Class-wise distribution in test_data_global:")
    for label, count in class_counts.items():
        logger.info("Class = %d, samples = %d" % (label, count))
        
    
    ##################################################################
    
    
    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    transform_train, transform_test = _data_transforms_cifar10()
    cache_train_data_set=CIFAR10(data_dir, train=True, transform=copy.deepcopy(transform_train), download=True)
    cache_test_data_set = CIFAR10(data_dir, train=False, transform=copy.deepcopy(transform_test), download=True)
    idx_test = [[] for i in range(10)]
    # checking
    for label in range(10):
        idx_test[label] = np.where(y_test == label)[0]
    test_dataidxs = [[] for i in range(client_number)]
    tmp_tst_num = math.ceil(len(cache_test_data_set) / client_number)
    for client_idx in range(client_number):
        for label in range(10):
            # each has 100 pieces of testing data
            label_num = math.ceil(traindata_cls_counts[client_idx][label] / sum(traindata_cls_counts[client_idx]) * tmp_tst_num)
            rand_perm = np.random.permutation(len(idx_test[label]))
            if len(test_dataidxs[client_idx]) == 0:
                test_dataidxs[client_idx] = idx_test[label][rand_perm[:label_num]]
            else:
                test_dataidxs[client_idx] = np.concatenate(
                    (test_dataidxs[client_idx], idx_test[label][rand_perm[:label_num]]))
        dataidxs = net_dataidx_map[client_idx]
        train_data_local, test_data_local = get_dataloader_cifar10( data_dir, batch_size, batch_size,
                                                 dataidxs,test_dataidxs[client_idx] ,cache_train_data_set=cache_train_data_set,cache_test_data_set=cache_test_data_set,logger=logger, seed = seed, partition_alpha = partition_alpha, client_idx = client_idx)
#        for data, labels in test_data_local:
#            logger.info("First batch labels: {}".format(labels))
#            break
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        logger.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    record_part(y_test, traindata_cls_counts, test_dataidxs, logger)

    return None, None, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, traindata_cls_counts
