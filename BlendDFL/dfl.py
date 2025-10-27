# -*- coding: utf-8 -*-

import warnings
import torch
import os
import logging
import random
import numpy as np
import pandas as pd
import time
import datetime
import torch.nn as nn
import copy
import math
import csv
import cProfile
import pstats
import torchvision
from topology_dfl import generate_nodes_info, compute_weights_dict
from torch.profiler import profile, ProfilerActivity
import tracemalloc
import torch.utils.data as data
from utils_dfl import *

from tqdm import tqdm

from plot import plot_acc
import itertools

import argparse
from args import add_args 

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10

from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from model.cnn_cifar import cnn_cifar10, cnn_cifar100

# Function to replace all BatchNorm2d with GroupNorm
def convert_bn_to_gn(model, max_groups=8):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            channels = module.num_features
            groups = get_safe_group_count(channels, max_groups)
            gn = nn.GroupNorm(groups, channels)
            setattr(model, name, gn)
        else:
            convert_bn_to_gn(module, max_groups)

def get_safe_group_count(num_channels, max_groups=8):
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return g
    return 1  # fallback



def copy_dict_efficient(dictionary):
    return {name: tensor.clone() for name, tensor in dictionary.items()}


def train_dfl(dataset, local_model_lists, weights_label, args):

    [train_data_num, test_data_num, train_data_global, test_data_global, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
    

   
    num_clients = args.client_num_in_total
    

    nodes_info, clients_degrees, centrality = generate_nodes_info( n_clients, args.client_num_per_round, args.topology, args.rows, args.columns)
    


    # Data volume of each device
    n_i_list = [train_data_local_num_dict[i] for i in range(num_clients)]
  


 
    optimizer_list = []
    lr_schedular_list = []
    

    
 #   ce_criterion = nn.CrossEntropyLoss(weight = weights_label.to(device).float())
    ce_criterion = nn.CrossEntropyLoss()
    
    tst_results_ths_round = dict() 
    generalization_tst_results_ths_round = dict()

    local_tst_results_ths_round = dict() 
    local_generalization_tst_results_ths_round = dict()
    
    
    previous_neighbors = {}
    for idx in range(num_clients):
        
        previous_neighbors[idx] = None
        
           
        if args.client_optimizer == "sgd":
            #optimizer = torch.optim.SGD(filter(lambda p: p.f, self.model.parameters()), lr=args.lr* (args.lr_decay**round), momentum=args.momentum,weight_decay=args.wd)
            #optimizer = torch.optim.SGD(local_model_lists[idx].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, local_model_lists[idx].parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
            
        elif args.client_optimizer == "adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, local_model_lists[idx].parameters()), lr=0.001)

        
        optimizer_list.append(optimizer)
        #This scheduler adjusts the learning rate by multiplying it by a factor called gamma every step_size epochs
        lr_schedular_list.append(torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_decay_rate)) if args.apply_lr_decay else lr_schedular_list.append(None)
     
 
        
    
    params = get_trainable_params(local_model_lists[idx])
    
    #=================================================================
           
            
    #==================================================================
    
    

    user_dict_list = [{i: {"params": None} for i in range(num_clients)} for _ in range(num_clients)]
    
    #==================================================================    
        

    previous_local_params = None
    

    last_params = {}
    #class_number = class_num
    class_number = len(dataset[-1][0])
    
    
    for epoch in tqdm(range(args.comm_round+1)):
        temp_user_dict_list = [{i: {"params": None} for i in range(num_clients)} for _ in range(num_clients)]
  
       # print(nodes_info)
        
        if epoch == 0 or (epoch)% args.round_number_evaluation == 0:
            logger.info("################Communication round : {}".format(epoch))
        #the function will randomly choose between 0 and 1
    
      
       

  
        for idx in range(num_clients):
 
            local_model_weights = [None] * num_clients
            if (epoch)% args.round_number_evaluation == 0:  
         
                logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(epoch, idx))

            aggregation_candidates = nodes_info[idx].copy()
            if idx not in aggregation_candidates:
                aggregation_candidates  = np.append(aggregation_candidates, idx).astype(int)
          

            if epoch> 0:
                for i in previous_neighbors[idx]:
 
                    local_model_weights[i] = user_dict_list[idx][i]["params"]
                    #local_model_weights[i] = copy_dict(local_model_lists[i].state_dict())
                
    #=============================== local training ======================================================

    
            if args.algorithm == 'blenddfl':
                w = local_training_kd(epoch, idx, args.model, local_model_lists[idx], local_model_weights, ce_criterion, optimizer_list[idx], train_data_local_dict[idx], test_data_local_dict[idx], previous_neighbors[idx], device, args, logger, previous_local_params, weights_label[idx], lr_schedular_list[idx])
               

            elif args.algorithm == 'dfedavgm':
                w = local_training_dfedavgm(epoch, idx, local_model_lists[idx], ce_criterion, optimizer_list[idx], train_data_local_dict[idx], device, args, logger, lr_schedular_list[idx])
                
            elif args.algorithm == 'dfedsam':
                w = local_training_dfedsam(epoch, idx, local_model_lists[idx], ce_criterion, optimizer_list[idx], train_data_local_dict[idx], device, args, logger, lr_schedular_list[idx])

            else:
                w = local_training(epoch, idx, local_model_lists[idx], ce_criterion, optimizer_list[idx], train_data_local_dict[idx], device, args, logger, args.algorithm, lr_schedular_list[idx])
                
  
 



            #================== select neighbors to send model =======================================
            

            selected_neighbors_to_send = nodes_info[idx].copy()
            if idx not in selected_neighbors_to_send:
                selected_neighbors_to_send  = np.append(selected_neighbors_to_send, idx).astype(int)
            #===================send models to neighbors ======================================================================

 

            for clnt in selected_neighbors_to_send:
      
                
                temp_user_dict_list[clnt][idx]["params"] = copy_dict(local_model_lists[idx].state_dict())

            del w, local_model_weights
            torch.cuda.empty_cache()
      #=================== evaluation after local training ================================
            # if (epoch)% args.round_number_evaluation == 0:
            # #if epoch == 490:
            #     logger.info('@@@@@@@@@@@@@@@@ Evaluation CM({}): {}'.format(epoch, idx))
            #     #class_number = len(class_num[idx])
                
            #     test_results = local_test(local_model_lists[idx], test_data_local_dict[idx], device, args, class_number , logger)
            
            #     local_tst_results_ths_round[idx] = test_results
             
            #     logger.info("personalization test acc on this client after training is {} / {} : {:.2f}".format(test_results['test_correct'], test_results['test_total'], test_results['test_acc']))
             
                
                
                
            #     generalization_test_results = local_test(local_model_lists[idx], test_data_global, device, args, class_number, logger)
            #     local_generalization_tst_results_ths_round[idx] = generalization_test_results
                
            #     logger.info("generalization test acc on this client after {} / {} : {:.2f}".format(generalization_test_results['test_correct'], generalization_test_results['test_total'], generalization_test_results['test_acc']))
                
              
                
            #     logger.info("Classwise Accuracy:")
            #     for i, acc in enumerate(generalization_test_results['classwise_accuracy']):
            #         logger.info("Class {}: {}".format(i, acc))
            

     

       ######=======================================================================================================   
            
        # After all clients have trained, commit updates to user_dict_list
        for idx in range(num_clients):
            for i in range(num_clients):
                if temp_user_dict_list[idx][i]["params"] is not None:
                    user_dict_list[idx][i]["params"] = temp_user_dict_list[idx][i]["params"]
        del temp_user_dict_list
        torch.cuda.empty_cache()

    #================ model aggregation =============================================================================================
        
    
        for idx in range(num_clients):
            local_model_weights = [None] * num_clients
                   
            aggregation_candidates = nodes_info[idx].copy()
            if idx not in aggregation_candidates:
                aggregation_candidates  = np.append(aggregation_candidates, idx).astype(int)
            
            
            
            #logger.info(f"client{idx}: aggregation candidates: {aggregation_candidates}")
            aggregation_candidates = sorted(aggregation_candidates)

    
         
            for i in aggregation_candidates:
 
                local_model_weights[i] = copy_dict_efficient(user_dict_list[idx][i]["params"])

            
            weights_dict = {neighbor: 1.0 / len(aggregation_candidates) for neighbor in aggregation_candidates}

            aggregated_weight = aggregation(idx, args.model, local_model_lists[idx], local_model_weights, aggregation_candidates, weights_dict)




 
            #======Update local models with aggregated model===================

               

            local_model_lists[idx].load_state_dict(aggregated_weight)
                

            #============================================================
 

            del local_model_weights
            torch.cuda.empty_cache()
  
              #=================== evaluation after aggregation ================================
            if (epoch)% args.round_number_evaluation == 0:
                logger.info('@@@@@@@@@@@@@@@@ Evaluation CM({}): {}'.format(epoch, idx))
                logger.info(f"neighbors: {aggregation_candidates}")
     
                
                generalization_test_results = local_test(local_model_lists[idx], test_data_global, device, args, class_number, logger)
                generalization_tst_results_ths_round[idx] = generalization_test_results
                
                logger.info("generalization test acc on this client after {} / {} : {:.2f}".format(generalization_test_results['test_correct'], generalization_test_results['test_total'], generalization_test_results['test_acc']))
                
              
                
                logger.info("Classwise Accuracy:")
                for i, acc in enumerate(generalization_test_results['classwise_accuracy']):
                    logger.info("Class {}: {}".format(i, acc))
            

       ######=======================================================================================================   
         
   
        if (epoch)% args.round_number_evaluation == 0:
            logger.info("################ average accuracy after aggregation: ############")
        #   _local_test_on_all_clients(list(tst_results_ths_round.values()), epoch)
            _local_test_on_all_clients(list(generalization_tst_results_ths_round.values()), epoch, generalization = True)
        
        for clt in range(num_clients):
            previous_neighbors[clt] = nodes_info[clt]
            if clt not in previous_neighbors[clt]:
                previous_neighbors[clt]  = np.append(previous_neighbors[clt], clt).astype(int)
         
        #nodes_info, clients_degrees, centrality = generate_nodes_info( n_clients, args.client_num_per_round,
        #                     args.topology, args.rows, args.columns)
    
    for idx in range(num_clients):
        model_path = f'saved_models/{idx}_{args.identity}.pth'
        torch.save(local_model_lists[idx].state_dict(), model_path)
      



def copy_dict_to_cpu(state_dict):
    # Create a new dictionary with tensors moved to the CPU
    return {key: value.cpu().clone() for key, value in state_dict.items()}

    
def get_model_params(model):
    return copy_dict(model.cpu().state_dict())


def get_trainable_params(model):
    dict= {}
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only include trainable parameters
            dict[name] = param
    return dict                





def copy_dict(dictionary):
    # Create a new dictionary with cloned tensors
    if dictionary != None:
        return {name: tensor.detach().cpu().clone() for name, tensor in dictionary.items()}
    else:
        return None
        


def _local_test_on_all_clients(tst_results_ths_round, round_idx, generalization = False):
        if generalization == True:
            logger.info("################generalization_test_on_all_clients in communication round: {}".format(round_idx))
        else:    
            logger.info("################local_test_on_all_clients in communication round: {}".format(round_idx))
        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        for client_idx in range(args.client_num_in_total):
            # test data
            print(f"tst_results_ths_round: {tst_results_ths_round}")
            test_metrics['num_samples'].append(tst_results_ths_round[client_idx]['test_total'])
            test_metrics['num_correct'].append(tst_results_ths_round[client_idx]['test_correct'])
            test_metrics['losses'].append(tst_results_ths_round[client_idx]['test_loss'])

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
           # if self.args.ci == 1:
           #     break

        # # test on test dataset
        test_acc = sum([test_metrics['num_correct'][i] / test_metrics['num_samples'][i] for i in range(args.client_num_in_total) ] )/args.client_num_in_total
        test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in range(args.client_num_in_total)])/args.client_num_in_total

        if generalization == True:
            stats = {'generalization_test_acc': test_acc, 'generalization_test_loss': test_loss}
            
        else:
            stats = {'test_acc': test_acc, 'test_loss': test_loss}

        logger.info(stats)
       
    


    
def select_neighbors_to_send(cur_clnt, neighbors_idx):
     random_neighbours_idx = np.random.choice([client_ for client_ in neighbors_idx if client_ != cur_clnt], args.n_neighbors_to_send, replace=False)
     #print("random_neighbour_idx", random_neighbour_idx)
     return random_neighbours_idx
    

def select_neighbors_to_receive(cur_clnt, neighbors_idx, last_params):
    utility = {}
    for clt in neighbors_idx:
        utility[clt] = 1 - torch.cosine_similarity(
        last_params[clt],
        last_params[cur_clnt],
        dim=-1,
        ).mean()

    sorted_dict = dict(sorted(utility.items(), key=lambda item: item[1], reverse=True))
            # Get the top k elements
    top_k_elements = dict(list(sorted_dict.items())[:1])
    selected_node = np.array(list(top_k_elements.keys()))

    return selected_node
    

          


def aggregation(idx, model_name, model, local_model_weights, neighbors_index, comm_weights):
#    logger.info('Doing local aggregation!')
  
    w = copy_dict_efficient(local_model_weights[idx])
   
   

    for key, param in model.named_parameters():
        if "trainable_layers" in key or model_name != 'resnet18_pretrained': 
            w[key] = w[key] * .0
      
            for clnt in neighbors_index:
                xx = local_model_weights[clnt][key] * comm_weights[clnt]

                w[key] += xx
            
   
    return w

def aggregation_all(models, num_clients, comm_weights):
#    logger.info('Doing local aggregation!')
  
    w = copy_dict(models[0].state_dict())
   

    for key, param in models[0].named_parameters():
        
        w[key] = w[key] * .0
      
        for clnt in range(num_clients):
            xx = copy_dict(models[clnt].state_dict())[key] * comm_weights[clnt]

            w[key] += xx
            
   
    return w
    

 

def logger_config(log_path, logging_name):
    #os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode='w',encoding='UTF-8')
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
    
    
def load_data(args, dataset_name, logger):
    if dataset_name == "cifar10":
        args.data_dir += "cifar10"
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_cifar10(args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger, args.seed)
    else:
        if dataset_name == "cifar100":
            args.data_dir += "cifar100"
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_data_cifar100(args.data_dir, args.partition_method,
                                                     args.partition_alpha, args.client_num_in_total,
                                                     args.batch_size, logger)
        elif dataset_name == "tiny":
            args.data_dir += "tiny_imagenet"
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_data_tiny(args.data_dir, args.partition_method,
                                                     args.partition_alpha, args.client_num_in_total,
                                                     args.batch_size, logger)
                                                     
        elif dataset_name == "mnist":
            args.data_dir += "mnist"
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_data_mnist(args.data_dir, args.partition_method,
                                                     args.partition_alpha, args.client_num_in_total,
                                                     args.batch_size, logger)
           
                                                     
        elif dataset_name == "emnist":
            args.data_dir += "emnist"
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_data_emnist(args.data_dir, args.partition_method,
                                                     args.partition_alpha, args.client_num_in_total,
                                                     args.batch_size, logger)

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset

def create_model(args, model_name,class_num):
    # logging.info("create_model. model_name = %s" % (model_name))
    print("number of classes: ", class_num)
    model = None
    if model_name == "lenet5":
        model = LeNet5(class_num)
    elif model_name == "cnn_cifar10":
        model = cnn_cifar10()
    elif model_name == "cnn_cifar100":
        model = cnn_cifar100()
    elif model_name == "resnet18" and args.dataset != 'tiny':
    #    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    #    num_classes = 10  # Change this to match your dataset (e.g., CIFAR-10)
     #   model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = customized_resnet18(class_num=class_num)
    elif model_name == "resnet18" and args.dataset == 'tiny':
        model = tiny_resnet18(class_num=class_num)
    elif model_name == "vgg11":
        model = vgg11(class_num)
    elif model_name == "cnn_mnist":
        model = cnn_mnist()
    elif model_name == "cnn_emnist":
        model = cnn_emnist(class_num)
    elif model_name == "resnet18_pretrained":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "shufflenet":
        model = shufflenet_v2_x1_0(pretrained=False, num_classes=100)
        
        convert_bn_to_gn(model)
        
        #model.fc = nn.Linear(model.fc.in_features, class_num)
    elif model_name == "shufflenet_0.5":
        # Load ShuffleNetV2 for CIFAR-100
        model = shufflenet_v2_x1_0(pretrained=False, num_classes=100)

        # Convert BatchNorm to GroupNorm
        convert_bn_to_gn(model, groups=8)
        model.fc = nn.Linear(model.fc.in_features, class_num)
    elif model_name == "mobilenet":    
        model = mobilenet_v2()
        replace_batchnorm_with_groupnorm(model)
    elif model_name == "hybrid_transformer":
        model = HybridTransformer(num_classes = 10)

        
        # Replace BatchNorm with GroupNorm
    #    model = replace_batchnorm_with_groupnorm(model, num_groups=8)
        
        
        #model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #    model.fc = torch.nn.Linear(512, class_num)
        
        
        # Modify the first convolutional layer for CIFAR-100
       # model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
       # model.maxpool = nn.Identity()  # Remove max pooling for smaller input sizes
       # model.fc = nn.Linear(512, 100)  # Update the final layer for CIFAR-100
        #model.fc = nn.Linear(model.fc.in_features, class_num)
    return model
    
    

class ClientResNet18(nn.Module):
    def __init__(self, shared_layers, trainable_layers):
        super(ClientResNet18, self).__init__()
        self.shared_layers = shared_layers  # Shared frozen layers
        self.trainable_layers = trainable_layers  # Client-specific trainable layers

    def forward(self, x):
        with torch.no_grad():  # No gradients for shared layers
            x = self.shared_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.trainable_layers(x)  # Compute trainable layers
        return x

class TrainableLayers(nn.Module):
    def __init__(self, resnet18):
        super(TrainableLayers, self).__init__()
   #     self.layer4 = resnet18.layer4  # Trainable layer
   #     self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(512, 10)  # Adjust output to match your task

    def forward(self, x):
      #  x = self.layer4(x)
      #  x = self.avgpool(x)
      #  x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    
if __name__ == '__main__':
   

    
    parser = add_args(argparse.ArgumentParser(description='asdfl'))
    args = parser.parse_args()

        
    device = torch.device("cuda:" + str(args.gpu))
    
    data_partition = args.partition_method
    
    if data_partition != "homo":
        data_partition += str(args.partition_alpha)
        
    if args.algorithm == 'blenddfl':    
        args.identity = args.description + args.algorithm  + "-temp"+ str(args.temperature) + "-kdw" + str(args.kd_weight) + "-"+args.dataset+"-"+data_partition 
    else:
        args.identity = args.description + args.algorithm  + "-"+ args.dataset+ "-"+data_partition
    args. client_num_per_round = int(args.client_num_in_total* args.frac)
    args.identity += "-mdl" + args.model
    args.identity += "-" + args.topology
    args.identity += "-cm" + str(args.comm_round) + "-clients" + str(args.client_num_in_total)
    #args.identity += "-neighbor" + str(args.client_num_per_round)
    args.identity += '-lr' + str(args.lr)
    args.identity += '-seed' + str(args.seed)
   # args.identity += '-type' + str(args.type)
     
    cur_dir = os.path.abspath(__file__).rsplit("/", 1)[0]

    log_dir = os.path.join(cur_dir, 'Log', args.dataset, args.topology)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, args.identity + '.log')
    logger = logger_config(log_path=log_path, logging_name=args.identity)

    #log_path = os.path.join(cur_dir, 'Log/' + args.dataset + '/' +'ring/'+ args.identity + '.log')
    #logger = logger_config(log_path='Log/' + args.dataset + '/' +'ring/'+ args.identity + '.log',logging_name=args.identity)
    
    logger.info(args)
    logger.info("running at device{}".format(device))
    

    
    comm_logits_size = 0
    
     
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    


    # load data
    dataset = load_data(args, args.dataset, logger)
    [train_data_num, test_data_num, train_data_global, test_data_global, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
    
    
    logger.info(f"Class distribution:{class_num}")
   # input()
    # Print the total number of test samples in the global test dataset
    logger.info(f"Global test dataset size: {len(test_data_global)}")

    # Print the number of test samples for each client
    for i in range(args.client_num_in_total):
        logger.info(f"Client {i} test dataset size: {len(test_data_local_dict[i])}")

    # Print the total number of local test samples across all clients
    total_local_test_size = sum(len(test_data_local_dict[i]) for i in range(args.client_num_in_total))
    logger.info(f"Total local test dataset size: {total_local_test_size}")


    
    
    ######## finding underrepresented classes ##############
    # Convert to a numpy array for easier processing
    class_number = np.array(class_num)

    # Step: Identify underrepresented classes for each client
    local_underrepresented = {}
    missing_classes = {}  # Dictionary to store missing classes for each client
    weights_underrep_label = {}
    weights_label = {}
    
    for client_idx, client_data in enumerate(class_num):
        total_client_samples = np.sum(client_data)  # Total samples for the client
        threshold = 0.1 * total_client_samples  # Local threshold: 5% of total samples
        
        # Print for verification
        print(f"Client {client_idx}: Total Samples = {total_client_samples}, Threshold = {threshold}")
        print(f"Client {client_idx} class distributions: {client_data}")
        
        
         # Compute class weights (inverse frequency)
        weights_label[client_idx] = torch.tensor([1.0 / math.sqrt(count) if count > 0 else 0 for count in client_data])
        weights_label[client_idx] /= weights_label[client_idx].sum()  # Normalize weights
                



    ########################################################################################

    
    number_of_labels = len(dataset[-1][0])
    #number_of_labels = class_num
    
    local_model_lists = [[] for _ in range(args.client_num_in_total)]
        
    n_clients = args.client_num_in_total
   

 
    global_model = create_model(args, model_name=args.model, class_num=number_of_labels)
    local_model_lists = [copy.deepcopy(global_model) for _ in range(n_clients)]



    logger.info(local_model_lists[0])



    train_dfl(dataset, local_model_lists, weights_label, args)
    
    

      
        
        
