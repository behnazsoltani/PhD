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

#from plot import plot_acc
import itertools

import argparse
from args import add_args 
from collections import defaultdict

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10

from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from model.cnn_cifar import cnn_cifar10, cnn_cifar100
from model.resnet import customized_resnet18


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
    

    nodes_info, clients_degrees, centrality = generate_nodes_info(0, n_clients, args.client_num_per_round, args.topology, args.rows, args.columns)
    


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
    # sparsity
    if args.sparse:
        w_spa = [args.dense_ratio for i in range(args.client_num_in_total)]
        if args.uniform:
            sparsities = calculate_sparsities(params,distribution="uniform", dense_ratio = args.dense_ratio)
        else:
            sparsities = calculate_sparsities(params,dense_ratio = args.dense_ratio)
    
            print("sparsities:", sparsities)
            
        
        if not args.different_initial_masks:
            temp = init_masks(params, sparsities)
            local_mask = [copy_dict(temp) for i in range(args.client_num_in_total)]
    
        else:
            local_mask = [copy_dict(init_masks(params, sparsities)) for i in range(args.client_num_in_total)]
            
    
    else:
        local_mask = [{} for _ in range(args.client_num_in_total)]

        for client_id in range(args.client_num_in_total):
            for name in params:
                # Initialize the mask with ones for each parameter for this client
                local_mask[client_id][name] = torch.ones_like(params[name])
                        
       
    #==================================================================
    
    
    user_dict_list = [
    {i: {"params": None, "mask": None} for i in range(num_clients)}
    for _ in range(num_clients)
]

    
    #==================================================================    
    
    previous_local_params = None
    

    last_params = {}
    #class_number = class_num
    class_number = len(dataset[-1][0])
    
    # assume all local models are initalized same    
    w_global = get_model_params(local_model_lists[0])
    w_per_mdls = []

    agg_weight = [None] * num_clients
    
    dense_weight = [copy_dict(w_global) for _ in range(num_clients)]


    
    for clnt in range(args.client_num_in_total):
        
        w_per_mdls.append(copy_dict(w_global))
        if args.sparse:
            for name in local_mask[clnt]:
                w_per_mdls[clnt][name] = w_global[name].to(device) * local_mask[clnt][name].to(device)
            
            local_model_lists[clnt].load_state_dict(w_per_mdls[clnt])
    
    #the mask of each client's last updated part to be shared
    #mask_pers_shared = [copy_dict(mask_pers_local[client_id]) for client_id in range(args.client_num_in_total)]
    
    aggregated_params = {}
    for epoch in tqdm(range(args.comm_round+1)):

        #nodes_info, clients_degrees, centrality = generate_nodes_info(epoch, n_clients, args.client_num_per_round, args.topology, args.rows, args.columns)
    
        #temp_user_dict_list = [{i: {"params": None} for i in range(num_clients)} for _ in range(num_clients)]
        temp_user_dict_list = [
        {i: {"params": None, "mask": None} for i in range(num_clients)}
        for _ in range(num_clients)
        ]

        # if epoch == 20:

        #     args.sparse = True
        #     for clnt in range(args.client_num_in_total):
        #         w_local = get_model_params(local_model_lists[clnt])  # dict[name] -> tensor
        #                 #shared_mask[idx] = copy.deepcopy(local_mask[idx])
        #         if args.algorithm == 'dispfl':
        #             #if not args.dis_gradient_check:
        #             gradient = screen_gradients(local_model_lists[clnt], train_data_local_dict[clnt], device)
        #             masks, num_remove = fire_mask(args, copy.deepcopy(local_mask[clnt]), w_local, epoch, device)
        #             new_masks = regrow_mask(args, masks, num_remove, device, gradient)
        #             local_mask[clnt] = copy.deepcopy(new_masks)
        #         #w_per_mdls = []  # reset every time you (re)build masked weights
                
                
        #        # w_per_mdls.append(copy_dict(w_local))

        #         with torch.no_grad():
        #             for name, m in local_mask[clnt].items():
        #                 t = w_local[name]
        #                 m = m.to(device=t.device, dtype=t.dtype)
        #                 w_local[name] = t * m

        #         local_model_lists[clnt].load_state_dict(w_local)

        
       # print(nodes_info)
        
        if epoch == 0 or (epoch)% args.round_number_evaluation == 0:
            logger.info("################Communication round : {}".format(epoch))
        #the function will randomly choose between 0 and 1
    
      
       

  
        for idx in range(num_clients):
 
            local_model_weights = [None] * num_clients
            shared_mask = [None] * num_clients
            if (epoch)% args.round_number_evaluation == 0:  
         
                logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(epoch, idx))

            aggregation_candidates = nodes_info[idx].copy()
            if idx not in aggregation_candidates:
                aggregation_candidates  = np.append(aggregation_candidates, idx).astype(int)
          

            if epoch> 0:
                for i in previous_neighbors[idx]:
 
                    local_model_weights[i] = user_dict_list[idx][i]["params"]
                    
                    shared_mask[i] = user_dict_list[idx][i]["mask"]
                    #local_model_weights[i] = copy_dict(local_model_lists[i].state_dict())
                
    #=============================== local training ======================================================

            aggregated_params[idx] = copy.deepcopy(local_model_lists[idx].state_dict())
            if args.algorithm == 'blenddfl':
                w = local_training_kd(epoch, idx, args.model, local_model_lists[idx], local_model_weights, ce_criterion, optimizer_list[idx], train_data_local_dict[idx], test_data_local_dict[idx], previous_neighbors[idx], device, args, logger, previous_local_params, weights_label[idx], lr_schedular_list[idx])
               

            elif args.algorithm == 'dfedavgm':
                w = local_training_dfedavgm(epoch, idx, local_model_lists[idx], ce_criterion, optimizer_list[idx], train_data_local_dict[idx], device, args, logger, lr_schedular_list[idx])
                
            elif args.algorithm == 'dfedsam':
                w = local_training_dfedsam(epoch, idx, local_model_lists[idx], ce_criterion, optimizer_list[idx], train_data_local_dict[idx], device, args, logger, lr_schedular_list[idx])

            elif args.algorithm == 'dispfl':
                w = local_training(epoch, idx, local_model_lists[idx], ce_criterion, optimizer_list[idx], train_data_local_dict[idx], device, args, logger, args.algorithm, lr_schedular_list[idx], masks = local_mask)
        
            elif args.algorithm == 'kdsparse':
                if epoch <= 20:
                    w = local_training(epoch, idx, local_model_lists[idx], ce_criterion, optimizer_list[idx], train_data_local_dict[idx], device, args, logger, args.algorithm, lr_schedular_list[idx], masks = local_mask)
                    
                else:
                    w = local_training_kdsparse(epoch, idx, args.model, local_model_lists[idx], local_model_weights, ce_criterion, optimizer_list[idx], train_data_local_dict[idx], test_data_local_dict[idx], previous_neighbors[idx], device, args, logger, previous_local_params, weights_label[idx], lr_schedular_list[idx], masks = local_mask)
                    #w = local_training(epoch, idx, local_model_lists[idx], ce_criterion, optimizer_list[idx], train_data_local_dict[idx], device, args, logger, args.algorithm, lr_schedular_list[idx], masks = local_mask)

            
            elif args.algorithm == 'freezing':
                if epoch <=args.kd_warmup:
                    #w = local_training_freezing(epoch, idx, local_model_lists[idx], ce_criterion, optimizer_list[idx], train_data_local_dict[idx], device, args, logger, args.algorithm, lr_schedular_list[idx], masks = local_mask[idx])
                    w = local_training(epoch, idx, local_model_lists[idx], ce_criterion, optimizer_list[idx], train_data_local_dict[idx], device, args, logger, args.algorithm, lr_schedular_list[idx])
                
                else:
                    #w = local_training_dynamic_kd_freezing(epoch, idx, local_model_lists[idx], ce_criterion, optimizer_list[idx], train_data_local_dict[idx],
                    #                            device, args, logger, args.algorithm, lr_schedular_list[idx], masks = local_mask[idx], prev_global_weights=aggregated_params[idx], temperature=2.0, alpha=args.kd_alpha)
                    w = local_training_kd_freezing(epoch, idx, local_model_lists[idx], ce_criterion, optimizer_list[idx], train_data_local_dict[idx],
                                                device, args, logger, args.algorithm, lr_schedular_list[idx], masks = local_mask[idx], prev_global_weights=aggregated_params[idx], temperature=args.temperature, alpha=args.kd_alpha)
                    
                    #w = local_training_freezing(epoch, idx, local_model_lists[idx], ce_criterion, optimizer_list[idx], train_data_local_dict[idx], device, args, logger, args.algorithm, lr_schedular_list[idx], masks = local_mask[idx])
            
            else:
                w = local_training(epoch, idx, local_model_lists[idx], ce_criterion, optimizer_list[idx], train_data_local_dict[idx], device, args, logger, args.algorithm, lr_schedular_list[idx])
                



            #================== select neighbors to send model =======================================
            

            selected_neighbors_to_send = nodes_info[idx].copy()
            if idx not in selected_neighbors_to_send:
                selected_neighbors_to_send  = np.append(selected_neighbors_to_send, idx).astype(int)
            #===================send models to neighbors ======================================================================
            if args.algorithm == 'freezing':
                
                for clnt in selected_neighbors_to_send:
                    # always send params
                    temp_user_dict_list[clnt][idx]["params"] = copy_dict(local_model_lists[idx].state_dict())

                    # decide which mask to send
                    if clnt == idx:
                        # dense mask (all ones) with same shapes as local_mask[idx]
                        temp_user_dict_list[clnt][idx]["mask"] = {
                            name: torch.ones_like(m) for name, m in local_mask[idx].items()
                        }
                    else:
                        # sparse mask
                        temp_user_dict_list[clnt][idx]["mask"] = copy_dict(local_mask[idx])
            elif args.algorithm == 'chunk':
                
                if epoch >= 40:
                    temp_user_dict_list = send_partial_params(local_model_lists[idx], selected_neighbors_to_send, epoch, idx, temp_user_dict_list)
                else:
                    for clnt in selected_neighbors_to_send:
                        temp_user_dict_list[clnt][idx]["params"] = copy_dict(local_model_lists[idx].state_dict())
                
                for clnt in selected_neighbors_to_send:
                   
                    temp_user_dict_list[clnt][idx]["mask"] = copy_dict(local_mask[idx])

            else:

                for clnt in selected_neighbors_to_send:
        
                    
                    temp_user_dict_list[clnt][idx]["params"] = copy_dict(local_model_lists[idx].state_dict())
                    temp_user_dict_list[clnt][idx]["mask"] = copy_dict(local_mask[idx])

            #del w, local_model_weights
            #torch.cuda.empty_cache()

        #==================== masks ===========================================================
            if args.sparse:
               
                #shared_mask[idx] = copy.deepcopy(local_mask[idx])
                if args.algorithm == 'dispfl':
                    if not args.dis_gradient_check:
                        gradient = screen_gradients(local_model_lists[idx], train_data_local_dict[idx], device)
                    masks, num_remove = fire_mask(args, copy.deepcopy(local_mask[idx]), w, epoch, device)
                    #new_masks = regrow_mask(args, masks, num_remove, device, gradient)
                    new_masks = regrow_mask_gradweight(args, masks, num_remove, device, gradient, w)
                    local_mask[idx] = copy.deepcopy(new_masks)
                elif args.algorithm == 'kdsparse':
                    if epoch >= 0:
                        #gradient = screen_kd_gradients(local_model_lists[idx], dense_weight[idx], train_data_local_dict[idx], device)
                        #gradient = screen_multikd_gradients(args, local_model_lists[idx],  local_model_weights, aggregation_candidates, dense_weight[idx], train_data_local_dict[idx], device, temperature=2.0, alpha=0.0)
    
                        #gradient = screen_gradients(local_model_lists[idx], train_data_local_dict[idx], device)
                        #masks, num_remove = fire_mask(args, copy.deepcopy(local_mask[idx]), w, epoch, device)
                        #masks, num_remove = fire_mask_dynamic_sparsity(copy.deepcopy(local_mask[idx]), w, gradient, epoch, device, args.comm_round)
                     
                        
                        #masks, num_remove = fire_mask_gradientandweight(args, copy.deepcopy(local_mask[idx]), w, epoch, device, gradient)
                        
                        #new_masks = regrow_mask(args, masks, num_remove, device, gradient)
                        # regrow_rate=0.02
                        # if epoch % 1 == 0:
                        #     new_masks = regrow_weights(masks, gradient, num_remove, regrow_rate, device) 
                        # 
                        
                        accum_gradient = screen_multikd_gradients(args, local_model_lists[idx],  local_model_weights, aggregation_candidates, dense_weight[idx], train_data_local_dict[idx], device, temperature=2.0, alpha=0.0)
    
                        new_masks = compute_per_weight_masks(local_model_lists[idx], accum_gradient, device,
                             target_sparsity=0.5) 
                      
                        local_mask[idx] = copy.deepcopy(new_masks)

                
   #######################################################################################                 
            if args.algorithm == 'freezing' and args.masking and epoch > 0 and epoch % 1 == 0:

                accum_gradient = screen_agg_gradients_freezing(epoch, args, local_model_lists[idx], aggregated_params[idx],
                                      train_data_local_dict[idx], device, temperature=args.temperature, alpha=args.kd_alpha,
                                     num_batches=5)
                #accum_gradient = screen_multikd_gradients_freezing(args, local_model_lists[idx],  local_model_weights, aggregation_candidates,
                 #                                           train_data_local_dict[idx], device,
                #                                             temperature=2.0, alpha=0.5, shared_mask = shared_mask , prev_global_weights = aggregated_params[idx])
                

                #accum_gradient = screen_multikd_gradients(args, local_model_lists[idx],  local_model_weights, aggregation_candidates, dense_weight[idx], train_data_local_dict[idx], device, temperature=2.0, alpha=0.0)
                #accum_gradient = screen_gradients(local_model_lists[idx], train_data_local_dict[idx], device)

                #new_masks = compute_per_weight_masks(local_model_lists[idx], accum_gradient, device,
                #        target_sparsity=0.5) 
                #new_masks = compute_energy_coverage_masks(local_model_lists[idx], accum_gradient, device, coverage=0.97)
                #new_masks = prune_regrow_dynamic(local_model_lists[idx], accum_gradient, local_mask[idx], coverage=0.99,
                 #            logger=logger, idx=idx, local_epoch=epoch)
                #new_masks = compute_global_energy_masks(local_model_lists[idx], accum_gradient, device)
                
                #new_masks = compute_global_energy_masks(local_model_lists[idx], accum_gradient, local_mask[idx], device, coverage=0.99,
                #            logger=logger, idx=idx, local_epoch=epoch)
                #new_masks = compute_layerwise_energy_masks(local_model_lists[idx], accum_gradient, local_mask[idx], device, logger=logger, idx=idx, local_epoch=epoch)
                #if epoch == 20:
                first_round = True
                #else:
                    #first_round = False
                #new_masks = compute_layerwise_prune_regrow_masks(local_model_lists[idx], accum_gradient, local_mask[idx],
                #                                                  device, first_round=first_round , logger=logger, idx=idx, local_epoch=epoch)
                #new_masks = compute_freeze_masks(local_model_lists[idx], accum_gradient,
                #                                                  device , logger=logger, idx=idx, local_epoch=epoch)
                
                
                #new_masks = compute_dynamic_sparsity_masks(epoch, args, local_model_lists[idx], accum_gradient, device,
                #         epoch, total_rounds=args.comm_round,
                        #  start_density=1.0, end_density=0.5, schedule="linear",
                        #  eps=1e-9,
                        #  logger=logger, idx=idx)
                
                
                new_masks = compute_dynamic_coverage_masks(epoch, args, local_model_lists[idx], accum_gradient, device,
                         epoch, total_rounds=args.comm_round,
                         start_cov=1.0, end_cov=args.end_coverage, schedule="linear",
                         eps=1e-9,
                         logger=logger, idx=idx)
                
                # new_masks = compute_comm_masks_kd(local_model_lists[idx], accum_gradient, device,
                #           aggregated_params[idx], train_data_local_dict[idx],
                #           candidates=(1.0, 0.98, 0.95, 0.9, 0.8, 0.7),
                #           tol_kd=0.2, kd_T=2.0,
                #           eps=1e-9,
                #           logger=logger, idx=idx)
                
                local_mask[idx] = copy.deepcopy(new_masks)
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
                
            #     logger.info("generalization test acc on this client after training is {} / {} : {:.2f}".format(generalization_test_results['test_correct'], generalization_test_results['test_total'], generalization_test_results['test_acc']))
                
   
            #     logger.info("Classwise Accuracy:")
            #     for i, acc in enumerate(generalization_test_results['classwise_accuracy']):
            #         logger.info("Class {}: {}".format(i, acc))
            
       ######=======================================================================================================   
            
        # After all clients have trained, commit updates to user_dict_list
        for idx in range(num_clients):
            for i in range(num_clients):
                if temp_user_dict_list[idx][i]["params"] is not None:
                    user_dict_list[idx][i]["params"] = temp_user_dict_list[idx][i]["params"]
                if temp_user_dict_list[idx][i]["mask"] is not None:
                    user_dict_list[idx][i]["mask"] = temp_user_dict_list[idx][i]["mask"]
        del temp_user_dict_list
        torch.cuda.empty_cache()

    #================ model aggregation =============================================================================================
        for idx in range(num_clients):
            local_model_weights = [None] * num_clients
            shared_mask = [None] * num_clients        
            aggregation_candidates = nodes_info[idx].copy()
            if idx not in aggregation_candidates:
                aggregation_candidates  = np.append(aggregation_candidates, idx).astype(int)
            
            
            
            #logger.info(f"client{idx}: aggregation candidates: {aggregation_candidates}")
            aggregation_candidates = sorted(aggregation_candidates)

    
         
            for i in aggregation_candidates:
              
                local_model_weights[i] = copy_dict_efficient(user_dict_list[idx][i]["params"])
                shared_mask[i] = copy_dict_efficient(user_dict_list[idx][i]["mask"])

            
            weights_dict = {neighbor: 1.0 / len(aggregation_candidates) for neighbor in aggregation_candidates}
            if args.algorithm == 'dpsgd':
                
                local_weight = aggregation(idx, args.model, local_model_lists[idx], local_model_weights, aggregation_candidates, weights_dict)
                #aggregated_weight = aggregate_after_training(epoch, idx, local_model_lists[idx], local_model_weights, train_data_local_dict[idx],
                #                          aggregation_candidates, device, args, weights_dict, optimizer_list[idx], lr_schedular_list[idx])
                #aggregated_weight = aggregate_from_chunk(local_model_lists[idx], local_model_weights, aggregated_params[idx], idx, aggregation_candidates)
            else:
                if args.algorithm == 'dispfl':
                    local_weight, agg_weight[idx]  = aggregate_with_mask(idx, local_model_weights, aggregation_candidates, shared_mask, local_mask, device)
                    #local_weight, agg_weight[idx]  = aggregate_with_union_fill(idx, local_model_weights, aggregation_candidates, shared_mask, local_mask, device)
                elif args.algorithm == 'kdsparse':
                    #local_weight, dense_weight[idx]  = aggregate_with_mask_fallback(idx, local_model_weights, dense_weight[idx], aggregation_candidates, shared_mask, local_mask)
                    #if epoch==2:
                     
                        #accum_gradient = screen_multikd_gradients(args, local_model_lists[idx],  local_model_weights, aggregation_candidates, dense_weight[idx], train_data_local_dict[idx], device, temperature=2.0, alpha=0.0)
                   
                        #new_masks = compute_per_weight_masks(local_model_lists[idx], accum_gradient, device,
                        #     target_sparsity=0.5)

                  
                    
                        #local_mask[idx] = copy.deepcopy(new_masks)
                
                    #if epoch<20:
                    local_weight, agg_weight[idx]  = aggregate_with_mask(idx, local_model_weights, aggregation_candidates, shared_mask, local_mask, device)
                 
                elif args.algorithm == 'freezing':
                    local_weight  = aggregate_with_freezing(idx, local_model_weights, aggregation_candidates, shared_mask, device)
                    #local_weight  = aggregate_with_freezing_prevagg(idx, local_model_weights, aggregation_candidates, shared_mask, aggregated_params[idx], device)
                      
                

 
            #======Update local models with aggregated model===================

            #if args.sparse:
            #if epoch<20:
            local_model_lists[idx].load_state_dict(local_weight)

            #else:
            #local_model_lists[idx].load_state_dict(aggregated_weight)
            #     #aggregated_params[idx] = aggregated_weight
                
            #============================================================

 

            
  
              #=================== evaluation after aggregation ================================
            if (epoch)% args.round_number_evaluation == 0:
                logger.info('@@@@@@@@@@@@@@@@ Evaluation CM({}): {}'.format(epoch, idx))
                logger.info(f"neighbors: {aggregation_candidates}")
     
                
                generalization_test_results = local_test(local_model_lists[idx], test_data_global, device, args, class_number, logger)
                generalization_tst_results_ths_round[idx] = generalization_test_results
                
                logger.info("generalization test acc on this client after aggregation is {} / {} : {:.2f}".format(generalization_test_results['test_correct'], generalization_test_results['test_total'], generalization_test_results['test_acc']))
                
              
                
                logger.info("Classwise Accuracy:")
                for i, acc in enumerate(generalization_test_results['classwise_accuracy']):
                    logger.info("Class {}: {}".format(i, acc))
            

       ######=======================================================================================================   
                            #if args.algorithm == 'kdsparse' and epoch==2:
                #local_weight, dense_weight[idx]  = aggregate_with_mask_fallback(idx, local_model_weights, dense_weight[idx], aggregation_candidates, shared_mask, local_mask)
            #prune_interval = 5
            #if epoch % prune_interval == 0 and epoch >= 20:
            if epoch == 20 and args.algorithm == 'kdsparse':
                accum_gradient = screen_multikd_gradients(args, local_model_lists[idx],  local_model_weights, aggregation_candidates, dense_weight[idx], train_data_local_dict[idx], device, temperature=2.0, alpha=0.0)
                #accum_gradient = screen_gradients(local_model_lists[idx], train_data_local_dict[idx], device)

                #new_masks = compute_per_weight_masks(local_model_lists[idx], accum_gradient, device,
                #        target_sparsity=0.5) 
                #new_masks = compute_energy_coverage_masks(local_model_lists[idx], accum_gradient, device, coverage=0.97)
                #new_masks = prune_regrow_dynamic(local_model_lists[idx], accum_gradient, local_mask[idx], coverage=0.99,
                 #            logger=logger, idx=idx, local_epoch=epoch)
                #new_masks = compute_global_energy_masks(local_model_lists[idx], accum_gradient, device)
                
                #new_masks = compute_global_energy_masks(local_model_lists[idx], accum_gradient, local_mask[idx], device, coverage=0.99,
                #            logger=logger, idx=idx, local_epoch=epoch)
                #new_masks = compute_layerwise_energy_masks(local_model_lists[idx], accum_gradient, local_mask[idx], device, logger=logger, idx=idx, local_epoch=epoch)
                if epoch == 20:
                    first_round = True
                else:
                    first_round = False
                new_masks = compute_layerwise_prune_regrow_masks(local_model_lists[idx], accum_gradient, local_mask[idx],
                                                                  device, first_round=first_round , logger=logger, idx=idx, local_epoch=epoch)
                local_mask[idx] = copy.deepcopy(new_masks)
                with torch.no_grad():
                    for name, p in local_model_lists[idx].named_parameters():
                        if name in local_mask[idx]:
                            p.mul_(local_mask[idx][name].to(device))  # in-place masking
            #============================================================

            del local_model_weights
            torch.cuda.empty_cache()
   
        if (epoch)% args.round_number_evaluation == 0:
            logger.info("################ average accuracy after aggregation: ############")
        #   _local_test_on_all_clients(list(tst_results_ths_round.values()), epoch)
            _local_test_on_all_clients(list(generalization_tst_results_ths_round.values()), epoch, generalization = True)
        #    _local_test_on_all_clients(list(local_generalization_tst_results_ths_round.values()), epoch, generalization = True)
        
        
        for clt in range(num_clients):
            previous_neighbors[clt] = nodes_info[clt]
            if clt not in previous_neighbors[clt]:
                previous_neighbors[clt]  = np.append(previous_neighbors[clt], clt).astype(int)
         
        #nodes_info, clients_degrees, centrality = generate_nodes_info( n_clients, args.client_num_per_round,
        #                     args.topology, args.rows, args.columns)
    
    # for idx in range(num_clients):
    #     model_path = f'saved_models/{idx}_{args.identity}.pth'
    #     torch.save(local_model_lists[idx].state_dict(), model_path)
      
def aggregate_with_freezing_prevagg(idx, local_model_weights, nei_idx, shared_mask,
                            prev_global_weights, device):
    """
    Aggregate neighbor parameters with freezing support.

    - Active (mask=1) params: use neighbor's current value.
    - Frozen (mask=0) params: use previous aggregated/global value for that neighbor.
    - Then aggregate across all neighbors.

    Args:
        idx: client index (int)
        local_model_weights: dict of all clients' weights
        nei_idx: list of neighbor indices
        shared_mask: dict of masks per client {clnt: {layer: tensor mask}}
        prev_global_weights: dict of previous aggregated parameters (for fallback)
        device: torch.device
    Returns:
        w_tmp: aggregated weights for client idx
    """
    # Initialize temporary weights with correct shapes
    w_tmp = copy_dict(local_model_weights[idx])

    for k in shared_mask[idx]:
        # Collect contributions from all neighbors
        contributions = []
        for clnt in nei_idx:
            mask = shared_mask[clnt][k].to(device)
            local_w = local_model_weights[clnt][k].to(device)
            global_w = prev_global_weights[k].to(device)

            # Neighbor contributes its weight if active, else global fallback
            contrib = mask * local_w + (1 - mask) * global_w
            contributions.append(contrib)

        # Average contributions across neighbors
        w_tmp[k] = torch.mean(torch.stack(contributions, dim=0), dim=0)

    return w_tmp


def send_partial_params(local_model, selected_neighbors_to_send, round_num, idx, temp_user_dict_list):
    """
    Send rotated parameter chunks to each neighbor.

    local_model: torch.nn.Module (the client model)
    selected_neighbors_to_send: list of neighbor IDs
    round_num: current round number
    idx: client index (to place updates in dict)
    temp_user_dict_list: the data structure storing params for neighbors
    """
    state_dict = local_model.state_dict()
    param_items = list(state_dict.items())

    num_neighbors = len(selected_neighbors_to_send) - 1
    chunk_size = (len(param_items) + num_neighbors - 1) // num_neighbors

    for i, clnt in enumerate(selected_neighbors_to_send):
        if clnt == idx:
            temp_user_dict_list[clnt][idx]["params"] = copy.deepcopy(state_dict)
        rotated_index = (i + round_num) % num_neighbors

        start = rotated_index * chunk_size
        end = min((rotated_index + 1) * chunk_size, len(param_items))

        # Only the selected chunk
        chunk_dict = dict(param_items[start:end])
        
        
        # Store in temp_user_dict_list
        temp_user_dict_list[clnt][idx]["params"] = copy.deepcopy(chunk_dict)

    return temp_user_dict_list


# def aggregate_from_chunk(local_model, model_weights, aggregated_params, client_id, aggregation_candidates):
#     """
#     Aggregate only over parameters received in temp_user_dict_list for this round.

#     local_model: torch.nn.Module
#         The client's current model (provides default values for missing params).
#     temp_user_dict_list: dict
#         temp_user_dict_list[client_id] contains messages from all senders.
#     client_id: int
#         ID of the client performing aggregation.

#     Returns:
#         new_state_dict: dict[str, Tensor]
#     """

#     # Get current model as baseline
#     local_state = copy.deepcopy(local_model.state_dict())

#     # Collect contributions for each parameter
#     param_accumulator = defaultdict(list)

#     # Always include self (full model is stored in temp_user_dict_list[client_id][client_id])
#     for clnt in aggregation_candidates:
#         params = model_weights[clnt]

#         for k, v in params.items():
#             param_accumulator[k].append(v)

#     # Start from local model
#     new_state_dict = copy.deepcopy(local_state)

#     # For each parameter that was received, average across senders
#     for k, tensors in param_accumulator.items():
#         new_state_dict[k] = sum(tensors) / len(tensors)

#     return new_state_dict

import copy
from collections import defaultdict

def aggregate_from_chunk(local_model, model_weights, aggregated_params, client_id, aggregation_candidates):
    """
    Aggregate only over parameters received this round.
    For parameters not received, use previous aggregated_params.

    local_model: torch.nn.Module
        The client's current model.
    model_weights: dict[int -> dict[str, Tensor]]
        model_weights[clnt] contains this round's parameters/chunks from client clnt.
    aggregated_params: dict[str, Tensor]
        The previous round's aggregated model (used as fallback).
    client_id: int
        ID of the client performing aggregation.
    aggregation_candidates: list[int]
        List of sender IDs to aggregate from.

    Returns:
        new_state_dict: dict[str, Tensor]
    """

    # Start from previous aggregate as baseline
    new_state_dict = copy.deepcopy(aggregated_params)

    # Collect contributions
    param_accumulator = defaultdict(list)

    for clnt in aggregation_candidates:
        params = model_weights[clnt]
        for k, v in params.items():
            param_accumulator[k].append(v)

    # For each parameter that was received, average across senders
    for k, tensors in param_accumulator.items():
        new_state_dict[k] = sum(tensors) / len(tensors)

    return new_state_dict





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
    
def calculate_sparsities(params, tabu=[], distribution="ERK", dense_ratio=0.5):
    # tabu: A list of layers that should not be sparsified (i.e., their sparsity is set to 0).
    spasities = {}
    if distribution == "uniform":
        for name in params:
            if name not in tabu:
                spasities[name] = 1 - dense_ratio
            #The layers in tabu will remain fully dense (sparsity = 0)
            else:
                spasities[name] = 0
    elif distribution == "ERK":
        
        # For each layer, a probability is calculated that determines how much sparsity it should have. This probability depends on the layer’s size and a scaling factor (args.erk_power_scale)
        # The goal is to calculate a factor called epsilon that adjusts the probabilities so that the total sparsity across all layers adds up correctly
        logger.info('initialize by ERK')
        total_params = 0
        for name in params:
            total_params += params[name].numel()
        is_epsilon_valid = False
   
        dense_layers = set()

        density = dense_ratio
        while not is_epsilon_valid:
            # We will start with all layers and try to find right epsilon. However if
            # any probablity exceeds 1, we will make that layer dense and repeat the
            # process (finding epsilon) with the non-dense layers.
            # We want the total number of connections to be the same. Let say we have
            # for layers with N_1, ..., N_4 parameters each. Let say after some
            # iterations probability of some dense layers (3, 4) exceeded 1 and
            # therefore we added them to the dense_layers set. Those layers will not
            # scale with erdos_renyi, however we need to count them so that target
            # paratemeter count is achieved. See below.
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name in params:
                if name in tabu:
                    dense_layers.add(name)
                #  The total number of parameters in the layer. It is calculated by multiplying the dimensions of the tensor (using np.prod).
                n_param = np.prod(params[name].shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (
                                                      np.sum(params[name].shape) / np.prod(params[name].shape)
                                              ) ** args.erk_power_scale
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        (f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name in params:
            if name in dense_layers:
                spasities[name] = 0
            else:
                spasities[name] = (1 - epsilon * raw_probabilities[name])
    return spasities        

def init_masks(params, sparsities):
    masks ={}
    for name in params:
        masks[name] = torch.zeros_like(params[name])
        # calculate how many elements should remain dense (non-zero)
        dense_numel = int((1-sparsities[name])*torch.numel(masks[name]))
        if dense_numel > 0:
            # flatten the mask into a 1D tensor
            temp = masks[name].view(-1)

            # generate a random permutation of indices
            perm = torch.randperm(len(temp))

            # pick the first `dense_numel` indices
            perm = perm[:dense_numel]
            temp[perm] =1
    return masks
          
# def aggregate_with_mask(idx, client_num_in_total, local_model_weights, nei_idx, mask, alpha = None):
#     logger.info('Doing local aggregation!')
     
#     weights =  copy_dict(mask)
#     if alpha is not None:
#         alpha = torch.tensor(alpha)
#     else:
#         alpha = torch.ones(client_num_in_total, device=device)
  
#     # Step 1: scale masks by alpha
#     #respecting both mask structure and client importance.
#     for clnt in nei_idx:
#         for k in mask[clnt]:
#             weights[clnt][k] = mask[clnt][k].to(device) * alpha[clnt]

#     # Step 2: total weights per layer
#     total_weights = {}
#     for clnt in nei_idx:
#         for layer, data in weights[clnt].items():
#             if layer not in total_weights:
#                 total_weights[layer] = data.clone()
#             else:
#                 total_weights[layer] += data
                
                
    
#     # Step 3: normalize weights
#     normalized_weights = [{} for _ in range(len(mask))]  # Adjusted for total clients
#     for clnt in nei_idx:
#         for layer in weights[clnt]:
#          # Directly using PyTorch, avoiding conversion to and from NumPy
#             normalized_weights[clnt][layer] = torch.where(
#             total_weights[layer] != 0,
#             weights[clnt][layer] / total_weights[layer],
#             torch.zeros_like(total_weights[layer])
#             )

#     # Step 4: aggregate model
#     w_tmp = copy_dict(local_model_weights[idx])
#     for k in mask[idx]:
#         w_tmp[k] = w_tmp[k].to(device) - w_tmp[k].to(device)
#         for clnt in nei_idx:
#             #weighted average aggregation (generalized FedAvg with mask+alpha)
#             w_tmp[k] += normalized_weights[clnt][k].to(device) * local_model_weights[clnt][k].to(device)
#     w_aggregated = copy.deepcopy(w_tmp)
#    # Step 5: apply local personalization mask
#     for name in mask[idx]:
#         w_tmp[name] = w_tmp[name] * mask[idx][name].to(device)
        

#     return w_tmp, w_aggregated


def aggregate_with_mask(idx, local_model_weights, nei_idx, shared_mask, local_mask, device):
    #logger.info('Doing local aggregation!')

    # Calculate total weights for each layer across all clients
    
  
    total_weights = {}
    for clnt in nei_idx:
        for layer, data in shared_mask[clnt].items():
            if layer not in total_weights:
                total_weights[layer] = data.clone()
            else:
                total_weights[layer] += data
  
                
                
    
    # Normalize each client's weights by the total weights for each layer
    normalized_weights = [{} for _ in range(len(shared_mask))]  # Adjusted for total clients
    for clnt in nei_idx:
        for layer in shared_mask[clnt]:
        
            normalized_weights[clnt][layer] = torch.where(
            total_weights[layer] != 0,
            shared_mask[clnt][layer] / total_weights[layer],
            torch.zeros_like(total_weights[layer])
            )


    w_tmp = copy_dict(local_model_weights[idx])
 
    for k in shared_mask[idx]:
        w_tmp[k] = torch.zeros_like(w_tmp[k])

        for clnt in nei_idx:
            #w_tmp[k] += torch.from_numpy(count_mask[k]) * local_model_weights[clnt][k]
            w_tmp[k] += normalized_weights[clnt][k] * local_model_weights[clnt][k]
    
    
    w_aggregated = copy.deepcopy(w_tmp)

    for name in local_mask[idx]:
        w_tmp[name] = w_tmp[name].to(device) * local_mask[idx][name].to(device)
        
    return w_tmp, w_aggregated    


def aggregate_with_union_fill(idx, local_model_weights, nei_idx, shared_mask, local_mask, device, lambda_fill=1.0):
    """
    Args:
        idx: index of current client
        local_model_weights: list of client models [dict]
        nei_idx: list of neighbor client indices
        shared_mask: dict of {client_idx: {layer_name: mask_tensor}}
        local_mask: dict of {client_idx: {layer_name: mask_tensor}}
        device: torch device
        lambda_fill: scaling factor for filling pruned params from neighbors (0 ≤ λ ≤ 1)
    
    Returns:
        (updated_weights_after_masking, full_aggregated_weights_before_masking)
    """

    # Step 1: Calculate total mask count per layer across neighbors
    total_mask = {}
    for clnt in nei_idx:
        for layer, mask in shared_mask[clnt].items():
            if layer not in total_mask:
                total_mask[layer] = mask.clone()
            else:
                total_mask[layer] += mask

    # Step 2: Compute normalized weights for each neighbor (mask-weighted)
    normalized_weights = [{} for _ in range(len(local_model_weights))]
    for clnt in nei_idx:
        for layer in shared_mask[clnt]:
            # Avoid division by zero
            total = torch.where(total_mask[layer] != 0, total_mask[layer], torch.ones_like(total_mask[layer]))
            normalized_weights[clnt][layer] = shared_mask[clnt][layer] / total

    # Step 3: Aggregate all parameters (union-based, full aggregation)
    w_tmp = copy.deepcopy(local_model_weights[idx])
    for layer in w_tmp:
        w_tmp[layer] = torch.zeros_like(w_tmp[layer])

        for clnt in nei_idx:
            if layer in normalized_weights[clnt]:
                w_tmp[layer] += normalized_weights[clnt][layer] * local_model_weights[clnt][layer]

    # Save unmasked version for analysis/debug
    w_aggregated = copy.deepcopy(w_tmp)

    # Step 4: Apply local mask with union-fill
    for layer in w_tmp:
        # Binary mask (1 = keep, 0 = prune)
        m_local = local_mask[idx][layer].to(device)

        # Union-fill: fill pruned positions with neighbor-aggregated values (scaled by lambda_fill)
        pruned = (1 - m_local)
        w_tmp[layer] = m_local * w_tmp[layer].to(device) + lambda_fill * pruned * w_aggregated[layer].to(device)

    return w_tmp, w_aggregated

def aggregate_with_freezing(idx, local_model_weights, nei_idx, shared_mask, device):
    #logger.info('Doing local aggregation!')

    # Calculate total weights for each layer across all clients
    
  
    total_weights = {}
    for clnt in nei_idx:
        for layer, data in shared_mask[clnt].items():
            data = data.to(device)
            if layer not in total_weights:

                total_weights[layer] = data.clone()
            else:
                total_weights[layer] += data
  
                
                
    
    # Normalize each client's weights by the total weights for each layer
    normalized_weights = [{} for _ in range(len(shared_mask))]  # Adjusted for total clients
    for clnt in nei_idx:
        for layer in shared_mask[clnt]:
        
            normalized_weights[clnt][layer] = torch.where(
            total_weights[layer] != 0,
            shared_mask[clnt][layer].to(device) / total_weights[layer],
            torch.zeros_like(total_weights[layer])
            )


    w_tmp = copy_dict(local_model_weights[idx])
 
    for k in shared_mask[idx]:
        w_tmp[k] = torch.zeros_like(w_tmp[k], device = device)

        for clnt in nei_idx:
            #w_tmp[k] += torch.from_numpy(count_mask[k]) * local_model_weights[clnt][k]
            w_tmp[k] += normalized_weights[clnt][k] * local_model_weights[clnt][k].to(device)
   
        
    return w_tmp    
@torch.no_grad()
def aggregate_with_mask_fallback(
    idx,
    local_model_weights,   # list[dict[layer->Tensor]] per client
    prev_dense,            # dict[layer->Tensor] previous dense teacher/aggregate
    nei_idx,               # iterable of neighbor client ids
    shared_mask,           # list[dict[layer->Tensor]] TRAINING masks m_{*,t-1} (0/1)
    local_mask,            # list[dict[layer->Tensor]] CURRENT masks m_{*,t} (0/1)
    include_self=True,
    beta = 0.7,
    eps=1e-8,
):
    # neighbor set (include self if desired)
    neighbors = list(nei_idx)
    if include_self and idx not in neighbors:
        neighbors.append(idx)

    w_dense  = {}
    w_sparse = {}

    for layer in shared_mask[idx].keys():
        # align dtype/device with weights
        base = local_model_weights[idx][layer]
        dtype, device = base.dtype, base.device

        # Denominator: union count per element (sum of training masks)
        D = torch.zeros_like(base, dtype=dtype, device=device)
        for j in neighbors:
            D.add_(shared_mask[j][layer].to(dtype=dtype, device=device))

        # Numerator: sum of masked weights that were trained
        num = torch.zeros_like(base, dtype=dtype, device=device)
        for j in neighbors:
            num.add_(shared_mask[j][layer].to(dtype=dtype, device=device) *
                     local_model_weights[j][layer].to(device))

        # Elementwise fallback: if nobody trained a coord (D==0), keep previous dense value
        dense_layer = torch.where(
        D > 0,
        (1 - beta) * prev_dense[layer].to(device) + beta * (num / (D + eps)),
        prev_dense[layer].to(device),
        )
        #dense_layer = torch.where(D > 0, num / (D + eps), prev_dense[layer].to(device))
        #dense_layer = torch.where(D > 0, num / (D + eps), 0)

        # Dense teacher/aggregate for this layer
        w_dense[layer] = dense_layer
        # My sparse view for the current round (apply current mask m_{k,t})
        w_sparse[layer] = dense_layer * local_mask[idx][layer].to(dtype=dtype, device=device)

    return w_sparse, w_dense



def aggregate_after_training(epoch, idx, model, model_weights, train_data,
                             aggregation_candidates, device, args, comm_weights,
                             optimizer_config, scheduler_config, masks=None):

    criterion = nn.CrossEntropyLoss()
    model_neighbor, optimizers, schedulers = {}, {}, {}

    for clnt in aggregation_candidates:
        neighbor_model = create_model(args, model_name=args.model, class_num=10)
        neighbor_model.load_state_dict(model_weights[clnt], strict=False)
        neighbor_model.to(device)
        model_neighbor[clnt] = neighbor_model

        # Use the same optimizer config as local training
        #lr = scheduler_config.get_last_lr()[0]  # StepLR returns a list of LRs
        lr = 0.001
        momentum = optimizer_config.param_groups[0]["momentum"]
        weight_decay = optimizer_config.param_groups[0]["weight_decay"]

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, neighbor_model.parameters()),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

        optimizers[clnt] = optimizer

        # Use the same scheduler config as local training
        # if scheduler_config["apply_lr_decay"]:
        #     schedulers[clnt] = torch.optim.lr_scheduler.StepLR(
        #         optimizer,
        #         step_size=scheduler_config["step_size"],
        #         gamma=scheduler_config["gamma"]
        #     )
        # else:
        #     schedulers[clnt] = None
        # Aggregate fine-tuned neighbor models

    #local_ep = args.local_epochs
    local_ep = 1
    for local_epoch in range(local_ep):
        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)

            for clnt in aggregation_candidates:
                if clnt == idx:  # skip self if needed
                    continue

                neighbor_model = model_neighbor[clnt]
                optimizer = optimizers[clnt]

                neighbor_model.train()
                optimizer.zero_grad()

                out = neighbor_model(x)
                loss = criterion(out, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(neighbor_model.parameters(), 10)
                optimizer.step()

    w = {k: torch.zeros_like(v).to(device) for k, v in model_weights[idx].items()}
    for clnt in aggregation_candidates:
        state_dict = model_neighbor[clnt].state_dict()
        for key in w:
            w[key] += state_dict[key].to(device) * comm_weights[clnt]

    return w

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
    #print("number of classes: ", class_num)
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
        #model = torchvision.models.resnet18(pretrained=True)
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
         #Modify conv1 to suit CIFAR-10
        #model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = nn.Linear(model.fc.in_features, class_num)
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
     
    cur_dir = os.path.dirname(os.path.abspath(__file__))
 


    log_dir = os.path.join(cur_dir,'dfl-sparse-log', args.dataset, args.topology)
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
    
    

      
        
        
