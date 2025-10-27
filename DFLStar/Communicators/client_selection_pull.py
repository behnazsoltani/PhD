import numpy as np
import time
import torch
from Communicators.CommHelpers import flatten_tensors, unflatten_tensors
from mpi4py import MPI
import random
import torch.nn as nn
from Utils.Misc import AverageMeter, compute_softmaxprob, test_accuracy, Recorder, test_loss
import copy
import struct
from Communicators.ConvergenceStatus import RoundCalculator

class clientSelection:
    """
    decentralized averaging according to a topology sequence
    For DSGD: Set i1 = 0 and i2 > 0 (any number it doesn't matter)
    For PD-SGD: Set i1 > 0 and i2 = 1
    For LD-SGD: Set i1 > 0 and i2 > 1
    """

    def __init__(self, rank, size, comm, topology, selection, utility_metric, ratio, i1, i2, layer):
        self.comm = comm
        self.rank = rank
        self.size = size
        self.topology = topology
        self.neighbor_list = self.topology.neighbor_list
        self.neighbor_weights = topology.neighbor_weights
        self.degree = len(self.neighbor_list)
        self.i1 = i1
        self.i2 = i2
        self.iter = 0
        
        self.iteration = 0
        self.comm_iter = 0
        self.comm_round = 0
        self.selection_mode=selection
        self.utility_metric = utility_metric
        self.epoch = None
        self.ratio = ratio
        
        self.param_neighbor = {}
        self.neighbor_param = {}
        self.utility = {}
        self.is_selected = {}
        self.selected_node = []
        self.greedy_selected_node = []
        self.unexplored = set()
        self.neighbor_loss = {}
        self.neighbor_info = {}
        self.softmax_probs = []
        self.loss = None
        self.info = None
        
        self.selfutility = None
        
        self.selection = True
        
        self.current_params = None
        self.last_params = None
        self.last_testloss = 0
        
        self.bestcount = 0
        self.selectioncount = 0
        self.round_weight_distance = {}
        
        num_gpus = torch.cuda.device_count()
        gpu_id = rank % num_gpus
        self.criterion = nn.CrossEntropyLoss().cuda(gpu_id)
        
        self.traffic = 0
        self.list_features = []
        
        self.round_calculator = RoundCalculator()
        self.accumaleted_weights = 0
        
        self.layer = layer
        


    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)
        self.train_buffer = torch.zeros_like(self.send_buffer)
        #if self.iteration == 2 and self.rank==0:
        #    print(f"rank {self.rank}: {self.send_buffer}")
    

    def greedyalgorithm(self, model, test_loader, selected_node=None):

    
        
        self.comm.Barrier()
        
        

        # compute self weight according to degree
        #selfweight = 1 - np.sum(self.neighbor_weights)
        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        #self.recv_buffer.add_(self.send_buffer, alpha=selfweight)
        
#        if self.rank == 0:
#            print(f"rank: {self.rank} self weight: {selfweight}")
#            print(f"rank: {self.rank} own parameter: {self.send_buffer}")

        send_buff = self.send_buffer.detach().numpy()
        
      
        #self.param = np.copy(send_buff)
        
        self.recv_tmp = np.empty_like(send_buff)
       
    
        #t_loss = {}
        test_acc_after = {}
        # decentralized averaging
        #test_acc_before = test_accuracy(model, test_loader, self.rank, greedy=True)
        for idx, node in enumerate(self.neighbor_list):
        
            #if self.rank == 0 and self.iteration ==4:
            #    print(f"sendbuff before testing is: {self.send_buffer}")
            self.recv_buffer2 = torch.zeros_like(self.send_buffer)
            self.recv_buffer2.add_(self.send_buffer, alpha=0.5)
            self.comm.Sendrecv(sendbuf=send_buff, source=node, recvbuf=self.recv_tmp, dest=node)
            # Aggregate neighbors' models: alpha * sum_j x_j
            #if self.rank == 0:
            #    print(f"rank: {self.rank} model parameter before averaging: {self.recv_buffer}")
            #self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=self.neighbor_weights[idx])
            
            self.recv_buffer2.add_(torch.from_numpy(self.recv_tmp), alpha=0.5)
            #t_loss_before = test_loss(model, test_loader, self.criterion)
            
            self.reset_model_test()
            #t_loss_after = test_loss(model, test_loader, self.criterion)
            test_acc_after[node] = test_accuracy(model, test_loader, self.rank, greedy=True)
            #t_loss[node] = test_loss(model, test_loader, self.criterion)
            #t_loss[node] = t_loss_before - t_loss_after
            #if self.rank == 7:
            #    print(f"node: {node} and selected node: {selected_node[0]}")
            
            #if node == selected_node[0]:
            #    selected_test = test_acc_after[node]
                #selected_test = t_loss[node]
                
            #self.param_neighbor[node] = np.copy(self.recv_tmp)
#            if self.rank==0:
#                print(f"rank: {self.rank} received from node: {node}") 
#                print(f"rank: {self.rank} sent to node: {node}")
#                print(f"rank: {self.rank} model parameter of neighbor: {torch.from_numpy(self.recv_tmp)}")
#                print(f"rank: {self.rank} model parameter after averaging: {self.recv_buffer}")
        max_key = max(test_acc_after, key=lambda k: test_acc_after[k])
        #max_key = min(t_loss, key=lambda k: t_loss[k])
        if selected_node == max_key:
            self.bestcount +=1
        self.selectioncount +=1
        
        #if self.rank == 7:
        
        #    print(f"selected nodes for {self.rank} is {selected_node[0]} with {selected_test}, greedy is {max_key} with {test_acc_after[max_key]}")
        
        #if self.rank == 7:
            #print(f" greedy is {max_key} with {test_acc[max_key]}")
        #    print(f" greedy is {max_key} with befor test acc is {test_acc_before} and after is {test_acc_after[max_key]}")
        
        #self.reset_model()
        #if self.iteration == 2 and self.rank==0:
            #print(f"rank {self.rank}: {self.send_buffer}")    
        
        self.comm.Barrier()
        return max_key
        
    
    
    def averaging2(self):

        
        
        self.comm.Barrier()
        tic = time.time()
        
        

        # compute self weight according to degree
        selfweight = 1 - np.sum(self.neighbor_weights)
        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        self.recv_buffer.add_(self.send_buffer, alpha=selfweight)
#        if self.rank == 0:
#            print(f"rank: {self.rank} self weight: {selfweight}")
#            print(f"rank: {self.rank} own parameter: {self.send_buffer}")

        send_buff = self.send_buffer.detach().numpy()
        
    
        #self.param = np.copy(send_buff)
        
        self.recv_tmp = np.empty_like(send_buff)
        # decentralized averaging
        for idx, node in enumerate(self.neighbor_list):
            self.comm.Sendrecv(sendbuf=send_buff, source=node, recvbuf=self.recv_tmp, dest=node)
            # Aggregate neighbors' models: alpha * sum_j x_j
#            if self.rank == 0:
#                print(f"rank: {self.rank} model parameter before averaging: {self.recv_buffer}")
            self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=self.neighbor_weights[idx])
        self.comm.Barrier()    
        
        
        self.train_buffer.add_(self.recv_buffer, alpha=selfweight)
        avg_buff = self.recv_buffer.detach().numpy()
        for idx, node in enumerate(self.neighbor_list):    
            self.comm.Sendrecv(sendbuf=avg_buff, source=node, recvbuf=self.recv_tmp, dest=node)
            self.train_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=self.neighbor_weights[idx])
            
            
            #self.param_neighbor[node] = np.copy(self.recv_tmp)
#            if self.rank==0:
#                print(f"rank: {self.rank} received from node: {node}") 
#                print(f"rank: {self.rank} sent to node: {node}")
#                print(f"rank: {self.rank} model parameter of neighbor: {torch.from_numpy(self.recv_tmp)}")
#                print(f"rank: {self.rank} model parameter after averaging: {self.recv_buffer}")
            

        self.comm.Barrier()
        toc = time.time()

        return toc - tic       
        
        
    def normalize_losses(self,loss_dict):
        #loss_dict[self.rank]= self.loss
        #inverse_losses = {k: 1 / (v + 1e-10) for k, v in loss_dict.items()}  # Adding a small value to avoid division by zero
        total_loss = sum(loss_dict.values())
        normalized_losses = {k: v / total_loss for k, v in loss_dict.items()}
        
        if self.rank == 0:
            print("normalized losses:",normalized_losses) 
        return normalized_losses
    
    def averaging(self):

       
        
        self.comm.Barrier()
        tic = time.time()
        
        

        # compute self weight according to degree
        selfweight = 1 - np.sum(self.neighbor_weights)
        #selfweight = weights[self.rank]
        #if self.rank == 0:
        #   print(f"selfweight: {selfweight}")
            
        
        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        self.recv_buffer.add_(self.send_buffer, alpha=selfweight)
#        if self.rank == 0:
#            print(f"rank: {self.rank} self weight: {selfweight}")
#            print(f"rank: {self.rank} own parameter: {self.send_buffer}")

        send_buff = self.send_buffer.detach().numpy()
        self.traffic += self.degree * (send_buff.nbytes / (1024.0 ** 2))
        #print(f"traffic: {self.traffic}, send_buff.nbytes: {send_buff.nbytes}")
        
     
        #self.param = np.copy(send_buff)
        
        self.recv_tmp = np.empty_like(send_buff)
        # decentralized averaging
        for idx, node in enumerate(self.neighbor_list):
            self.comm.Sendrecv(sendbuf=send_buff, source=node, recvbuf=self.recv_tmp, dest=node)
            # Aggregate neighbors' models: alpha * sum_j x_j
#            if self.rank == 0:
#                print(f"rank: {self.rank} model parameter before averaging: {self.recv_buffer}")

            self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=self.neighbor_weights[idx])
            #self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=weights[node])
            #if self.rank == 0:
            #    print(f"weight for node {node}: {weights[node]}")
            
            self.param_neighbor[node] = np.copy(self.recv_tmp)
#            if self.rank==0:
#                print(f"rank: {self.rank} received from node: {node}") 
#                print(f"rank: {self.rank} sent to node: {node}")
#                print(f"rank: {self.rank} model parameter of neighbor: {torch.from_numpy(self.recv_tmp)}")
#                print(f"rank: {self.rank} model parameter after averaging: {self.recv_buffer}")
            

        self.comm.Barrier()
        toc = time.time()

        return toc - tic
        
    
    def select_averaging(self, loss):

        self.comm.Barrier()
        tic = time.time()
        
        

        # compute self weight according to degree
        self.n_weights = self.getwight()
        weights_list = [value for value in self.n_weights.values()]
        selfweight = 1 - np.sum(weights_list)
        #print(f"rank: {self.rank}, number of push:{len(self.is_selected)}")
        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        self.recv_buffer.add_(self.send_buffer, alpha=selfweight)
        #self.recv_buffer.add_(self.send_buffer, alpha=0.75)
#        if self.rank == 0:
#            print(f"rank: {self.rank} own parameter: {self.send_buffer}")
#            print(f"rank: {self.rank} own weight: {selfweight}")

        send_buff = self.send_buffer.detach().numpy()
        #print("the content of send buffer: ", send_buff) 
        
     
        #self.param = np.copy(send_buff)
        
        #self.recv_tmp = np.empty_like(send_buff)
        self.recv_tmp = np.zeros_like(send_buff)

        #print("the size of recv buffer: ", self.recv_tmp.nbytes)
        #print("the rank: ", self.rank)
        # decentralized averaging
        for idx, node in enumerate(self.neighbor_list):
            #print (f"nodeeeeeeeeeeeeeeeeeeee: {node}, rank: {self.rank} ")
          
            if self.rank in self.is_selected[node] and node in self.selected_node:
                self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=self.recv_tmp, source=node)
                self.traffic += send_buff.nbytes / (1024.0 ** 2)
                #print(f"traffic: {self.traffic}, send_buff.nbytes: {send_buff.nbytes}")
#                if self.rank == 0:
#                    print(f"rank: {self.rank} model parameter before averaging: {self.recv_buffer}")
                self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=self.n_weights[node])
                #self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=0.25)
#                if self.rank==0:
#                    print(f"rank: {self.rank} received from node: {node}") 
#                    print(f"rank: {self.rank} sent to node: {node}")
#                    print(f"rank: {self.rank} model parameter of neighbor: {torch.from_numpy(self.recv_tmp)} and its weight: {self.n_weights[node]}")
#                    print(f"rank: {self.rank} model parameter after averaging: {self.recv_buffer}")
                #self.param_neighbor[node] = np.copy(self.recv_tmp)
                #self.unexplored.add(node)
                #continue     
            
            
            
            else:
                if self.rank in self.is_selected[node]:
                    #print(f"rank: {self.rank} is sending to node: {node}") 
                    req = self.comm.Isend(send_buff, dest=node) 
                    req.Wait()
                    self.traffic += send_buff.nbytes / (1024.0 ** 2)
                    #if self.rank==0:
                        #print(f"rank: {self.rank} sent to node: {node}")             
            
                if node in self.selected_node:

                    #print(f"rank: {self.rank} is waiting for node: {node}")
                    req = self.comm.Irecv(self.recv_tmp, source=node)
                    req.Wait()

                
                    #print("the content of recv buffer: ", self.recv_tmp)
                    #print(f"rank: {self.rank} received from node: {node}") 
                    
#                    if self.rank == 0:
#                        print(f"rank: {self.rank} model parameter before averaging: {self.recv_buffer}")
                    self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=self.n_weights[node])
                    #self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=0.25)
#                    if self.rank==0:
#                        print(f"rank: {self.rank} received from node: {node}") 
#                        print(f"rank: {self.rank} model parameter of neighbor: {torch.from_numpy(self.recv_tmp)} and its weight: {self.n_weights[node]}")
#                        print(f"rank: {self.rank} model parameter after averaging: {self.recv_buffer}")
                        
                    #self.param_neighbor[node] = np.copy(self.recv_tmp)
                    #self.unexplored.add(node)   
                
                

            #self.comm.Sendrecv(sendbuf=send_buff, source=node, recvbuf=self.recv_tmp, dest=node)
            # Aggregate neighbors' models: alpha * sum_j x_j

            

        self.comm.Barrier()
        toc = time.time()
        #if self.rank==0:
            #print("finished averaging")

        return toc - tic
        
    def select_weightedaveraging(self, loss, weights):
        self.comm.Barrier()
        tic = time.time()
        
        

        # compute self weight according to degree
        #self.n_weights = self.getwight()
        #weights_list = [value for value in self.n_weights.values()]
        selfweight = weights[self.rank]
        #print("selfweight:", selfweight)
        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        self.recv_buffer.add_(self.send_buffer, alpha=selfweight)
        #self.recv_buffer.add_(self.send_buffer, alpha=0.75)
#        if self.rank == 0:
#            print(f"rank: {self.rank} own parameter: {self.send_buffer}")
#            print(f"rank: {self.rank} own weight: {selfweight}")

        send_buff = self.send_buffer.detach().numpy()
        #print("the content of send buffer: ", send_buff) 
        
       
        #self.param = np.copy(send_buff)
        
        #self.recv_tmp = np.empty_like(send_buff)
        self.recv_tmp = np.zeros_like(send_buff)

        #print("the size of recv buffer: ", self.recv_tmp.nbytes)
        #print("the rank: ", self.rank)
        # decentralized averaging
        for idx, node in enumerate(self.neighbor_list):
            #print (f"nodeeeeeeeeeeeeeeeeeeee: {node}, rank: {self.rank} ")
          
            if self.rank in self.is_selected[node] and node in self.selected_node:
                self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=self.recv_tmp, source=node)
                self.traffic += send_buff.nbytes / (1024.0 ** 2)
#                if self.rank == 0:
#                    print(f"rank: {self.rank} model parameter before averaging: {self.recv_buffer}")
                self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=weights[node])
                #self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=0.25)
#                if self.rank==0:
#                    print(f"rank: {self.rank} received from node: {node}") 
#                    print(f"rank: {self.rank} sent to node: {node}")
#                    print(f"rank: {self.rank} model parameter of neighbor: {torch.from_numpy(self.recv_tmp)} and its weight: {self.n_weights[node]}")
#                    print(f"rank: {self.rank} model parameter after averaging: {self.recv_buffer}")
                #self.param_neighbor[node] = np.copy(self.recv_tmp)
                #self.unexplored.add(node)
                #continue     
            
            
            
            else:
                if self.rank in self.is_selected[node]:
                    #print(f"rank: {self.rank} is sending to node: {node}") 
                    req = self.comm.Isend(send_buff, dest=node) 
                    req.Wait()
                    self.traffic += send_buff.nbytes / (1024.0 ** 2)
                    #if self.rank==0:
                        #print(f"rank: {self.rank} sent to node: {node}")             
            
                if node in self.selected_node:

                    #print(f"rank: {self.rank} is waiting for node: {node}")
                    req = self.comm.Irecv(self.recv_tmp, source=node)
                    req.Wait()

                
                    #print("the content of recv buffer: ", self.recv_tmp)
                    #print(f"rank: {self.rank} received from node: {node}") 
                    
#                    if self.rank == 0:
#                        print(f"rank: {self.rank} model parameter before averaging: {self.recv_buffer}")
                    self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=weights[node])
                    #self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=0.25)
#                    if self.rank==0:
#                        print(f"rank: {self.rank} received from node: {node}") 
#                        print(f"rank: {self.rank} model parameter of neighbor: {torch.from_numpy(self.recv_tmp)} and its weight: {self.n_weights[node]}")
#                        print(f"rank: {self.rank} model parameter after averaging: {self.recv_buffer}")
                        
                    #self.param_neighbor[node] = np.copy(self.recv_tmp)
                    #self.unexplored.add(node)   
                
                

            #self.comm.Sendrecv(sendbuf=send_buff, source=node, recvbuf=self.recv_tmp, dest=node)
            # Aggregate neighbors' models: alpha * sum_j x_j

            

        self.comm.Barrier()
        toc = time.time()
        #if self.rank==0:
            #print("finished averaging")

        return toc - tic
            
        

    def reset_model(self):
        # Reset local models to be the averaged model
        for f, t in zip(unflatten_tensors(
                self.recv_buffer.cuda(), self.tensor_list),
                #self.train_buffer.cuda(), self.tensor_list),
                self.tensor_list):
            with torch.no_grad():
                t.set_(f)
                
    def reset_model_test(self):
        # Reset local models to be the averaged model
        for f, t in zip(unflatten_tensors(
                self.recv_buffer2.cuda(), self.tensor_list),
                #self.train_buffer.cuda(), self.tensor_list),
                self.tensor_list):
            with torch.no_grad():
                t.set_(f)

    def communicate(self, model, test_loader, loss, current_data_list, ps):
    
#        if self.iteration == 0 and self.iter == 0:
#            #print("self.iter: ", self.iter)
#            #print("self.iteration: ", self.iteration)  
#            self.tensor_lst = list()
#            for param in model.parameters():
#                self.tensor_lst.append(param)
#            print(f"rank: {self.rank} parameters: {flatten_tensors(self.tensor_lst).cpu()}")
    
        #self.epoch = epoch

        # Have to have this here because of the case that i1 = 0 (cant do 0 % 0)
        self.iter += 1
        
        
        self.loss = loss
      
        comm_time = 0
        select_t = 0
        

        # I1: Number of Local Updates Communication Set
        if self.iter % (self.i1+1) == 0:
            #self.iteration += 1
            self.comm_iter += 1
            
            #self.comm_round +=1
            
            
            
            
#            self.accumaleted_distribution += difference_distribution

            #normalized_weights = self.weightsforavg(difference_distribution)
            #print(f"normalized weights: {normalized_weights}")

            
            # stack all model parameters into one tensor list
            self.tensor_list = list()
            for param in model.parameters():
                self.tensor_list.append(param)
                

            # necessary preprocess
            self.prepare_comm_buffer()
            
            
            #self.select_client(model, test_loader)
            

         # no local  self.comm_round +=1
           


            # decentralized averaging according to activated topology
            # record the communication time

            #comm_time += self.averaging2()  

            #comm_time +=self.select_averaging()
            
#            if self.comm_round < 4:
#                comm_time += self.averaging()
                
#            else:
            
            if ps == 1:
                select_start = time.time()
                
                self.select_client(model, test_loader, loss, current_data_list)
                
            #self.comm.Barrier()
                select_t = time.time() - select_start
                comm_time +=self.select_averaging(loss)
                
            else:
                comm_time += self.averaging()
            
            #normalized_weights = self.weightsforavgnoaccumulation(difference_distribution)
#            
            #comm_time +=self.select_weightedaveraging(loss, normalized_weights)
#            comm_time +=self.select_averaging(loss)
#            for _ in range(2):
#                comm_time += self.averaging()
#                self.reset_model()
#                self.prepare_comm_buffer()
            
            


            # update local models
            #fedprox and feddc
            self.reset_model()
            
            self.global_params = [p.detach().clone() for p in model.parameters()]
          #  self.global_params = unflatten_tensors(self.recv_buffer.cuda(), self.tensor_list)
          
#            self.global_parameter = None
#            for param in model.parameters():
#                if not isinstance(self.global_parameter, torch.Tensor):
#                # Initially nothing to concatenate
#                    self.global_parameter = param.reshape(-1)
#                else:
#                    self.global_parameter = torch.cat((self.global_parameter, param.reshape(-1)), 0)
             

            # I2: Number of DSGD Communication Set
            if self.comm_iter % self.i2 == 0:
                self.comm_iter = 0
                self.comm_round +=1
            else:
                # decrease iteration by one in order to run another one update and average step (I2 communication)
                self.iter -= 1
                
          


        return comm_time, select_t, 

    def weightsforavg(self, difference_distribution):
        weights ={}
        self.accumaleted_weights += 1/(difference_distribution+1)
        for node in self.neighbor_list:
        #for node in self.selected_node:
            weights[node] = self.comm.sendrecv(self.accumaleted_weights, dest=node, source=node)
            #self.accumaleted_distribution[node] = self.accumaleted_distribution.get(key, 0) + diff[node]
            
            
        weights[self.rank] = self.accumaleted_weights
#            if self.rank == 0:
#                print("diffffffffffffffffffffffffff:", diff)
        #diff[self.rank] = difference_distribution
        

        # Normalize weights
        sum_of_weights = sum(weights.values())
        normalized_weights = {key: value / sum_of_weights for key, value in weights.items()}
        
        for node in self.neighbor_list:
            self.accumaleted_weights += weights[node]
        
        
        # Convert values to numpy array
#            difference_values = np.array(list(diff.values()))
#
#              # Sum of the differences
#            sum_of_differences = np.sum(difference_values)
#
#              # Check if the sum is non-zero to avoid division by zero
#            if sum_of_differences != 0:
#              # Normalize the differences
#                normalized_differences = {key: value / sum_of_differences for key, value in diff.items()}
#            else:
#              # If the sum is zero, set all values to 0
#                normalized_differences = {key: 0 for key in differences}

        return normalized_weights        
        
    def weightsforavgnoaccumulation(self, difference_distribution):
        diff ={}
        for node in self.neighbor_list:

        
            if self.rank in self.is_selected[node] and node in self.selected_node:
                diff[node] = self.comm.sendrecv(difference_distribution, dest=node, source=node)
                self.traffic += difference_distribution.nbytes / (1024.0 ** 2)
                           
            else:
                if self.rank in self.is_selected[node]:
                    #print(f"rank: {self.rank} is sending to node: {node}") 
                    self.comm.send(difference_distribution, dest=node) 
                   
                    self.traffic += difference_distribution.nbytes / (1024.0 ** 2)
                    #if self.rank==0:
                        #print(f"rank: {self.rank} sent to node: {node}")             
            
                if node in self.selected_node:

                    #print(f"rank: {self.rank} is waiting for node: {node}")
                    diff[node] = self.comm.recv(source=node)
                   
                        
#        diff[node] = self.comm.sendrecv(difference_distribution, dest=node, source=node)
#            #self.accumaleted_distribution[node] = self.accumaleted_distribution.get(key, 0) + diff[node]
#            
#            
#        diff[self.rank] = difference_distribution
##            if self.rank == 0:
##                print("diffffffffffffffffffffffffff:", diff)
        diff[self.rank] = difference_distribution
        
        max_difference = max(diff.values())
        weights = {key: (max_difference - value + 1) for key, value in diff.items()}

        # Normalize weights
        sum_of_weights = sum(weights.values())
        normalized_weights = {key: value / sum_of_weights for key, value in weights.items()}
        
        
        # Convert values to numpy array
#            difference_values = np.array(list(diff.values()))
#
#              # Sum of the differences
#            sum_of_differences = np.sum(difference_values)
#
#              # Check if the sum is non-zero to avoid division by zero
#            if sum_of_differences != 0:
#              # Normalize the differences
#                normalized_differences = {key: value / sum_of_differences for key, value in diff.items()}
#            else:
#              # If the sum is zero, set all values to 0
#                normalized_differences = {key: 0 for key in differences}

        return normalized_weights
            
                
     
    
    
    def utility_function(self, model, test_loader, loss, current_data_list): 
    
        cosine_similarity = {}
        utility = {}
        
#        if self.utility_metric == 'weight_difference':
#            self.param = flatten_tensors(self.tensor_list).cpu().detach().numpy()
#            #print ("self param: ", self.param)
#
#     
#            magnitude1 = np.linalg.norm(self.param)
#            #print("magnitude: ", magnitude1)
#    
#            for node, neighbor_p in self.param_neighbor.items():
#                dot_product = np.dot(self.param, neighbor_p)
#                magnitude2 = np.linalg.norm(neighbor_p)
#                cosine_similarity[node] = dot_product / (magnitude1 * magnitude2)
#
#                #print("Cosine Similarity for node %d:" % self.rank)
#                #print(cosine_similarity)
#                utility[node] = 1/cosine_similarity[node]

        if self.utility_metric == 'rounds_testloss_difference':
        
            t_loss = test_loss(model, test_loader, self.criterion)
            self.sendloss(self.last_testloss - t_loss)
            
            
            
            for node in self.neighbor_list:
                #print("neighbor_loss: ", self.neighbor_loss[node])
                utility[node] = self.neighbor_loss[node]  
                
            self.last_testloss = t_loss    
        
        elif self.utility_metric == 'rounds_weight_difference':
            #self.current_params = self.tensor_list.copy()
            self.current_params = flatten_tensors(self.tensor_list).cpu().detach().numpy()
            #if self.rank == 0:
             #   for i, param in enumerate(self.current_params):
             #       print(f"Shape of tensor {i} in self.current_params: {param.shape}")
               # print("hiiiiiiiiii:", self.current_params)
           
            
            
#            if self.last_params is None:
#                #self.last_params = [torch.zeros_like(param) for param in self.current_params]
#                self.last_params = np.zeros_like(self.current_params)
                
                #if self.rank == 0:
                 #   for i, param in enumerate(self.current_params):
                 #       print(f"HIIIIIII Shape of tensor {i} in self.current_params: {param.shape}")
                    #print("hiiiiiiiiii:", self.last_params)

                
                
            distance = 0
            #for p1, p2 in zip(self.current_params, self.last_params):
                #distance += torch.norm(p1 - p2, p=2)
            #distance = sum(np.linalg.norm(p1.cpu().detach().numpy() - p2.cpu().detach().numpy()) for p1, p2 in zip(self.current_params, self.last_params))
            
            
            #current_params_cpu = [param.cpu().detach().numpy() for param in self.current_params]
            
            
            #last_params_cpu = [param.cpu().detach().numpy() for param in self.last_params]
            
            distance = np.linalg.norm(self.current_params - self.last_params)
            self.round_weight_distance[self.rank] = distance
                
                
            #difference_params = self.current_params - self.last_params
            #send_buff = np.linalg.norm(difference_params)
            

     
            neighbor_params_difference = {}
            self.selfutility = distance
            
            for node in self.neighbor_list:
                utility[node] = self.comm.sendrecv(distance, dest=node, source=node)
                self.round_weight_distance[node] = utility[node]
                
            self.traffic += self.calculate_value_size(distance)
                
            self.last_params = self.current_params.copy()
                
        elif self.utility_metric == 'rounds_weight_difference_window':
            self.current_params = flatten_tensors(self.tensor_list).cpu().detach().numpy()
            

            
            distance = 0   
            distance = np.linalg.norm(self.current_params - self.last_params)
#            self.traffic += self.calculate_value_size(distance)
            
            self.round_calculator.add_round(self.current_params)
            convergence_status = self.round_calculator.calculate_cs()

            self.traffic += self.calculate_value_size(convergence_status)
            
            for node in self.neighbor_list:
                utility[node] = self.comm.sendrecv(convergence_status, dest=node, source=node)
                #self.round_weight_distance[node] = utility[node]
                
          
                
            self.last_params = self.current_params.copy()
            
            #if self.rank == 7:
               # print(f"convergence status: {convergence_status}, distance: {distance}")
                                   
        
        
        
        elif self.utility_metric == 'rounds_weight_difference_similarity':
            self.current_params = flatten_tensors(self.tensor_list).cpu().detach().numpy()
            

            
            distance = 0   
            distance = np.linalg.norm(self.current_params - self.last_params)
#            self.traffic += self.calculate_value_size(distance)
            
            self.round_calculator.add_round(self.current_params)
            convergence_status = self.round_calculator.calculate_cs()

            self.traffic += self.calculate_value_size(convergence_status)
            
            for node in self.neighbor_list:
                self.round_weight_distance[node] = self.comm.sendrecv(distance, dest=node, source=node)
                #self.round_weight_distance[node] = utility[node]
                
          
                
            self.last_params = self.current_params.copy()
            
            #if self.rank == 7:
               # print(f"convergence status: {convergence_status}, distance: {distance}")
            
            #-----------------------------------------------
            for name, layer in model.named_modules():
                #if isinstance(layer, nn.Linear):
                if isinstance(layer, nn.Conv2d):
                    last_params = layer.weight.data
                    last_layer = layer
                    
            last_params_numpy = (last_params).cpu().detach().numpy()
            send_buff = np.copy(last_params_numpy)
            
            self.traffic += self.degree * (send_buff.nbytes / (1024.0 ** 2))
            
            self.recv_buf = np.zeros_like(send_buff)
            #last_params_sum = np.copy(last_params_numpy)
            
            num_params_last_conv = last_layer.weight.numel()
            #print("number of parameters of last conv layer:", num_params_last_conv)

            dissimilarity = {}
            for node in self.neighbor_list:
                self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=self.recv_buf, source=node)
                n_params = np.copy(self.recv_buf)
                #difference = np.sum(np.abs(self.param - self.neighbor_param[node]))
                
                
                
                diff_vector = last_params_numpy - n_params
                dissimilarity[node] = np.linalg.norm(diff_vector)
                
                self.neighbor_param[node] = torch.from_numpy(n_params).cuda()
#                #self.neighbor_param[node] = [torch.tensor(t) for t in neighbor_param_unflatten]
#                
                dissimilarity[node] = 1 - torch.cosine_similarity(
                        self.neighbor_param[node],
                        last_layer.weight.data,
                        dim=-1,
                    ).mean()
                    
                utility[node] = dissimilarity[node] * self.round_weight_distance[node]
                
                #if self.rank == 7:
                #    print(f"node {node}: similarity: {dissimilarity[node]}, weight_distance: {self.round_weight_distance[node]}, utility: {utility[node]}")
                
                    #print("last layer of node 7:", last_layer.weight.data)
                
                

            
        
        elif self.utility_metric == 'weight_distribution':
            neighbor_hist = {}
            kl_divergence = {}
            self.param = flatten_tensors(self.tensor_list).cpu().detach().numpy()
            selfhist, _ = np.histogram(self.param, bins=10, density=True)
            #selfhist /= np.sum(selfhist)
            send_buff = selfhist
            self.recv_buf = np.zeros_like(selfhist)
            for node in self.neighbor_list:
                self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=self.recv_buf, source=node)
                neighbor_hist[node] = np.copy(self.recv_buf)
                #print(neighbor_hist[node])
#                neighbor_hist[node] = np.clip(neighbor_hist[node], 1e-10, None)
#                assert not np.any(neighbor_hist[node] == 0.0), "Division by zero encountered"
#                kl_divergence[node] = np.sum(selfhist * np.log2(selfhist / neighbor_hist[node]))
#                utility[node] = kl_divergence[node]
                
                diff_vector = selfhist - neighbor_hist[node]
                l2_distance = np.linalg.norm(diff_vector)  # Calculate L2 (Euclidean) norm
                utility[node] = l2_distance
            
            
            #for node in self.neighbor_list:
                #print("neighbor_loss: ", self.neighbor_loss[node])
                #utility[node] = abs(self.neighbor_info[node]-self.info)    

            
        elif self.utility_metric == 'weight_meanstd':
            self.neighbor_info = self.sendinfo(model)
            info_array = np.array(self.info)
            for node in self.neighbor_list:
                neighbor_info_array = np.array(self.neighbor_info[node])
                diff_vector = neighbor_info_array - info_array
                l2_distance = np.linalg.norm(diff_vector)  # Calculate L2 (Euclidean) norm
                utility[node] = l2_distance
                
        
        elif self.utility_metric == 'trainloss':
            self.sendloss(loss)
  
            for node in self.neighbor_list:
                #print("neighbor_loss: ", self.neighbor_loss[node])
                utility[node] = self.neighbor_loss[node]
        elif self.utility_metric == 'testloss':
            
            t_loss = test_loss(model, test_loader, self.criterion)
            self.sendloss(t_loss)
    
            for node in self.neighbor_list:
                #print("neighbor_loss: ", self.neighbor_loss[node])
                utility[node] = 1/self.neighbor_loss[node]        
                
        elif self.utility_metric == 'test_train_loss':
            
            t_loss = test_loss(model, test_loader, self.criterion)
            self.sendloss(loss - t_loss)
 
            for node in self.neighbor_list:
                #print("neighbor_loss: ", self.neighbor_loss[node])
                utility[node] = self.neighbor_loss[node] 
        elif self.utility_metric == 'test_accuracy':
            t_acc= test_accuracy(model, test_loader, self.rank)
            self.sendloss(t_acc)
      
            for node in self.neighbor_list:
                #print("neighbor_loss: ", self.neighbor_loss[node])
                utility[node] = self.neighbor_loss[node]
            #print(f"node:{self.rank}, utility: {utility}")    
            
        elif self.utility_metric == 'weight_difference': 
            self.param = flatten_tensors(self.tensor_list).cpu().detach().numpy()
            send_buff = self.param
            self.traffic += self.degree * (send_buff.nbytes / (1024.0 ** 2))
            
            
            self.recv_buf = np.zeros_like(send_buff)
            for node in self.neighbor_list:
                self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=self.recv_buf, source=node)
                self.neighbor_param[node] = np.copy(self.recv_buf)
                #difference = np.sum(np.abs(self.param - self.neighbor_param[node]))
                difference = np.linalg.norm(self.param - self.neighbor_param[node])
                utility[node] = difference
                
        elif self.utility_metric == 'euclidweight_difference':
       
            self.neighbor_param = {}
                            
            for name, layer in model.named_modules():
                if isinstance(layer, nn.Linear):
                #if isinstance(layer, nn.Conv2d):
                    last_params = layer.weight.data
                    last_layer = layer
                    
                    
                    
                    
                    #last_conv_layer_name = name
                   # break
                
#                if self.rank == 2 and self.iteration==2:
#                    total_params = 0
#
#                    # Iterate through the named modules and print the number of parameters
#                    for name, module in model.named_modules():
#                        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#                        # Count the parameters for convolutional and linear (fully connected) layers
#                            num_params = sum(p.numel() for p in module.parameters())
#                            total_params += num_params
#                            print(f"{name}: {num_params} parameters")
            
                        # Print the total number of parameters in the model
#                    print(f"Total Parameters: {total_params}")
            #num_params_last_conv = last_conv_layer.weight.numel()
            #print(f"Number of parameters in the last convolutional layer {last_conv_layer_name}: {num_params_last_conv}")
            last_params_numpy = last_params.cpu().detach().numpy()
            send_buff = np.copy(last_params_numpy)
            
            self.traffic += self.degree * (send_buff.nbytes / (1024.0 ** 2))
            
                
            
        
        
#            self.param = flatten_tensors(self.tensor_list).cpu().detach().numpy()
#            send_buff = self.param
            self.recv_buf = np.zeros_like(send_buff)
            last_params_sum = np.copy(last_params_numpy)
            for node in self.neighbor_list:
                self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=self.recv_buf, source=node)
                self.neighbor_param[node] = np.copy(self.recv_buf)
                #difference = np.sum(np.abs(self.param - self.neighbor_param[node]))
                
                diff_vector = last_params_numpy - self.neighbor_param[node]
                l2_distance = np.linalg.norm(diff_vector)
                utility[node] = l2_distance
                #print(f"last_params_numpy: {last_params_numpy}, neighbor_param: {self.neighbor_param[node]}")
                
#                dot_product = np.dot(last_params_numpy, self.neighbor_param[node].T)
#
#                magnitude_last = np.linalg.norm(last_params_numpy)
#                magnitude_neighbor = np.linalg.norm(self.neighbor_param[node])
#
#                # Calculate the cosine similarity
#                cosine_similarity = dot_product / (magnitude_last * magnitude_neighbor)
#
#                utility[node] = 1 - (np.mean(cosine_similarity))
                #print("utility: ", utility[node])
                               

                #last_params_sum += self.neighbor_param[node]
           # average_last_params = last_params_sum / (self.degree + 1)
            #average_params = torch.from_numpy(average_last_params).cuda()
            #last_layer.weight.data = average_params 
                
                
                
        elif self.utility_metric == 'weightloss':
            
            self.param = flatten_tensors(self.tensor_list).cpu().detach().numpy()
            send_buff = self.param
            self.recv_buf = np.zeros_like(self.param)
            difference = {}
            loss_utility = {}
            for node in self.neighbor_list:
                self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=self.recv_buf, source=node)
                self.neighbor_param[node] = np.copy(self.recv_buf)
                #difference[node] = np.sum(np.abs(self.param - self.neighbor_param[node]))
                diff_vector = self.param - self.neighbor_param[node]
                l2_distance = np.linalg.norm(diff_vector)  # Calculate L2 (Euclidean) norm
                difference[node] = l2_distance
                loss_utility[node]=1/self.neighbor_loss[node]
                

            
            normalized_diff = {}
            normalized_loss = {}
            for node in self.neighbor_list:
                normalized_diff[node] = (difference[node] - min(difference.values())) / (max(difference.values()) - min(difference.values()))
                
                normalized_loss[node] = (loss_utility[node] - min(loss_utility.values())) / (max(loss_utility.values()) - min(loss_utility.values()))
                utility[node] = (0.5 * normalized_diff[node]) + (0.5 * (normalized_loss[node]))
                
        
        
        
        elif self.utility_metric == 'loss_difference_on_client': 
            self.param = flatten_tensors(self.tensor_list).cpu().detach().numpy()
            send_buff = self.param
            self.traffic += self.degree * (send_buff.nbytes / (1024.0 ** 2))
            
            
            self.recv_buf = np.zeros_like(self.param)
            neighbor_param = {}
            #copied_model = copy.deepcopy(model)
            original_state_dict = model.state_dict()
            for node in self.neighbor_list:
                self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=self.recv_buf, source=node)
                #neighbor_param[node] = np.copy(self.recv_buf)
                
                n_params = torch.from_numpy(self.recv_buf)
                
                #if self.comm_round == 0 and self.rank ==5:
                    #print("model before changing", model.state_dict()['conv1.weight'])
                for f, t in zip(unflatten_tensors(n_params.cuda(), self.tensor_list),
                self.tensor_list):
                    with torch.no_grad():
                        t.set_(f)
                        
                #if self.comm_round == 0 and self.rank ==0:
                    #print("model after changing", model.state_dict()['conv1.weight'])
                
                model.eval()
                top1 = AverageMeter()
                for data, target in current_data_list:
                    
                # compute output
                    with torch.no_grad():
                        outputs, _ = model(data)
                    loss = self.criterion(outputs, target)
                    top1.update(loss.item(), data.size(0))
                utility[node] = top1.avg
                #print("utility[node]", utility[node])
                    
                model.load_state_dict(original_state_dict)    

            model.train()

                
                    
        
        
        
        elif self.utility_metric == 'softmax_probs_testing':
                        
         
            softmax_probs = {}
            normalized_probs = {}
            loss_utility = {}
            kl_divergence = {}
            probs = compute_softmaxprob(model, test_loader)
            
            self.traffic += self.degree * calculate_value_size(probs)
            
            ref_dist = 0.1 * np.ones(10)
            #normalized_selfprobs = probs / np.sum(probs)
            for node in self.neighbor_list:
               
                softmax_probs[node] = self.comm.sendrecv(probs, dest=node, source=node)
                #print(f"softmax {softmax_probs[node]} for {node} and own softmax {self.rank} is {probs}")
                
                #utility[node] = np.linalg.norm(probs - softmax_probs[node])
                #print(f"utility {utility[node]} for {node} in {self.rank}") 
                
                normalized_probs[node]= softmax_probs[node] / np.sum(softmax_probs[node])
                selfnormalized_probs= probs / np.sum(probs)
              #  assert not np.any(normalized_probs[node] == 0.0), "Division by zero encountered"
                kl_divergence[node] = np.sum(ref_dist * np.log2(ref_dist / normalized_probs[node]))
                selfkl_divergence = np.sum(ref_dist * np.log2(ref_dist / selfnormalized_probs))
                utility[node] = 1/kl_divergence[node]
                self.selfutility = 1/selfkl_divergence
                
        elif self.utility_metric == 'softmax_probs_training':
            probs = {}
            normalized_probs = {}
            loss_utility = {}
            kl_divergence = {}
            ref_dist = 0.1 * np.ones(10)
            #normalized_selfprobs = self.softmax_probs / np.sum(self.softmax_probs)
            
            self.traffic += self.degree * calculate_value_size(self.softmax_probs)
            
            for node in self.neighbor_list:
               
                probs[node] = self.comm.sendrecv(self.softmax_probs, dest=node, source=node)
                #print(f"softmax {probs[node]} for {node} and own softmax {self.rank} is {self.softmax_probs}")
                
                utility[node] = np.linalg.norm(probs[node] - self.softmax_probs)
                #print(f"utility {utility[node]} for {node} in {self.rank}") 
                
#                normalized_probs[node]= probs[node] / np.sum(probs[node])
#                assert not np.any(normalized_probs[node] == 0.0), "Division by zero encountered"
#                kl_divergence[node] = np.sum(ref_dist * np.log2(ref_dist / normalized_probs[node]))
#                utility[node] = kl_divergence[node] 
        elif self.utility_metric == 'cosine_similarity_weight':
            self.neighbor_param = {}
                            
            for name, layer in model.named_modules():
                #if isinstance(layer, nn.Linear):
                if self.layer == 'first_cv' or self.layer == 'last_cv':
                    if isinstance(layer, nn.Conv2d):
                        last_params = layer.weight.data
                        last_layer = layer
                        if self.layer == 'first_cv':
                            break
                elif self.layer == 'first_fc' or self.layer == 'last_fc':
                    if isinstance(layer, nn.Linear):
                        last_params = layer.weight.data
                        last_layer = layer
                        if self.layer == 'first_fc':
                            break
                    
                    
                    
                    
                    #last_conv_layer_name = name
                   # break
                
#                if self.rank == 2 and self.iteration==2:
#                    total_params = 0
#
#                    # Iterate through the named modules and print the number of parameters
#                    for name, module in model.named_modules():
#                        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#                        # Count the parameters for convolutional and linear (fully connected) layers
#                            num_params = sum(p.numel() for p in module.parameters())
#                            total_params += num_params
#                            print(f"{name}: {num_params} parameters")
            
                        # Print the total number of parameters in the model
#                    print(f"Total Parameters: {total_params}")
#            num_params_last_conv = last_layer.weight.numel()
#            print(f"Number of parameters in the last convolutional layer: {num_params_last_conv}")
            last_params_numpy = (last_params).cpu().detach().numpy()
            send_buff = np.copy(last_params_numpy)
            
            self.traffic += self.degree * (send_buff.nbytes / (1024.0 ** 2))
            #print(f"utility traffic: {self.traffic}, send_buff.nbytes: {send_buff.nbytes}")    
            
        
        
#            self.param = flatten_tensors(self.tensor_list).cpu().detach().numpy()
#            send_buff = self.param
            self.recv_buf = np.zeros_like(send_buff)
            last_params_sum = np.copy(last_params_numpy)
            for node in self.neighbor_list:
                self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=self.recv_buf, source=node)
                n_params = np.copy(self.recv_buf)
                #difference = np.sum(np.abs(self.param - self.neighbor_param[node]))
                
                self.neighbor_param[node] = torch.from_numpy(n_params).cuda()
                #self.neighbor_param[node] = [torch.tensor(t) for t in neighbor_param_unflatten]
                
                utility[node] = 1 - torch.cosine_similarity(
                        self.neighbor_param[node],
                        last_layer.weight.data,
                        dim=-1,
                    ).mean()
                    
              #  if self.rank == 0:
               #   print(f"utility of node {node} is {utility[node]}")
                
          
                #print(f"last_params_numpy: {last_params_numpy}, neighbor_param: {self.neighbor_param[node]}")
                
#                dot_product = np.dot(last_params_numpy, self.neighbor_param[node].T)
#
#                magnitude_last = np.linalg.norm(last_params_numpy)
#                magnitude_neighbor = np.linalg.norm(self.neighbor_param[node])
#
#                # Calculate the cosine similarity
#                cosine_similarity = dot_product / (magnitude_last * magnitude_neighbor)
#
#                utility[node] = 1 - (np.mean(cosine_similarity))
                #print("utility: ", utility[node])
                               

                last_params_sum += self.neighbor_param[node].cpu().detach().numpy()
                
#            if self.rank == 0:
#                print("last layer before updating:", last_layer.weight.data)     
#            average_last_params = last_params_sum / (self.degree + 1)
#            average_params = torch.from_numpy(average_last_params).cuda()
#            with torch.no_grad():
#                last_layer.weight.data = average_params
#            self.prepare_comm_buffer()
                
#            if self.rank == 0:
#                print("last layer after updating:", last_layer.weight.data) 

                        
        
        elif self.utility_metric == 'cosine_similarity_features':
            neighbor_feature = {}
            
            numpy_features = np.array(self.list_features)
            #print("numpy features:",numpy_features) 
    
            send_buff = np.copy(numpy_features)
            
            self.traffic += self.degree * (send_buff.nbytes / (1024.0 ** 2))
            #print("number of elements:", send_buff.size)
            #4 bytes
            #print("traffic:", self.traffic)    
            
        
        
#            self.param = flatten_tensors(self.tensor_list).cpu().detach().numpy()
#            send_buff = self.param
            self.recv_buf = np.zeros_like(send_buff)
            #last_params_sum = np.copy(last_params_numpy)
            for node in self.neighbor_list:
                self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=self.recv_buf, source=node)
                n_features = np.copy(self.recv_buf)
                #difference = np.sum(np.abs(self.param - self.neighbor_param[node]))
                
                neighbor_feature[node] = torch.from_numpy(n_features).cuda()
                torch_features = torch.from_numpy(numpy_features).cuda()
                #print("torch_features.shape", torch_features.shape)
                #self.neighbor_param[node] = [torch.tensor(t) for t in neighbor_param_unflatten]
                
                utility[node] = 1 - torch.cosine_similarity(
                        neighbor_feature[node],
                        torch_features,
                        dim=-1,
                    ).mean()
                    
            self.list_features.clear()                
            
            
        
        
        else:
            print("errorrrrrrrrrrrrrrrrrrrrrrrrr")
                  
                
                
        self.utility = utility
        
        
    def calculate_value_size(self, value):
        # Convert the value to bytes using struct.pack
        value_bytes = struct.pack('f', value)

        # Get the size of the value in bytes
        size_bytes = len(value_bytes)

        # Convert bytes to megabytes
        size_mb = size_bytes / (1024.0 ** 2)

        return size_mb

            
            
    def select_client(self, model, test_loader, loss, current_data_list):
        #self.epoch = epoch      
        #if self.selection_mode == 'random' or self.iteration==1:
        if self.selection_mode == 'random':
            self.selected_node = random.sample(self.neighbor_list, 1)
            self.selected_node = np.array(self.selected_node)
            
        elif self.selection_mode == 'greedy':
            self.selected_node = np.array([self.greedyalgorithm(model, test_loader)])

        elif self.selection_mode == 'bnz':
            self.utility_function(model, test_loader, loss, current_data_list)
            
            sorted_dict = dict(sorted(self.utility.items(), key=lambda item: item[1], reverse=True))
            # Get the top k elements
            top_k_elements = dict(list(sorted_dict.items())[:1])
            self.selected_node = np.array(list(top_k_elements.keys()))
            
            
            
#            greatest_utility = max(self.utility.values())
#            threshold = greatest_utility * 0.9
#            # Select devices whose performance is above the threshold
#            selected_nodes = [node for node, utl in self.utility.items() if utl > threshold]
#            #print("selected nodes:", selected_nodes)
#            self.selected_node = random.sample(selected_nodes, 1)
            #if self.rank == 7:
            #    print(f"selected node:{self.selected_node}")

            
            
            #self.greedy_selected_node = np.array([self.greedyalgorithm(model, test_loader, self.selected_node)])


            
            
#            final_elements = {}
#            
#            for key in top_k_elements.keys():
#                if top_k_elements[key] > self.loss:
#                    break
#                else:
#                    final_elements[key] = top_k_elements[key]

            
            
            
            
            
            #if(self.rank ==0):
                #print ("sample: ", top_k_elements.keys())
            
        #self.recv_buff = np.empty_like(self.selected_node) 
#        for node in self.neighbor_list:
#            status = MPI.Status()
#            recv_size = self.comm.recv(source=node, tag=0, status=status)
#            self.recv_buff = [0] * recv_size
#            self.comm.Sendrecv(sendbuf=self.selected_node, dest=node, recvbuf=self.recv_buff, source=node)
#            #print("recv buff: ", self.recv_buff)
#            self.is_selected[node] = list(self.recv_buff)
#            #print("selected node: ", self.is_selected[node])
            #print(f"sending the selection information to neighbors from: {self.rank} selected node: {self.selected_node} destination: {node}")

 
        for node in self.neighbor_list:
            #request_send = None
            #request_receive = None
            #status = MPI.Status()
            #message_available = self.comm.Iprobe(source=node, status=status)
            #if message_available:
                #recv_size = self.comm.recv(source=node, status=status)
                #self.recv_buff = [0] * recv_size
            request_receive = self.comm.irecv(source=node)
            if len(self.selected_node) > 0:
                #print("The number of selected nodes: ", self.selected_node.size)
                request_send = self.comm.isend(self.selected_node, dest=node)
            else:
                print("Errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
                null_message = bytearray()
                request_send = self.comm.isend(null_message, dest=node)
                #print("null messageeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee: ",null_message) 
                

            #if request_send is not None:
            request_send.wait()
            #if request_receive is not None:
            received_data = request_receive.wait()
            #print("received data: ", received_data)

            self.is_selected[node] = received_data
                
#                if self.selected_node.size > 0:
#                    self.comm.Sendrecv(sendbuf=self.selected_node, dest=node, recvbuf=self.recv_buff, source=node)
#                    #print("recv buff: ", self.recv_buff)
#                else:
#                    self.comm.Recv(self.recv_buff, source=node)
#                self.is_selected[node] = list(self.recv_buff)
#            else:
#                if self.selected_node.size > 0:
#                    self.comm.Send(self.selected_node, dest=node)
                        
 
            #print(f"sending the selection information to neighbors from: {self.rank} selected node: {self.selected_node} destination: {node}")
    
           
        #for node in self.neighbor_list:
            #data = self.comm.recv(source=node)
            
            #self.comm.Recv(self.recv_buff, source=node)

            #print(f"receiving the selected node information from neighbors to: {self.rank} selected node: {self.is_selected[node]} source: {node}")
            

            
    def sendloss(self, loss):
        self.loss = loss
       # send_buff = np.array(loss)
        #send_buff = loss
       # recv_buff = np.empty(1, dtype=np.float32)
        #recv_buff = None
        
        
        #print(f"rank: {self.rank}, neighbor loss")
        for node in self.neighbor_list:
            #self.comm.Sendrecv(sendbuf=send_buff, source=self.rank, recvbuf=recv_buff, dest=node)
            
            #self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=recv_buff, source=node)
            self.neighbor_loss[node] = self.comm.sendrecv(loss, dest=node, source=node) 
            #print(f"rank: {self.rank}, neighbor loss for {node}: ", self.neighbor_loss[node])
            #self.comm.send(loss, dest=node)
        #received_value = comm.recv(source=src)
        
        self.traffic += self.degree * self.calculate_value_size(loss)
        
        self.comm.Barrier()
    def sendinfo(self,model):
#        last_conv_layer = None
#        for name, layer in model.named_modules():
#            if isinstance(layer, nn.Conv2d):
#                last_conv_layer = layer
#        #params = last_conv_layer.state_dict()
#        params = last_conv_layer.weight.data
#        mean_value = torch.mean(params)
#        self.info=mean_value.item()
        layer_mean_values = []
        layer_values = []
        n_info ={}
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                params = layer.weight.data
                mean_value = torch.mean(params)
                #layer_mean_values[name] = mean_value.item()
                std_value = torch.std(params)
                comb_values = 0.2 * std_value.item() + 0.8 * mean_value.item() 
                #layer_mean_values.append(mean_value.item())
                #layer_std_values.append(std_value.item())
                layer_values.append(comb_values)
        self.info=layer_values

        # Print the mean parameter values for each convolutional layer
        #for name, mean_value in layer_mean_values.items():
            #print(f"Layer: {name}, Mean Value: {mean_value}")

        for node in self.neighbor_list:
            n_info[node] = self.comm.sendrecv(layer_values, dest=node, source=node) 
        return n_info
            
            
#    def sendweight(self):
#        self.param = flatten_tensors(self.tensor_list).cpu().detach().numpy()
#        self.recv_buf = np.zeros_like(self.param)
#        for node in self.neighbor_list:
#            self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=self.recv_buf, source=node)
#            self.neighbor_param[node] = np.copy(self.recv_buf)
#            difference = np.sum(np.abs(self.param - self.neighbor_param[node]))
#            utility[node] = difference
#        self.utility= utility
        
            
#    def sendsoftmax(self, probs):
#        for node in self.neighbor_list:
#            self.neighbor_probs[node] = self.comm.sendrecv(probs, dest=node, source=node) 
              
        
    def getwight(self):
        weights = {}
        num_selectedneighbors = len(self.selected_node)
#        total_utility = self.selfutility
#        for node in self.selected_node:
#            total_utility += self.utility[node]
            
        for node in self.selected_node:
            #if self.epoch==0:
            weights[node] = (1/(num_selectedneighbors+1))
            #weights[node] = self.utility[node] / total_utility
            #else:
                #weights[node] = (1/self.neighbor_loss[node])/((1/self.neighbor_loss[node])+(1/self.loss))
        return weights 
        
        
    def count_selected_nodes(self, current_node):
        count = sum(1 for nodes in self.is_selected.values() if current_node in nodes)
        return count
        
                   
                
        
        

