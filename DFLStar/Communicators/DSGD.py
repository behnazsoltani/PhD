import numpy as np
import time
import torch
from Communicators.CommHelpers import flatten_tensors, unflatten_tensors
from mpi4py import MPI
import random
import torch.nn as nn
from Utils.Misc import compute_softmaxprob


class decenCommunicator:
    """
    decentralized averaging according to a topology sequence
    For DSGD: Set i1 = 0 and i2 > 0 (any number it doesn't matter)
    For PD-SGD: Set i1 > 0 and i2 = 1
    For LD-SGD: Set i1 > 0 and i2 > 1
    """

    def __init__(self, rank, size, comm, topology, selection, utility_metric, ratio, i1, i2):
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
        #Behnaz
        self.iteration = 0
        self.comm_iter = 0
        self.selection_mode=selection
        self.utility_metric = utility_metric
        self.epoch = None
        self.ratio = ratio
        
        self.param_neighbor = {}
        self.neighbor_param = {}
        self.utility = {}
        self.is_selected = {}
        self.selected_node = []
        self.unexplored = set()
        self.neighbor_loss = {}
        self.neighbor_info = {}
        self.softmax_probs = []
        self.loss = None
        self.info = None

    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu() 
        self.recv_buffer = torch.zeros_like(self.send_buffer)
        self.train_buffer = torch.zeros_like(self.send_buffer)
        #if self.iteration == 1:
            #print(f"rank {self.rank}: {self.send_buffer}")
    

    def averaging(self):

        #print ("Behnazzzzzzzzzzzzzzzzzzzzzz") 
        
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
        
        #Behnaz
        #self.param = np.copy(send_buff)
        
        self.recv_tmp = np.empty_like(send_buff)
        # decentralized averaging
        for idx, node in enumerate(self.neighbor_list):
            self.comm.Sendrecv(sendbuf=send_buff, source=node, recvbuf=self.recv_tmp, dest=node)
            # Aggregate neighbors' models: alpha * sum_j x_j
#            if self.rank == 0:
#                print(f"rank: {self.rank} model parameter before averaging: {self.recv_buffer}")
            self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=self.neighbor_weights[idx])
            self.param_neighbor[node] = np.copy(self.recv_tmp)
#            if self.rank==0:
#                print(f"rank: {self.rank} received from node: {node}") 
#                print(f"rank: {self.rank} sent to node: {node}")
#                print(f"rank: {self.rank} model parameter of neighbor: {torch.from_numpy(self.recv_tmp)}")
#                print(f"rank: {self.rank} model parameter after averaging: {self.recv_buffer}")
            

        self.comm.Barrier()
        toc = time.time()

        return toc - tic
        
    def averaging2(self):

        #print ("Behnazzzzzzzzzzzzzzzzzzzzzz") 
        
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
        
        #Behnaz
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
        
        
    def select_averaging(self):

        self.comm.Barrier()
        tic = time.time()
        
        

        # compute self weight according to degree
        self.n_weights = self.getwight()
        weights_list = [value for value in self.n_weights.values()]
        selfweight = 1 - np.sum(weights_list)
        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        self.recv_buffer.add_(self.send_buffer, alpha=selfweight)
#        if self.rank == 0:
#            print(f"rank: {self.rank} own parameter: {self.send_buffer}")
#            print(f"rank: {self.rank} own weight: {selfweight}")

        send_buff = self.send_buffer.detach().numpy()
        #print("the content of send buffer: ", send_buff) 
        
        #Behnaz
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
#                if self.rank == 0:
#                    print(f"rank: {self.rank} model parameter before averaging: {self.recv_buffer}")
                self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=self.n_weights[node])
#                if self.rank==0:
#                    print(f"rank: {self.rank} received from node: {node}") 
#                    print(f"rank: {self.rank} sent to node: {node}")
#                    print(f"rank: {self.rank} model parameter of neighbor: {torch.from_numpy(self.recv_tmp)} and its weight: {self.n_weights[node]}")
#                    print(f"rank: {self.rank} model parameter after averaging: {self.recv_buffer}")
                self.param_neighbor[node] = np.copy(self.recv_tmp)
                self.unexplored.add(node)
                continue     
            
            
            
            else:
                if self.rank in self.is_selected[node]:
                    #print(f"rank: {self.rank} is sending to node: {node}") 
                    req = self.comm.Isend(send_buff, dest=node) 
                    req.Wait()
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
#                    if self.rank==0:
#                        print(f"rank: {self.rank} received from node: {node}") 
#                        print(f"rank: {self.rank} model parameter of neighbor: {torch.from_numpy(self.recv_tmp)} and its weight: {self.n_weights[node]}")
#                        print(f"rank: {self.rank} model parameter after averaging: {self.recv_buffer}")
                        
                    self.param_neighbor[node] = np.copy(self.recv_tmp)
                    self.unexplored.add(node)   
                
                

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

    def communicate(self, model, epoch, test_loader):
    
#        if self.iteration == 0 and self.iter == 0:
#            #print("self.iter: ", self.iter)
#            #print("self.iteration: ", self.iteration)  
#            self.tensor_lst = list()
#            for param in model.parameters():
#                self.tensor_lst.append(param)
#            print(f"rank: {self.rank} parameters: {flatten_tensors(self.tensor_lst).cpu()}")
    
        self.epoch = epoch

        # Have to have this here because of the case that i1 = 0 (cant do 0 % 0)
        self.iter += 1
      
        comm_time = 0

        # I1: Number of Local Updates Communication Set
        if self.iter % (self.i1+1) == 0:
            self.iteration += 1
            self.comm_iter += 1
            
            #Behnaz
            #self.select_client(epoch, model, test_loader)
            
            # stack all model parameters into one tensor list
            self.tensor_list = list()
            for param in model.parameters():
                self.tensor_list.append(param)
                

            # necessary preprocess
            self.prepare_comm_buffer()
            
            #Behnaz
            
           


            # decentralized averaging according to activated topology
            # record the communication time
            comm_time += self.averaging()
            #comm_time += self.averaging2()  

            #comm_time +=self.select_averaging() 

            # update local models
            self.reset_model()
            
             

            # I2: Number of DSGD Communication Set
            if self.comm_iter % self.i2 == 0:
                self.comm_iter = 0
            else:
                # decrease iteration by one in order to run another one update and average step (I2 communication)
                self.iter -= 1
                
            #Behnaz


        return comm_time
        
        
    def utility_function(self, model, test_loader): 
    
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
        if self.utility_metric == 'weight_distribution':
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
                
        
        elif self.utility_metric == 'test_loss':
            for node in self.neighbor_list:
                #print("neighbor_loss: ", self.neighbor_loss[node])
                utility[node] = 1/self.neighbor_loss[node]
                
        elif self.utility_metric == 'test_accuracy':
            for node in self.neighbor_list:
                #print("neighbor_loss: ", self.neighbor_loss[node])
                utility[node] = self.neighbor_loss[node]
            #print(f"node:{self.rank}, utility: {utility}")    
            
        elif self.utility_metric == 'weight_difference': 
            self.param = flatten_tensors(self.tensor_list).cpu().detach().numpy()
            send_buff = self.param

            
            
            self.recv_buf = np.zeros_like(last_conv_params_numpy)
            for node in self.neighbor_list:
                self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=self.recv_buf, source=node)
                self.neighbor_param[node] = np.copy(self.recv_buf)
                difference = np.sum(np.abs(self.param - self.neighbor_param[node]))
                utility[node] = difference
                
        elif self.utility_metric == 'euclidweight_difference':
            #print("Behnazzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
            self.neighbor_param = {}
                            
            for name, layer in model.named_modules():
                if isinstance(layer, nn.Conv2d):
                    last_conv_params = layer.weight.data
                    last_conv_layer = layer
            #num_params_last_conv = last_conv_layer.weight.numel()
            #print(f"Number of parameters in the last convolutional layer: {num_params_last_conv}")
            last_conv_params_numpy = last_conv_params.cpu().detach().numpy()
            send_buff = last_conv_params_numpy
            
                
            
        
        
#            self.param = flatten_tensors(self.tensor_list).cpu().detach().numpy()
#            send_buff = self.param
            self.recv_buf = np.zeros_like(send_buff)
            last_conv_params_sum = np.copy(last_conv_params_numpy)
            for node in self.neighbor_list:
                self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=self.recv_buf, source=node)
                self.neighbor_param[node] = np.copy(self.recv_buf)
                #difference = np.sum(np.abs(self.param - self.neighbor_param[node]))
                diff_vector = last_conv_params_numpy - self.neighbor_param[node]
                l2_distance = np.linalg.norm(diff_vector)  # Calculate L2 (Euclidean) norm
                utility[node] = l2_distance
                               

                last_conv_params_sum += self.neighbor_param[node]
            average_last_conv_params = last_conv_params_sum / (self.degree + 1)
            average_params = torch.from_numpy(average_last_conv_params).cuda()
            last_conv_layer.weight.data = average_params 
                
                
                
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
                
        elif self.utility_metric == 'softmax_probs_testing':
                        #Behnaz
            #print("Behnazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
            softmax_probs = {}
            normalized_probs = {}
            loss_utility = {}
            kl_divergence = {}
            probs = compute_softmaxprob(model, test_loader)
            ref_dist = 0.1 * np.ones(10)
            #normalized_selfprobs = probs / np.sum(probs)
            for node in self.neighbor_list:
               
                softmax_probs[node] = self.comm.sendrecv(probs, dest=node, source=node)
                #print(f"softmax {softmax_probs[node]} for {node} and own softmax {self.rank} is {probs}")
                
                #utility[node] = np.linalg.norm(probs - softmax_probs[node])
                #print(f"utility {utility[node]} for {node} in {self.rank}") 
                
                normalized_probs[node]= softmax_probs[node] / np.sum(softmax_probs[node])
                assert not np.any(normalized_probs[node] == 0.0), "Division by zero encountered"
                kl_divergence[node] = np.sum(ref_dist * np.log2(ref_dist / normalized_probs[node]))
                utility[node] = kl_divergence[node] 
                
        elif self.utility_metric == 'softmax_probs_training':
            probs = {}
            normalized_probs = {}
            loss_utility = {}
            kl_divergence = {}
            ref_dist = 0.1 * np.ones(10)
            #normalized_selfprobs = self.softmax_probs / np.sum(self.softmax_probs)
            for node in self.neighbor_list:
               
                probs[node] = self.comm.sendrecv(self.softmax_probs, dest=node, source=node)
                #print(f"softmax {probs[node]} for {node} and own softmax {self.rank} is {self.softmax_probs}")
                
                utility[node] = np.linalg.norm(probs[node] - self.softmax_probs)
                #print(f"utility {utility[node]} for {node} in {self.rank}") 
                
#                normalized_probs[node]= probs[node] / np.sum(probs[node])
#                assert not np.any(normalized_probs[node] == 0.0), "Division by zero encountered"
#                kl_divergence[node] = np.sum(ref_dist * np.log2(ref_dist / normalized_probs[node]))
#                utility[node] = kl_divergence[node] 
        
        else:
            print("errorrrrrrrrrrrrrrrrrrrrrrrrr")
                  
                
                
        self.utility = utility

            
            
    def select_client(self, epoch, model, test_loader):
        self.epoch = epoch      
        if self.selection_mode == 'random' or self.iteration==1:
            self.selected_node = random.sample(self.neighbor_list, 1)
            self.selected_node = np.array(self.selected_node)

        elif self.selection_mode == 'bnz':
            self.utility_function(model, test_loader)
            
            sorted_dict = dict(sorted(self.utility.items(), key=lambda item: item[1], reverse=True))
            # Get the top k elements
            top_k_elements = dict(list(sorted_dict.items())[:1])
            self.selected_node = np.array(list(top_k_elements.keys()))
            
#            greatest_utility = max(self.utility.values())
#            threshold = greatest_utility * 0.95
#            # Select devices whose performance is above the threshold
#            selected_nodes = [node for node, utl in self.utility.items() if utl > threshold]
#            self.selected_node = random.sample(selected_nodes, 1)
#            
            
            
            


            
            
#            final_elements = {}
#            
#            for key in top_k_elements.keys():
#                if top_k_elements[key] > self.loss:
#                    break
#                else:
#                    final_elements[key] = top_k_elements[key]

            
            
            
            #print(f"selected nodes for {self.rank} is {self.selected_node}")
            
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
        send_buff = np.array(loss)
        #send_buff = loss
        recv_buff = np.empty(1, dtype=np.float32)
        #recv_buff = None
        
        
        #print(f"rank: {self.rank}, neighbor loss")
        for node in self.neighbor_list:
            #self.comm.Sendrecv(sendbuf=send_buff, source=self.rank, recvbuf=recv_buff, dest=node)
            
            #self.comm.Sendrecv(sendbuf=send_buff, dest=node, recvbuf=recv_buff, source=node)
            self.neighbor_loss[node] = self.comm.sendrecv(loss, dest=node, source=node) 
            #print(f"rank: {self.rank}, neighbor loss for {node}: ", self.neighbor_loss[node])
            #self.comm.send(loss, dest=node)
        #received_value = comm.recv(source=src)
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
        for node in self.selected_node:
            #if self.epoch==0:
            weights[node] = (1/(num_selectedneighbors+1))
            #else:
                #weights[node] = (1/self.neighbor_loss[node])/((1/self.neighbor_loss[node])+(1/self.loss))
        return weights            
                
        
        

