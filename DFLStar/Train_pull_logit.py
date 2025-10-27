import numpy as np
import time
import argparse
from GDM.Resnet import ResNet
#from GDM.models_pens import CIFAR10Net
from GDM.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNN_OriginalFedAvg, CNN_Cifar10, CNN_CIFAR100, CNN_emnist
#from GDM.resnet18 import ResNet
from GDM.Alexnet import AlexNet
from GDM.mobilenetv2 import MobileNetV2 
from GDM.GraphConstruct import GraphConstruct
from Communicators.AsyncCommunicator import AsyncDecentralized
from Communicators.client_selection_pull import clientSelection
from Communicators.DSGD import decenCommunicator
from mpi4py import MPI
from GDM.Dirichlet import partition_dataset
from Communicators.CommHelpers import flatten_tensors
from Utils.Misc import AverageMeter, Recorder, test_accuracy, test_loss, compute_accuracy, compute_softmax_train
import os
import torch
import torch.utils.data.distributed
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
from minimizers import SAM, ASAM
from scipy.special import kl_div
from scipy.stats import wasserstein_distance
from collections import Counter
cudnn.benchmark = True 


def count_samples_per_label(train_loader):
    label_counts = Counter()

    for data, labels in train_loader:
        label_counts.update(labels.tolist())  # Convert labels to a list and update the counter

    return label_counts

def run(rank, size):

    # set random seed
    torch.manual_seed(args.randomSeed + rank)
    torch.manual_seed(args.randomSeed)
    np.random.seed(args.randomSeed)
    torch.cuda.manual_seed(args.randomSeed)
    random.seed(args.randomSeed)
    #print("random seeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed: ", args.randomSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # select neural network model
    if args.dataset == 'cifar10' or args.dataset == 'mnist' or args.dataset == 'fashion_mnist' or args.dataset == 'cinic10':
      model = CNN_Cifar10()
      num_class = 10
      #model = ResNet(args.resSize, num_class)
    elif args.dataset == 'emnist':
      num_class = 47
      model = CNN_emnist()  
    elif args.dataset == 'cifar100':
      num_class = 100
      #model = ResNet(args.resSize, num_class)
      #model = MobileNetV2()
      model = CNN_CIFAR100()
    #model = Resnet.ResNet(args.resSize, num_class)
    #model = CNN_Cifar10()
    
    
#    tensorlist =[]
#    #for name, param in model.named_parameters():
#    for param in model.parameters():
#        tensorlist.append(param)
#    print(f"rank: {rank}, param:{flatten_tensors(tensorlist)}")
      


    # split up GPUs
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus

    # initialize the GPU being used
    torch.cuda.set_device(gpu_id)
    model = model.cuda(gpu_id)


    # model loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=1e-4,
                          nesterov=args.nesterov) 


    
    
    # guarantee all local models start from the same point
    init_model = sync_allreduce(model, size, MPI.COMM_WORLD)
    


    # load data
    val_split = 0
    train_loader, test_loader, DataRatios  = partition_dataset(rank, size, MPI.COMM_WORLD, args) 
   # print(f"rank: {rank}, local sample ratio: {DataRatios[rank]}") 
   
    #label_counts = count_samples_per_label(train_loader)
    #print(label_counts)


    # load base network topology
    p = 3/size
    GP = GraphConstruct(rank, size, MPI.COMM_WORLD, args.graph, args.weight_type, p=p, num_c=args.num_clusters, rows=args.rows, columns=args.columns)

    if args.comm_style == 'pd-sgd':
        communicator = decenCommunicator(rank, size, MPI.COMM_WORLD, GP, args.selection, args.utility_metric, DataRatios[rank], args.i1, 1)
    elif args.comm_style == 'pd-sgd-selection':
        communicator = clientSelection(rank, size, MPI.COMM_WORLD, GP, args.selection, args.utility_metric, DataRatios[rank], args.i1, args.i2, args.layer)
    elif args.comm_style == 'd-sgd':
        communicator = decenCommunicator(rank, size, MPI.COMM_WORLD, GP, args.selection, 0, 1)
 

    # init recorder
    comp_time = 0
    comm_time = 0
    recorder = Recorder(args, rank)
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.noniid:
        #d_epoch = 200
        d_epoch = 500
    else:
        d_epoch = 100

    MPI.COMM_WORLD.Barrier()
    # start training
    #for epoch in range(args.epoch):
    
    #minimizer = SAM(optimizer, model, args.rho, 0.1)
    iteration = 0
    termination = False
    selection_time = 0
    d_comm_time = 0
    features = []
    batch_list = []
    current_data_list = []
    n_batch = 0
    record_time = 0
    
    parametersize = sum(p.numel() for p in model.parameters())
    print("Total parameters:", parametersize)
    while termination==False:
        
        init_time = time.time()
        
        
    
        model.train()
        
      
        #t_loss = test_loss(model, test_loader, criterion)
        #print(f"test loss for {rank} is {t_loss}")
        #communicator.sendloss(t_loss)
        
        #communicator.select_client(epoch, model)
        
        global_params = [p.detach().clone() for p in model.parameters()]
        
        #category_counts = np.zeros(10)
        total_category_counts = np.zeros(10)
        n_data = 0

        # Start training each epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            
            if communicator.comm_round >= args.epoch:
                MPI.COMM_WORLD.Barrier()
                termination = True
                break
                
#            category_counts = {i: 0 for i in range(10)}
#            for label in target:
#                label_int = int(label.item())  # Convert PyTorch tensor to Python int
#                if label_int in category_counts:
#                    category_counts[label_int] += 1
#                else:
#                    print(f"Warning: Label {label_int} is not in the expected range. Ignoring.")
              
                
            # Convert dictionary values to a NumPy array
#            category_array = np.array(list(category_counts.values()))


            # Normalize the array
            #normalized_category_array = category_array / category_array.sum()
            #category_distribution_normilized = {key: value / value.sum(axis=0) for key, value in category_counts.items()}
            
            #category_distribution = category_counts / len(target)
            
            #total_category_counts += normalized_category_array
#            total_category_counts += category_array
#            n_data += len(target)
            #if rank  == 0:
            #    print("category:", total_category_counts)
#            model.train()
            
#            if iteration == 0:
#                tensor_list = list()
#                for param in model.parameters():
#                    tensor_list.append(param)
#                communicator.last_params = flatten_tensors(tensor_list).cpu().detach().numpy()
#                
#            iteration +=1
             
            
                
#            tensor_listm = []
#            for param in model.parameters():
#                tensor_listm.append(param)
#                break
#            modelm = flatten_tensors(tensor_listm).cpu().detach().numpy()
            #if batch_idx==0:
                #print(f"rank {rank}, parameters: {modelm[-20:]}")
    
            #if batch_idx == 10:
                #print (f"rank:  {rank}, label: {target}")
                #print("target: ", target)
            #communicator.select_client(epoch)
            start_time = time.time()
            # data loading
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            current_data_list.append((data,target))
            
            
            #Fedprox
#            proximal_term = 0.0
#            for local_weights, global_weights in zip(model.parameters(), global_params):
#                proximal_term += torch.square((local_weights - global_weights).norm(2))

            # forward pass
            output, _ = model(data)
           
            #print(f"Appended feature with shape: {getfeature(data,model).shape}")
            #if rank == 0 and iteration ==1:
                #print("communicator.list_features:", communicator.list_features)
#logit               
            loss_local = criterion(output, target)
#            if rank == 6:
#                print(f"local loss before {loss_local}, data {data}")

        
            if args.KD == 1:
                #print("kd is trueeeeeeeeeeeeee")
                with torch.no_grad():
                    #local_params = [p.detach().clone() for p in model.parameters()]
                    #model.load_state_dict(global_params.state_dict())
                    if args.dataset == 'cifar10' or args.dataset == 'mnist' or args.dataset == 'fashion_mnist' or args.dataset == 'cinic10':
                        model_evaluation = CNN_Cifar10().cuda(gpu_id)
                        #model_evaluation = ResNet(args.resSize, num_class).cuda(gpu_id)
                    elif args.dataset == 'emnist':
                        model_evaluation = CNN_emnist().cuda(gpu_id)
                    elif args.dataset == 'cifar100':
                        #model_evaluation = MobileNetV2().cuda(gpu_id)
                        #model_evaluation = ResNet(args.resSize, num_class).cuda(gpu_id)
                        model_evaluation = CNN_CIFAR100().cuda(gpu_id)
                    model_evaluation.load_state_dict({name: param for name, param in zip(model.state_dict(), global_params)})
                    model_evaluation.eval()
                    output_global, _ = model_evaluation(data)
                    #loss_global = criterion(output, target)
                    
                #loss = loss_local + 0.05*(output - output_global).norm(2)
                loss = loss_local + args.lambda_kd*(output - output_global).norm(2)
                    
            else:
                loss = loss_local
                
            
            
                
          
                
            #print(f"local output: {output}, global output: {output_global}")
            #print(f"local output: {output}")
#            if communicator.comm_round < 500:
#                loss = loss_local + 0.01*(output - output_global).norm(2)
#            else:
#            initial_value = 0.01
#            maximum_value = 0.1
#            #parameter_value = initial_value + (maximum_value - initial_value) * (communicator.comm_round / args.epoch)
#            # Calculate decay constant (k) based on the desired decay
#            final_value = 0.05
#            decay_constant = -np.log(final_value / initial_value) / args.epoch
#
#            # Calculate decayed values over the rounds
#            decay_value = initial_value * np.exp(-decay_constant * communicator.comm_round)

            #print("parameter value:", parameter_value)
#            loss = loss_local + 0.05*(output - output_global).norm(2)
            

#            loss = loss_local
#            if rank == 6:
#                print(f"local loss after {loss_local}, data {data}")
            
            
            #fedprox
#            loss = criterion(output, target) +  (0.2 / 2) * proximal_term
            
            #if rank == 12:
            #    print(f"proximal term is {proximal_term.item()} and loss is {loss.item()}")
            
          
            #soft_probs = compute_softmax_train(output, target)
            #communicator.softmax_probs = soft_probs

            # record training loss and accuracy
            record_start = time.time()
            acc1 = compute_accuracy(output, target)
            losses.update(loss_local.item(), data.size(0))
            top1.update(acc1[0].item(), data.size(0))
            #print("rank: %d, epoch: %.3f, loss: %.3f, train_acc: %.3f" % (rank, epoch, losses.avg, top1.avg))
            


            record_end = time.time() - record_start
            record_time += record_end
            
       
            #communicator.sendloss(losses.avg)
            
            
            

            # backward pass
            loss.backward()
            
            
            #SAM
#            minimizer.ascent_step()           
#            criterion(model(data), target).backward()
#            minimizer.descent_step()
            
            
            # gradient step
            optimizer.step()
            optimizer.zero_grad()
            
            
            
#            if len(data) == 32 and n_batch < 4:
#                batch_list.append(data)
#                n_batch +=1
            
            
#            if communicator.iter % (communicator.i1+1) == 0:
#                #category_distribution = category_counts / len(target)
#                average_category_distribution = (total_category_counts / n_data) + 1e-10
#                total_category_counts = np.zeros(10)
#                n_data = 0
#                global_distribution = (np.ones(10) / 10) + 1e-10
#                #difference_distribution = np.linalg.norm(average_category_distribution-global_distribution)
#                kl_divergence = np.sum(kl_div(average_category_distribution, global_distribution))
#                #print(f"average category dist: {average_category_distribution} , global_distribution:{global_distribution}")
#                
#                #kl_divergence = np.sum(average_category_distribution * np.log(average_category_distribution / global_distribution))
#                #print("kl divergence:", kl_divergence)
#                #earth_mover_distance = wasserstein_distance(average_category_distribution, global_distribution)
#
#  
#               
#                #category_counts = np.zeros(10)
#                for d in batch_list:           
#                    communicator.list_features.append(getfeature(d,model))
#                
#                batch_list.clear()
#                model.train()
#                n_batch = 0


            # communication happens here
            comm_start = time.time()
            d_comm_time, selection_time = communicator.communicate(model, test_loader, losses.avg, current_data_list, args.PS)
            #if rank ==7:
            #    print("communication time:", d_comm_time)
            comm_t = time.time() - comm_start
            
            #print(f"rank: {rank}, communication time: {comm_t}")
            
            
        
            

            

            
            # gradient step
#            optimizer.step()
 #           optimizer.zero_grad()
            end_time = time.time()

            # compute computational time
            comp_time += (end_time - start_time - comm_t)

            # compute communication time
            comm_time += d_comm_time
            
            
            
            
            #if d_comm_time > 0 and communicator.comm_iter == 0:
#            if args.i1==0:
#                evaluation_round = communicator.comm_round%50
#            else:
#                evaluation_round = communicator.comm_round%10

            if d_comm_time > 0:
                
                global_params = communicator.global_params

            if d_comm_time > 0 and communicator.comm_round%50 == 0:
            #if d_comm_time > 0:
           # if communicator.comm_round >= args.epoch:
                
            
        
            #if communicator.iter % (communicator.i1+1) == 0:
                #communicator.comm_round +=1
            

                t = time.time()
                #t_loss = test_loss(model, test_loader, criterion)
     
                test_acc = test_accuracy(model, test_loader, rank)
                test_time = time.time() - t
                
                comp_time -= record_time
                #Selection_time added
                epoch_time = comp_time + comm_time + selection_time
                #print("selection time:", selection_time)
    
                print("rank: %d, comm_round: %.3f, loss: %.3f, train_acc: %.3f, test_loss: %.3f, test_acc: %0.3f, comp time: %.3f, "
              "epoch time: %.3f" % (rank, communicator.comm_round, losses.avg, top1.avg, 0, test_acc, comp_time, epoch_time))
            
                recorder.add_new(comp_time, comm_time, epoch_time, (time.time() - init_time)-test_time,
                         top1.avg, losses.avg, 0, test_acc)
                         
                         
                comp_time, comm_time = 0, 0
                
                #communicator.sendloss(losses.avg)
                losses.reset()
                top1.reset()
                
                current_data_list.clear()
                
                record_time = 0
                
                
                
                recorder.save_to_file()
                
                #global_params = communicator.global_params
                    
                    
                if not args.customLR:
                    update_learning_rate(optimizer, communicator.comm_round, drop=0.5, epochs_drop=200.0, decay_epoch=d_epoch,
                                        itr_per_epoch=len(train_loader))
#                else:
#                    args.lr *= 0.9992
#                    #args.lr *= 0.99
#                    #args.lr *= 0.98
#                    for param_group in optimizer.param_groups:
#                        param_group["lr"] = args.lr
#                    if rank == 0:
#                        print("learning rate is:", args.lr)




                init_time = time.time()

                
                
                         
            
      
            #t_loss = test_loss(model, test_loader, criterion)
            
            #test_acc = test_accuracy(model, test_loader)
            
            #communicator.sendloss(t_loss)
            #communicator.sendinfo(model)
            #if rank == 12:
                #test_acc = test_accuracy(model, test_loader, rank)
                #print("the accuracy after local training in node 12:", test_acc)
            

        #print(f"node {rank}, optimal selectiom: {communicator.bestcount}/{communicator.selectioncount}") 
        #communicator.bestcount = 0
        #communicator.selectioncount = 0    

        # update learning rate here
#        if not args.customLR:
#            update_learning_rate(optimizer, communicator.comm_round, drop=0.5, epochs_drop=10.0, decay_epoch=d_epoch,
#                                    itr_per_epoch=len(train_loader))
#        else:
#            args.lr *= 0.998
#            #args.lr *= 0.98
#            for param_group in optimizer.param_groups:
#                param_group["lr"] = args.lr
#            print("lr is:", args.lr)
                
                      
#            if epoch == 81 or epoch == 122:
#                args.lr *= 0.1
#                for param_group in optimizer.param_groups:
#                    param_group["lr"] = args.lr

        # evaluate test accuracy at the end of each epoch
#         t = time.time()
#         t_loss = test_loss(model, test_loader, criterion)
#        
#         test_acc = test_accuracy(model, test_loader, rank)
#         test_time = time.time() - t
        #print(f"last test loss for {rank} is {t_loss}")
        
        #communicator.sendloss(test_acc)
        
        #communicator.sendloss(losses.avg)
        #communicator.sendinfo(model)
       # communicator.select_client()

        # evaluate validation accuracy at the end of each epoch
        # val_acc = test_accuracy(model, val_loader)


        # total time spent in algorithm
#        comp_time -= record_time
#        epoch_time = comp_time + comm_time
#
#        print("rank: %d, comm_round: %.3f, loss: %.3f, train_acc: %.3f, test_loss: %.3f, test_acc: %0.3f, comp time: %.3f, "
#              "epoch time: %.3f" % (rank, communicator.comm_round, losses.avg, top1.avg, t_loss, test_acc, comp_time, epoch_time))

      #  recorder.add_new(comp_time, comm_time, epoch_time, (time.time() - init_time)-test_time,
      #                   top1.avg, losses.avg, t_loss, test_acc)
                         
#        if epoch == 99:
#            save_model(model, 'model_parameters100.pth')
#        if epoch == 199:
#            save_model(model, 'model_parameters200.pth')
#        if epoch == 299:
#            save_model(model, 'model_parameters300.pth')
                         
        #communicator.sendloss(t_loss)
        

        # reset recorders
#        comp_time, comm_time = 0, 0
#        losses.reset()
#        top1.reset()
        


    # Save data to output folder
#    recorder.save_to_file()



    MPI.COMM_WORLD.Barrier()
    test_acc = test_accuracy(model, test_loader, rank)
    print("rank %d: Test Accuracy before syncing %.3f" % (rank, test_acc))
    test_acc = test_accuracy(model, test_loader, rank)
    print("rank %d: Test Accuracy again for test %.3f" % (rank, test_acc))
    sync_allreduce(model, size, MPI.COMM_WORLD)
    test_acc = test_accuracy(model, test_loader, rank)
    print("rank %d: Test Accuracy %.3f" % (rank, test_acc))
    with open(f'traffic_info_{rank}.txt', 'w') as file:
        file.write(f"{communicator.traffic}")
    #print(f"Trrafic: {communicator.trrafic} MB")

def getfeature(data,model):
    model.eval()
        
    _, extracted_features = model(data) 
    #print("extracted_features:",extracted_features)
    
    # Using .size()
    #feature_size = extracted_features.size()

    # Alternatively, using .shape
    #feature_shape = extracted_features.shape

    #print("Feature Size:", feature_size)
    #print("Feature Shape:", feature_shape)
    #print("output Shape:", output.shape)
    
    #num_elements = extracted_features.numel()
    #print("Number of Elements:", num_elements)
    
    #size = sum(p.numel() for p in model.parameters())
    #print("Total parameters:", size)
    
    #n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print("Total Trainable Parameters in the Model:", n_params)

        
    extracted_features_numpy = (extracted_features.cpu().detach().numpy())

    #model.train()
    return extracted_features_numpy


def update_learning_rate(optimizer, epoch, drop, epochs_drop, decay_epoch, itr=None, itr_per_epoch=None):
    """
    1) Linearly warmup to reference learning rate (5 epochs)
    2) Decay learning rate exponentially starting at decay_epoch
    ** note: args.lr is the reference learning rate from which to scale up
    ** note: minimum global batch-size is 256
    """
    base_lr = 0.1
    lr = args.lr

    if args.warmup and epoch < 5:  # warmup to scaled lr
        if lr > base_lr:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (lr - base_lr) * (count / (5 * itr_per_epoch))
            lr = base_lr + incr
    elif epoch >= decay_epoch:
        #lr *= np.power(drop, np.floor((1 + epoch - decay_epoch) / epochs_drop))
        lr *= np.power(drop, (1 + epoch - decay_epoch) / epochs_drop)
        

    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #print("lr is:", lr)


def sync_allreduce(model, size, comm):
    senddata = {}
    recvdata = {}
    for param in model.parameters():
        tmp = param.data.cpu()
        senddata[param] = tmp.numpy()
        recvdata[param] = np.empty(senddata[param].shape, dtype=senddata[param].dtype)
    torch.cuda.synchronize()
    comm.Barrier()

    for param in model.parameters():
        comm.Allreduce(senddata[param], recvdata[param], op=MPI.SUM)
    torch.cuda.synchronize()
    comm.Barrier()

    tensor_list = list()
    for param in model.parameters():
        tensor_list.append(param)
        param.data = torch.Tensor(recvdata[param]).cuda()
        param.data = param.data / float(size)
        

    #comm.Barrier()

    # flatten tensors
    initial_model = flatten_tensors(tensor_list).cpu().detach().numpy()
    #print(f"parameters: {flatten_tensors(tensor_list).cpu().detach().numpy()}")

    return initial_model
    
# Save model parameters to a file
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

# Load model parameters from a file
def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
#    model.eval()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--name', '-n', default="default", type=str, help='experiment name')
    parser.add_argument('--description', type=str, help='experiment description')

    parser.add_argument('--model', default="res", type=str, help='model name: res/VGG/wrn')
    parser.add_argument('--comm_style', default='pd-sgd-selection', type=str, help='baseline communicator')
    parser.add_argument('--resSize', default=18, type=int, help='res net size')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate to start from \
                        (if not customLR then lr always 0.1)')
    parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
    parser.add_argument('--epoch', '-e', default=10, type=int, help='total epoch')
    parser.add_argument('--bs', default=64, type=int, help='batch size on each worker')
    parser.add_argument('--noniid', default=1, type=int, help='use non iid data or not')
    parser.add_argument('--degree_noniid', default=0.7, type=float, help='how distributed are labels (0 is random)')
    parser.add_argument('--weight_type', default='uniform', type=str, help='how do workers average with each other')
    parser.add_argument('--unordered_epochs', default=1, type=int, help='calculate consensus after the first n models')

    # Specific async arguments
    parser.add_argument('--wb', default=0, type=int, help='proportionally increase neighbor weights or self replace')
    parser.add_argument('--memory_efficient', default=0, type=int, help='DO store all neighbor local models')
    parser.add_argument('--max_sgd', default=10, type=int, help='max sgd steps per worker')
    parser.add_argument('--personalize', default=0, type=int, help='use personalization or not')

    parser.add_argument('--i1', default=0, type=int, help='i1 comm set, number of local updates no averaging')
    parser.add_argument('--i2', default=1, type=int, help='i2 comm set, number of d-sgd updates')
    parser.add_argument('--sgd_steps', default=1, type=int, help='baseline sgd steps per worker')
    parser.add_argument('--num_clusters', default=1, type=int, help='number of clusters in graph')
    parser.add_argument('--graph', default='ring', type=str, help='graph topology')

    parser.add_argument('--warmup', action='store_true', help='use lr warmup or not')
    parser.add_argument('--nesterov', action='store_true', help='use nesterov momentum or not')
    parser.add_argument('--dataset', default='cifar10', type=str, help='the dataset')
    parser.add_argument('--datasetRoot', default='Data', type=str, help='the path of dataset')
    parser.add_argument('--downloadCifar', default=0, type=int, help='change to 1 if needing to download Cifar')
    parser.add_argument('--p', '-p', action='store_true', help='partition the dataset or not')
    parser.add_argument('--savePath', type=str, help='save path')
    parser.add_argument('--outputFolder', default='Output', type=str, help='save folder')
    parser.add_argument('--randomSeed', default=9001, type=int, help='random seed')
    parser.add_argument('--customLR', default=0, type=int, help='custom learning rate strategy, 1 if using multi-step')
   
    parser.add_argument('--selection', default='random', type=str, help='selection method')
    parser.add_argument('--utility_metric', default='test_loss', type=str, help='selection metric')
    parser.add_argument('--rows', default=4, type=int, help='number of rows')
    parser.add_argument('--columns', default=4, type=int, help='number of columns')
    parser.add_argument('--alpha', default=0.2, type=float, help='control the non-iidness of dataset')
    parser.add_argument('--rho', default=0.1, type=float, help='Rho for SAM')
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--KD', default=0, type=int, help="knowledge distillation")
    parser.add_argument('--lambda_kd', default=0.05, type=float, help="knowledge distillation constant")
    parser.add_argument('--PS', default='1', type=int, help="Is there PS?")
    parser.add_argument('--layer', default='last_cv', type=str, help="layer for participant selection")


    args = parser.parse_args()

    if not args.description:
        print('Please input an experiment description. Exiting!')
        exit()

    if not os.path.isdir(args.outputFolder):
        os.mkdir(args.outputFolder)

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    run(rank, size)
