import torch
import numpy as np
import os
import pdb
from Communicators.CommHelpers import flatten_tensors
import torch.nn as nn
import torch.distributed as dist
from mpi4py import MPI
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Recorder(object):
    def __init__(self, args, rank):
        # self.record_valacc = list()
        self.record_timing = list()
        self.record_total_timing = list()
        self.record_comp_timing = list()
        self.record_comm_timing = list()
        self.record_losses = list()
        self.record_trainacc = list()
        
        self.record_testacc = list()
        
        self.record_testloss = list()
        self.total_record_timing = list()
        self.args = args
        self.rank = rank
        self.saveFolderName = args.outputFolder + '/' + self.args.name + '-' + str(self.args.graph) + '-' \
                              + str(self.args.sgd_steps) + 'sgd-' + str(self.args.epoch) + 'epochs'
        if rank == 0 and not os.path.isdir(self.saveFolderName):
            os.mkdir(self.saveFolderName)

    def add_new(self, comp_time, comm_time, epoch_time, total_time, top1, losses, test_loss, test_accuracy):
        self.record_timing.append(epoch_time)
        self.record_total_timing.append(total_time)
        self.record_comp_timing.append(comp_time)
        self.record_comm_timing.append(comm_time)
        self.record_trainacc.append(top1)
        self.record_losses.append(losses)
        # self.record_valacc.append(val_acc)
        self.record_testloss.append(test_loss)
        
        self.record_testacc.append(test_accuracy)

    def save_to_file(self):
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-epoch-time.log', self.record_timing, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-total-time.log', self.record_total_timing,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-comptime.log', self.record_comp_timing,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-commtime.log', self.record_comm_timing,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-losses.log', self.record_losses, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-tacc.log', self.record_trainacc, delimiter=',')
       
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-testacc.log', self.record_testacc, delimiter=',')
        
        # np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-vacc.log', self.record_valacc, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-testloss.log', self.record_testloss, delimiter=',')
        with open(self.saveFolderName + '/ExpDescription', 'w') as f:
            f.write(str(self.args) + '\n')
            f.write(self.args.description + '\n')


def compute_accuracy(output, target, topk=(1,)):
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
def test_accuracy_test(model, save_inputs, save_targets, rank, greedy=False): 
    model.eval()
    top1 = AverageMeter()
    tensor_listm = list()
    
    target_list = []
    #for batch_idx, (inputs, targets) in enumerate(test_loader):
        #if greedy == True and batch_idx == 1:
            #break
            
        #if batch_idx == 1 and rank == 2:
        #    print(f"rank:, {rank}, input: {inputs[:1]}, target: {targets[:1]}")
        
        #target_list.append(targets)
    inputs, targets = save_inputs.cuda(non_blocking=True), save_targets.cuda(non_blocking=True)
    
    
    with torch.no_grad():
        outputs, _ = model(inputs)
        
    if rank == 0 or rank == 1 or rank == 4:
        print(f"rank {rank}, Running Mean of model: {model.layer1[0].bn1.running_mean}")   
    #print(f"rank:, {rank}, input details: {inputs[:1].cpu().numpy()}, target details: {targets[:1].cpu().numpy()}")
        # compute output
        
           # output2 = model.layer1(output2)

#            if batch_idx == 1 and (rank == 0 or rank == 1 or rank == 4):
#                print(f"Rank {rank}, Output2 of conv1 : {output2[:1]}")
                #print("batch size:", targets.size(0))

    
            # Continue for other layers...
    
            #final_output = model.final_layer(outputN)
            #print(f"Rank {rank}, Final Output: {final_output[:1]}")
            
    # flatten tensors
    acc1 = compute_accuracy(outputs, targets)
        #print(f"rank {rank} acc {acc1}")
    top1.update(acc1[0].item(), inputs.size(0))
    #idx = batch_idx
        
    #print(f"targets:{target_list}")

    #print("batch_idx:", idx)
    #pdb.set_trace()
    return top1.avg



def test_accuracy(model, test_loader, rank, greedy=False): 
    model.eval()
    top1 = AverageMeter()
    tensor_listm = list()
#    for param in model.parameters():
#        tensor_listm.append(param)
#        break
    #modelm = flatten_tensors(tensor_listm).cpu().detach().numpy()
   # print(f"rank {rank}, parameters: {modelm[-20:]}")
    #comm = MPI.COMM_WORLD
    #for layer in model.modules():
          
      #  if isinstance(layer, nn.BatchNorm2d):
        # Move the batch normalization statistics from GPU to CPU
#         running_mean_cpu = layer.running_mean.cpu()
#         running_var_cpu = layer.running_var.cpu()
#         
#         # Convert the tensors to NumPy arrays
#         running_mean_np = np.array(running_mean_cpu, dtype=np.float32)
#         running_var_np = np.array(running_var_cpu, dtype=np.float32)
#         
#         # Gather running_mean and running_var values from all ranks
#         all_running_mean = np.empty((comm.Get_size(), running_mean_np.shape[0]), dtype=np.float32)
#         all_running_var = np.empty((comm.Get_size(), running_var_np.shape[0]), dtype=np.float32)
#         
#         comm.Allgather([running_mean_np, MPI.FLOAT], [all_running_mean, MPI.FLOAT])
#         comm.Allgather([running_var_np, MPI.FLOAT], [all_running_var, MPI.FLOAT])
#         
#         # Calculate the average running_mean and running_var
#         average_running_mean = torch.from_numpy(np.mean(all_running_mean, axis=0))
#         average_running_var = torch.from_numpy(np.mean(all_running_var, axis=0))
#         
#         # Set the synchronized statistics back to the batch normalization layer
#         layer.running_mean = average_running_mean
#         layer.running_var = average_running_var
           # print(f"rank {rank}, Running Mean: {layer.running_mean}")
            
    target_list = []
    for batch_idx, (inputs, targets) in enumerate(test_loader):
    
#        if batch_idx == 0 and (rank == 0 or rank == 1):
#            debug_input_info(inputs, batch_idx, rank)
        #if greedy == True and batch_idx == 1:
            #break
            
        #if batch_idx == 1 and rank == 2:
        #    print(f"rank:, {rank}, input: {inputs[:1]}, target: {targets[:1]}")
        
        target_list.append(targets)
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        
       
            #print(f"rank:, {rank}, input details: {inputs[:1].cpu().numpy()}, target details: {targets[:1].cpu().numpy()}")
        # compute output
        
        with torch.no_grad():
            outputs, _ = model(inputs)
            #output2 = model.conv1(inputs)
           # output2 = model.layer1(output2)

#            if batch_idx == 1 and (rank == 0 or rank == 1 or rank == 4):
#                print(f"Rank {rank}, Output2 of conv1 : {output2[:1]}")
                #print("batch size:", targets.size(0))

    
            # Continue for other layers...
    
            #final_output = model.final_layer(outputN)
            #print(f"Rank {rank}, Final Output: {final_output[:1]}")
            
    # flatten tensors
        acc1 = compute_accuracy(outputs, targets)
        #print(f"rank {rank} acc {acc1}")
        top1.update(acc1[0].item(), inputs.size(0))
        idx = batch_idx
        
    #print(f"targets:{target_list}")

    #print("batch_idx:", idx)
    #pdb.set_trace()
    return top1.avg


def debug_input_info(inputs, batch_idx, rank):
    """
    Print debug information for a given input batch.
    """
    # Print the shape of the inputs
    print(f"Rank {rank}, Batch {batch_idx}, Input Shape: {inputs.shape}")

    # Print the first few elements of the inputs for comparison
    print(f"Rank {rank}, Batch {batch_idx}, Input Sample: {inputs.view(-1)[:10].tolist()}")


def test_loss(model, test_loader, criterion):
    model.eval()
    top1 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        # compute output
        with torch.no_grad():
            outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        top1.update(loss.item(), inputs.size(0))
    return top1.avg

def compute_softmaxprob(model, test_loader):
    softmax_probs_list = []
    model.eval()
    softmax_prob = AverageMeter() 
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        # compute output
        with torch.no_grad():
            outputs, _ = model(inputs)
        softmax_probs = torch.softmax(outputs, dim=1)
        softmax_probs_numpy = softmax_probs.cpu().numpy() 
        softmax_probs_list.append(softmax_probs_numpy)
        #print("softmax_probs_numpy:", softmax_probs_numpy)
        
    total_class_probs = np.zeros(10)

    for batch_probs in softmax_probs_list:
        total_class_probs += batch_probs.sum(axis=0)
    #num_batches = len(softmax_probs_list)
    total_samples = sum(len(batch_probs) for batch_probs in softmax_probs_list)
    mean_class_probs = total_class_probs / total_samples

    #mean_class_probs = total_class_probs / num_batches
    #print("mean_class_probs: ", mean_class_probs)
    model.train()
    return mean_class_probs
    
def compute_softmax_train(output, target):
    total_class_probs = np.zeros(10)
    with torch.no_grad():
        batch_size = target.size(0)
        softmax_probs = torch.softmax(output, dim=1)
        softmax_probs_numpy = softmax_probs.cpu().numpy()
    total_class_probs = softmax_probs_numpy.sum(axis=0)
    mean_class_probs = total_class_probs / batch_size
    return mean_class_probs
    
    
class KL_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = (
            self.T
            * self.T
            * nn.KLDivLoss(reduction="batchmean")(output_batch, teacher_outputs)
        )

        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss



