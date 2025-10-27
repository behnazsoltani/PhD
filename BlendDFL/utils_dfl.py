import time
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
import copy
import math
from dfedsam import DFedSAM
from model.cnn_cifar import cnn_cifar10, cnn_cifar100
# do local training and return parameters (ordered dict)
#def local_training(model, criterion, optimizer,images, labels, device, args, schedular=None):

def local_training_dfedsam(epoch, idx, model, criterion, optimizer, train_data, device, args, logger, scheduler=None):

    num_comm_params = 0

    model.to(device)
    model.train()

    dfedsam = DFedSAM(model, optimizer, rho=0.01) 

    # Local training using SAM
    local_state_dict = dfedsam.local_update(
        model.state_dict(), train_data, epoch, args.local_epochs, criterion, device
    )
    
    model.load_state_dict(local_state_dict)
    
    if scheduler:
        scheduler.step()


    return model.state_dict()

    
def local_training(epoch, idx, model, criterion, optimizer,train_data, device, args, logger, algorithm, scheduler=None, masks=None):
 
    model.to(device)
    model.train()
#    init_model = copy.deepcopy(model.state_dict())
    bs = args.batch_size
    batch_loss = []
    # start = time.time()

    for local_epoch in range(args.local_epochs):
        epoch_loss = []
    
        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
            
            assert not torch.isnan(x).any(), "Input features contain NaN values"
            assert not torch.isnan(labels).any(), "Labels contain NaN values"
            
        
                
    
            model.zero_grad()
            
            out = model(x)

            loss = criterion(out, labels.long())
        

        
            loss.backward()

 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)


            optimizer.step()

            
            epoch_loss.append(loss.item())
            
            if masks != None:
                            
                for name, param in model.named_parameters():
                    if name in masks:
                        param.data *= masks[name].to(device)
        
        if (epoch)% args.round_number_evaluation == 0:
            logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(idx, local_epoch, sum(epoch_loss) / len(epoch_loss)))

            

            
    # print('exact time used in training: {}'.format(time.time()-start))
    if scheduler != None:
        scheduler.step()


    
   
          


    return model.state_dict()
        
def local_training_dfedavgm(epoch, idx, model, criterion, optimizer, train_data, device, args, logger, scheduler=None):

    num_comm_params = 0

    model.to(device)
    model.train()
#    init_model = copy.deepcopy(model.state_dict())
  
    # start = time.time()
    

    # Store previous model parameters for momentum computation
    curr_params = {name: param.clone().detach() for name, param in model.named_parameters()}
    prev_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    
    # Compute decayed learning rate inside local_training()
    # if args.apply_lr_decay:
    #     lr = args.lr * (args.lr_decay_rate ** (epoch // args.lr_scheduler_step_size))
    # else:
    #     lr = args.lr
    lr = optimizer.param_groups[0]['lr']
    #print("lr:",lr)
    for local_epoch in range(args.local_epochs):
        epoch_loss = []
    
        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
    
            assert not torch.isnan(x).any(), "Input features contain NaN values"
            assert not torch.isnan(labels).any(), "Labels contain NaN values"
    
            model.zero_grad()
            out = model(x)
            loss = criterion(out, labels.long())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)


            
    
            # Update model parameters using DFedAvgM formula
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                 

                    # Heavy-ball momentum update
                    param.data = curr_params[name] - lr * grad + \
                                args.dfedavgm_momentum * (curr_params[name] - prev_params[name])

                    # Update prev and curr for next iteration
                    prev_params[name] = curr_params[name].clone()
                    curr_params[name] = param.data.clone()
    
            epoch_loss.append(loss.item())
    
        if epoch % 10 == 0:
            logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(idx, local_epoch, sum(epoch_loss) / len(epoch_loss)))
    
#        torch.cuda.empty_cache()
    
#    # Scheduler step if applicable
    if scheduler is not None:
        scheduler.step()


    return model.state_dict()
def copy_dict(dictionary):
    # Create a new dictionary with cloned tensors
    return {name: tensor.detach().cpu().clone() for name, tensor in dictionary.items()}
    


def local_training_kd(epoch, idx, model_name, model, model_weights, criterion, optimizer, train_data, test_data, aggregation_candidates, device, args, logger, before_agg_local_params, weights_label, scheduler=None, masks=None):

    #last_model = copy_dict(model.state_dict())
    # last_model = copy.deepcopy(model)
    # if epoch > 0:
    #   last_model.load_state_dict(model_weights[idx])
    num_comm_params = 0
 
    
    
    # Ensure that all model parameters are updated when you load the new state_dict
    
    

    
    bs = args.batch_size
    batch_loss = []
    # start = time.time()
    
    logits = {}
    neighbor_accuracies = {}
    neighbor_losses = {}
    
#c    ###################### class_wise accuracy #################################
#    
    model_neighbor = {}
    if args.dataset == "cifar10":
        class_num = 10
    elif args.dataset == "cifar100":
        class_num = 100
#    w = copy.deepcopy(model_weights[idx])
    if epoch > 0:
    
        


        for clnt in aggregation_candidates:
    #        if clnt == idx:
    #            continue
            
            

            model_neighbor[clnt] = create_model(args, model_name=args.model, class_num = class_num)
         
            trainable_state_dict = {}
            for name, param in model_neighbor[clnt].named_parameters():
                if param.requires_grad:  # Only include parameters that are trainable

                    trainable_state_dict[name] = model_weights[clnt][name]
  
            model_neighbor[clnt].load_state_dict(trainable_state_dict)

       

  
##########################################################

    local_ep = args.local_epochs
    for local_epoch in range(local_ep):
        
        epoch_loss = []
        agg_distillation_loss = []
        feature_distillation = []
        
        


       #===================== training on private data ====================
    
        features = []  # To store captured features


        if len(train_data) == 0:
            raise ValueError("Train data loader is empty!")
        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
       
           #     input()
           
            assert not torch.isnan(x).any(), "Input features contain NaN values"
            assert not torch.isnan(labels).any(), "Labels contain NaN values"
            
            model.train()
            model.to(device)
     
            model.zero_grad()

            
            output = model(x)
 
 
 #============= adaptive underrepresented class weighting ==========================================
            aggregated_logits = None
            aggregated_distillation_loss = torch.tensor(0.0, device=device)
            total_weight = 0.0
            num_classes = output.shape[1]
            sample_weights = compute_adaptive_class_weights(labels, num_classes, epoch, args.comm_round, max_weight=5.0)
                    

            if epoch > 0:
                # === Aggregate neighbor logits ===
                for clnt in aggregation_candidates:
                    teacher = model_neighbor[clnt]
                    teacher.to(device)
                    teacher.eval()
                    with torch.no_grad():
                        out = teacher(x)  # Teacher's logits for the current batch
                    #teacher.to("cpu")
                    weight_client = 1.0  # Uniform weighting per neighbor
                    total_weight += weight_client

                    if aggregated_logits is None:
                        aggregated_logits = weight_client * out
                    else:
                        aggregated_logits += weight_client * out

                if aggregated_logits is not None:
                    aggregated_logits /= total_weight  # Average teacher logits

                 
         
                    T = args.temperature
                    
                    kl_loss = F.kl_div(
                        F.log_softmax(output / T, dim=1),
                        F.softmax(aggregated_logits / T, dim=1),
                        reduction="none"
                    ).sum(dim=1)  # shape: [batch_size]

                    #sample_weights = adaptive_weights[labels]  # [batch_size]
                    weighted_loss  = (kl_loss * sample_weights).sum() / sample_weights.sum()
                                            
                    aggregated_distillation_loss = weighted_loss * (T ** 2)
      ########################################################

            #================================================================================
            

            ce_loss = (F.cross_entropy(output, labels, reduction='none') * sample_weights).mean()

            kd_weight = args.kd_weight

            #ablation
            #kd_weight = 0

         
            loss = ce_loss + (kd_weight * aggregated_distillation_loss)
   
            loss.backward() 
            # to avoid nan loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            
           
            
            epoch_loss.append(loss.item())
            agg_distillation_loss.append(aggregated_distillation_loss.item())
            #feature_distillation.append(feature_distillation_loss.item())
           
           # del loss, output
            #torch.cuda.empty_cache()
       
   
        if (epoch)% args.round_number_evaluation == 0:
            logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(idx, local_epoch, sum(epoch_loss) / len(epoch_loss)))
            logger.info('Client Index = {}\tEpoch: {}\tlogit_distillation_loss: {:.6f}'.format(idx, local_epoch, sum(agg_distillation_loss) / len(agg_distillation_loss)))
            #logger.info('Client Index = {}\tEpoch: {}\tfeature_distillation_loss: {:.6f}'.format(idx, local_epoch, sum(feature_distillation) / len(feature_distillation)))
     #---------------------------------------------------

    if scheduler != None:
        scheduler.step()
        



    

    return model.state_dict()

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

    return model



def compute_adaptive_class_weights(labels, num_classes, epoch, total_rounds, max_weight=5.0):
    # Step 1: Count class occurrences
    class_counts = torch.bincount(labels, minlength=num_classes).float()

    # Step 2: Compute base weights only for seen classes
    seen_classes = class_counts > 0
    base_weights = torch.zeros(num_classes, dtype=torch.float, device=labels.device)
    base_weights[seen_classes] = 1.0 / class_counts[seen_classes]
    
    # Step 3: Normalize only over seen classes
    base_weights_sum = base_weights[seen_classes].sum()
    base_weights[seen_classes] = base_weights[seen_classes] / base_weights_sum * seen_classes.sum()

    # Step 4: Compute scaling factor
    scaling = epoch / total_rounds

    #ablation
    #scaling = 0
    # Step 5: Compute adaptive weights
    adaptive_weights = 1.0 + scaling * (base_weights - 1.0)

    #ablation
    #adaptive_weights = torch.ones_like(labels, dtype=torch.float)
    # Step 6: Return sample-level weights
    return adaptive_weights[labels]


 
    
def get_model_params(model):
    return copy.deepcopy(model.cpu().state_dict())


def local_test(model, test_data, device, args, class_num, logger):
   
   
   
    model.to(device)
    model.eval()


    metrics = {
        'test_correct': 0,
        'test_acc':0.0,
        'test_loss': 0,
        'test_total': 0,
        'classwise_loss': torch.zeros(class_num, dtype=torch.float).to(device),  # Loss per class
        'classwise_correct': torch.zeros(class_num, dtype=torch.int).to(device),  # Track correct per class
        'classwise_total': torch.zeros(class_num, dtype=torch.int).to(device)  # Track total per class
    }

    criterion = nn.CrossEntropyLoss(reduction='none').to(device)  # ? Per-sample loss

    #print(f"length of test data: {len(test_data)}")
    #safe_test_loader = iterate_safely(test_data)
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data):
          
            x = x.to(device)
            target = target.to(device)
       
            
            pred = model(x)
           # logger.info(f"pred: {pred}")  
            
            #pred = model(x)
            loss = criterion(pred, target.long())

            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            metrics['test_correct'] += correct.item()
            #metrics['test_loss'] += loss.item() * target.size(0)
            metrics['test_loss'] += loss.sum().item() 
            metrics['test_total'] += target.size(0)
            
     
 

            for i in range(target.size(0)):  # Iterate over batch
                t = target[i]
                metrics['classwise_total'][t] += 1
                metrics['classwise_loss'][t] += loss[i]  # Add loss for this sample's class
                if predicted[i] == t:
                    metrics['classwise_correct'][t] += 1

                    
        #    logger.info(f"metrics: {metrics}")
        #break
    
    metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']                
     # Calculate per-class accuracy
    classwise_accuracy = metrics['classwise_correct'].float() / metrics['classwise_total'].float()
    
    # Add per-class accuracy to the metrics dictionary
    metrics['classwise_accuracy'] = classwise_accuracy.cpu().numpy()  # Convert to numpy for easy reading
    
     # Compute per-class average loss (to avoid division by zero, add a small value)
    metrics['classwise_loss'] = (metrics['classwise_loss'] / (metrics['classwise_total'] + 1e-8)).cpu().numpy()
    model.train()
 #   model.shared_layers.to('cpu')
    return metrics



    
def calculate_model_size(num_parameters, dtype='float32'):
    dtype_size = 4 if dtype == 'float32' else 8  # float32 = 4 bytes, float64 = 8 bytes
    model_size = num_parameters * dtype_size  # Size in bytes
    return model_size / 1e6  # Convert to MB

def count_communication_params(update): 
    num_non_zero_weights = 0
    
    for name in update:
        num_non_zero_weights += torch.count_nonzero(update[name]).item()
    return num_non_zero_weights

def count_total_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params
    



def replace_batchnorm_with_groupnorm(model, num_groups=8):
    """
    Recursively replace BatchNorm2d layers with GroupNorm in a model.
    
    Args:
        model (nn.Module): The model containing BatchNorm2d layers.
        num_groups (int): Number of groups for GroupNorm.

    Returns:
        nn.Module: The updated model with GroupNorm layers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            # Replace with GroupNorm
            setattr(model, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        elif isinstance(module, nn.Sequential):  
            # Modify layers inside Sequential properly
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module, nn.BatchNorm2d):
                    num_channels = sub_module.num_features
                    module._modules[sub_name] = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
                else:
                    replace_batchnorm_with_groupnorm(sub_module)  # Recursively process
        else:
            replace_batchnorm_with_groupnorm(module)  # Recursively process other modules
            
    return model

