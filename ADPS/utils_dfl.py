import time
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
import copy
import math
from dfedsam import DFedSAM
from model.cnn_cifar import cnn_cifar10, cnn_cifar100
from model.resnet import customized_resnet18
# do local training and return parameters (ordered dict)
#def local_training(model, criterion, optimizer,images, labels, device, args, schedular=None):


def screen_gradients( model, train_data, device):
    #model = self.model
    model.to(device)
    model.eval()
    # # # train and update
    criterion = nn.CrossEntropyLoss().to(device)
    # # sample one epoch  of data
    model.zero_grad()
    (x, labels) = next(iter(train_data))
    x, labels = x.to(device), labels.to(device)
    log_probs = model.forward(x)
    loss = criterion(log_probs, labels.long())
    loss.backward()
    gradient={}
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradient[name] = param.grad.to("cpu")
    return gradient
    
    
#    for batch_idx, (x, labels) in enumerate(train_data):
#        x, labels = x.to(device), labels.to(device)
#        log_probs = model.forward(x)
#        loss = criterion(log_probs, labels.long())
#        loss.backward()
#        gradient={}
#        for name, param in model.named_parameters():
#            if name not in gradient.keys():
#                gradient[name] = param.grad.to("cpu")
#            else:
#                gradient[name] += param.grad.to("cpu")
#    return gradient

def fire_mask(args, masks, weights, round, device):
    # anneal_factor: max drop ratio
    # more pruning early, less pruning later
    drop_ratio = args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / args.comm_round))
    #drop_ratio = 0.5
   
    new_masks = copy.deepcopy(masks)

    num_remove = {}
    for name in masks:
        num_non_zeros = torch.sum(masks[name])
        num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
        temp_weights = torch.where(masks[name].to(device) > 0, torch.abs(weights[name].to(device)), 100000 * torch.ones_like(weights[name].to(device)))
        x, idx = torch.sort(temp_weights.view(-1).to(device))
        new_masks[name].view(-1)[idx[:num_remove[name]]] = 0
    return new_masks, num_remove



def compute_per_weight_masks(model, grad_accum, device, target_sparsity=0.3, eps=1e-9):
    """
    Compute binary pruning masks for each parameter in the model using
    global thresholding on normalized scores.

    Args:
        model (torch.nn.Module): the model containing parameters.
        grad_accum (dict): dict mapping param names -> accumulated gradients.
        device (torch.device): device for computation.
        target_sparsity (float): fraction of weights to prune overall (default 0.3).
        eps (float): small constant to avoid div by zero in normalization.

    Returns:
        dict: param_name -> mask tensor (same shape as parameter).
    """
    # --- compute raw importance scores ---
    scores = {}
    for name, p in model.named_parameters():
        if p.requires_grad and name in grad_accum:
            # importance = |weight| * |gradient|
            scores[name] = (p.data.detach().abs() * grad_accum[name].to(device).detach().abs())

    # --- normalize per layer (so layers are comparable) ---
    norm_scores = {}
    for name, s in scores.items():
        median = s.median()
        norm_scores[name] = s / (median + eps)

    # --- concatenate all scores ---
    all_scores = torch.cat([s.flatten() for s in norm_scores.values()])
    N = all_scores.numel()
    k = int((1 - target_sparsity) * N)  # number of weights to keep

    # --- find global threshold ---
    if k <= 0:
        threshold = float("inf")  # prune all
    elif k >= N:
        threshold = float("-inf")  # keep all
    else:
        threshold = all_scores.kthvalue(N - k + 1).values

    # --- build masks per parameter ---
    masks = {}
    for name, s in norm_scores.items():
        mask = (s >= threshold).to(dtype=torch.int)
        masks[name] = mask.view_as(s)

    return masks
import torch

def compute_energy_coverage_masks(model, grad_accum, device, coverage=0.97, eps=1e-9, verbose=True):
    """
    Compute pruning masks using the energy-coverage rule:
    keep the smallest set of weights whose importance scores
    sum to >= coverage * total score mass in each layer.

    Args:
        model (torch.nn.Module): the model containing parameters.
        grad_accum (dict): dict mapping param names -> accumulated gradients.
        device (torch.device): device for computation.
        coverage (float): fraction of total score mass to retain (0 < coverage <= 1).
        eps (float): small constant to avoid div-by-zero.
        verbose (bool): if True, prints per-layer sparsity summary.

    Returns:
        dict: param_name -> mask tensor (same shape as parameter).
    """
    masks = {}
    summary = {}

    for name, p in model.named_parameters():
        if p.requires_grad and name in grad_accum:
            # importance scores: |w| * |grad|
            scores = (p.data.detach().abs() * grad_accum[name].to(device).detach().abs())
            flat_scores = scores.view(-1)

            total_score = flat_scores.sum()
            if total_score.item() < eps:
                # all scores are ~0 → prune everything
                masks[name] = torch.zeros_like(scores, dtype=torch.bool)
                summary[name] = (0, flat_scores.numel())
                continue

            # sort scores descending
            vals, _ = torch.sort(flat_scores, descending=True)

            # cumulative sum
            csum = torch.cumsum(vals, dim=0)

            # find the smallest k that covers the target fraction
            k = int((csum >= coverage * total_score).nonzero(as_tuple=False)[0].item() + 1)

            # threshold = kth largest kept score
            thresh = vals[k-1]

            # build mask
            mask = (scores >= thresh)
            masks[name] = mask.view_as(scores)

            # record summary
            kept = int(mask.sum().item())
            total = mask.numel()
            summary[name] = (kept, total)

    # --- print summary ---
    if verbose:
        print("\n[Energy Coverage Pruning Summary]")
        for name, (kept, total) in summary.items():
            pruned = total - kept
            print(f"{name:30s} kept {kept:6d}/{total:6d} "
                  f"({100*kept/total:5.1f}% kept, {100*pruned/total:5.1f}% pruned)")
        print("----------------------------------------------------\n")

    return masks

# def compute_per_weight_masks(model, grad_accum, device, target_sparsity=0.3):
#     """
#     Compute binary pruning masks for each parameter in the model based on
#     weight * gradient magnitude scores (a form of importance).
    
#     Args:
#         model (torch.nn.Module): the model containing parameters.
#         grad_accum (dict): dict mapping param names -> accumulated gradients.
#         device (torch.device): device for computation.
#         target_sparsity (float): fraction of weights to prune (default 0.3).
    
#     Returns:
#         dict: param_name -> mask tensor (same shape as parameter).
#     """
#     # --- compute importance scores ---
#     scores = {}
#     for name, p in model.named_parameters():
#         if p.requires_grad and name in grad_accum:
#             # importance = |weight| * |gradient|
#             scores[name] = (p.data.abs() * grad_accum[name].to(device).abs())

#     # --- build per-weight masks ---
#     masks = {}
#     for name, s in scores.items():
#         # Flatten scores
#         v = s.view(-1)

#         # Number of weights to keep
#         k = int((1 - target_sparsity) * v.numel())
#         k = max(k, 1)  # avoid 0

#         # Threshold = kth largest score
#         thresh = torch.topk(v, k, largest=True).values.min()

#         # Create mask
#         mask = (s >= thresh)

#         # Reshape to original param shape
#         masks[name] = mask.view_as(s)

#     return masks


# def fire_mask_gradientandweight(args, masks, weights, round, device, gradient):
#     # anneal_factor: max drop ratio
#     # more pruning early, less pruning later
#     drop_ratio = args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / args.comm_round))
# #    drop_ratio = 0.5
   
#     new_masks = copy.deepcopy(masks)

#     num_remove = {}
#     for name in masks:
#         num_non_zeros = torch.sum(masks[name])
#         num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
#         temp_weights = torch.where(masks[name].to(device) > 0, torch.abs(weights[name].to(device)) * torch.abs(gradient[name].to(device)), 100000 * torch.ones_like(weights[name].to(device)))
#         x, idx = torch.sort(temp_weights.view(-1).to(device))
#         new_masks[name].view(-1)[idx[:num_remove[name]]] = 0
#     return new_masks, num_remove


#def fire_mask_gradientandweight(args, masks, weights, round, device, gradient, min_keep_ratio=0.1):
    # """
    # Prune the bottom 10% of weights that are below the average saliency score,
    # but ensure at least `min_keep_ratio` of active weights are retained per layer.

    # - saliency = |w| * |KD-grad|
    # - threshold: bottom 10% of scores that are < mean score
    # - min_keep_ratio: minimum fraction of weights to retain
    # """
    # new_masks = copy.deepcopy(masks)
    # num_remove = {}

    # for name in masks:
    #     # Move tensors to device
    #     mask = masks[name].to(device)
    #     weight = weights[name].to(device)
    #     grad = gradient[name].to(device)

    #     # Only consider currently active weights
    #     active_mask = (mask > 0)
    #     num_active = int(active_mask.sum().item())
    #     total_params = mask.numel()

    #     if num_active == 0:
    #         num_remove[name] = 0
    #         continue

    #     # Compute saliency: |w| * |grad|
    #     saliency = torch.abs(weight) * torch.abs(grad)

    #     # Mean saliency over active weights
    #     mean_score = saliency[active_mask].mean()

    #     # Get weights below average
    #     below_avg_mask = (saliency < mean_score) & active_mask
    #     below_avg_scores = saliency[below_avg_mask]
    #     num_below_avg = below_avg_scores.numel()

    #     if num_below_avg == 0:
    #         num_remove[name] = 0
    #         continue

    #     # Compute bottom 10% of below-average weights
    #     num_to_prune = math.floor(0.10 * num_below_avg)

    #     # Ensure we don’t violate the minimum keep ratio
    #     min_allowed = math.ceil(min_keep_ratio * total_params)
    #     max_allowed_to_prune = max(0, num_active - min_allowed)

    #     # Final number of weights to prune
    #     num_to_prune = min(num_to_prune, max_allowed_to_prune)

    #     if num_to_prune == 0:
    #         num_remove[name] = 0
    #         continue

    #     # Sort and get indices of smallest scores
    #     sorted_scores, sorted_indices = torch.sort(below_avg_scores.view(-1))
    #     prune_indices_in_subset = sorted_indices[:num_to_prune]

    #     # Map to full tensor indices
    #     below_avg_flat = below_avg_mask.view(-1)
    #     full_mask_flat = new_masks[name].view(-1)
    #     global_indices = below_avg_flat.nonzero(as_tuple=False).view(-1)
    #     to_prune_indices = global_indices[prune_indices_in_subset]

    #     # Prune selected weights
    #     full_mask_flat[to_prune_indices] = 0
    #     num_remove[name] = int(len(to_prune_indices))

    # return new_masks, num_remove, to_prune_indices


def get_cosine_sparsity(round, total_rounds, max_sparsity):
    """
    Cosine scheduler to smoothly increase target sparsity over time.
    """
    ratio = round / total_rounds
    return max_sparsity * 0.5 * (1 - math.cos(math.pi * ratio))


def fire_mask_dynamic_sparsity(
    masks,
    weights,
    gradient,
    round,
    device,
    total_rounds,
    max_sparsity=0.5,
    min_keep_ratio=0.05,
    weight_percentile=20,
    grad_percentile=20
):
    """
    Cosine-annealed dynamic sparsity pruning:
    - Prunes weights that are both small in |w| and small in |grad|
    - Prunes up to the dynamically scheduled sparsity rate
    - Ensures minimum keep ratio
    - Returns updated mask, pruning counts, and just-pruned mask
    """
    new_masks = copy.deepcopy(masks)
    num_removed = {}
    just_pruned_masks = {}

    target_sparsity = get_cosine_sparsity(round, total_rounds, max_sparsity)

    for name in masks:
        mask = masks[name].to(device)
        weight = weights[name].to(device)
        grad = gradient[name].to(device)

        active_mask = mask > 0
        num_active = int(active_mask.sum().item())
        total_params = mask.numel()

        if num_active == 0:
            num_removed[name] = 0
            just_pruned_masks[name] = torch.zeros_like(mask)
            continue

        # Dynamic thresholds via percentiles
        w_abs = torch.abs(weight[active_mask])
        g_abs = torch.abs(grad[active_mask])

      
        threshold_w = torch.quantile(w_abs, weight_percentile / 100.0)
        threshold_g = torch.quantile(g_abs, grad_percentile / 100.0)

        # threshold_w = 0.8 * torch.abs(weight).mean()
        # threshold_g = 0.8 * torch.abs(grad).mean()

        prune_candidates = (torch.abs(weight) < threshold_w) & \
                            (torch.abs(grad) < threshold_g) & active_mask
        
        num_candidates = int(prune_candidates.sum().item())

        # Calculate how many we are allowed to prune this round
        current_sparsity = 1.0 - (num_active / total_params)
        target_active = int((1 - target_sparsity) * total_params)
        min_keep = int(min_keep_ratio * total_params)
        target_active = max(target_active, min_keep)
        max_to_prune = max(0, num_active - target_active)
        num_to_prune = min(num_candidates, max_to_prune)

        if num_to_prune == 0:
            num_removed[name] = 0
            just_pruned_masks[name] = torch.zeros_like(mask)
            continue

        # Prioritize pruning weakest saliency scores
        saliency = torch.abs(weight) * torch.abs(grad)
        saliency_flat = saliency.view(-1)
        candidate_flat = prune_candidates.view(-1)
        candidate_indices = candidate_flat.nonzero(as_tuple=False).view(-1)
        scores = saliency_flat[candidate_indices]
        _, sorted_idx = torch.sort(scores)
        to_prune_indices = candidate_indices[sorted_idx[:num_to_prune]]

        new_masks[name].view(-1)[to_prune_indices] = 0

        # Track just-pruned mask for regrowth
        just_pruned = torch.zeros_like(mask)
        just_pruned.view(-1)[to_prune_indices] = 1

        num_removed[name] = int(len(to_prune_indices))
        just_pruned_masks[name] = just_pruned

    return new_masks, num_removed




def regrow_weights(
    masks,
    gradients,
    num_remove,
    regrow_rate=0.02,
    device='cuda'
):
    """
    Regrow pruned weights based on gradient magnitude.
    Assumes regrow is not called every round, so no need to exclude just-pruned weights.

    Args:
        masks (dict): Binary mask dict for each layer (0 = pruned, 1 = active).
        gradients (dict): Corresponding gradient dict for each layer.
        regrow_rate (float): Fraction of total weights to regrow per layer.
        device (str): Device to perform operations on.

    Returns:
        new_masks (dict): Updated masks with regrown weights set to 1.
        regrow_stats (dict): Number of weights regrown per layer.
        regrow_masks (dict): Binary mask (1 = newly regrown) for tracking.
    """
    new_masks = {}
    regrow_stats = {}
    regrow_masks = {}

    for name in masks:
        # Move tensors to device for computation
        mask = masks[name].to(device)
        grad = gradients[name].to(device)

        # Candidates for regrowth: currently pruned weights
        pruned_mask = (mask == 0)
        num_candidates = int(pruned_mask.sum().item())
        total_params = mask.numel()
        num_to_regrow = int(regrow_rate * total_params)

        if num_candidates == 0 or num_to_regrow == 0:
            regrow_stats[name] = 0
            regrow_masks[name] = torch.zeros_like(mask)
            new_masks[name] = mask.clone()
            continue

        # Gradients of pruned weights
        grad_values = torch.abs(grad)[pruned_mask]

        # Top-K strongest gradients
        # topk_vals, topk_idx = torch.topk(
        #     grad_values.view(-1), min(num_to_regrow, grad_values.numel())
        # )
        topk_vals, topk_idx = torch.topk(
            grad_values.view(-1), int(0.5 * num_remove[name])
        )

        # Get flat indices to regrow
        pruned_flat = pruned_mask.view(-1)
        candidate_indices = pruned_flat.nonzero(as_tuple=False).view(-1)
        regrow_indices = candidate_indices[topk_idx]

        # Create updated mask
        new_mask = mask.clone().view(-1)
        new_mask[regrow_indices] = 1
        new_mask = new_mask.view_as(mask)

        # # Track regrown positions
        # regrow_mask = torch.zeros_like(mask)
        # regrow_mask.view(-1)[regrow_indices] = 1

        # Store updated info
        new_masks[name] = new_mask
     #   regrow_masks[name] = regrow_mask
        regrow_stats[name] = len(regrow_indices)

    return new_masks



def regrow_mask_gradient(args, mask,  num_remove, device, grad=None):
  
    # w = pruned weights → w = 0
    # Use only gradient to evaluate regrowth potential

    pruned_mask = (mask == 0)
    pruned_grads = torch.abs(grad[pruned_mask])
    mean_active_grad = torch.abs(grad[mask > 0]).mean()
    threshold = mean_active_grad

    # Select top gradients among pruned weights above threshold
    regrow_candidates = (torch.abs(grad) > threshold) & pruned_mask

    # Regrow top-k of those (e.g., 2% of total pruned weights)
    num_to_regrow = int(0.02 * pruned_mask.sum().item())
    flat_idx = regrow_candidates.view(-1).nonzero(as_tuple=False).view(-1)
    top_idx = torch.topk(torch.abs(grad.view(-1)[flat_idx]), k=num_to_regrow).indices
    regrow_indices = flat_idx[top_idx]

    # Set mask to 1 to regrow
    mask.view(-1)[regrow_indices] = 1
    return new_masks


def regrow_mask(args, masks,  num_remove, device, gradient=None):
  
    new_masks = copy.deepcopy(masks)
    
    for name in masks:
       
    # if name not in public_layers:
        # if "conv" in name:
        #If args.dis_gradient_check is False, we use gradient magnitude to decide which parameters to regrow
        if not args.dis_gradient_check:
            temp = torch.where(masks[name].to(device) == 0, torch.abs(gradient[name].to(device)), -100000 * torch.ones_like(gradient[name].to(device)))
            sort_temp, idx = torch.sort(temp.view(-1).to(device), descending=True)
            new_masks[name].view(-1)[idx[:num_remove[name]]] = 1
        else:
            temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]),torch.zeros_like(masks[name]) )
            idx = torch.multinomial( temp.flatten().to(device),num_remove[name], replacement=False)
            new_masks[name].view(-1)[idx]=1
    return 
    
def regrow_mask_gradweight(args, masks,  num_remove, device, gradient=None, weight = None):
  
    new_masks = copy.deepcopy(masks)
    
    for name in masks:
       
    # if name not in public_layers:
        # if "conv" in name:
        #If args.dis_gradient_check is False, we use gradient magnitude to decide which parameters to regrow
        if not args.dis_gradient_check:
            temp = torch.where(masks[name].to(device) == 0, torch.abs(gradient[name].to(device)) * torch.abs(weight[name].to(device)), -100000 * torch.ones_like(gradient[name].to(device)))
            sort_temp, idx = torch.sort(temp.view(-1).to(device), descending=True)
            new_masks[name].view(-1)[idx[:num_remove[name]]] = 1
        else:
            temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]),torch.zeros_like(masks[name]) )
            idx = torch.multinomial( temp.flatten().to(device),num_remove[name], replacement=False)
            new_masks[name].view(-1)[idx]=1
    return new_masks



def screen_kd_gradients(local_model, agg_params, train_data, device, temperature=2.0, alpha=0.0):
    """
    Compute gradients of KD loss between local model and aggregated parameters.
    
    Args:
        local_model: the local PyTorch model
        agg_params: dictionary of aggregated parameters (state_dict)
        train_data: DataLoader or iterable providing (x, labels)
        device: computation device
        temperature: temperature for KD
        alpha: weighting factor between CE loss and KD loss (optional)
    Returns:
        gradient: dictionary of parameter gradients
    """
    # Copy aggregated parameters into a teacher model
    teacher_model = copy.deepcopy(local_model)
    teacher_model.load_state_dict(agg_params)
    teacher_model.to(device)
    teacher_model.eval()
    
    local_model.to(device)
    local_model.train()
   
    x, labels = next(iter(train_data))
    x, labels = x.to(device), labels.to(device)
    
    # Forward pass
    student_logits = local_model(x)
    with torch.no_grad():
        teacher_logits = teacher_model(x)
    
    # KD loss: KL divergence between teacher and student (soft targets)
    kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
    kd_loss = kd_loss_fn(
        nn.functional.log_softmax(student_logits / temperature, dim=1),
        nn.functional.softmax(teacher_logits / temperature, dim=1)
    ) * (temperature ** 2)
    
    # Optional: combine with standard CE loss
    ce_loss_fn = nn.CrossEntropyLoss()
    ce_loss = ce_loss_fn(student_logits, labels.long())
    
    loss = alpha * ce_loss + (1 - alpha) * kd_loss
    
    # Backward pass
    local_model.zero_grad()
    loss.backward()
    
    gradient = {}
    for name, param in local_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gradient[name] = param.grad.detach().cpu()
    
    return gradient
import torch
import torch.nn as nn

def screen_agg_gradients_freezing(epoch, args, local_model, prev_global_weights,
                                      train_data, device, temperature=2.0, alpha=0.0,
                                      num_batches=5):
    """
    Compute accumulated gradients of KD + CE loss between local model
    and the aggregated/global teacher model.

    Frozen params are implicitly handled by using prev_global_weights.

    Args:
        args: experiment args (with dataset, model info).
        local_model: PyTorch model of current client.
        prev_global_weights: dict {param_name: tensor}, previous aggregated model.
        train_data: DataLoader.
        device: torch.device.
        temperature: KD temperature.
        alpha: weight between KD and CE loss.
        num_batches: number of batches to accumulate grads.

    Returns:
        accume_grad: dict {param_name: accumulated gradient tensor}.
    """
    # --- Setup global teacher model ---
    if args.dataset == "cifar10":
        class_num = 10
    elif args.dataset == "cifar100":
        class_num = 100
    else:
        raise ValueError("Unsupported dataset")

    teacher_model = create_model(args, model_name=args.model, class_num=class_num)
    teacher_model.load_state_dict(prev_global_weights)
    teacher_model.to(device)
    teacher_model.eval()

    # --- Prepare local model ---
    local_model.to(device)
    local_model.train()

    accume_grad = {}
    data_iter = iter(train_data)

    for i in range(num_batches):
        try:
            x, labels = next(data_iter)
        except StopIteration:
            break
        x, labels = x.to(device), labels.to(device)

        # Forward student
        student_logits = local_model(x)

        # Forward teacher (global aggregated)
        with torch.no_grad():
            teacher_logits = teacher_model(x)

        # --- KD loss ---
        kd_loss_f = nn.KLDivLoss(reduction='batchmean')
        kd_loss = kd_loss_f(
            nn.functional.log_softmax(student_logits / temperature, dim=1),
            nn.functional.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)

        # --- CE loss ---
        ce_loss_fn = nn.CrossEntropyLoss()
        ce_loss = ce_loss_fn(student_logits, labels.long())

        
        if args.kd_schedule == "linear":
            alpha_t = args.kd_alpha_min + ((epoch+1) / args.comm_round) * (args.kd_alpha_max - args.kd_alpha_min)
        elif args.kd_schedule == "exp":
            alpha_t = get_alpha(epoch+1, args.comm_round, alpha_min=args.kd_alpha_min, alpha_max=args.kd_alpha_max, beta=5.0, warmup_frac=0.2)
        else:
            alpha_t = alpha
        # --- Final loss ---
        loss = alpha_t * kd_loss + (1 - alpha_t) * ce_loss

        # --- Backward pass ---
        local_model.zero_grad()
        loss.backward()

        # Accumulate gradients
        for name, param in local_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in accume_grad:
                    accume_grad[name] = torch.zeros_like(param.grad.detach().cpu())
                accume_grad[name] += param.grad.detach().cpu()

    # --- Normalize accumulated gradients ---
    for name in accume_grad:
        accume_grad[name] /= num_batches

    return accume_grad


def screen_multikd_gradients_freezing(args, local_model, model_weights,
                             aggregation_candidates,
                             train_data, device, temperature=2.0, alpha=0.0,
                             num_batches=5, shared_mask=None, prev_global_weights=None):
    """
    Compute accumulated gradients of KD + CE loss between local model
    and aggregated neighbor models.

    Frozen params (mask=0) for each neighbor are replaced with prev_global_weights
    instead of being ignored.

    Args:
        args: experiment args (with dataset, model info).
        local_model: PyTorch model of current client.
        model_weights: dict {client_id: {param_name: tensor}}
        aggregation_candidates: list of neighbor client ids.
        agg_params: unused here (kept for compatibility).
        train_data: DataLoader.
        device: torch.device.
        temperature: KD temperature.
        alpha: weight between KD and CE loss.
        num_batches: number of batches to accumulate grads.
        shared_mask: dict {client_id: {param_name: mask_tensor}}.
        prev_global_weights: dict {param_name: tensor}, previous aggregated model.
    Returns:
        accume_grad: dict {param_name: accumulated gradient tensor}.
    """
    # --- Setup teacher models ---
    model_neighbor = {}
    if args.dataset == "cifar10":
        class_num = 10
    elif args.dataset == "cifar100":
        class_num = 100
    else:
        raise ValueError("Unsupported dataset")

    for clnt in aggregation_candidates:
        model_neighbor[clnt] = create_model(args, model_name=args.model, class_num=class_num)

        # Build state_dict with freeze handling
        trainable_state_dict = {}
        for name, param in model_neighbor[clnt].named_parameters():
            if param.requires_grad:
                if shared_mask is not None and prev_global_weights is not None:
                    mask = shared_mask[clnt][name].to(device)
                    local_w = model_weights[clnt][name].to(device)
                    global_w = prev_global_weights[name].to(device)
                    # Active params = neighbor’s weight, Frozen params = prev global
                    trainable_state_dict[name] = mask * local_w + (1 - mask) * global_w
                else:
                    trainable_state_dict[name] = model_weights[clnt][name]

        model_neighbor[clnt].load_state_dict(trainable_state_dict)

    # --- Prepare local model ---
    local_model.to(device)
    local_model.train()

    accume_grad = {}
    data_iter = iter(train_data)

    for i in range(num_batches):
        try:
            x, labels = next(data_iter)
        except StopIteration:
            break  # not enough data left
        x, labels = x.to(device), labels.to(device)

        # Forward student
        student_logits = local_model(x)

        # --- Aggregate teacher logits fresh for each batch ---
        aggregated_logits = None
        total_weight = 0.0
        for clnt in aggregation_candidates:
            teacher = model_neighbor[clnt].to(device)
            teacher.eval()
            with torch.no_grad():
                out = teacher(x)
            total_weight += 1.0
            aggregated_logits = out if aggregated_logits is None else aggregated_logits + out

        aggregated_logits /= total_weight

        # --- KD loss ---
        kd_loss_f = nn.KLDivLoss(reduction='batchmean')
        kd_loss = kd_loss_f(
            nn.functional.log_softmax(student_logits / temperature, dim=1),
            nn.functional.softmax(aggregated_logits / temperature, dim=1)
        ) * (temperature ** 2)

        # --- CE loss ---
        ce_loss_fn = nn.CrossEntropyLoss()
        ce_loss = ce_loss_fn(student_logits, labels.long())

        # --- Final loss ---
        loss = alpha * kd_loss + (1 - alpha) * ce_loss

        # --- Backward pass ---
        local_model.zero_grad()
        loss.backward()

        # Accumulate gradients
        for name, param in local_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in accume_grad:
                    accume_grad[name] = torch.zeros_like(param.grad.detach().cpu())
                accume_grad[name] += param.grad.detach().cpu()

    # --- Normalize accumulated gradients ---
    for name in accume_grad:
        accume_grad[name] /= num_batches

    return accume_grad


def screen_multikd_gradients(args, local_model, model_weights,
                             aggregation_candidates, agg_params,
                             train_data, device, temperature=2.0, alpha=0.0,
                             num_batches=5):
    """
    Compute accumulated gradients of KD + CE loss between local model
    and aggregated neighbor models.

    Returns:
        accume_grad: dict {param_name -> accumulated gradient tensor}
    """
    # --- Setup teacher models ---
    model_neighbor = {}
    if args.dataset == "cifar10":
        class_num = 10
    elif args.dataset == "cifar100":
        class_num = 100
    else:
        raise ValueError("Unsupported dataset")

    for clnt in aggregation_candidates:
        model_neighbor[clnt] = create_model(args, model_name=args.model, class_num=class_num)
        # Load weights
        trainable_state_dict = {name: model_weights[clnt][name]
                                for name, param in model_neighbor[clnt].named_parameters()
                                if param.requires_grad}
        model_neighbor[clnt].load_state_dict(trainable_state_dict)

    # --- Prepare local model ---
    local_model.to(device)
    local_model.train()

    accume_grad = {}
    data_iter = iter(train_data)

    for i in range(num_batches):
        try:
            x, labels = next(data_iter)
        except StopIteration:
            break  # not enough data left
        x, labels = x.to(device), labels.to(device)

        # Forward student
        student_logits = local_model(x)

        # --- Aggregate teacher logits fresh for each batch ---
        aggregated_logits = None
        total_weight = 0.0
        for clnt in aggregation_candidates:
            teacher = model_neighbor[clnt].to(device)
            teacher.eval()
            with torch.no_grad():
                out = teacher(x)
            weight_client = 1.0  # uniform weighting
            total_weight += weight_client
            aggregated_logits = out if aggregated_logits is None else aggregated_logits + out

        aggregated_logits /= total_weight

        # --- KD loss ---
        kd_loss_f = nn.KLDivLoss(reduction='batchmean')
        kd_loss = kd_loss_f(
            nn.functional.log_softmax(student_logits / temperature, dim=1),
            nn.functional.softmax(aggregated_logits / temperature, dim=1)
        ) * (temperature ** 2)

        # --- CE loss ---
        ce_loss_fn = nn.CrossEntropyLoss()
        ce_loss = ce_loss_fn(student_logits, labels.long())

        # --- Final loss ---
        loss = alpha * kd_loss + (1 - alpha) * ce_loss

        # --- Backward pass ---
        local_model.zero_grad()
        loss.backward()

        # Accumulate gradients
        for name, param in local_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in accume_grad:
                    accume_grad[name] = torch.zeros_like(param.grad.detach().cpu())
                accume_grad[name] += param.grad.detach().cpu()

    # --- Normalize accumulated gradients ---
    for name in accume_grad:
        accume_grad[name] /= num_batches

    return accume_grad


# def screen_multikd_gradients(args, local_model,  model_weights, aggregation_candidates, agg_params, train_data, device, temperature=2.0, alpha=0.0):
#     """
#     Compute gradients of KD loss between local model and aggregated parameters.
    
#     Args:
#         local_model: the local PyTorch model
#         agg_params: dictionary of aggregated parameters (state_dict)
#         train_data: DataLoader or iterable providing (x, labels)
#         device: computation device
#         temperature: temperature for KD
#         alpha: weighting factor between CE loss and KD loss (optional)
#     Returns:
#         gradient: dictionary of parameter gradients
#     """
#     model_neighbor = {}
#     if args.dataset == "cifar10":
#         class_num = 10
#     elif args.dataset == "cifar100":
#         class_num = 100



#     for clnt in aggregation_candidates:


#         model_neighbor[clnt] = create_model(args, model_name=args.model, class_num = class_num)
        
#         trainable_state_dict = {}
#         for name, param in model_neighbor[clnt].named_parameters():
#             if param.requires_grad:  # Only include parameters that are trainable

#                 trainable_state_dict[name] = model_weights[clnt][name]

#         model_neighbor[clnt].load_state_dict(trainable_state_dict)

#     aggregated_logits = None
#     aggregated_distillation_loss = torch.tensor(0.0, device=device)
#     total_weight = 0.0

#     local_model.to(device)
#     local_model.train()

#     x, labels = next(iter(train_data))
#     x, labels = x.to(device), labels.to(device)

#     # Forward pass
#     student_logits = local_model(x)
#     num_classes = student_logits.shape[1]
#     #sample_weights = compute_adaptive_class_weights(labels, num_classes, epoch, args.comm_round, max_weight=5.0)
            

        
#     # === Aggregate neighbor logits ===
#     for clnt in aggregation_candidates:
#         teacher = model_neighbor[clnt]
#         teacher.to(device)
#         teacher.eval()
#         with torch.no_grad():
#             out = teacher(x)  # Teacher's logits for the current batch
#         #teacher.to("cpu")
#         weight_client = 1.0  # Uniform weighting per neighbor
#         total_weight += weight_client

#         if aggregated_logits is None:
#             aggregated_logits = weight_client * out
#         else:
#             aggregated_logits += weight_client * out

#     if aggregated_logits is not None:
#         aggregated_logits /= total_weight  # Average teacher logits

        

#     #T = args.temperature
    
#     # kl_loss = F.kl_div(
#     #     F.log_softmax(student_logits / T, dim=1),
#     #     F.softmax(aggregated_logits / T, dim=1),
#     #     reduction="none"
#     # ).sum(dim=1)  # shape: [batch_size]

#     kd_loss_f = nn.KLDivLoss(reduction='batchmean')
#     kd_loss = kd_loss_f(
#         nn.functional.log_softmax(student_logits / temperature, dim=1),
#         nn.functional.softmax(aggregated_logits / temperature, dim=1)
#     ) * (temperature ** 2)

#     #sample_weights = adaptive_weights[labels]  # [batch_size]
#     #weighted_loss  = (kl_loss * sample_weights).sum() / sample_weights.sum()
                            
#     #aggregated_distillation_loss = kl_loss * (T ** 2)
#     aggregated_distillation_loss = kd_loss
# ########################################################

#     #================================================================================
    

#     #ce_loss = (F.cross_entropy(output, labels, reduction='none') * sample_weights).mean()

#     #kd_weight = args.kd_weight

#     #ablation
#     #kd_weight = 0

    
#     # loss = ce_loss + (kd_weight * aggregated_distillation_loss)
#     loss = aggregated_distillation_loss
    
        
            
           
  
    
   
    
    
   
#     # KD loss: KL divergence between teacher and student (soft targets)
#     # kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
#     # kd_loss = kd_loss_fn(
#     #     nn.functional.log_softmax(student_logits / temperature, dim=1),
#     #     nn.functional.softmax(teacher_logits / temperature, dim=1)
#     # ) * (temperature ** 2)
    
#     # # Optional: combine with standard CE loss
#     # ce_loss_fn = nn.CrossEntropyLoss()
#     # ce_loss = ce_loss_fn(student_logits, labels.long())
    
#     # loss = alpha * ce_loss + (1 - alpha) * kd_loss
    
#     # Backward pass
#     local_model.zero_grad()
#     loss.backward()
    
#     gradient = {}
#     for name, param in local_model.named_parameters():
#         if param.requires_grad and param.grad is not None:
#             gradient[name] = param.grad.detach().cpu()
    
#     return gradient

    


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
                    #if name in masks:
                    param.data *= masks[name].to(device)
        
        if (epoch)% args.round_number_evaluation == 0:
            logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(idx, local_epoch, sum(epoch_loss) / len(epoch_loss)))

            

            
    # print('exact time used in training: {}'.format(time.time()-start))
    if scheduler != None:
        scheduler.step()


    
   
          


    return model.state_dict()
        

def local_training_freezing(epoch, idx, model, criterion, optimizer,train_data, device, args, logger, algorithm, scheduler=None, masks=None):
 
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

            if masks is not None:
                for name, param in model.named_parameters():
                    #if name in masks:
                    if param.grad is not None:
                        param.grad *= masks[name].to(device)  # zero grad where frozen



            optimizer.step()

            
            epoch_loss.append(loss.item())
            
            # send_params = {}
            # if masks != None:
                            
            #     for name, param in model.named_parameters():
            #         if name in masks:
            #             send_params[name] = param.data * masks[name].to(device)
        
        if (epoch)% args.round_number_evaluation == 0:
            logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(idx, local_epoch, sum(epoch_loss) / len(epoch_loss)))

            

            
    # print('exact time used in training: {}'.format(time.time()-start))
    if scheduler != None:
        scheduler.step()


    
   
          


    return model.state_dict()


def local_training_dynamic_kd_freezing(epoch, idx, model, criterion, optimizer,
                               train_data, device, args, logger, algorithm,
                               scheduler=None, masks=None, prev_global_weights=None,
                               temperature=2.0, alpha=0.5):
    """
    Local training with parameter freezing and dynamically scheduled KD weight.

    Args:
        prev_global_weights: dict {param_name: tensor}, previous aggregated model (teacher).
        temperature: KD temperature.
        alpha: base KD vs CE loss balance (used only if no dynamic schedule).
    """
    model.to(device)
    model.train()
    bs = args.batch_size
    batch_loss = []

    # --- Setup teacher (previous global model as teacher) ---
    if prev_global_weights is not None:
        if args.dataset == "cifar10":
            class_num = 10
        elif args.dataset == "cifar100":
            class_num = 100
        else:
            raise ValueError("Unsupported dataset")

        teacher_model = create_model(args, model_name=args.model, class_num=class_num)
        teacher_model.load_state_dict(prev_global_weights)
        teacher_model.to(device)
        teacher_model.eval()
    else:
        teacher_model = None

    # --- Dynamic KD weight schedule ---
    # linearly scale alpha_t from alpha_min to alpha_max over total rounds
    #alpha_min = args.kd_alpha_min
    #alpha_max = args.kd_alpha_max

    if args.kd_schedule == "linear":
        alpha_t = args.kd_alpha_min + (epoch / args.comm_round) * (args.kd_alpha_max - args.kd_alpha_min)
    elif args.kd_schedule == "exp":
        alpha_t = get_alpha(epoch, args.comm_round, alpha_min=args.kd_alpha_min, alpha_max=args.kd_alpha_max, beta=5.0, warmup_frac=0.2)

    #alpha_t = alpha_min + (epoch / args.comm_round) * (alpha_max - alpha_min)
    
  



    for local_epoch in range(args.local_epochs):
        epoch_loss = []

        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
            model.zero_grad()

            # Student forward pass
            student_logits = model(x)
            ce_loss = criterion(student_logits, labels.long())

            # --- Knowledge Distillation Loss ---
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(x)

                kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
                kd_loss = kd_loss_fn(
                    nn.functional.log_softmax(student_logits / temperature, dim=1),
                    nn.functional.softmax(teacher_logits / temperature, dim=1)
                ) * (temperature ** 2)

                loss = alpha_t * kd_loss + (1 - alpha_t) * ce_loss
            else:
                loss = ce_loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            # Optional: apply freeze masks
            # if masks is not None:
            #     for name, param in model.named_parameters():
            #         if param.grad is not None and name in masks:
            #             param.grad *= masks[name].to(device)

            optimizer.step()
            epoch_loss.append(loss.item())

        # Logging per local epoch
        if (epoch) % args.round_number_evaluation == 0:
            logger.info(
                f'Client {idx}\tLocal Epoch {local_epoch}\tLoss: {sum(epoch_loss)/len(epoch_loss):.6f}\t'
                f'Dynamic α_t={alpha_t:.3f}'
            )

    if scheduler is not None:
        scheduler.step()

    return model.state_dict()


def local_training_kd_freezing(epoch, idx, model, criterion, optimizer,
                            train_data, device, args, logger, algorithm,
                            scheduler=None, masks=None, prev_global_weights=None,
                            temperature=2.0, alpha=0.5):
    """
    Local training with parameter freezing and global model as teacher.

    Args:
        prev_global_weights: dict {param_name: tensor}, previous aggregated model (teacher).
        temperature: KD temperature.
        alpha: KD vs CE loss balance (0=only CE, 1=only KD).
    """
    model.to(device)
    model.train()
    bs = args.batch_size
    batch_loss = []

    # --- Setup teacher (global aggregated model) ---
    if prev_global_weights is not None:
        if args.dataset == "cifar10":
            class_num = 10
        elif args.dataset == "cifar100":
            class_num = 100
        else:
            raise ValueError("Unsupported dataset")

        teacher_model = create_model(args, model_name=args.model, class_num=class_num)
        teacher_model.load_state_dict(prev_global_weights)
        teacher_model.to(device)
        teacher_model.eval()
    else:
        teacher_model = None

    for local_epoch in range(args.local_epochs):
        epoch_loss = []

        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)

            model.zero_grad()

            # Forward student
            student_logits = model(x)

            # --- Compute loss ---
            ce_loss = criterion(student_logits, labels.long())

            if teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(x)
                kd_loss_f = nn.KLDivLoss(reduction='batchmean')
                kd_loss = kd_loss_f(
                    nn.functional.log_softmax(student_logits / temperature, dim=1),
                    nn.functional.softmax(teacher_logits / temperature, dim=1)
                ) * (temperature ** 2)
                loss = alpha * kd_loss + (1 - alpha) * ce_loss
            else:
                loss = ce_loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            # Apply freezing mask (zero grad for frozen params)
            # if masks is not None:
            #     for name, param in model.named_parameters():
            #         if param.grad is not None and name in masks:
            #             param.grad *= masks[name].to(device)

            optimizer.step()
            epoch_loss.append(loss.item())

        # Logging
        if (epoch) % args.round_number_evaluation == 0:
            logger.info(
                'Client Index = {}\tLocal Epoch: {}\tLoss: {:.6f}'.format(
                    idx, local_epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

    if scheduler is not None:
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

def aggregate_after_training(epoch, idx, model_name, model, model_weights, criterion, optimizer, train_data, test_data, aggregation_candidates, device, args, logger, before_agg_local_params, weights_label, scheduler=None, masks=None):

   
 
    bs = args.batch_size

#c    ###################### class_wise accuracy #################################
#    
    model_neighbor = {}

    for clnt in aggregation_candidates:

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
     


        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
       

            for clnt in aggregation_candidates:
                
                model_neighbor[clnt].to(device)
                model_neighbor[clnt].train()
                model_neighbor[clnt].to(device)
    
                model_neighbor[clnt].zero_grad()
                out = model_neighbor[clnt](x)  # Teacher's logits for the current batch
                loss = nn.CrossEntropyLoss()(output, labels)
                loss.backward() 
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
 

   

    if scheduler != None:
        scheduler.step()
        



    

    return model.state_dict()

def local_training_kdsparse(epoch, idx, model_name, model, model_weights, criterion, optimizer, train_data, test_data, aggregation_candidates, device, args, logger, before_agg_local_params, weights_label, scheduler=None, masks=None):

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
            #sample_weights = compute_adaptive_class_weights(labels, num_classes, epoch, args.comm_round, max_weight=5.0)
                    

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
                    kd_loss_f = nn.KLDivLoss(reduction='batchmean')
                    kd_loss = kd_loss_f(
                    nn.functional.log_softmax(output / T, dim=1),
                    nn.functional.softmax(aggregated_logits / T, dim=1)
                    ) * (T ** 2)


                    # kl_loss = F.kl_div(
                    #     F.log_softmax(output / T, dim=1),
                    #     F.softmax(aggregated_logits / T, dim=1),
                    #     reduction="none"
                    # ).sum(dim=1)  # shape: [batch_size]

                    #sample_weights = adaptive_weights[labels]  # [batch_size]
                    #weighted_loss  = (kl_loss * sample_weights).sum() / sample_weights.sum()
                                            
                    aggregated_distillation_loss = kd_loss
                    #aggregated_distillation_loss = kl_loss * (T ** 2)
      ########################################################

            #================================================================================
            

            #ce_loss = (F.cross_entropy(output, labels, reduction='none') * sample_weights).mean()
            ce_loss = nn.CrossEntropyLoss()(output, labels)
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
            if masks != None:
                            
                for name, param in model.named_parameters():
                    #if name in masks:
                    param.data *= masks[name].to(device)
   
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

import torch
import torch.nn as nn

@torch.no_grad()
def kd_proxy_loss(student, teacher, loader, device, T=3.0, max_batches=2):
    """
    Compute KD loss (KL divergence) between student and teacher logits
    on a small calibration loader.
    """
    kd_loss_f = nn.KLDivLoss(reduction="batchmean")
    student.eval(); teacher.eval()
    total, n = 0.0, 0
    for i, (xb, *_) in enumerate(loader):
        if i >= max_batches: break
        xb = xb.to(device)
        s = student(xb)
        t = teacher(xb)
        kd = kd_loss_f(
            torch.log_softmax(s / T, dim=1),
            torch.softmax(t / T, dim=1)
        ) * (T * T)
        total += kd.item(); n += 1
    return total / max(n, 1)

def masked_student_for_comm(student, masks, device):
    """
    Simulate the communicated model for intersection-based aggregation:
    - Active params (mask=1): student values
    - Inactive params (mask=0): dropped (set to 0, ignored in aggregation)
    """
    sdict = student.state_dict()
    new_state = {}
    for name, param in sdict.items():
        if name in masks:
            m = masks[name].to(device)
            new_state[name] = m * param.to(device)
        else:
            new_state[name] = param.to(device)
    comm_model = type(student)()
    comm_model.load_state_dict(new_state)
    comm_model.to(device).eval()
    return comm_model

def get_skip_layers(model):
    """
    Return parameter names to skip:
    - First layer (weights + bias)
    - Last layer (weights + bias)
    - All biases
    - All normalization layers (bn, norm, gn, running stats)
    """
    all_params = [name for name, _ in model.named_parameters()]
    if not all_params:
        return []

    first_layer = all_params[0].split(".")[0]   # module name of first param
    last_layer = all_params[-1].split(".")[0]   # module name of last param

    skip_keywords = ["bias", "bn", "norm", "gn", "running_mean", "running_var"]

    skip_list = []
    for name, _ in model.named_parameters():
        # Skip if bias/norm
        if any(kw in name.lower() for kw in skip_keywords):
            skip_list.append(name)
            continue

        # Skip first and last layer (weights + bias)
        if name.startswith(first_layer) or name.startswith(last_layer):
            skip_list.append(name)

    return skip_list

def build_masks_by_coverage(model, grad_accum, coverage, device, eps=1e-9):
    """
    Build masks per-layer by keeping top `coverage` fraction of gradient energy.
    """
    #skip_keywords = ["bias", "bn", "norm", "gn", "conv1", "fc3"]
    
    new_masks = {}
    # for name, p in model.named_parameters():
    #     if any(kw in name.lower() for kw in skip_keywords):
    #         new_masks[name] = torch.ones_like(p.data).float()
    #         continue
    #     if not p.requires_grad or name not in grad_accum:
    #         continue

    skip_layers = set(get_skip_layers(model))
    for name, p in model.named_parameters():
        if name in skip_layers:
            # Always keep these trainable
            new_masks[name] = torch.ones_like(p.data).float()
            continue

        if not p.requires_grad:
            continue


        g = grad_accum[name].detach().to(device)
        scores = g.abs().view(-1)

        vals, _ = torch.sort(scores, descending=True)
        csum = torch.cumsum(vals, dim=0)
        target = coverage * scores.sum()
        print(f"coverage: {coverage}")
        print(f"csum: {csum}")
        print(f"target: {target}")
       

        if target == 0:
            k_keep = 0
        else:
            idxs = (csum >= target).nonzero(as_tuple=False)
            k_keep = vals.numel() if idxs.numel() == 0 else int(idxs[0].item() + 1)
      
        thresh = vals[k_keep-1] if k_keep > 0 else torch.inf
        keep_mask = (scores >= thresh).float()

        # k_keep = int(round(coverage * scores.numel()))
        # k_keep = min(max(k_keep, 1), scores.numel())
        # thresh = torch.topk(scores, k_keep, largest=True).values.min()
        # keep_mask = (scores >= thresh).float()


        
        new_masks[name] = keep_mask.view_as(p.data)
    return new_masks

def build_masks_by_energy_weight(model, coverage, device, eps=1e-9):
    """
    Build masks using global energy coverage based on weight magnitudes.
    Keeps the smallest set of parameters that cover `coverage` fraction
    of total absolute weight energy.
    """
    skip_keywords = ["bias", "bn", "norm", "gn", "running_mean", "running_var", "conv1", "fc3"]
    new_masks = {}
    layer_scores = {}

    # --- collect scores per layer ---
    for name, p in model.named_parameters():
        if any(kw in name.lower() for kw in skip_keywords):
            new_masks[name] = torch.ones_like(p.data).float()
            continue
        if not p.requires_grad:
            continue

        w = p.detach().to(device)
        s = w.abs()  # energy = |w|
        layer_scores[name] = s

    if not layer_scores:
        return new_masks

    # --- concatenate all scores ---
    all_scores = torch.cat([s.flatten() for s in layer_scores.values()])
    total_energy = all_scores.sum()

    # sort all scores
    vals, _ = torch.sort(all_scores, descending=True)
    csum = torch.cumsum(vals, dim=0)

    # target energy to cover
    target = coverage * total_energy
    idxs = (csum >= target).nonzero(as_tuple=False)
    k_keep = vals.numel() if idxs.numel() == 0 else int(idxs[0].item() + 1)
    k_keep = min(k_keep, vals.numel())
    global_thresh = vals[k_keep - 1]

    # --- assign masks per layer ---
    for name, s in layer_scores.items():
        mask = (s >= global_thresh).float()
        new_masks[name] = mask.view_as(s)

    return new_masks


def compute_comm_masks_kd(model, grad_accum, device,
                          agg_params, local_loader,
                          candidates=(1.0, 0.9, 0.8, 0.7, 0.6),
                          tol_kd=0.05, kd_T=3.0,
                          eps=1e-9,
                          logger=None, idx=None):
    """
    Compute communication masks using KD-based selection under intersection aggregation.

    Args:
        model: local student model.
        grad_accum: dict {param_name -> accumulated gradients}.
        device: torch.device.
        agg_params: last aggregated model parameters (teacher).
        local_loader: small local calibration loader.
        candidates: coverage levels to test.
        tol_kd: max relative KD increase allowed.
        kd_T: temperature for KD.
    """
    # --- Build teacher model from agg_params ---
    teacher_model = type(model)()
    teacher_model.load_state_dict(agg_params)
    teacher_model.to(device).eval()

    # --- Baseline KD (send everything) ---
    full_masks = {name: torch.ones_like(p.data).float()
                  for name, p in model.named_parameters()}
    comm_model_full = masked_student_for_comm(model, full_masks, device)
    # Cache calibration data once
    data_iter = iter(local_loader)
    cached_batches = [next(data_iter) for _ in range(2)]  # max_batches=2
    base_kd = kd_proxy_loss(comm_model_full, teacher_model, cached_batches, device, T=kd_T)

    best_cov, best_masks = 1.0, full_masks
    best_kd = base_kd
    best_rel_increase = 0.0  # by definition baseline

    # --- Try candidates ---
    for cov in sorted(candidates, reverse=True):  # from 1.0 down to sparse
        if cov == 1.0:
            continue
        cand_masks = build_masks_by_energy_weight(model, cov, device, eps)
     
  
        comm_model = masked_student_for_comm(model, cand_masks, device)
        kd_val = kd_proxy_loss(comm_model, teacher_model, cached_batches, device, T=kd_T)
   
        rel_increase = (kd_val - base_kd) / max(base_kd, eps)

   

        if rel_increase <= tol_kd:
            best_cov, best_masks, best_kd, best_rel_increase = cov, cand_masks, kd_val, rel_increase

    # --- Collect per-layer stats for best_masks ---
    best_summary = {}
    for name, mask in best_masks.items():
        kept = int(mask.sum().item())
        total = mask.numel()
        best_summary[name] = (kept, total)

    # --- Logging ---
    if logger is not None and idx is not None:
        kept_all = sum(k for k, t in best_summary.values())
        total_all = sum(t for k, t in best_summary.values())
        for name, (kept, total) in best_summary.items():
            pruned = total - kept
            logger.info(
                'Client Index = {}\tLayer: {}\tKept: {}/{}\t'
                '({:.2f}% kept, {:.2f}% not sent)\tCoverage={:.2f}'.format(
                    idx, name, kept, total,
                    100 * kept / total, 100 * pruned / total,
                    best_cov
                )
            )
        logger.info(
            'Client Index = {}\tTOTAL\tKept: {}/{}\t'
            '({:.2f}% kept, {:.2f}% not sent)\tCoverage={:.2f}\t'
            'KD={:.6f}\tBaselineKD={:.6f}\tRelIncrease={:.2%}\tTolKD={:.2f}'.format(
                idx, kept_all, total_all,
                100 * kept_all / total_all, 100 * (1 - kept_all / total_all),
                best_cov, best_kd, base_kd, best_rel_increase, tol_kd
            )
        )

    return best_masks



def compute_layerwise_prune_regrow_masks(model, grad_accum, masks, device,
                                         prune_coverage=0.99,
                                         eps=1e-9,
                                         first_round=False,
                                         logger=None, idx=None, local_epoch=None):
    """
    Layer-wise dynamic prune + regrow:
    - Prune active weights until 'prune_coverage' of energy mass is kept.
    - Regrow pruned weights whose gradients are above their mean in that layer.
    - If first_round=True, only prune (no regrowth).

    Args:
        model (torch.nn.Module)
        grad_accum (dict): {param_name -> accumulated gradients}.
        masks (dict): {param_name -> current mask}.
        device (torch.device)
        prune_coverage (float): fraction of active energy to retain per layer.
        eps (float): stability constant.
        first_round (bool): if True, skip regrowth step.
    """
    skip_keywords = ["bias", "bn", "norm", "gn", "conv1", "fc3"]
    new_masks = {}
    summary = {}

    for name, p in model.named_parameters():
        if any(kw in name.lower() for kw in skip_keywords):
            # Force skip layers to mask=1
            new_masks[name] = torch.ones_like(p.data).float()
            summary[name] = (p.data.numel(), p.data.numel())  # all kept
            continue
        if not p.requires_grad or name not in grad_accum:
            continue

        w = p.data.detach().to(device)
        g = grad_accum[name].detach().to(device)
        old_mask = masks.get(name, torch.ones_like(w)).to(device).float()

        # ----------------------
        # Phase 1: Prune actives
        # ----------------------
        active_idx = old_mask.view(-1).nonzero(as_tuple=False).squeeze()
        prune_mask_flat = old_mask.view(-1).clone()

        if active_idx.numel() > 0:
            #active_scores = (w.abs() * g.abs()).view(-1)[active_idx]
            active_scores = g.abs().view(-1)[active_idx]

            vals, _ = torch.sort(active_scores, descending=True)
            csum = torch.cumsum(vals, dim=0)
            target = prune_coverage * active_scores.sum()

            if target == 0:
                k_keep = 0
            else:
                idxs = (csum >= target).nonzero(as_tuple=False)
                k_keep = vals.numel() if idxs.numel() == 0 else int(idxs[0].item() + 1)

            thresh = vals[k_keep-1] if k_keep > 0 else torch.inf
            active_keep = (active_scores >= thresh).float()

            prune_mask_flat[active_idx] = active_keep

        # ----------------------
        # Phase 2: Regrow pruned
        # ----------------------
        if not first_round:
            pruned_idx = (1 - old_mask).view(-1).nonzero(as_tuple=False).squeeze()
            regrow_mask_flat = torch.zeros_like(prune_mask_flat)

            if pruned_idx.numel() > 0:
                pruned_scores = g.abs().view(-1)[pruned_idx]
                thresh = pruned_scores.mean()  # mean-threshold rule
                regrow_mask_flat[pruned_idx] = (pruned_scores >= thresh).float()
        else:
            regrow_mask_flat = torch.zeros_like(prune_mask_flat)

        # ----------------------
        # Merge prune + regrow
        # ----------------------
        final_mask_flat = torch.clamp(prune_mask_flat + regrow_mask_flat, 0, 1)
        new_masks[name] = final_mask_flat.view_as(w)

        kept = int(final_mask_flat.sum().item())
        total = w.numel()
        summary[name] = (kept, total)

    # ----------------------
    # Logging
    # ----------------------
    if logger is not None and idx is not None and local_epoch is not None:
        kept_all = sum(k for k, t in summary.values())
        total_all = sum(t for k, t in summary.values())
        for name, (kept, total) in summary.items():
            pruned = total - kept
            logger.info(
                'Client Index = {}\tEpoch: {}\tLayer: {}\tKept: {}/{}\t'
                '({:.2f}% kept, {:.2f}% pruned)'.format(
                    idx, local_epoch, name, kept, total,
                    100 * kept / total, 100 * pruned / total
                )
            )
        logger.info(
            'Client Index = {}\tEpoch: {}\tTOTAL\tKept: {}/{}\t'
            '({:.2f}% kept, {:.2f}% pruned)'.format(
                idx, local_epoch, kept_all, total_all,
                100 * kept_all / total_all, 100 * (1 - kept_all/total_all)
            )
        )

    return new_masks

def compute_freeze_masks(model, grad_accum, device,
                         freeze_coverage=0.90,
                         eps=1e-9,
                         logger=None, idx=None, local_epoch=None):
    """
    Compute binary masks for freezing parameters.
    - Keeps top 'freeze_coverage' fraction of gradient energy mass per layer active.
    - Remaining parameters are frozen (mask=0).
    - No regrowth step.

    Args:
        model (torch.nn.Module)
        grad_accum (dict): {param_name -> accumulated gradients}.
        device (torch.device)
        freeze_coverage (float): fraction of gradient energy to retain per layer.
        eps (float): stability constant.
    """
    skip_keywords = ["bias", "bn", "norm", "gn", "conv1", "fc3"]
    new_masks = {}
    summary = {}

    for name, p in model.named_parameters():
        if any(kw in name.lower() for kw in skip_keywords):
            # always keep these trainable
            new_masks[name] = torch.ones_like(p.data).float()
            summary[name] = (p.data.numel(), p.data.numel())
            continue
        if not p.requires_grad or name not in grad_accum:
            continue

        g = grad_accum[name].detach().to(device)

        # Importance = gradient magnitude
        scores = g.abs().view(-1)

        # Sort scores
        vals, _ = torch.sort(scores, descending=True)
        csum = torch.cumsum(vals, dim=0)
        target = freeze_coverage * scores.sum()

        if target == 0:
            k_keep = 0
        else:
            idxs = (csum >= target).nonzero(as_tuple=False)
            k_keep = vals.numel() if idxs.numel() == 0 else int(idxs[0].item() + 1)

        thresh = vals[k_keep-1] if k_keep > 0 else torch.inf
        keep_mask = (scores >= thresh).float()

        new_masks[name] = keep_mask.view_as(p.data)

        kept = int(new_masks[name].sum().item())
        total = p.data.numel()
        summary[name] = (kept, total)

    # Logging
    if logger is not None and idx is not None and local_epoch is not None:
        kept_all = sum(k for k, t in summary.values())
        total_all = sum(t for k, t in summary.values())
        for name, (kept, total) in summary.items():
            pruned = total - kept
            logger.info(
                'Client Index = {}\tEpoch: {}\tLayer: {}\tKept: {}/{}\t'
                '({:.2f}% kept, {:.2f}% frozen)'.format(
                    idx, local_epoch, name, kept, total,
                    100 * kept / total, 100 * pruned / total
                )
            )
        logger.info(
            'Client Index = {}\tEpoch: {}\tTOTAL\tKept: {}/{}\t'
            '({:.2f}% kept, {:.2f}% frozen)'.format(
                idx, local_epoch, kept_all, total_all,
                100 * kept_all / total_all, 100 * (1 - kept_all/total_all)
            )
        )

    return new_masks


def get_alpha(t, total_rounds, alpha_min=0.5, alpha_max=0.9, beta=5.0, warmup_frac=0.2):
    T_w = warmup_frac * total_rounds
    if t >= T_w:
        return alpha_max
    return alpha_max - (alpha_max - alpha_min) * math.exp(-beta * t / T_w)

def compute_dynamic_coverage_masks(epoch, args, model, grad_accum, device,
                         round_idx=0, total_rounds=100,
                         start_cov=1.0, end_cov=0.7, schedule="linear",
                         eps=1e-9,
                         logger=None, idx=None):
    """
    Compute binary masks for freezing parameters with scheduling.
    - Early rounds: high coverage (keep more params active).
    - Later rounds: low coverage (freeze more).
    - No regrowth step.

    Args:
        model (torch.nn.Module)
        grad_accum (dict): {param_name -> accumulated gradients}.
        device (torch.device)
        round_idx (int): current training round.
        total_rounds (int): total communication rounds planned.
        start_cov (float): starting coverage (fraction of gradient energy kept).
        end_cov (float): final coverage.
        schedule (str): "linear" or "exp" for decay type.
        eps (float): stability constant.
    """
    def get_dynamic_coverage(t, T, start_cov, end_cov, schedule):
        if schedule == "linear":
            return max(end_cov,
                       start_cov - (t / T) * (start_cov - end_cov))
        elif schedule == "exp":
            
            beta = 5.0
            return end_cov + (start_cov - end_cov) * torch.exp(-beta * t / T)
        else:
            raise ValueError("schedule must be 'linear' or 'exp'")

    freeze_coverage = get_dynamic_coverage(round_idx, total_rounds,
                                           start_cov, end_cov, schedule)
    
    #freeze_coverage = 0.8

    skip_keywords = ["bias", "bn", "norm", "gn", "conv1", "fc3"]
    new_masks = {}
    summary = {}

    #skip_layers = set(get_skip_layers(model))
    

    # for name, p in model.named_parameters():
    #     if name in skip_layers:
    #         # Always keep these trainable
    #         new_masks[name] = torch.ones_like(p.data).float()
    #         continue

    #     if not p.requires_grad:
    #         continue


    for name, p in model.named_parameters():


        # if freeze_coverage == 1 :
        #     new_masks[name] = torch.ones_like(p.data).float()
        #     summary[name] = (p.data.numel(), p.data.numel())

        #     continue

        if any(kw in name.lower() for kw in skip_keywords):
            # always keep these trainable
            new_masks[name] = torch.ones_like(p.data).float()
            summary[name] = (p.data.numel(), p.data.numel())
            continue
        if not p.requires_grad or name not in grad_accum:
            continue

        g = grad_accum[name].detach().to(device)

        # Importance = gradient magnitude
        scores = g.abs().view(-1)


 

        # w = p.detach().to(device)
        # scores = w.abs().view(-1)  # energy = |w|
   

        # Sort scores
        vals, _ = torch.sort(scores, descending=True)
        csum = torch.cumsum(vals, dim=0)
        target = freeze_coverage * scores.sum()

        if target == 0:
            k_keep = 0
        else:
            idxs = (csum >= target).nonzero(as_tuple=False)
            k_keep = vals.numel() if idxs.numel() == 0 else int(idxs[0].item() + 1)

        thresh = vals[k_keep-1] if k_keep > 0 else torch.inf
        keep_mask = (scores >= thresh).float()

        new_masks[name] = keep_mask.view_as(p.data)

        kept = int(new_masks[name].sum().item())
        total = p.data.numel()
        summary[name] = (kept, total)

    # Logging
    if logger is not None and idx is not None:
        logger.info('@@@@@@@@@@@@@@@@ Parameter Selection CM({}): {}'.format(epoch, idx))
        kept_all = sum(k for k, t in summary.values())
        total_all = sum(t for k, t in summary.values())
        for name, (kept, total) in summary.items():
            pruned = total - kept
            logger.info(
                'Layer: {}\tsent: {}/{}\t'
                '({:.2f}% sent, {:.2f}% pruned)\tCoverage={:.2f}'.format(
                    name, kept, total,
                    100 * kept / total, 100 * pruned / total,
                    freeze_coverage
                )
            )
        logger.info(
            'Client Index = {}\tTOTAL\tsent: {}/{}\t'
            '({:.2f}% sent, {:.2f}% pruned)\tCoverage={:.2f}'.format(
                idx, kept_all, total_all,
                100 * kept_all / total_all, 100 * (1 - kept_all/total_all),
                freeze_coverage
            )
        )

    return new_masks




def compute_dynamic_sparsity_masks(
    epoch, args, model, grad_accum, device,
    round_idx=0, total_rounds=1000,
    start_density=1.0, end_density=0.7, schedule="linear",
    eps=1e-8, logger=None, idx=None
):
    """
    Compute binary masks that keep a *count-based* fraction (density) of params.
    - No gradient/energy criteria. Strictly top-|w| by count.
    - No regrowth.
    - 'density' is the kept ratio per tensor.

    Args:
        epoch, args: for periodic logging (args.round_number_evaluation)
        model (torch.nn.Module)
        device (torch.device)
        round_idx (int): current communication round (0-based).
        total_rounds (int): total planned rounds.
        start_density (float), end_density (float): kept fraction schedule.
        schedule (str): "linear" or "exp".
        eps (float): numerical tolerance for boundary cases.
        logger, idx: optional logging.
    """

    def get_dynamic_density(t, T, start_density, end_density, schedule):
        if schedule == "linear":
            return max(end_density,
                       start_density - (t / max(T, 1)) * (start_density - end_density))
        elif schedule == "exp":
            beta = 5.0
            # clamp to [end_density, start_density]
            d = end_density + (start_density - end_density) * torch.exp(torch.tensor(-beta * t / max(T, 1), dtype=torch.float32))
            return float(torch.clamp(d, min=min(start_density, end_density), max=max(start_density, end_density)).item())
        else:
            raise ValueError("schedule must be 'linear' or 'exp'")

    density = float(get_dynamic_density(round_idx, total_rounds, start_density, end_density, schedule))


    skip_keywords = ["bias", "bn", "norm", "gn", "conv1", "fc3"]
    new_masks, summary = {}, {}

    for name, p in model.named_parameters():
        # Always keep these trainable / communicated
        if any(kw in name.lower() for kw in skip_keywords):
            m = torch.ones_like(p.data, device=p.device, dtype=torch.float32)
            new_masks[name] = m
            summary[name] = (m.numel(), m.numel())
            continue

        if not p.requires_grad:
            # Keep if it doesn't require grad (no pruning applied)
            m = torch.ones_like(p.data, device=p.device, dtype=torch.float32)
            new_masks[name] = m
            summary[name] = (m.numel(), m.numel())
            continue

        numel = p.data.numel()
        if numel == 0:
            continue  # nothing to do

        # Boundary cases for density
        # if density >= 1.0 - eps:
        #     m = torch.ones_like(p.data, device=p.device, dtype=torch.float32)
        #     new_masks[name] = m
        #     summary[name] = (m.numel(), m.numel())
        #     continue
        # if density <= eps:
        #     m = torch.zeros_like(p.data, device=p.device, dtype=torch.float32)
        #     new_masks[name] = m
        #     summary[name] = (0, m.numel())
        #     continue

        
        g = grad_accum[name].detach().to(device)

        # Importance = gradient magnitude
        scores = g.abs().view(-1)


        # # Count-based pruning: keep exactly ~density * N by |w|
        # w = p.data.detach()
        # scores = w.abs().reshape(-1)

        k_keep = min(numel, max(0, math.ceil(density * numel)))
        if k_keep == 0:
            keep_mask_flat = torch.zeros_like(scores, dtype=torch.float32, device=p.device)
        elif k_keep == numel:
            keep_mask_flat = torch.ones_like(scores, dtype=torch.float32, device=p.device)
        else:
            # Top-k by magnitude
            topk_vals = torch.topk(scores, k_keep, largest=True, sorted=True).values
            thresh = topk_vals[-1]  # threshold at the kth largest
            # Keep >= threshold (ties may slightly exceed k_keep; that's OK)
            keep_mask_flat = (scores >= thresh).to(torch.float32)

        m = keep_mask_flat.view_as(p.data)
        new_masks[name] = m
        kept = int(m.sum().item())
        summary[name] = (kept, numel)

    # Logging (periodic)
    if (epoch % args.round_number_evaluation == 0) and logger is not None and idx is not None:
        kept_all = sum(k for k, t in summary.values())
        total_all = sum(t for k, t in summary.values())
        for name, (kept, total) in summary.items():
            pruned = total - kept
            logger.info(
                'Client Index = {}\tLayer: {}\tSent: {}/{}\t'
                '({:.2f}% sent, {:.2f}% pruned)\tDensity={:.4f}'.format(
                    idx, name, kept, total,
                    100.0 * kept / max(total, 1), 100.0 * pruned / max(total, 1),
                    density
                )
            )
        logger.info(
            'Client Index = {}\tTOTAL\tSent: {}/{}\t'
            '({:.2f}% sent, {:.2f}% pruned)\tDensity={:.4f}'.format(
                idx, kept_all, total_all,
                100.0 * kept_all / max(total_all, 1), 100.0 * (1 - kept_all / max(total_all, 1)),
                density
            )
        )

    return new_masks


def compute_global_energy_masks(model, grad_accum, masks, device,
                                coverage=0.99,
                                normalize="median",
                                min_keep_ratio=0.1,
                                eps=1e-9,
                                logger=None, idx=None):
    """
    Global energy coverage pruning with per-layer normalization,
    regrowth-aware scores, and per-layer minimum keep ratio.
    Only computes masks (does not apply them).
    """
    regrow_scores = {}
    skip_keywords = ["bias", "bn", "norm", "gn"]

    # --- regrowth-aware scores ---
    for name, p in model.named_parameters():
        if any(kw in name.lower() for kw in skip_keywords):
            continue
        if p.requires_grad and name in grad_accum:
            w = p.data.detach().to(device)
            g = grad_accum[name].detach().to(device)
            old_mask = masks.get(name, torch.ones_like(p.data)).to(device).float()
            # alive: |w|*|g| ; pruned: |g|
            regrow_scores[name] = (w.abs() * g.abs()) * old_mask + g.abs() * (1.0 - old_mask)

    if not regrow_scores:
        return masks

    # --- per-layer normalization ---
    norm_scores = {}
    for name, s in regrow_scores.items():
        if normalize == "median":
            scale = s.median()
        elif normalize == "mean":
            scale = s.mean()
        elif normalize == "max":
            scale = s.max()
        else:
            raise ValueError("normalize must be 'median', 'mean', or 'max'")
        norm_scores[name] = s / (scale + eps)

    # --- global energy threshold ---
    all_scores = torch.cat([s.flatten() for s in norm_scores.values()])
    total_score = all_scores.sum()
    vals, _ = torch.sort(all_scores, descending=True)
    csum = torch.cumsum(vals, dim=0)
    target = coverage * total_score
    idxs = (csum >= target).nonzero(as_tuple=False)
    if idxs.numel() == 0:  # coverage=1.0 edge case
        k = vals.numel()
    else:
        k = int(idxs[0].item() + 1)
    k = min(k, vals.numel())
    global_thresh = vals[k-1]

    # --- build masks with min_keep_ratio enforcement ---
    new_masks = {}
    summary = {}
    for name, s in norm_scores.items():
        mask = (s >= global_thresh).float()
        total = mask.numel()
        kept = int(mask.sum().item())

        min_kept = int(min_keep_ratio * total)
        if kept < min_kept:
            flat = s.view(-1)
            th = torch.topk(flat, min_kept, largest=True).values.min()
            mask = (s >= th).float()
            kept = int(mask.sum().item())

        new_masks[name] = mask.view_as(s)
        summary[name] = (kept, total)

    # --- logging ---
    if logger is not None and idx is not None:
        kept_all = sum(k for k, t in summary.values())
        total_all = sum(t for k, t in summary.values())
        for name, (kept, total) in summary.items():
            pruned = total - kept
            logger.info(
                'Client Index = {}\tLayer: {}\tKept: {}/{}\t({:.2f}% kept, {:.2f}% pruned)'.format(
                    idx, name, kept, total,
                    100 * kept / total, 100 * pruned / total
                )
            )
        logger.info(
            'Client Index = {}\tTOTAL\tKept: {}/{}\t({:.2f}% kept, {:.2f}% pruned)'.format(
                idx, kept_all, total_all,
                100 * kept_all / total_all, 100 * (1 - kept_all/total_all)
            )
        )

    return new_masks


def prune_regrow_dynamic(model, grad_accum, masks, coverage=0.97, device="cuda", eps=1e-9,
                         logger=None, idx=None, local_epoch=None, epoch_loss=None):
    """
    Dynamic pruning with regrowth using energy coverage.
    Sparsity is not fixed: depends on score distribution.

    Args:
        model (torch.nn.Module): model to prune.
        grad_accum (dict): dict {param_name -> accumulated gradients}.
        masks (dict): dict {param_name -> current binary mask (0/1)}.
        coverage (float): fraction of score mass to keep (0 < coverage <= 1).
        device (str): device string.
        eps (float): numerical stability.
        logger: logger object (e.g., Python logging).
        idx (int): client index (for logging).
        local_epoch (int): local epoch number (for logging).
        epoch_loss (list/float): loss values (for logging).

    Returns:
        dict: updated masks {param_name -> new mask}.
    """
    new_masks = {}
    summary = {}

    for name, p in model.named_parameters():
        if not p.requires_grad or name not in grad_accum:
            continue

        w = p.data.detach().to(device)
        g = grad_accum[name].detach().to(device)

        mask = masks[name].to(device).float()

        # Scores: active use |w|*|grad|, pruned use |grad|
        scores = (w.abs() * g.abs()) * mask + g.abs() * (1.0 - mask)
        flat_scores = scores.view(-1)

        total_score = flat_scores.sum()
        if total_score.item() < eps:
            new_mask = torch.zeros_like(p.data, dtype=torch.float32)
            new_masks[name] = new_mask
            summary[name] = (0, flat_scores.numel())
            continue

        # Sort scores descending
        vals, _ = torch.sort(flat_scores, descending=True)
        csum = torch.cumsum(vals, dim=0)

        # Find smallest k that covers "coverage" fraction of total score mass
        k = int((csum >= coverage * total_score).nonzero(as_tuple=False)[0].item() + 1)
        #k = vals.numel()
        thresh = vals[k-1]

        # Build new mask
        new_mask = (scores >= thresh).float().view_as(p.data)
        new_masks[name] = new_mask

        # Enforce pruning
        p.data.mul_(new_mask)

        kept = int(new_mask.sum().item())
        total = new_mask.numel()
        summary[name] = (kept, total)

    # --- logging instead of printing ---
    if logger is not None and idx is not None and local_epoch is not None:
        for name, (kept, total) in summary.items():
            pruned = total - kept
            logger.info(
                'Client Index = {}\tEpoch: {}\tLayer: {}\tKept: {}/{}\t({:.2f}% kept, {:.2f}% pruned)'
                .format(idx, local_epoch, name, kept, total, 100*kept/total, 100*pruned/total)                       
            )

    return new_masks


 
    
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

