import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

def normalize(tensor):
    norm = torch.norm(tensor, p=2) + 1e-16  # Avoid division by zero
    return tensor / norm

class DFedSAM:
    def __init__(self, model, optimizer, rho=0.01, W=None, neighbors=None):
        self.model = model
        self.optimizer = optimizer
        self.rho = rho  # Perturbation factor (set to 0.01 as in the paper)
        self.W = W  # Adjacency matrix
        self.neighbors = neighbors  # Dict mapping each client to its neighbors
        self.state = defaultdict(dict)
    
    def local_update(self, x_t, data_loader, epoch, K, criterion, device):
        """Perform local training with SAM."""
        y_t = x_t.copy()
        self.model.load_state_dict(x_t)
        self.model.to(device)
        self.model.train()
        
        perturbation = {}
        
        for k in range(K):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(data_loader):
                x, labels = x.to(device), labels.to(device)

            # Load current local model y_t into self.model
                self.model.load_state_dict(y_t)
                self.model.zero_grad()

                # 1. Compute g_t^k (original gradient)
                out = self.model(x)
                loss = criterion(out, labels)
                loss.backward()

                # Compute global L2 norm of all gradients
                grad_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in self.model.parameters() if p.grad is not None))
                scale = self.rho / (grad_norm + 1e-12)

                # Apply perturbation: w ← w + ρ * grad / ||grad||
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.add_(scale * p.grad)

                # 2. Compute perturbed gradient ∇F(y + δ)
                self.model.zero_grad()
                out_perturbed = self.model(x)
                loss_perturbed = criterion(out_perturbed, labels)
                loss_perturbed.backward()

                # 3. Manual SGD update: y ← y - η * perturbed_grad
                lr = self.optimizer.param_groups[0]['lr']
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        y_t[name] = y_t[name] - lr * param.grad

            #torch.cuda.empty_cache()

        # Final local model after K steps
        return y_t


    # def local_update(self, x_t, data_loader, epoch, K, criterion, device):
    #     """Perform local training with SAM."""
    #     y_t = x_t.copy()
    #     self.model.load_state_dict(x_t)
    #     self.model.to(device)
    #     self.model.train()
        
    #     perturbation = {}
        
    #     for k in range(K):
    #         epoch_loss = []
    #         for batch_idx, (x, labels) in enumerate(data_loader):
    #             x, labels = x.to(device), labels.to(device)

    #             self.optimizer.zero_grad()
    #             out = self.model(x)
    #             loss = criterion(out, labels.long())
    #             loss.backward()

    #             # Store the original gradients
    #             original_gradients = {p: p.grad.clone() for p in self.model.parameters() if p.grad is not None}

    #             # Apply perturbation to model weights
    #             with torch.no_grad():
                  
    #                 for p in self.model.parameters():
    #                     if p.grad is not None:
    #                         perturbation[p] = normalize(p.grad) * self.rho
    #                         p.add_(perturbation[p])

    #             # Compute new gradient after perturbation
    #             self.optimizer.zero_grad()
    #             perturbed_out = self.model(x)
    #             perturbed_loss = criterion(perturbed_out, labels.long())
    #             perturbed_loss.backward()

    #             # Restore original model weights
    #             with torch.no_grad():
    #                 for p in self.model.parameters():
    #                     if p.grad is not None:     
    #                         p.sub_(perturbation[p])  # Undo perturbation

                
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
    #             # Apply optimizer step with perturbed gradients
    #             self.optimizer.step()

    #             epoch_loss.append(loss.item())

    #         if epoch % 10 == 0:
    #             print(f"Epoch: {k}, Loss: {sum(epoch_loss) / len(epoch_loss)}")

    #         torch.cuda.empty_cache()
        
    #     return self.model.state_dict()
