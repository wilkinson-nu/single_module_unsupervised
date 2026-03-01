import torch
from torch import nn

@torch.no_grad()
def alignment(z_cat):
    z_cat = nn.functional.normalize(z_cat, dim=1)
    batch_size = z_cat.shape[0] // 2
    z_i, z_j = z_cat[:batch_size], z_cat[batch_size:]    
    return ((z_i - z_j).pow(2).sum(dim=1)).mean()

@torch.no_grad()
def uniformity(z, t=2):
    z = nn.functional.normalize(z, dim=1)
    sq_pdist = torch.cdist(z, z, p=2).pow(2)
    # mask out diagonal (self-pairs)
    mask = 1 - torch.eye(z.size(0), device=z.device)
    uniformity = torch.exp(-t * sq_pdist) * mask
    return torch.log(uniformity.sum() / (z.size(0) * (z.size(0)-1)))
    
@torch.no_grad()
def argmax_consistency(c_cat):
    batch_size = c_cat.shape[0] // 2
    c_i, c_j = c_cat[:batch_size], c_cat[batch_size:]
    
    argmax_i = torch.argmax(c_i, dim=1)
    argmax_j = torch.argmax(c_j, dim=1)
    
    same = (argmax_i == argmax_j).float()
    return same.mean()

@torch.no_grad()
def topk_consistency(c_cat, k=2):
    batch_size = c_cat.shape[0] // 2
    c_i, c_j = c_cat[:batch_size], c_cat[batch_size:]
    
    # Top-k indices for each view
    topk_i = torch.topk(c_i, k, dim=1).indices
    topk_j = torch.topk(c_j, k, dim=1).indices
    
    # For each sample, check if there's an overlap in the sets
    overlap = (topk_i.unsqueeze(2) == topk_j.unsqueeze(1))
    same = overlap.any(dim=(1,2)).float()
    
    return same.mean().item()
