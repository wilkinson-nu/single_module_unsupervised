import torch
from torch import nn

@torch.no_grad()
def alignment(z_cat):
    z_cat = nn.functional.normalize(z_cat, dim=1)
    batch_size = z_cat.shape[0] // 2
    z_i, z_j = z_cat[:batch_size], z_cat[batch_size:]    
    cos = (z_i * z_j).sum(dim=1)
    return (2 - 2 * cos).mean()

@torch.no_grad()
def uniformity(z, t=2):
    z = z.float()
    z = nn.functional.normalize(z, dim=1)
    sq_pdist = torch.cdist(z, z, p=2).pow(2)
    # mask out diagonal (self-pairs)
    mask = ~torch.eye(z.size(0), device=z.device, dtype=torch.bool)
    vals = torch.exp(-t * sq_pdist)[mask]
    return torch.log(vals.mean())
    
@torch.no_grad()
def argmax_consistency(c_cat):
    batch_size = c_cat.shape[0] // 2
    c_i, c_j = c_cat[:batch_size], c_cat[batch_size:]
    
    argmax_i = torch.argmax(c_i, dim=1)
    argmax_j = torch.argmax(c_j, dim=1)
    
    same = (argmax_i == argmax_j).float()
    return same.mean()
