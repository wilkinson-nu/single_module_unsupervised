import torch

@torch.no_grad()
def argmax_consistency(c_cat, device=None):
    batch_size = c_cat.shape[0] // 2
    c_i, c_j = c_cat[:batch_size], c_cat[batch_size:]
    
    argmax_i = torch.argmax(c_i, dim=1)
    argmax_j = torch.argmax(c_j, dim=1)
    
    same = (argmax_i == argmax_j).float()
    mean_same = same.mean()
    if device is not None: mean_same = mean_same.to(device)
    return mean_same

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
