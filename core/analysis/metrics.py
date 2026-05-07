import torch
from torch import nn
from core.losses.gather import GatherLayer
import torch.distributed as dist

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


@torch.no_grad()
def simclr_geometry_metrics(buffer, device):

    '''
    Each element in buffer is the concatenation of the two views in a batch
    Loop over the buffer and calculate values for each batch and then average
    '''

    dim = buffer[0].shape[1]
    cov = torch.zeros(dim, dim, device=device)
    n = 0
    
    ## Keep track of values
    pos_buffer = []
    neg_buffer = []
    
    ## loop over buffer
    for emb_cat_cpu in buffer:

        emb_cat = emb_cat_cpu.to(device, non_blocking=True)
        
        batch_size = emb_cat.shape[0]//2
        z_cat = emb_cat / (emb_cat.norm(dim=1, keepdim=True) + 1e-8)
        z_i, z_j = z_cat[:batch_size], z_cat[batch_size:]

        z_i_all = torch.cat(GatherLayer.apply(z_i), dim=0)
        z_j_all = torch.cat(GatherLayer.apply(z_j), dim=0)
        total_batch = z_i_all.shape[0]
        z_all = torch.cat([z_i_all, z_j_all], dim=0)
        
        #######################
        ### Geometry metrics ##
        #######################

        sim = torch.mm(z_all, z_all.t())
        mask = torch.eye(2*total_batch, device=z_all.device, dtype=torch.bool)
        sim.masked_fill_(mask, -float("inf"))

        idx = torch.arange(2*total_batch, device=z_all.device)
        pos_idx = (idx + total_batch) % (2*total_batch)
        pos_buffer .append(sim[idx, pos_idx])

        ## Now modify sim for calculating hard negatives
        sim[idx, pos_idx] = -float("inf")
        neg_buffer .append(sim.max(dim=1).values)

        #######################
        # Effective dimension #
        #######################
        
        z_all = z_all - z_all.mean(dim=0, keepdim=True)
        cov += z_all.T @ z_all
        n += z_all.shape[0]

    ## Now calculate the covariance info
    cov = cov / (n - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    deff = (eigvals.sum() ** 2) / (eigvals.pow(2).sum())
    lambda1_ratio = eigvals.max() / eigvals.sum()

    ## Calculate the SimCLR geometry values
    all_pos = torch.cat(pos_buffer, dim=0)
    all_neg = torch.cat(neg_buffer, dim=0)        
    gap = all_pos - all_neg
    pos_mean = all_pos.mean()
    neg_mean = all_neg.mean()
    gap_mean = gap.mean()
    gap_std  = gap.std(unbiased=False)

    return {"pos": pos_mean.item(),
            "hard_neg": neg_mean.item(),
            "gap": gap_mean.item(),
            "gap_std": gap_std.item(),
            "deff": deff.item(),
            "l1_ratio": lambda1_ratio.item(),
            "eigvals": eigvals.flip(0)[:10].cpu()
            }


@torch.no_grad()
def basic_geometry_metrics(buffer, device):
    '''
    A simplified version of the simclr geometry metrics, no need for complex gather layers etc
    '''
    
    dim = buffer[0].shape[1]

    ## First pass: calculate global mean across all batches and GPUs
    sum_z = torch.zeros(dim, device=device)
    n = 0
    for emb_cat_cpu in buffer:
        emb_cat = emb_cat_cpu.to(device, non_blocking=True)
        z = emb_cat / (emb_cat.norm(dim=1, keepdim=True) + 1e-8)
        sum_z += z.sum(dim=0)
        n += z.shape[0]

    ## Gather across GPUs if we're in a multi-GPU setting
    if dist.is_initialized():
        dist.all_reduce(sum_z, op=dist.ReduceOp.SUM)
        n_tensor = torch.tensor(n, device=device, dtype=torch.float32)
        dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
        n = n_tensor.item()

    ## Find the global mean
    global_mean = sum_z / n

    ## Second pass, calculate the covariance
    cov = torch.zeros(dim, dim, device=device)
    for emb_cat_cpu in buffer:
        emb_cat = emb_cat_cpu.to(device, non_blocking=True)
        z = emb_cat / (emb_cat.norm(dim=1, keepdim=True) + 1e-8)
        z = z - global_mean
        cov += z.T @ z

    if dist.is_initialized():
        dist.all_reduce(cov, op=dist.ReduceOp.SUM)

    ## Calculate the covariance info
    cov = cov / (n - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    deff = (eigvals.sum() ** 2) / (eigvals.pow(2).sum())
    lambda1_ratio = eigvals.max() / eigvals.sum()

    return {"deff": deff.item(),
            "l1_ratio": lambda1_ratio.item(),
            "eigvals": eigvals.flip(0)[:10].cpu()
            }
