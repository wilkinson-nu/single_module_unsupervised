import torch
from torch import nn
import torch.distributed as dist
from core.losses.gather import GatherLayer
import torch.nn.functional as F

class NTXentMerged(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, emb_cat):
        """
        emb_cat are the concatenated batches of pairs emb_cat = z_i + z_j
        """
        batch_size = emb_cat.shape[0]//2
        z_cat = nn.functional.normalize(emb_cat, dim=1)

        sim = torch.mm(z_cat, z_cat.t()) / self.temperature
        mask = torch.eye(2*batch_size, device=z_cat.device, dtype=torch.bool)
        sim.masked_fill_(mask, -float("inf"))
        
        positives = torch.arange(2*batch_size, device=z_cat.device)
        positives = (positives + batch_size) % (2*batch_size)
        
        loss = self.cross_entropy(sim, positives)
        
        return loss, {}


class NTXentMergedMultiGPU(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_cat):
        batch_size = emb_cat.shape[0] // 2
        z = F.normalize(emb_cat, dim=1)
        z_i_all = torch.cat(GatherLayer.apply(z[:batch_size]), dim=0)
        z_j_all = torch.cat(GatherLayer.apply(z[batch_size:]), dim=0)
        N = z_i_all.shape[0]
        z_all = torch.cat([z_i_all, z_j_all], dim=0)          # 2N x D
    
        rank = dist.get_rank()
        idx = torch.cat([
            torch.arange(rank * batch_size, (rank + 1) * batch_size, device=z.device),
            torch.arange(N + rank * batch_size, N + (rank + 1) * batch_size, device=z.device),
        ])
        sim = z_all[idx] @ z_all.t() / self.temperature       # 2B x 2N
        sim[torch.arange(2 * batch_size, device=z.device), idx] = -float("inf")
        return F.cross_entropy(sim, (idx + N) % (2 * N)), {}
        
