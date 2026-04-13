import torch
from torch import nn
import torch.distributed as dist
from core.losses.gather import GatherLayer

class NTXentMerged(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_cat):
        """
        emb_cat are the concatenated batches of pairs emb_cat = z_i + z_j
        """
        batch_size = emb_cat.shape[0]//2
        z_cat = nn.functional.normalize(emb_cat, dim=1)
        z_i, z_j = z_cat[:batch_size], z_cat[batch_size:]

        negatives_mask = (~torch.eye(batch_size*2, batch_size*2, dtype=bool, device=emb_cat.device)).float()
        representations = torch.cat([z_i, z_j], dim=0)

        z = nn.functional.normalize(representations, dim=1)
        similarity_matrix = torch.mm(z, z.t())

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.mean(loss_partial)

        return loss


class NTXentMergedMultiGPU(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, emb_cat):
        """
        emb_cat: concatenated embeddings of shape (2*B_per_gpu, D), stacked as [z_i, z_j]
        """

        if not torch.isfinite(emb_cat).all():
            print("Projection head output contains NaN/Inf")
        
        batch_size = emb_cat.shape[0]//2
        z_cat = emb_cat / (emb_cat.norm(dim=1, keepdim=True) + 1e-8)
        z_i, z_j = z_cat[:batch_size], z_cat[batch_size:]
        
        z_i_all = torch.cat(GatherLayer.apply(z_i), dim=0)
        z_j_all = torch.cat(GatherLayer.apply(z_j), dim=0)
        total_batch = z_i_all.shape[0]


        z_all = torch.cat([z_i_all, z_j_all], dim=0)

        if not torch.isfinite(z_all).all():
            print("Embedding contains NaN/Inf")
            print(z_all.min(), z_all.max())
            raise RuntimeError("Bad embeddings")
        
        sim = torch.mm(z_all, z_all.t()) / self.temperature
        mask = torch.eye(2*total_batch, device=z_all.device, dtype=torch.bool)
        sim.masked_fill_(mask, -float("inf"))

        positives = torch.arange(2*total_batch, device=z_all.device)
        positives = (positives + total_batch) % (2*total_batch)

        loss = self.cross_entropy(sim, positives)
        
        return loss

