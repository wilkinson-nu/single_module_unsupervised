import torch
from torch import nn
import torch.distributed as dist

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
        similarity_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2*batch_size)

        return loss


class NTXentMergedMultiGPU(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def concat_all_gather(tensor):
        """
        Gathers a tensor from all GPUs and concatenates along the batch dimension.
        Gradients do NOT flow through remote GPUs.
        """
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size == 1:
            return tensor
        tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensors_gather, tensor)
        return torch.cat(tensors_gather, dim=0)

    def forward(self, emb_cat):
        """
        emb_cat: concatenated embeddings of shape (2*B_per_gpu, D), stacked as [z_i, z_j]
        """
        batch_size = emb_cat.shape[0]//2
        z_cat = nn.functional.normalize(emb_cat, dim=1)
        z_i, z_j = z_cat[:batch_size], z_cat[batch_size:]

        # Gather embeddings across GPUs
        z_i_all = self.concat_all_gather(z_i)
        z_j_all = self.concat_all_gather(z_j)
        total_batch = representations.shape[0]//2

        negatives_mask = (~torch.eye(total_batch*2, total_batch_size*2, device=emb_cat.device, dtype=torch.bool)).float()        
        representations = torch.cat([z_i_all, z_j_all], dim=0)
        sim_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(sim_matrix, total_batch)
        sim_ji = torch.diag(sim_matrix, -total_batch)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (total_batch)        
        
        return loss

