import math
import torch
from torch import nn
import torch.distributed as dist
from core.losses.gather import GatherLayer

class ClusteringLossMerged(nn.Module):
    def __init__(self, temperature=0.5, entropy_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.entropy_weight = entropy_weight

    def forward(self, c_cat):

        batch_size = c_cat.shape[0]//2
        class_num = c_cat.shape[1]
        c_i, c_j = c_cat[:batch_size], c_cat[batch_size:]

        ## Start with the entropy term
        p_i = c_i.sum(dim=0)
        p_j = c_j.sum(dim=0)
        p_i = p_i/p_i.sum()
        p_j = p_j/p_j.sum()

        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i + 1e-10)).sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j + 1e-10)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        
        negatives_mask = (~torch.eye(class_num*2, class_num*2, dtype=bool, device=c_cat.device)).float()
        representations = torch.cat([c_i, c_j], dim=0)

        c = nn.functional.normalize(representations, dim=1)
        similarity_matrix = torch.mm(c, c.t())

        # similarity_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, class_num)
        sim_ji = torch.diag(similarity_matrix, -class_num)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2*class_num)

        return loss, ne_loss*self.entropy_weight


class ClusteringLossMergedMultiGPU(nn.Module):
    def __init__(self, temperature=0.5, entropy_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.entropy_weight = entropy_weight
    
    def forward(self, c_cat):

        batch_size = c_cat.shape[0]//2
        class_num = c_cat.shape[1]
        c_i, c_j = c_cat[:batch_size], c_cat[batch_size:]

        c_i_all = torch.cat(GatherLayer.apply(c_i), dim=0)
        c_j_all = torch.cat(GatherLayer.apply(c_j), dim=0)
        total_batch = c_i_all.shape[0]

        ## Start with the entropy term
        p_i = c_i_all.sum(dim=0)
        p_j = c_j_all.sum(dim=0)
        p_i = p_i/p_i.sum()
        p_j = p_j/p_j.sum()

        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i + 1e-10)).sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j + 1e-10)).sum()
        ne_loss = ne_i + ne_j

        c_i_all = c_i_all.t()
        c_j_all = c_j_all.t()
        
        negatives_mask = (~torch.eye(class_num*2, class_num*2, dtype=bool, device=c_cat.device)).float()
        representations = torch.cat([c_i_all, c_j_all], dim=0)

        c = nn.functional.normalize(representations, dim=1)
        similarity_matrix = torch.mm(c, c.t())

        # similarity_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, class_num)
        sim_ji = torch.diag(similarity_matrix, -class_num)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2*class_num)

        return loss, ne_loss*self.entropy_weight



class SharpenedClusterLoss(nn.Module):
    def __init__(self, T_pred=0.3, T_target=0.05, entropy_weight=1.0):
        super().__init__()
        self.T_pred = T_pred
        self.T_target = T_target
        self.entropy_weight = entropy_weight

    def forward(self, logits):

        batch_size = logits.shape[0] // 2

        z_i, z_j = logits[:batch_size], logits[batch_size:]

        p_i = nn.functional.softmax(z_i / self.T_pred, dim=1)
        p_j = nn.functional.softmax(z_j / self.T_pred, dim=1)

        q_i = nn.functional.softmax(z_i / self.T_target, dim=1).detach()
        q_j = nn.functional.softmax(z_j / self.T_target, dim=1).detach()

        loss = (
            -(q_i * torch.log(p_j + 1e-10)).sum(dim=1).mean()
            -(q_j * torch.log(p_i + 1e-10)).sum(dim=1).mean()
        )

        # entropy regularization (same idea as your code)
        p = torch.cat([p_i, p_j], dim=0)
        p_mean = p.mean(dim=0)

        ne_loss = math.log(p_mean.size(0)) + (p_mean * torch.log(p_mean + 1e-10)).sum()

        return loss, ne_loss * self.entropy_weight
