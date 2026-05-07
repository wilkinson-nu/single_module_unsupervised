import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from core.losses.gather import GatherLayer

class SimDINOLoss(nn.Module):
    """
    From the MCRLoss: https://github.com/RobinWu218/SimDINO/blob/main/simdino/main_dino.py#L497
    Simplified for the single crop case here
    """
    
    def __init__(self, eps=0.5, coeff=1.0):
        super().__init__()
        self.eps = eps
        self.coeff = coeff

    def forward(self, student_feat, teacher_feat):
        """
        Expansion Loss and Compression Loss between features of the teacher and student networks.
        Both are tensors with shape (2*N, D)
        """

        N = student_feat.shape[0] // 2
        D = student_feat.shape[-1]

        s1, s2 = student_feat[:N], student_feat[N:]
        t1, t2 = teacher_feat[:N], teacher_feat[N:]

        comp_loss = self.calc_compression(s1, s2, t1, t2)
        expa_loss = self.calc_expansion(s1, s2)

        loss = -self.coeff*comp_loss - expa_loss
        return loss, comp_loss.detach(), expa_loss.detach()        
            
    def calc_compression(self, s1, s2, t1, t2):
        """
        Compute compression loss between student and teacher features.
        Simplified by the 2 global view assumption
        """
        sim_1 = F.cosine_similarity(s1, t2, dim=-1).mean()
        sim_2 = F.cosine_similarity(s2, t1, dim=-1).mean()
        return (sim_1 + sim_2) / 2
    
    def calc_expansion(self, s1, s2):
        """
        Compute expansion loss using Coding Rate estimation.
        """
        N = 1
        if dist.is_initialized():
            N = dist.get_world_size()
            s1 = torch.cat(GatherLayer.apply(s1), dim=0)
            s2 = torch.cat(GatherLayer.apply(s2), dim=0)

        m, p = s1.shape
            
        ## Per-view covariances
        cov1 = s1.T @ s1
        cov2 = s2.T @ s2

        scalar = p / (m * self.eps)
        I = torch.eye(p, device=s1.device, dtype=s1.dtype)

        ## Coding rate for each view via Cholesky log-determinant
        rate1 = torch.linalg.cholesky_ex(I + scalar * cov1)[0].diagonal().log().sum()
        rate2 = torch.linalg.cholesky_ex(I + scalar * cov2)[0].diagonal().log().sum()

        loss = (rate1 + rate2) / 2

        ## Balancing factor gamma (heuristic that could be a hyperparameter)
        loss *= (p + m) / (p * m)

        return loss

