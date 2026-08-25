import torch
import torch.nn as nn
import torch.nn.functional as F
from core.losses.gather import GatherLayer

class VICRegLossDistributed(nn.Module):
    def __init__(
        self,
        sim_coeff=25.0,
        std_coeff=25.0,
        cov_coeff=1.0,
        eps=1e-4,
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps

    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, emb_cat):
        assert emb_cat.shape[0] % 2 == 0

        batch_size = emb_cat.shape[0] // 2
        z1 = emb_cat[:batch_size]
        z2 = emb_cat[batch_size:]

        # Paired-view invariance. DDP averaging gives the global mean
        # when local batch sizes are equal.
        invariance_loss = F.mse_loss(z1, z2)

        # Variance and covariance should use the global distributed batch.
        z1_all = torch.cat(GatherLayer.apply(z1), dim=0)
        z2_all = torch.cat(GatherLayer.apply(z2), dim=0)

        variance_loss = (
            torch.mean(
                F.relu(
                    1.0 - torch.sqrt(z1_all.var(dim=0, unbiased=False) + self.eps)
                )
            )
            + torch.mean(
                F.relu(
                    1.0 - torch.sqrt(z2_all.var(dim=0, unbiased=False) + self.eps)
                )
            )
        )

        z1_centered = z1_all - z1_all.mean(dim=0)
        z2_centered = z2_all - z2_all.mean(dim=0)

        n = z1_all.shape[0]
        d = z1_all.shape[1]

        cov1 = z1_centered.T @ z1_centered / max(n - 1, 1)
        cov2 = z2_centered.T @ z2_centered / max(n - 1, 1)

        covariance_loss = (
            self.off_diagonal(cov1).pow(2).sum() / d
            + self.off_diagonal(cov2).pow(2).sum() / d
        )

        loss = (
            self.sim_coeff * invariance_loss
            + self.std_coeff * variance_loss
            + self.cov_coeff * covariance_loss
        )

        return loss, {
            "invariance": invariance_loss.detach(),
            "variance": variance_loss.detach(),
            "covariance": covariance_loss.detach(),
        }
