import torch
from torch import nn

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
