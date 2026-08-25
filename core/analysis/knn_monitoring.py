import torch
import MinkowskiEngine as ME
import torch.distributed as dist
import torch.nn.functional as F

MONITOR_LABELS = {'nproton': 4, 'npipm': 3, 'npi0': 3,
                  'nem': 3, 'ncluster': 4, 'ncharged': 6}

@torch.no_grad()
def extract_features(encoder, loader, device, label_names):
    was_training = encoder.training
    encoder.eval()
    fs, ls = [], {n: [] for n in label_names}
    for bcoords, bfeats, blabels, bs in loader:
        bcoords = bcoords.to(device, non_blocking=True)
        bfeats  = bfeats.to(device,  non_blocking=True)
        batch   = ME.SparseTensor(bfeats, bcoords, device=device)
        fs.append(encoder.module(batch, bs).float())
        for n in label_names:
            ls[n].append(blabels[n].to(device).long())
    if was_training:
        encoder.train()

    f = torch.cat(fs)
    world = dist.get_world_size()
    gf = [torch.zeros_like(f) for _ in range(world)]
    dist.all_gather(gf, f.contiguous())
    f = torch.cat(gf)

    out_l = {}
    for n in label_names:
        l  = torch.cat(ls[n])
        gl = [torch.zeros_like(l) for _ in range(world)]
        dist.all_gather(gl, l.contiguous())
        out_l[n] = torch.cat(gl)
    return f, out_l


@torch.no_grad()
def knn_votes(q, bank, bank_lab, n_classes, k=20, T=0.1, chunk=2048):

    ## Center the features
    center = bank.mean(dim=0, keepdim=True)

    qn, bn = F.normalize(q - center, dim=1), F.normalize(bank - center, dim=1)
    
    out = []
    for i in range(0, qn.shape[0], chunk):
        sim    = qn[i:i + chunk] @ bn.t()
        sk, ik = sim.topk(min(k, bn.shape[0]), dim=1)
        w      = (sk / T).exp()
        oh     = F.one_hot(bank_lab[ik], n_classes).float()
        out.append((oh * w.unsqueeze(-1)).sum(1))
    return torch.cat(out)
