import torch
import MinkowskiEngine as ME
import torch.distributed as dist
import torch.nn.functional as F
from core.supervised import ClassificationMetrics, SupervisedHead, supervised_loss
from core.utils import print0

@torch.no_grad()
def extract_features(encoder, loader, device, label_names):
    was_training = encoder.training
    encoder.eval()
    fs, ls = [], {n: [] for n in label_names}

    print0("Looping over events...")
    for bcoords, bfeats, blabels, bs in loader:
        bcoords = bcoords.to(device, non_blocking=True)
        bfeats  = bfeats.to(device,  non_blocking=True)
        batch   = ME.SparseTensor(bfeats, bcoords, device=device)
        fs.append(encoder.module(batch, bs).float())
        for n in label_names:
            ls[n].append(blabels[n].to(device).long())
    if was_training:
        encoder.train()

    print0("Finished loop over events...")
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
    print0("Finished extract_features gather")
    return f, out_l


@torch.no_grad()
def knn_neighbors(q, bank, k=20, chunk=2048):
    center = bank.mean(dim=0, keepdim=True)

    qn = F.normalize(q - center, dim=1)
    bn = F.normalize(bank - center, dim=1)

    similarities = []
    indices = []

    k = min(k, bn.shape[0])

    for i in range(0, qn.shape[0], chunk):
        sim = qn[i:i + chunk] @ bn.t()
        sk, ik = sim.topk(k, dim=1)

        similarities.append(sk)
        indices.append(ik)

    return torch.cat(similarities), torch.cat(indices)


@torch.no_grad()
def knn_votes_from_neighbors(
    similarities,
    indices,
    bank_labels,
    n_classes,
    temperature=0.1,
):
    # Subtracting the row maximum is numerically safer. It does not
    # change the winning class because it scales every row uniformly.
    weights = torch.exp(
        (similarities - similarities[:, :1]) / temperature
    )

    neighbor_labels = bank_labels[indices]

    one_hot = F.one_hot(
        neighbor_labels,
        num_classes=n_classes,
    ).float()

    return (one_hot * weights.unsqueeze(-1)).sum(dim=1)


def evaluate_knn(
        bank_features,
        bank_labels,
        query_features,
        query_labels,
        *,
        classifier_config,
        device,
        k,
        temperature,
):
    metrics = ClassificationMetrics(
        classifier_config,
        device=device,
    )

    similarities, indices = knn_neighbors(
        query_features,
        bank_features,
        k=k,
    )
    
    votes = {
        name: knn_votes_from_neighbors(
            similarities,
            indices,
            bank_labels[name],
            cfg["n_classes"],
            temperature=temperature,
        )
        for name, cfg in classifier_config.items()
    }

    metrics.update(votes, query_labels)

    # No reduce: features and labels are already globally gathered,
    # and this calculation runs on rank 0.
    return metrics.compute()

def fit_linear_probe(
    bank_features,
    bank_labels,
    query_features,
    query_labels,
    *,
    classifier_config,
    device,
    epochs=20,
    batch_size=1024,
    lr=1e-2,
    seed=12345,
):
    # Keep the extracted feature dataset on CPU.
    bank_features = bank_features.detach().float().cpu()
    query_features = query_features.detach().float().cpu()

    bank_labels = {
        name: labels.detach().long().cpu()
        for name, labels in bank_labels.items()
        if name in classifier_config
    }
    query_labels = {
        name: labels.detach().long().cpu()
        for name, labels in query_labels.items()
        if name in classifier_config
    }

    # Fit feature preprocessing using the bank only.
    mean = bank_features.mean(dim=0, keepdim=True)
    std = bank_features.std(
        dim=0,
        unbiased=False,
        keepdim=True,
    )
    
    # Prevent very low-variance dimensions from receiving huge amplification.
    std_floor = 0.01 * std.median()
    std_safe = std.clamp_min(std_floor)
    
    bank_features = (bank_features - mean) / std_safe
    query_features = (query_features - mean) / std_safe

    # Avoid probe initialization/training changing the main training RNG.
    cuda_devices = (
        [device.index]
        if device.type == "cuda"
        else []
    )

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)

        probe = SupervisedHead(
            encoder_dim=bank_features.shape[1],
            classifier_config=classifier_config,
        ).to(device)

        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=lr,
            weight_decay=0.0,
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        probe.train()

        for i in range(epochs):
            
            #sum_loss = torch.zeros((), device=device)
            #num_samples = 0
            
            permutation = torch.randperm(
                bank_features.shape[0],
                generator=generator,
            )

            for start in range(
                0,
                bank_features.shape[0],
                batch_size,
            ):
                indices = permutation[start:start + batch_size]

                features = bank_features[indices].to(device, non_blocking=True)
                labels = {
                    name: values[indices].to(device, non_blocking=True)
                    for name, values in bank_labels.items()
                }

                outputs = probe(features)

                loss, _ = supervised_loss(outputs, labels, classifier_config)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                #this_batch_size = features.shape[0]
                #sum_loss += loss.detach().double()*this_batch_size
                #num_samples += this_batch_size

            ## Report the average loss for this epoch
            #av_loss = sum_loss/max(num_samples, 1)
            #print0(f"{i}: loss = {av_loss.item()}")
                
        # Evaluate on the query split.
        probe.eval()

        probe_metrics = ClassificationMetrics(
            classifier_config,
            device=device,
        )

        with torch.no_grad():
            for start in range(
                0,
                query_features.shape[0],
                batch_size,
            ):
                end = start + batch_size

                features = query_features[start:end].to(device, non_blocking=True)
                labels = {
                    name: values[start:end].to(device, non_blocking=True)
                    for name, values in query_labels.items()
                }

                outputs = probe(features)
                probe_metrics.update(outputs, labels)

        # Do not call reduce(): this probe runs on rank 0 using globally
        # gathered bank and query features.
        results = probe_metrics.compute()

    del probe, optimizer
    return results
