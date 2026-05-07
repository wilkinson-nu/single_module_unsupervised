import torch

## A simply logging utility
def log_scalar(writer, metrics, name, value, step):
    if writer is not None:
        writer.add_scalar(name, value, step)
    metrics[name].append(value)

## Some diagnostic functions
@torch.no_grad()
def log_grad_norm(module, tag, writer, iteration):
    if writer is None: return
    device = next(module.parameters()).device
    total = torch.tensor(0.0, device=device)
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        total += p.grad.pow(2).sum()

    grad_norm = total.sqrt().item()
    writer.add_scalar(f'grads/{tag}/sqrt_sum_grad_l2', grad_norm, iteration)
    return

@torch.no_grad()
def log_grad_rms(module, tag, writer, iteration):
    if writer is None: return

    device = next(module.parameters()).device
    total = torch.tensor(0.0, device=device)
    count = 0
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        total += p.grad.pow(2).mean()
        count += 1

    mean_rms = (total / count).sqrt()
    writer.add_scalar(f'grads/{tag}/mean_grad_rms', mean_rms, iteration)
    return

@torch.no_grad()
def log_grad_over_wgt(module, tag, writer, iteration, eps=1e-12):
    if writer is None: return
    device = next(module.parameters()).device
    g2 = torch.tensor(0.0, device=device)
    w2 = torch.tensor(0.0, device=device)
    
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        g2 += p.grad.pow(2).sum()
        w2 += p.data.pow(2).sum()
        

    ratio = (g2.sqrt() / (w2.sqrt() + eps)).item()
    writer.add_scalar(f'grads/{tag}/sum_grad_over_wgt', ratio, iteration)
    return
