import torch
import math
from core.training.lars import LARS, LARS_LRScheduler

def get_opt_and_sched(args, encoder, heads, total_steps):

    lr_scheduler = None
    wd_scheduler = None
    optimizer    = None

    ## Sort out the parameter groups
    param_groups = build_param_groups(encoder,
                                      heads,
                                      args.weight_decay,
                                      args.weight_decay_final,
                                      bool(args.weight_decay_head))
    
    ## Sort out the optimizer (one for each GPU...)
    if args.optimizer == 'lars':
        corr_lr = args.lr * (args.batch_size*args.world_size / 256)
        optimizer = LARS(
            param_groups,
            lr=corr_lr,
            momentum=args.lars_momentum,
            trust_coef=args.lars_trust_coeff,
        )

        warmup_steps = int(0.05 * total_steps)
        scheduler = LARS_LRScheduler(optimizer, warmup_steps, total_steps, lr_max=corr_lr, lr_min=0.0)

    ## Default to adam
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
        )
        if args.scheduler == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr*500, total_steps=total_steps, cycle_momentum=False)
        if args.scheduler == "step":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       milestones=[150,300,450],
                                                       gamma=0.1,
                                                       last_epoch=-1,
                                                       verbose=False)

    
    return optimizer, scheduler


def build_param_groups(encoder,
                       heads,
                       weight_decay,
                       weight_decay_final=-1,
                       weight_decay_head=False):
    """
    This works for both Adam and LARS.
    """
    
    enc_params  = []
    omit_params = []
    head_params = []

    for name, param in encoder.named_parameters():
        if not param.requires_grad:
            continue
        
        if param.ndim == 1 or name.endswith(".bias"):
            omit_params.append(param)
        else:
            enc_params .append(param)

    for module in list(heads.values()):
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim == 1 or name.endswith(".bias"):
                omit_params.append(param)
            else:
                head_params.append(param)

    ## sort out weight scheduler logic
    weight_sched = weight_decay_final > 0
    head_weight_decay = weight_decay if weight_decay_head else 0.0
            
    return [
        {"params": enc_params,  "weight_decay": weight_decay,      "weight_sched": weight_sched},
        {"params": omit_params, "weight_decay": 0.0,               "weight_sched": False, "lars_exclude": True},
        {"params": head_params, "weight_decay": head_weight_decay, "weight_sched": weight_sched and weight_decay_head},
    ]

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def update_weight_decay(optimizer, 
                        weight_decay, 
                        weight_decay_final, 
                        step, 
                        total_steps):

    if weight_decay_final <= 0: return weight_decay

    # Cosine schedule from weight_decay to weight_decay_final
    wd = weight_decay_final + 0.5 * (weight_decay - weight_decay_final) * (
        1 + math.cos(math.pi * step / total_steps)
    )

    for group in optimizer.param_groups:
        if group.get("weight_sched", False):
            group["weight_decay"] = wd

    return wd
