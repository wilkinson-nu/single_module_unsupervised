import torch
import math
from torch import optim
from core.training.lars import LARS, LARS_LRScheduler

def is_final_residual_gamma(name):
    return (
        name.endswith("norm3.bn.weight")  # v1 Bottleneck
        or name.endswith("norm2.bn.weight")  # v1 BasicBlock
    )

def get_opt_and_sched(args, encoder, heads, total_steps, world_size, print_debug=False):

    lr_scheduler = None
    wd_scheduler = None
    optimizer    = None

    ## Sort out the parameter groups
    param_groups = build_param_groups(encoder,
                                      heads,
                                      args.weight_decay,
                                      args.weight_decay_final,
                                      bool(args.weight_decay_head),
                                      print_debug)
    
    ## Sort out the optimizer (one for each GPU...)
    if args.optimizer == 'lars':
        if print_debug: print("Optimizer = LARS; LR =", args.lr, "(", args.lr * (args.batch_size*world_size / 256), "); trust =", args.lars_trust_coeff, "; mom =", args.lars_momentum)
        corr_lr = args.lr * (args.batch_size*world_size / 256)
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
        #if args.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                  max_lr=args.lr,
                                                  total_steps=total_steps,
                                                  pct_start=0.1,      # 10% warmup
                                                  div_factor=25,      # start at max_lr/25
                                                  final_div_factor=1e4,  # end at max_lr/1e4
                                                  cycle_momentum=False)
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
                       weight_decay_head=False,
                       print_debug=False):
    """
    This works for both Adam and LARS.
    """
    
    enc_params  = []
    omit_params = []
    head_params = []
    residual_gamma_params = []

    enc_names  = []
    omit_names = []
    head_names = []
    residual_gamma_names = []
    
    for name, param in encoder.named_parameters():
        if not param.requires_grad:
            continue

        if is_final_residual_gamma(name)
            residual_gamma_params.append(param)
            residual_gamma_names.append(name)
        elif param.ndim == 1 or name.endswith(".bias"):
            omit_params.append(param)
            omit_names.append(name)
        else:
            enc_params .append(param)
            enc_names .append(name)
            
    for module in list(heads.values()):
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim == 1 or name.endswith(".bias"):
                omit_params.append(param)
                omit_names.append(name)
            else:
                head_params.append(param)
                head_names.append(name)
                
    ## sort out weight scheduler logic
    weight_sched = weight_decay_final > 0
    head_weight_decay = weight_decay if weight_decay_head else 0.0

    if print_debug:
        print("Parameter groups:")
        print(f"ENCODER ({len(enc_params)} parameters):")
        for name, param in zip(enc_names, enc_params):
            print(f"  LARS + WD:       {name:60s} {tuple(param.shape)}")

        print(f"RES GAMMA ({len(residual_gamma_params)} parameters):")
        for name, param in zip(residual_gamma_names, residual_gamma_params):
            print(f"  LARS + WD:       {name:60s} {tuple(param.shape)}")        
            
        print(f"OMITTED ({len(omit_params)} parameters):")
        for name, param in zip(omit_names, omit_params):
            print(f"  NO LARS / NO WD:  {name:60s} {tuple(param.shape)}")

        print(f"HEAD ({len(head_params)} parameters):")
        for name, param in zip(head_names, head_params):
            print(
                f"  LARS + "
                f"{'WD' if weight_decay_head else 'NO WD':<6}:    "
                f"{name:60s} {tuple(param.shape)}"
            )
    
    return [
        {
            "group_name": "enc",
            "params": enc_params,
            "names": enc_names,
            "weight_decay": weight_decay,
            "weight_sched": weight_sched,
            "lars_exclude": False
        },
        {
            "group_name": "omit",
            "params": omit_params,
            "names": omit_names,
            "weight_decay": 0.0,
            "weight_sched": False,
            "lars_exclude": True,
            "lr_scale": args.non_lars_lr_scale
        },
        {
            "group_name": "head",
            "params": head_params,
            "names": head_names,
            "weight_decay": head_weight_decay,
            "weight_sched": weight_sched and weight_decay_head,
            "lars_exclude": False
        },
        {
            "group_name": "residual_gamma",
            "params": residual_gamma_params,
            "names": residual_gamma_names,
            "weight_decay": 0.0,
            "weight_sched": False,
            "lars_exclude": True,
            "lr_scale": args.non_lars_lr_scale,
        }
    ]

## This is essentially redundant code used by the DINO implementation, remove at some point to use a consistent scheduler
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
