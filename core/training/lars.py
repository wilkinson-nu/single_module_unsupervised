import torch
from math import cos, pi

class LARS(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr,
                 momentum=0.9,
                 weight_decay=0,
                 trust_coef=0.001,
                 eps=1e-8):
        defaults = dict(lr=lr,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        trust_coef=trust_coef,
                        eps=eps,
                        lars_exclude=False)
        super().__init__(params, defaults)

        # This will hold the diagnostics from the most recent optimizer step.
        #
        # We store simple Python floats here rather than tensors so that
        # they don't remain attached to the computation graph or GPU memory.
        self.last_stats = {}
        
    @torch.no_grad()
    def step(self):

        # Clear statistics from the previous optimizer step.
        self.last_stats = {}
        
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            trust_coef = group['trust_coef']
            eps = group['eps']
            lars_exclude = group.get('lars_exclude', False)
            names = group.get('names', None)
            
            # We will keep statistics for this parameter group separately.
            group_stats = []
            
            for param_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = p.grad

                # ---------------------------------------------------------
                # Basic norms
                # ---------------------------------------------------------

                weight_norm = torch.norm(p)
                grad_norm = torch.norm(grad)

                # Start with the raw gradient.
                
                update = p.grad

                ## Only apply to layers where LARS is appropriate
                if not lars_exclude:

                    ## Only apply WD to layers where LARS is applied
                    if weight_decay != 0:
                        update = update.add(p, alpha=weight_decay)

                    ## More logging
                    update_norm_before_lars = torch.norm(update)
                        
                    p_norm = torch.norm(p)
                    u_norm = torch.norm(update)
                    ones = torch.ones_like(p_norm)

                    if p_norm > 0 and u_norm > 0:
                        q = trust_coef * p_norm / (u_norm + eps)

                        ## Logging
                        trust_ratio = q
                        update = update.mul(q)

                else:
                    # BN parameters / biases:
                    # no weight decay and no LARS trust-ratio scaling.
                    trust_ratio = torch.tensor(
                        1.0,
                        device=p.device,
                        dtype=p.dtype,
                    )

                # Norm AFTER LARS scaling.
                lars_update_norm = torch.norm(update)
    
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(update).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(update)

                p.add_(buf, alpha=-lr)

                ## More logging:
                # p <- p - lr * momentum_buffer
                step_norm = lr * torch.norm(buf)

                # Fractional change in this parameter tensor:
                #
                #     ||delta w|| / ||w||
                #
                # This is probably the single most useful diagnostic
                # for determining whether LARS is making excessively
                # large updates.
                if weight_norm > 0:
                    relative_step = step_norm / weight_norm
                else:
                    relative_step = torch.tensor(
                        0.0,
                        device=p.device,
                        dtype=p.dtype,
                    )

                # ---------------------------------------------------------
                # Save diagnostics
                # ---------------------------------------------------------
                if names is not None:
                    name = names[param_idx]
                else:
                    name = f"group{group_idx}_param{param_idx}"
                    
                group_stats.append({
                    'name': name,
                    'weight_norm': weight_norm.item(),
                    'grad_norm': grad_norm.item(),
                    'trust_ratio': trust_ratio.item(),
                    'lars_update_norm': lars_update_norm.item(),
                    'step_norm': step_norm.item(),
                    'relative_step': relative_step.item(),
                })

            self.last_stats[group_idx] = group_stats

class LARS_LRScheduler:
    """Linear warmup from ~0 to lr_max over warmup_steps, then cosine decay
    to lr_min over the remaining (total_steps - warmup_steps) steps.

    Convention: call scheduler.step() AFTER optimizer.step().
    lr_max is the ALREADY-SCALED peak LR (e.g. ~4.8), not the base 0.3.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, lr_max, lr_min=0.0):
        assert 0 <= warmup_steps < total_steps, "need 0 <= warmup < total"
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.step_count = 0
        # Set the LR for the very first optimizer step (step index 0).
        self._set_lr(self._lr_at(0))

    def _lr_at(self, step):
        if step < self.warmup_steps:
            # Linear warmup: reaches lr_max at step == warmup_steps.
            return self.lr_max * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + cos(pi * progress))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine

    def _set_lr(self, lr):
        for group in self.optimizer.param_groups:
            if group.get('lars_exclude', False):
                group['lr'] = lr * group.get('lr_scale', 1.0)
            else:
                group['lr'] = lr
        #group['lr'] = lr

    def step(self):
        self.step_count += 1
        self._set_lr(self._lr_at(self.step_count))

    def get_last_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {'step_count': self.step_count}

    def load_state_dict(self, sd):
        self.step_count = sd['step_count']
        self._set_lr(self._lr_at(self.step_count))

                

## class LARS_LRScheduler:
##     def __init__(self,
##                  optimizer,
##                  warmup_steps,
##                  total_steps,
##                  lr_max,
##                  lr_min=0.0):
##         self.optimizer = optimizer
##         self.warmup_steps = warmup_steps
##         self.total_steps = total_steps
##         self.lr_max = lr_max
##         self.lr_min = lr_min
##         self.step_count = 0
## 
##         ## Ensure we start with a sensible value
##         initial_lr = self.lr_min + (self.lr_max - self.lr_min) * (1. / self.warmup_steps)
##         for g in optimizer.param_groups:
##             g['lr'] = initial_lr
##         
##         self._last_lr = [group['lr'] for group in optimizer.param_groups]
## 
##         
##         
##     def step(self):
##         self.step_count += 1
## 
##         # linear warmup
##         if self.step_count <= self.warmup_steps:
##             lr_factor = self.step_count / self.warmup_steps
##         else:
##             # cosine decay after warmup
##             progress = min(1.0,
##                            (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
##                            )
##             lr_factor = 0.5 * (1 + cos(pi * progress))
## 
##         lr = self.lr_min + (self.lr_max - self.lr_min) * lr_factor
## 
##         # update global LR in all param groups
##         for group in self.optimizer.param_groups:
##             group['lr'] = lr
## 
##         self._last_lr = [lr] * len(self.optimizer.param_groups)
## 
##     def get_last_lr(self):
##         return self._last_lr            
