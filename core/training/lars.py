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
                        eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            trust_coef = group['trust_coef']
            eps = group['eps']
            lars_exclude = group.get('lars_exclude', False)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if lars_exclude:
                    update = grad
                    effective_lr = group['lr']
                else:
                    update = grad
                    if weight_decay != 0:
                        update = update.add(p, alpha=weight_decay)

                    p_norm = torch.norm(p)
                    update_norm = torch.norm(update)
                    # Guard: fall back to local_lr = 1.0 if either norm is 0.
                    if p_norm > 0 and update_norm > 0:
                        local_lr = trust_coef * p_norm / (update_norm + eps)
                    else:
                        local_lr = torch.ones((), device=p.device, dtype=p.dtype)
                    effective_lr = lr * local_lr

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(update).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(update)

                p.add_(buf, alpha=-effective_lr)


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
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine

    def _set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group['lr'] = lr

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
