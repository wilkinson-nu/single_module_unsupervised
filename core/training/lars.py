import torch
from math import cos, pi

class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.9, weight_decay=1e-4, trust_coef=0.001, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        trust_coef=trust_coef, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                param = p
                grad = p.grad
                
                update = grad + group['weight_decay'] * param
                param_norm = torch.norm(param)
                update_norm = torch.norm(update)
                local_lr = group['trust_coef'] * param_norm / (update_norm + group['eps'])

                if 'momentum_buffer' not in self.state[p]:
                    buf = self.state[p]['momentum_buffer'] = torch.zeros_like(p)
                else:
                    buf = self.state[p]['momentum_buffer']

                buf.mul_(group['momentum']).add_(local_lr * update)
                p.add_(-group['lr'] * buf.to(p.dtype))
                

class LARS_LRScheduler:
    def __init__(self,
                 optimizer,
                 warmup_steps,
                 total_steps,
                 lr_max,
                 lr_min=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.step_count = 0

        ## Ensure we start with a sensible value
        initial_lr = self.lr_min + (self.lr_max - self.lr_min) * (1. / self.warmup_steps)
        for g in optimizer.param_groups:
            g['lr'] = initial_lr
        
        self._last_lr = [group['lr'] for group in optimizer.param_groups]

        
        
    def step(self):
        self.step_count += 1

        # linear warmup
        if self.step_count <= self.warmup_steps:
            lr_factor = self.step_count / self.warmup_steps
        else:
            # cosine decay after warmup
            progress = min(1.0,
                           (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                           )
            lr_factor = 0.5 * (1 + cos(pi * progress))

        lr = self.lr_min + (self.lr_max - self.lr_min) * lr_factor

        # update global LR in all param groups
        for group in self.optimizer.param_groups:
            group['lr'] = lr

        self._last_lr = [lr] * len(self.optimizer.param_groups)

    def get_last_lr(self):
        return self._last_lr            
