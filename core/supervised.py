import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import math

## These config options are interrelated, so I'm putting them here for now
LABEL_CLAMP = {
    'nproton':   3,
    'npipm':     2,
    'npi0':      2,
    'nem':       2,
    'ncluster':  3,
    'ncharged':  5,
    'nkapm':     1,
    'nka0':      1,
    'nlambda0':  1,
}

def _ncharged(l):
    return l['nproton'] + l['npipm'] + l['nkapm']

def _ncluster(l):
    return l['ndeuteron'] + l['nalpha'] + l['nhelium3'] + l['ntritium'] + l['nnuclfrag']

DERIVED_LABELS = {
    'ncharged': _ncharged,
    'ncluster': _ncluster,
}

DEFAULT_CLASSIFIER_CONFIG = {
    'nproton':   {'n_classes': 4, 'weight': 1.0, 'cap': 3},
    'npipm':     {'n_classes': 3, 'weight': 1.0, 'cap': 2},
    'npi0':      {'n_classes': 3, 'weight': 1.0, 'cap': 2},
    'nem':       {'n_classes': 3, 'weight': 1.0, 'cap': 2},
    'ncluster':  {'n_classes': 4, 'weight': 1.0, 'cap': 3},
    'nlambda0':  {'n_classes': 2, 'weight': 5.0, 'cap': 1},  # upweight rare events
    'nkapm':     {'n_classes': 2, 'weight': 5.0, 'cap': 1},  # upweight rare events    
    'nka0':      {'n_classes': 2, 'weight': 5.0, 'cap': 1},  # upweight rare events    
    'ncharged':  {'n_classes': 6, 'weight': 1.0, 'cap': 5},
}

class SupervisedHead(nn.Module):
    def __init__(self, encoder_dim, classifier_config):
        super().__init__()
        self.classifier_config = classifier_config
        self.heads = nn.ModuleDict({
            name: nn.Linear(encoder_dim, cfg['n_classes'])
            for name, cfg in classifier_config.items()
        })

    def forward(self, features):
        return {name: head(features) for name, head in self.heads.items()}

    
def supervised_loss(outputs, labels, classifier_config):
    total_loss = 0.0
    loss_dict = {}
    for name, cfg in classifier_config.items():
        loss = F.cross_entropy(outputs[name], labels[name].long())
        weighted_loss = cfg['weight'] * loss
        total_loss += weighted_loss
        loss_dict[name] = loss.detach()
    return total_loss, loss_dict


class ClassificationMetrics:
    def __init__(self, classifier_config, device):
        self.classifier_config = classifier_config
        self.device = device
        self.reset()

    def reset(self):
        self.correct = {
            name: torch.tensor(0, device=self.device)
            for name in self.classifier_config
        }
        self.total = {
            name: torch.tensor(0, device=self.device)
            for name in self.classifier_config
        }
        self.per_class_correct = {
            name: torch.zeros(cfg['n_classes'], device=self.device)
            for name, cfg in self.classifier_config.items()
        }
        self.per_class_total = {
            name: torch.zeros(cfg['n_classes'], device=self.device)
            for name, cfg in self.classifier_config.items()
        }
        self.abs_error = {
            name: torch.tensor(0.0, device=self.device)
            for name in self.classifier_config
        }
        # Removed mae_total — it's identical to total, no need to track separately

    def update(self, outputs, labels):
        for name, logits in outputs.items():
            with torch.no_grad():
                preds   = logits.argmax(dim=-1)
                targets = labels[name].long()
                n_classes = self.classifier_config[name]['n_classes']
                
                self.correct[name] += (preds == targets).sum()
                self.total[name]   += targets.numel()
                
                # Vectorized per-class counts — eliminates Python loop + GPU syncs
                # one_hot: (N, n_classes), then mask by correct predictions
                one_hot = torch.nn.functional.one_hot(targets, num_classes=n_classes).float()  # (N, C)
                correct_mask = (preds == targets).float().unsqueeze(1)                          # (N, 1)
                self.per_class_correct[name] += (one_hot * correct_mask).sum(dim=0)
                self.per_class_total[name]   += one_hot.sum(dim=0)
                
                self.abs_error[name] += (preds.float() - targets.float()).abs().sum()

    def reduce(self):
        # Batch all tensors into a single all_reduce per classifier to reduce
        # communication overhead
        for name in self.classifier_config:
            # Stack scalars + per_class vectors into one tensor
            packed = torch.cat([
                self.correct[name].unsqueeze(0).float(),
                self.total[name].unsqueeze(0).float(),
                self.abs_error[name].unsqueeze(0),
                self.per_class_correct[name],
                self.per_class_total[name],
            ])
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)

            # Unpack
            n = self.classifier_config[name]['n_classes']
            self.correct[name]           = packed[0].long()
            self.total[name]             = packed[1].long()
            self.abs_error[name]         = packed[2]
            self.per_class_correct[name] = packed[3:3 + n]
            self.per_class_total[name]   = packed[3 + n:3 + 2 * n]

    def compute(self):
        # All .item() calls are deferred to here — called once per epoch,
        # so syncs are acceptable
        results = {}
        for name, cfg in self.classifier_config.items():
            n_classes = cfg['n_classes']

            total            = self.total[name].item()
            correct          = self.correct[name].item()
            per_class_correct = self.per_class_correct[name].cpu()  # single transfer
            per_class_total   = self.per_class_total[name].cpu()

            acc = correct / max(total, 1)

            # Vectorized per-class accuracy — no Python loop
            valid_mask     = per_class_total > 0
            per_class_acc_tensor = torch.where(
                valid_mask,
                per_class_correct / per_class_total.clamp(min=1),
                torch.full_like(per_class_correct, float('nan')),
            )
            per_class_acc = per_class_acc_tensor.tolist()

            valid_accs         = per_class_acc_tensor[valid_mask]
            mean_per_class_acc = valid_accs.mean().item() if valid_mask.any() else float('nan')

            mae = self.abs_error[name].item() / max(total, 1)

            # Vectorized recall — slice off class 0
            tp             = per_class_correct[1:].sum().item()
            fn             = (per_class_total[1:] - per_class_correct[1:]).sum().item()
            recall_nonzero = tp / max(tp + fn, 1)

            results[name] = {
                'accuracy':           acc,
                'mean_per_class_acc': mean_per_class_acc,
                'per_class_acc':      per_class_acc,
                'mae':                mae,
                'recall_nonzero':     recall_nonzero,
            }
        return results
