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
    'nproton':   {'n_classes': 4, 'weight': 1.0},
    'npipm':     {'n_classes': 3, 'weight': 1.0},
    'npi0':      {'n_classes': 3, 'weight': 1.0},
    'nem':       {'n_classes': 3, 'weight': 1.0},
    'ncluster':  {'n_classes': 4, 'weight': 1.0},
    'nlambda0':  {'n_classes': 2, 'weight': 5.0},  # upweight rare events
    'nkapm':     {'n_classes': 2, 'weight': 5.0},  # upweight rare events    
    'nka0':      {'n_classes': 2, 'weight': 5.0},  # upweight rare events    
    'ncharged':  {'n_classes': 6, 'weight': 1.0},
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
        self.correct = {name: torch.zeros((), dtype=torch.long, device=self.device) for name in self.classifier_config}
        self.total = {name: torch.zeros((), dtype=torch.long, device=self.device) for name in self.classifier_config}
        self.per_class_correct = {
            name: torch.zeros(cfg['n_classes'], device=self.device)
            for name, cfg in self.classifier_config.items()
        }
        self.per_class_total = {
            name: torch.zeros(cfg['n_classes'], device=self.device)
            for name, cfg in self.classifier_config.items()
        }
        self.abs_error = {name: torch.zeros((), device=self.device) for name in self.classifier_config}
        self.mae_total = {name: torch.zeros((), dtype=torch.long,   device=self.device) for name in self.classifier_config}


    def update(self, outputs, labels):
        for name, logits in outputs.items():
            preds   = logits.argmax(dim=-1)
            targets = labels[name].long()
            
            correct = (preds == targets)
            self.correct[name] += correct.sum()
            self.total[name]   += targets.numel()
            
            pc_total   = self.per_class_total[name]
            pc_correct = self.per_class_correct[name]
            pc_total.scatter_add_(0, targets, torch.ones_like(targets, dtype=pc_total.dtype))
            pc_correct.scatter_add_(0, targets, correct.to(pc_correct.dtype))
            
            self.abs_error[name] += (preds.float() - targets.float()).abs().sum()
            self.mae_total[name] += targets.numel()
            

    def reduce(self):
        for name in self.classifier_config:
            dist.all_reduce(self.correct[name],           op=dist.ReduceOp.SUM)
            dist.all_reduce(self.total[name],             op=dist.ReduceOp.SUM)
            dist.all_reduce(self.per_class_correct[name], op=dist.ReduceOp.SUM)
            dist.all_reduce(self.per_class_total[name],   op=dist.ReduceOp.SUM)
            dist.all_reduce(self.abs_error[name],         op=dist.ReduceOp.SUM)
            dist.all_reduce(self.mae_total[name],         op=dist.ReduceOp.SUM)


    def compute(self):
        results = {}
        for name, cfg in self.classifier_config.items():
            n_classes = cfg['n_classes']

            correct          = self.correct[name].item()
            total            = self.total[name].item()
            pc_correct       = self.per_class_correct[name].cpu().tolist()
            pc_total         = self.per_class_total[name].cpu().tolist()
            abs_error        = self.abs_error[name].item()
            mae_total        = self.mae_total[name].item()

            acc = correct / max(total, 1)

            per_class_acc = [
                (pc_correct[c] / pc_total[c]) if pc_total[c] > 0 else float('nan')
                for c in range(n_classes)
            ]

            valid = [x for x in per_class_acc if not math.isnan(x)]
            mean_per_class_acc = sum(valid) / len(valid) if valid else float('nan')

            mae = abs_error / max(mae_total, 1)

            tp = sum(pc_correct[1:n_classes])
            fn = sum(pc_total[c] - pc_correct[c] for c in range(1, n_classes))
            recall_nonzero = tp / max(tp + fn, 1)

            results[name] = {
                'accuracy':           acc,
                'mean_per_class_acc': mean_per_class_acc,
                'per_class_acc':      per_class_acc,
                'mae':                mae,
                'recall_nonzero':     recall_nonzero,
            }
        return results
