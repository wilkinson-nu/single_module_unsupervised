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
        self.correct = {name: torch.tensor(0, device=self.device) for name in self.classifier_config}
        self.total = {name: torch.tensor(0, device=self.device) for name in self.classifier_config}
        self.per_class_correct = {
            name: torch.zeros(cfg['n_classes'], device=self.device)
            for name, cfg in self.classifier_config.items()
        }
        self.per_class_total = {
            name: torch.zeros(cfg['n_classes'], device=self.device)
            for name, cfg in self.classifier_config.items()
        }
        self.abs_error = {name: torch.tensor(0.0, device=self.device) for name in self.classifier_config}
        self.mae_total = {name: torch.tensor(0,   device=self.device) for name in self.classifier_config}

    def update(self, outputs, labels):
        for name, logits in outputs.items():
            preds   = logits.argmax(dim=-1)
            targets = labels[name].long()

            self.correct[name] += (preds == targets).sum()
            self.total[name]   += targets.numel()

            n_classes = self.classifier_config[name]['n_classes']
            for c in range(n_classes):
                mask = targets == c
                self.per_class_correct[name][c] += (preds[mask] == targets[mask]).sum()
                self.per_class_total[name][c]   += mask.sum()

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

            acc = self.correct[name].item() / max(self.total[name].item(), 1)

            per_class_acc = []
            for c in range(n_classes):
                total_c = self.per_class_total[name][c].item()
                if total_c > 0:
                    per_class_acc.append(
                        self.per_class_correct[name][c].item() / total_c
                    )
                else:
                    per_class_acc.append(float('nan'))

            valid = [x for x in per_class_acc if not math.isnan(x)]
            mean_per_class_acc = sum(valid) / len(valid) if valid else float('nan')

            mae = self.abs_error[name].item() / max(self.mae_total[name].item(), 1)

            tp = sum(
                self.per_class_correct[name][c].item()
                for c in range(1, n_classes)
            )
            fn = sum(
                (self.per_class_total[name][c] - self.per_class_correct[name][c]).item()
                for c in range(1, n_classes)
            )
            recall_nonzero = tp / max(tp + fn, 1)

            results[name] = {
                'accuracy':           acc,
                'mean_per_class_acc': mean_per_class_acc,
                'per_class_acc':      per_class_acc,
                'mae':                mae,
                'recall_nonzero':     recall_nonzero,
            }
        return results
