from torch import nn
import torch.nn.functional as F
from core.models.utils import get_act_from_string
from collections import OrderedDict

class ProjectionHeadTwoLayerDINO(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int,
                 nhidden: int,
                 hidden_act_fn : object = nn.ReLU,
                 apply_bn : bool = False):
        super().__init__()

        self.nchan = nchan
        self.nlatent = nlatent
        self.nhidden = nhidden
        self.apply_bn = apply_bn
        self.hidden_act_fn = hidden_act_fn
        
        self.proj = self.make_network()
        self.initialize_weights()
        

    def make_network(self):
        proj = OrderedDict()
        proj['lin1'] = nn.Linear(self.nchan, self.nhidden, bias=True)
        if self.apply_bn: proj['norm1'] = nn.BatchNorm1d(self.nhidden)
        proj['act1'] = self.hidden_act_fn()
        ## Bias doesn't matter because of the L2 norm at the end
        proj['lin2'] = nn.Linear(self.nhidden, self.nlatent, bias=True)
        return nn.Sequential(proj)
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_hidden=False):

        ## Separate execution so hidden layers can be returned
        h1_logits = self.proj.lin1(x)
        h1 = h1_logits

        if self.apply_bn: h1 = self.proj.norm1(h1)
        h1 = self.proj.act1(h1)
        
        out = self.proj.lin2(h1)
        ## L2 normalize at the end
        out = F.normalize(out, dim=-1, p=2)
        
        if return_hidden:
            return {"proj_layer1": F.normalize(h1_logits, dim=-1, p=2),
                    "proj_final": out}
        return out


class ProjectionHeadThreeLayerDINO(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int,
                 nhidden: int,
                 hidden_act_fn : object = nn.ReLU,
                 apply_bn : bool = False):
        super().__init__()


        self.nchan = nchan
        self.nlatent = nlatent
        self.nhidden = nhidden
        self.apply_bn = apply_bn
        self.hidden_act_fn = hidden_act_fn

        self.proj = self.make_network()
        self.initialize_weights()

    def make_network(self):
        proj = OrderedDict()
        proj['lin1'] = nn.Linear(self.nchan, self.nhidden, bias=True)
        if self.apply_bn: proj['norm1'] = nn.BatchNorm1d(self.nhidden)
        proj['act1'] = self.hidden_act_fn()
        proj['lin2'] = nn.Linear(self.nhidden, self.nhidden, bias=True)
        if self.apply_bn: proj['norm2'] = nn.BatchNorm1d(self.nhidden)
        proj['act2'] = self.hidden_act_fn()
        proj['lin3'] = nn.Linear(self.nhidden, self.nlatent, bias=True)
        return nn.Sequential(proj)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_hidden=False):

        ## Separate execution so hidden layers can be returned
        h1_logits = self.proj.lin1(x)
        h1 = h1_logits

        if self.apply_bn: h1 = self.proj.norm1(h1)
        h1 = self.proj.act1(h1)
        h2_logits = self.proj.lin2(h1)
        h2 = h2_logits

        if self.apply_bn: h2 = self.proj.norm2(h2)
        h2 = self.proj.act2(h2)

        out = self.proj.lin3(h2)
        ## L2 normalize at the end
        out = F.normalize(out, dim=-1, p=2)
        
        if return_hidden:
            return {"proj_layer1": F.normalize(h1_logits, dim=-1, p=2),
                    "proj_layer2": F.normalize(h2_logits, dim=-1, p=2),
                    "proj_final": out}
        return out

        
        
def get_dino_projhead(nchan, args):
    hidden_act_fn = get_act_from_string(args.enc_act)
    
    if args.proj_arch == "two":
        proj_head = ProjectionHeadTwoLayerDINO(nchan, args.latent, args.nhidden, hidden_act_fn, apply_bn=False)
    elif args.proj_arch == "twobn":
        proj_head = ProjectionHeadTwoLayerDINO(nchan, args.latent, args.nhidden, hidden_act_fn, apply_bn=True)
    elif args.proj_arch == "three":
        proj_head = ProjectionHeadThreeLayerDINO(nchan, args.latent, args.nhidden, hidden_act_fn, apply_bn=False)
    elif args.proj_arch == "threebn":
        proj_head = ProjectionHeadThreeLayerDINO(nchan, args.latent, args.nhidden, hidden_act_fn, apply_bn=True)
    return proj_head
