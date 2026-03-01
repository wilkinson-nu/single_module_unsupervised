from torch import nn
from core.models.utils import get_act_from_string
from collections import OrderedDict

class ProjectionHeadOneLogits(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int):
        super().__init__()

        self.proj = nn.Linear(nchan, nlatent, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)

    ## return_logits does nothing for a single layer
    def forward(self, x, return_logits=False):
        return self.proj(x)

class ProjectionHeadTwoLayer(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int,
                 nhidden: int = -1,
                 hidden_act_fn : object = nn.ReLU,
                 apply_bn : bool = False):
        super().__init__()

        ## Slightly dodgy to retain previous default behaviour
        self.nchan = nchan
        self.nlatent = nlatent
        self.nhidden = nhidden if nhidden != -1 else nchan//4
        self.apply_bn = apply_bn
        self.hidden_act_fn = hidden_act_fn
        
        self.proj = self.make_network()
        self.initialize_weights()

    def make_network(self):
        proj = OrderedDict()
        proj['lin1'] = nn.Linear(self.nchan, self.nhidden, bias=True)
        if self.apply_bn: proj['norm1'] = nn.BatchNorm1d(self.nhidden)
        proj['act1'] = self.hidden_act_fn()
        proj['lin2'] = nn.Linear(self.nhidden, self.nlatent, bias=False)
        return nn.Sequential(proj)
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
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

        if return_hidden: return [h1_logits, out]
        return out


class ProjectionHeadThreeLayer(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int,
                 nhidden: int = -1,
                 hidden_act_fn : object = nn.ReLU,
                 apply_bn : bool = False):
        super().__init__()


        self.nchan = nchan
        self.nlatent = nlatent
        ## Slightly dodgy to retain previous default behaviour
        self.nhidden = nhidden if nhidden != -1 else nchan//4
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
        proj['lin3'] = nn.Linear(self.nhidden, self.nlatent, bias=False)
        return nn.Sequential(proj)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
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

        if return_hidden: return [h1_logits, h2_logits, out]
        return out

        
        
def get_projhead(nchan, args):
    hidden_act_fn = get_act_from_string(args.enc_act)
    
    if args.proj_arch in ["logits", "two"]:
        proj_head = ProjectionHeadTwoLayer(nchan, args.latent, getattr(args, "nhidden", -1), hidden_act_fn, apply_bn=False)
    elif args.proj_arch in ["logitsbn", "twobn"]:
        proj_head = ProjectionHeadTwoLayer(nchan, args.latent, getattr(args, "nhidden", -1), hidden_act_fn, apply_bn=True)
    elif args.proj_arch == "three":
        proj_head = ProjectionHeadThreeLayer(nchan, args.latent, getattr(args, "nhidden", -1), hidden_act_fn, apply_bn=False)
    elif args.proj_arch == "threebn":
        proj_head = ProjectionHeadThreeLayer(nchan, args.latent, getattr(args, "nhidden", -1), hidden_act_fn, apply_bn=True)
    elif args.proj_arch == "one":
        proj_head = ProjectionHeadOneLayer(nchan, args.latent)
    return proj_head
