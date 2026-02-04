from torch import nn
from core.models.utils import get_act_from_string

class ProjectionHeadOneLogits(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int):
        super().__init__()

        self.proj = nn.Linear(nchan, nlatent, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.proj(x)

class ProjectionHeadLogitsBN(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int,
                 nhidden: int = -1,
                 hidden_act_fn : object = nn.ReLU):
        super().__init__()

        ## Slightly dodgy to retain previous default behaviour
        self.hidden = nhidden if nhidden != -1 else nchan//4

        self.proj = nn.Sequential(
            nn.Linear(nchan, self.hidden, bias=False),
            nn.BatchNorm1d(self.hidden),
            hidden_act_fn(),
            nn.Linear(self.hidden, nlatent, bias=True),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.proj(x)

class ProjectionHeadLogits(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int,
                 nhidden: int = -1,
                 hidden_act_fn : object = nn.ReLU):
        super().__init__()

        ## Slightly dodgy to retain previous default behaviour
        self.hidden = nhidden if nhidden != -1 else nchan//4
        
        self.proj = nn.Sequential(
            nn.Linear(nchan, self.hidden),
            hidden_act_fn(),
            nn.Linear(self.hidden, nlatent),
        )
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.proj(x)

class ProjectionHead(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int,
                 nhidden: int = -1,
                 hidden_act_fn : object = nn.ReLU,
                 latent_act_fn : object = nn.Tanh):
        super().__init__()

        ## Slightly dodgy to retain previous default behaviour
        self.hidden = nhidden if nhidden != -1 else nchan//4
        
        self.proj = nn.Sequential(
            nn.Linear(nchan, self.hidden),
            hidden_act_fn(),
            nn.Linear(self.hidden, nlatent),
            latent_act_fn(),
        )
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.proj(x)

def get_projhead(nchan, args):
    hidden_act_fn = get_act_from_string(args.enc_act)
    latent_act_fn=nn.Tanh
    if args.proj_arch == "logits":
        proj_head = ProjectionHeadLogits(nchan, args.latent, getattr(args, "nhidden", -1), hidden_act_fn)
    elif args.proj_arch == "logitsbn":
        proj_head = ProjectionHeadLogitsBN(nchan, args.latent, getattr(args, "nhidden", -1), hidden_act_fn)
    elif args.proj_arch == "one":
        proj_head = ProjectionHeadOneLogits(nchan, args.latent)
    else:
        proj_head = ProjectionHead(nchan, args.latent, getattr(args, "nhidden", -1), hidden_act_fn, latent_act_fn)
    return proj_head
