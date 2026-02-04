from torch import nn
from core.models.utils import get_act_from_string

class ClusteringHeadTwoLayer(nn.Module):
    def __init__(self, 
                 nchan : int, 
                 nclusters : int,
                 softmax_temp : float,
                 hidden_act_fn : object = nn.ReLU):
        super().__init__()

        self.middle_layer = max(nchan//2, nclusters*2)
        self.softmax_temp = softmax_temp
        
        self.proj = nn.Sequential(
            nn.Linear(nchan, self.middle_layer),
            hidden_act_fn(),
            nn.Linear(self.middle_layer, nclusters),
        )
        self.initialize_weights()
        self.softmax = nn.Softmax(dim=1)
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = self.proj(x)
        x = self.softmax(x/self.softmax_temp)
        return x

    
class ClusteringHeadTwoLayerBN(nn.Module):
    def __init__(self,
                 nchan : int,
                 nclusters : int,
                 nhidden: int = -1,
                 softmax_temp : float = 1.0,
                 hidden_act_fn : object = nn.ReLU):
        super().__init__()

        ## Slightly dodgy to retain previous default behaviour
        self.hidden = nhidden if nhidden != -1 else nchan//2

        self.softmax_temp = softmax_temp

        self.proj = nn.Sequential(
            nn.Linear(nchan, self.hidden, bias=False),
            nn.BatchNorm1d(self.hidden),
            hidden_act_fn(),
            nn.Linear(self.hidden, nclusters, bias=True),
        )
        self.initialize_weights()
        self.softmax = nn.Softmax(dim=1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.proj(x)
        x = self.softmax(x/self.softmax_temp)
        return x
    

    
class ClusteringHeadOneLayer(nn.Module):
    def __init__(self,
                 nchan : int,
                 nclusters : int,
                 softmax_temp : float):
        super().__init__()

        self.softmax_temp = softmax_temp
        self.linear = nn.Linear(nchan, nclusters)
        self.softmax = nn.Softmax(dim=1)        
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x/self.softmax_temp)
        return x

def get_clusthead(nchan, args):

    hidden_act_fn = get_act_from_string(args.enc_act)
    if args.clust_arch == "none":
        clust_head = None
    elif args.clust_arch == "one":
        clust_head = ClusteringHeadOneLayer(nchan, args.nclusters, args.softmax_temp)
    elif args.clust_arch == "twobn":
        clust_head = ClusteringHeadTwoLayerBN(nchan, args.nclusters, getattr(args, "nhidden", -1), args.softmax_temp, hidden_act_fn)
    else:
        clust_head = ClusteringHeadTwoLayer(nchan, args.nclusters, args.softmax_temp)
    return clust_head
