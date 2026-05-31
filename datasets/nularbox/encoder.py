from torch import nn
import torch
import MinkowskiEngine as ME
import datasets.nularbox.resnet_encoder as resnet
from datasets.nularbox.custom_encoder import CCEncoderVarDepth, CCEncoderVarDepthBIG
from core.models.utils import get_act_from_string_ME

def get_encoder(args):

    stem_norm = bool(getattr(args, "enc_stem_norm", 0))
    init_stem_stride = getattr(args, "enc_init_stem_stride", 2)
    final_stem_stride = getattr(args, "enc_final_stem_stride", 2)
    stem_pool = getattr(args, "enc_stem_pool", 'none')
    stem_deep = bool(getattr(args, "enc_stem_deep", 0))
    res_pool = bool(getattr(args, "enc_res_pool", 0))
    final_pool = getattr(args, "enc_arch_pool", "avg")
    layer1_norm = bool(getattr(args, "enc_layer1_norm", 1))
    bottleneck_dim = getattr(args, "enc_final_linear", -1)
    enc_act = getattr(args, "enc_act", "relu")
    stem_channels = getattr(args, "enc_stem_channels", -1)
    
    ## Check for ResNet
    if "ResNet" in args.enc_arch:    
        if args.enc_arch in ["ResNet18", "ResNet18v2"]:
            enc = resnet.ResNet18v2
        elif args.enc_arch in ["ResNet34", "ResNet34v2"]:
            enc = resnet.ResNet34v2
        elif args.enc_arch in ["ResNet50", "ResNet50v2"]:
            enc = resnet.ResNet50v2
        elif args.enc_arch in ["ResNet101", "ResNet101v2"]:
            enc = resnet.ResNet101v2
        elif args.enc_arch in ["ResNet152", "ResNet152v2"]:
            enc = resnet.ResNet152v2
        elif args.enc_arch == "ResNet18v1":
            enc = resnet.ResNet18v1
        elif args.enc_arch == "ResNet34v1":
            enc = resnet.ResNet34v1
        elif args.enc_arch == "ResNet50v1":
            enc = resnet.ResNet50v1
        elif args.enc_arch == "ResNet101v1":
            enc = resnet.ResNet101v1
        elif args.enc_arch == "ResNet152v1":
            enc = resnet.ResNet152v1
            
        encoder = enc(enc_act=enc_act,
                      stem_pool=stem_pool,
                      init_stem_stride=init_stem_stride,
                      final_stem_stride=final_stem_stride,
	              stem_norm=stem_norm,
                      stem_deep=stem_deep,
                      res_pool=res_pool,
                      pool=final_pool,
                      layer1_norm=layer1_norm,
                      bottleneck_dim=bottleneck_dim,
                      stem_channels=stem_channels)
        
        return encoder

    ## Else, default to the old custom encoders...
    print(args.enc_arch)
    enc = None
    ## Only one architecture for now
    if args.enc_arch == "d6":
        enc = CCEncoderVarDepth 
        depth = 6
    elif args.enc_arch == "d5":
        enc = CCEncoderVarDepth
        depth = 5
    elif args.enc_arch == "d4":
        enc = CCEncoderVarDepth
        depth = 4
    elif args.enc_arch == "d3":
        enc = CCEncoderVarDepth
        depth =	3
    elif args.enc_arch == "bigd6":
        enc = CCEncoderVarDepthBIG 
        depth = 6
    elif args.enc_arch == "bigd5":
        enc = CCEncoderVarDepthBIG
        depth = 5
    elif args.enc_arch == "bigd4":
        enc = CCEncoderVarDepthBIG
        depth = 4
    elif args.enc_arch == "bigd3":
        enc = CCEncoderVarDepthBIG
        depth =	3
        
    enc_act_fn=get_act_from_string_ME(args.enc_act)
    encoder = enc(nchan=args.nchan, \
                  act_fn=enc_act_fn, \
                  first_kernel=args.enc_arch_first_kernel, \
                  pool=args.enc_arch_pool, \
                  slow_growth=bool(args.enc_arch_slow_growth),
                  final_linear=args.enc_arch_final_linear,
                  depth=depth)
    return encoder
