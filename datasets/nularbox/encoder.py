import datasets.nularbox.resnet_encoder as resnet

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
    
    ## Only support ResNet for now 
    if "ResNet" not in args.enc_arch:
        raise ValueError(f"Unknown encoder architecture: {self.enc_arch}")
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
