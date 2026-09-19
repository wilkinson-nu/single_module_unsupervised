from torch import nn
import MinkowskiEngine as ME

def get_act_from_string_ME(act_name):
    if act_name == "relu":
        return ME.MinkowskiReLU
    if act_name == "leakyrelu":
        return ME.MinkowskiLeakyReLU
    if act_name == "gelu":
        return ME.MinkowskiGELU
    if act_name in ["silu", "swish"]:
        return ME.MinkowskiSiLU
    if act_name == "selu":
        return ME.MinkowskiSELU
    if act_name == "tanh":
        return ME.MinkowskiTanh
    if act_name == "softsign":
        return ME.MinkowskiSoftsign
    return None

def get_act_from_string(act_name):
    if act_name == "relu":
        return nn.ReLU
    if act_name == "leakyrelu":
        return nn.LeakyReLU
    if act_name == "gelu":
        return nn.GELU
    if act_name in ["silu", "swish"]:
        return nn.SiLU
    if act_name == "selu":
        return nn.SELU
    if act_name == "tanh":
        return nn.Tanh
    if act_name == "softsign":
        return nn.Softsign
    return None

