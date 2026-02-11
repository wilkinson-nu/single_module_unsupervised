import torch
import argparse

## Temporarily override for testing
# from core.models.encoder import get_encoder
from datasets.nularbox.encoder import get_encoder
from core.models.projection_head import get_projhead
from core.models.clustering_head import get_clusthead

def load_checkpoint(state_file_name):
    checkpoint = torch.load(state_file_name, map_location='cpu')
    
    # Reconstruct args Namespace
    args = argparse.Namespace(**checkpoint['args'])
    return checkpoint, args

def get_models_from_checkpoint(state_file_name):

    checkpoint, args = load_checkpoint(state_file_name)

    ## Get the models
    encoder = get_encoder(args)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    ## Dictionary of heads and load saved model parameters
    heads = {}
    
    heads["proj"] = get_projhead(encoder.get_nchan_instance(), args)
    heads["proj"] .load_state_dict(checkpoint['proj_head_state_dict'])

    ## Optionally load the clustering head
    if args.clust_arch != "none":
        heads["clust"] = get_clusthead(encoder.get_nchan_cluster(), args)
        heads["clust"] .load_state_dict(checkpoint['clust_head_state_dict']) 

    return encoder, heads, args

