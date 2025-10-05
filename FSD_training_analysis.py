## This file trains a simple encoder with a contrastive loss. It should work with any number of GPUs distributed across nodes
import numpy as np
import argparse
from torch import optim
import sys

## Make matplotlib do things in batch mode
import matplotlib
matplotlib.use("Agg")

import torchvision.transforms.v2 as transforms
import MinkowskiEngine as ME
import torch
import time
import math

## Use a GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## The parallelisation libraries
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import ConcatDataset
from torch import nn

## Includes from my libraries for this project
from ME_NN_libs import CCEncoderFSD12x4Opt, ProjectionHead, ClusteringHeadTwoLayer, ClusteringHeadOneLayer, ProjectionHeadLogits

## Import analysis functions
from ME_analysis_libs import plot_metric_data_vs_sim, plot_metric_by_cluster, plot_cluster_bigblock, run_tsne_cuml, run_umap_cuml

## Seeding
SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

## Import transformations
from ME_dataset_libs import DoNothing, get_transform, FirstRegionCrop

## Import dataset
from ME_dataset_libs import SingleModuleImage2D_solo_ME, solo_ME_collate_fn

def load_checkpoint(state_file_name):
    checkpoint = torch.load(state_file_name, map_location='cpu')
    
    # Reconstruct args Namespace
    args = argparse.Namespace(**checkpoint['args'])
    return checkpoint, args


def get_act_from_string(act_name):
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


def get_models_from_checkpoint(state_file_name):

    checkpoint, args = load_checkpoint(state_file_name)

    ## Get the models
    encoder = get_encoder(args)
    proj_head = get_projhead(encoder.get_nchan_instance(), args)
    clust_head = get_clusthead(encoder.get_nchan_cluster(), args)

    ## Load saved model parameters
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    proj_head.load_state_dict(checkpoint['proj_head_state_dict'])
    clust_head.load_state_dict(checkpoint['clust_head_state_dict']) 

    return encoder, proj_head, clust_head, args


## Function to deal with all of the dataset handling
def get_dataset(input_dir, nevents):

    print("Loading", nevents," events from", input_dir)
    nom_transform = transforms.Compose([
            FirstRegionCrop((800, 256), (768, 256)),
            ])
    
    dataset = SingleModuleImage2D_solo_ME(input_dir, \
                                          transform=nom_transform, \
                                          max_events=nevents)

    loader = torch.utils.data.DataLoader(dataset,
                                         collate_fn=solo_ME_collate_fn,
                                         batch_size=1024,
                                         shuffle=False,
                                         num_workers=4)
    return dataset, loader

def get_encoder(args):
    
    ## Only one architecture for now
    #if args.enc_arch == "12x4":
    enc = CCEncoderFSD12x4Opt
        
    enc_act_fn=get_act_from_string(args.enc_act)
    encoder = enc(nchan=args.nchan, \
                  act_fn=enc_act_fn, \
                  first_kernel=args.enc_arch_first_kernel, \
                  flatten=bool(args.enc_arch_flatten), \
                  pool=args.enc_arch_pool, \
                  slow_growth=bool(args.enc_arch_slow_growth),
                  sep_heads=bool(args.enc_arch_sep_heads))
    return encoder

def get_projhead(nchan, args):
    hidden_act_fn = nn.SiLU
    latent_act_fn = nn.Tanh
    if args.proj_arch == "logits":
        proj_head = ProjectionHeadLogits(nchan, args.latent, hidden_act_fn)
    else:
        proj_head = ProjectionHead(nchan, args.latent, hidden_act_fn, latent_act_fn)
    return proj_head

def get_clusthead(nchan, args):

    if args.clust_arch == "one":
        clust = ClusteringHeadOneLayer
    else:
        clust = ClusteringHeadTwoLayer
    
    clust_head = clust(nchan, args.nclusters, args.softmax_temp)
    return clust_head


def image_loop(encoder, proj_head, clust_head, loader):

    latent = []
    cluster = []
    nhits = []
    maxQ = []
    sumQ = []
    labels = []
    
    encoder.eval()
    proj_head.eval()
    clust_head.eval()
    
    ## Loop over the images
    for batch_coords, batch_feats, batch_labels in loader:
        
        batch_size = len(batch_labels)
        batch_coords = batch_coords.to(device)
        batch_feats = batch_feats.to(device)
        orig_batch = ME.SparseTensor(batch_feats, batch_coords, device=device)            
        
        ## Now do the forward passes            
        with torch.no_grad(): 
            encoded_instance_batch, encoded_cluster_batch = encoder(orig_batch, batch_size)
            clust_batch = clust_head(encoded_cluster_batch)
            proj_batch = proj_head(encoded_instance_batch)
            
        nhits += [i.shape[0] for i in orig_batch.decomposed_features]
        sumQ += [i.sum().item() for i in orig_batch.decomposed_features]
        maxQ += [i.max().item() for i in orig_batch.decomposed_features]
        cluster += [x[np.newaxis, :] for x in clust_batch.detach().cpu().numpy()]
        latent += [x[np.newaxis, :] for x in proj_batch.detach().cpu().numpy()]
        labels += [i for i in batch_labels]

    ## Derive some other useful quantities
    np_clust = np.vstack(cluster)
    sorted_idx  = np.argsort(np_clust, axis=1)[:, ::-1]
    top3_idx = sorted_idx[:, :3]
    clust_top3 = np.take_along_axis(np_clust, top3_idx, axis=1)

    ## Return a dictionary to make my life easier
    return {
        "nhits": np.array(nhits),
        "sumQ": np.array(sumQ),
        "maxQ": np.array(maxQ),
        "labels": np.array(labels),
        "latent": np.vstack(latent),
        "clust": np_clust,
        "clust_index": np.argmax(np_clust, axis=1),
        "clust_top3": clust_top3,
        "clust_max": np.max(np_clust, axis=1),
    }


## Function to reorder the order of clusters in the processed data
def reorder_clusters(data_processed, sim_processed):

    ## How frequently is each cluster selected in data
    unique, counts = np.unique(data_processed['clust_index'], return_counts=True)

    ## Order from most common to least common
    order = np.argsort(-counts)

    ## map for reordering cluster indices
    relabel_map = {old: new for new, old in enumerate(unique[order])}

    data_processed['clust_index'] = np.array([relabel_map[c] for c in data_processed['clust_index']])
    sim_processed['clust_index'] = np.array([relabel_map[c] for c in sim_processed['clust_index']])

    data_processed['clust'] = data_processed['clust'][:,order]
    sim_processed['clust'] = sim_processed['clust'][:,order]

    
def run_analysis(args):

    ## Setup the encoder
    encoder, proj_head, clust_head, training_args = get_models_from_checkpoint(args.input_file)

    encoder.to(device)
    proj_head.to(device)
    clust_head.to(device)
    
    ## Set up the datasets and loaders
    data_dataset, data_loader = get_dataset(training_args.data_dir, args.ndata)
    sim_dataset, sim_loader = get_dataset(training_args.sim_dir, args.nsim)

    ## Get the processed vectors of interest from the datasets
    data_processed = image_loop(encoder, proj_head, clust_head, data_loader)
    sim_processed = image_loop(encoder, proj_head, clust_head, sim_loader)

    ## Do some magic to re-order the clusters for presentation purposes
    reorder_clusters(data_processed, sim_processed)
    
    ## Make histograms
    plot_metric_data_vs_sim(data_processed['clust_index'], \
                            sim_processed['clust_index'], \
                            sim_processed['labels'],\
                            xtitle="Max. cluster index", \
                            save_name=args.out_name_root+"_clust_index.png")

    
    plot_metric_data_vs_sim(data_processed['clust_max'], \
                            sim_processed['clust_max'], \
                            sim_processed['labels'],\
                            xtitle="Max. cluster value", \
                            save_name=args.out_name_root+"_clust_max.png")
    plot_metric_by_cluster(data_processed['clust_max'], \
                           data_processed['clust_index'],\
                           xtitle="Max. cluster value", \
                           save_name=args.out_name_root+"_clust_max_dataclust.png")
    plot_metric_by_cluster(sim_processed['clust_max'], \
                           sim_processed['clust_index'],\
                           xtitle="Max. cluster value", \
                           save_name=args.out_name_root+"_clust_max_simclust.png")
    
    
    plot_metric_data_vs_sim(data_processed['nhits'],\
                            sim_processed['nhits'], \
                            sim_processed['labels'],\
                            nbinsx=70, x_max=1400,\
                            xtitle="N. hits", \
                            save_name=args.out_name_root+"_nhits.png")
    plot_metric_by_cluster(data_processed['nhits'],\
                           data_processed['clust_index'],\
                           nbinsx=70, x_max=1400,\
                           xtitle="N. hits", \
                           save_name=args.out_name_root+"_nhits_dataclust.png")
    plot_metric_by_cluster(sim_processed['nhits'],\
                           sim_processed['clust_index'],\
                           nbinsx=70, x_max=1400,\
                           xtitle="N. hits", \
                           save_name=args.out_name_root+"_nhits_simclust.png")

    
    plot_metric_data_vs_sim(data_processed['sumQ'], \
                            sim_processed['sumQ'], \
                            sim_processed['labels'],\
                            nbinsx=70, x_max=1400,\
                            xtitle="Sum Q", \
                            save_name=args.out_name_root+"_sumQ.png")
    plot_metric_by_cluster(data_processed['sumQ'], \
                           data_processed['clust_index'],\
                           nbinsx=70, x_max=1400,\
                           xtitle="Sum Q", \
                           save_name=args.out_name_root+"_sumQ_dataclust.png")
    plot_metric_by_cluster(sim_processed['sumQ'], \
                           sim_processed['clust_index'],\
                           nbinsx=70, x_max=1400,\
                           xtitle="Sum Q", \
                           save_name=args.out_name_root+"_sumQ_simclust.png")

    
    plot_metric_data_vs_sim(data_processed['maxQ'], \
                            sim_processed['maxQ'], \
                            sim_processed['labels'],\
                            nbinsx=100, x_min=1.5, x_max=2.5,\
                            xtitle="Max. Q", \
                            save_name=args.out_name_root+"_maxQ.png")
    plot_metric_by_cluster(data_processed['maxQ'], \
                           data_processed['clust_index'],\
                           nbinsx=100, x_min=1.5, x_max=2.5,\
                           xtitle="Max. Q", \
                           save_name=args.out_name_root+"_maxQ_dataclust.png")
    plot_metric_by_cluster(sim_processed['maxQ'], \
                           sim_processed['clust_index'],\
                           nbinsx=100, x_min=1.5, x_max=2.5,\
                           xtitle="Max. Q", \
                           save_name=args.out_name_root+"_maxQ_simclust.png")

    ## t-SNE examples
    # _ = run_tsne_cuml(data_processed['clust'], data_processed['clust_index'], perp=100, exag=50, lr=5000, alpha_vect=data_processed['clust_max'], \
    #save_name=args.out_name_root+"_tSNE_data.png")
    ## UMAP
    run_umap_cuml(data_processed['clust'], data_processed['clust_index'], alpha_vect=data_processed['clust_max'], \
                  n_neighbors=100, min_distance=0.05, n_epochs=500,\
                  save_name=args.out_name_root+"_UMAP_data.png")
    
    ## Plot some examples for each cluster:
    if args.example_cluster_images:
        for n in range(training_args.nclusters):
            plot_cluster_bigblock(data_dataset, data_processed['clust_index'], n, 4, 10, \
                                  cluster_probs=data_processed['clust_max'], \
                                  save_name=args.out_name_root+"_data_example"+str(n)+"_top.png")
            plot_cluster_bigblock(data_dataset, data_processed['clust_index'], n, 4, 10, \
                                  save_name=args.out_name_root+"_data_example"+str(n)+"_all.png")
            plot_cluster_bigblock(sim_dataset, sim_processed['clust_index'], n, 4, 10, \
                                  cluster_probs=sim_processed['clust_max'], \
                                  save_name=args.out_name_root+"_sim_example"+str(n)+"_top.png")
            plot_cluster_bigblock(sim_dataset, sim_processed['clust_index'], n, 4, 10, \
                                  save_name=args.out_name_root+"_sim_example"+str(n)+"_all.png")

        
## Do the business
if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("Model analysis")

    # Require an input file name and location to dump plots
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--out_name_root', type=str)

    ## Give a sensible default for the number of events to process
    parser.add_argument('--ndata', type=int, default=100000, nargs='?')
    parser.add_argument('--nsim', type=int, default=100000, nargs='?')

    ## Options for controlling the plots to make
    parser.add_argument('--example_cluster_images', type=int, choices=[0,1], default=0, nargs='?')
    
    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))
    
    run_analysis(args)
