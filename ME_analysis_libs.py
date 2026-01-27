import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse
from sklearn.manifold import TSNE
import MinkowskiEngine as ME
import torchvision.transforms.v2 as transforms
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from ME_dataset_libs import make_dense, make_dense_from_tensor, Label
from cuml.manifold import TSNE as cuML_TSNE
import cupy as cp
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from cuml.preprocessing import StandardScaler as cuMLScaler
from cuml.manifold import UMAP as cuML_UMAP
from matplotlib.ticker import MaxNLocator
from ME_dataset_libs import SingleModuleImage2D_solo_ME, solo_ME_collate_fn, solo_ME_collate_fn_with_meta
from ME_dataset_libs import DoNothing, get_transform, FirstRegionCrop
from ME_NN_libs import get_encoder, get_projhead, get_clusthead
import faiss

## For clustering studies
#import spherecluster
#from spherecluster import VonMisesFisherMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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


## Function to deal with all of the dataset handling
## Should probably move to ME_dataset_libs...
def get_dataset(input_dir, nevents, return_metadata=False):

    print("Loading", nevents," events from", input_dir)
    nom_transform = transforms.Compose([
            FirstRegionCrop((800, 256), (768, 256)),
            ])
    
    dataset = SingleModuleImage2D_solo_ME(input_dir, \
                                          transform=nom_transform, \
                                          max_events=nevents,\
                                          return_metadata=return_metadata)
    this_collate = solo_ME_collate_fn
    if return_metadata: this_collate = solo_ME_collate_fn_with_meta
    
    loader = torch.utils.data.DataLoader(dataset,
                                         collate_fn=this_collate,
                                         batch_size=2048,
                                         shuffle=False,
                                         num_workers=8)
    return dataset, loader


def image_loop(encoder, heads, loader, detailed_info=True):

    latent = []    ## This is the instance clustering space
    enc_latent = []    ## This is after the encoder (as passed to the clustering head) 
    cluster = []
    nhits = []
    maxQ = []
    sumQ = []
    labels = []
    y_range = []
    x_range = []
    filenames = []
    event_ids = []
    
    encoder.eval()
    for h in heads.values(): h.eval()
    
    ## Loop over the images (discard any extra info returned by loader)
    for batch in loader:

        ## Start with the items that are always there
        batch_coords, batch_feats, batch_labels = batch[:3]
        ## If the loader returns the file names and event ids, collect those too
        batch_filenames = batch[3] if len(batch) > 3 else None
        batch_eventids = batch[4] if len(batch) > 4 else None
        
        batch_size = len(batch_labels)
        batch_coords = batch_coords.to(device)
        batch_feats = batch_feats.to(device)
        orig_batch = ME.SparseTensor(batch_feats, batch_coords, device=device)            

        dec_coords = orig_batch.decomposed_coordinates
        dec_feats  = orig_batch.decomposed_features
        
        ## Now do the forward passes            
        with torch.no_grad(): 
            encoded_instance_batch, encoded_cluster_batch = encoder(orig_batch, batch_size)
            if "clust" in heads: clust_batch = heads["clust"](encoded_cluster_batch)
            proj_batch = heads["proj"](encoded_instance_batch)

        ## Move to the CPU
        if "clust" in heads: cluster.append(clust_batch.detach().cpu())
        latent.append(proj_batch.detach().cpu())
        enc_latent.append(encoded_cluster_batch.detach().cpu())
        labels.extend(batch_labels)
        
        ## If desired, add a load more info, but this slows things down a lot...
        if detailed_info is True:
            nhits_batch = torch.tensor([f.shape[0] for f in dec_feats], device=device)
            sumQ_batch  = torch.stack([f.sum() for f in dec_feats])
            maxQ_batch  = torch.stack([f.max() for f in dec_feats])
        
            y_max_batch = torch.stack([c[:,0].max() for c in dec_coords])
            y_min_batch = torch.stack([c[:,0].min() for c in dec_coords])
            x_max_batch = torch.stack([c[:,1].max() for c in dec_coords])
            x_min_batch = torch.stack([c[:,1].min() for c in dec_coords])

            y_range_batch = y_max_batch - y_min_batch
            x_range_batch = x_max_batch - x_min_batch

            # Move everything to the CPU
            nhits.append(nhits_batch.cpu())
            sumQ.append(sumQ_batch.cpu())
            maxQ.append(maxQ_batch.cpu())
            y_range.append(y_range_batch.cpu())
            x_range.append(x_range_batch.cpu())

        if batch_filenames is not None:
            filenames.extend(batch_filenames)
        if batch_eventids is not None:
            event_ids.extend(batch_eventids)

    ## Turn into numpy arrays 
    latent = torch.cat(latent).numpy()
    enc_latent = torch.cat(enc_latent).numpy()

    ## Return a dictionary to make my life easier
    out = {
        "labels": np.array(labels),
        "latent": latent,
        "enc_latent": enc_latent,
    }
    
    if "clust" in heads:
        cluster = torch.cat(cluster).numpy()

        ## Derive some other useful quantities
        sorted_idx  = np.argsort(cluster, axis=1)[:, ::-1]
        top3_idx = sorted_idx[:, :3]
        clust_top3 = np.take_along_axis(cluster, top3_idx, axis=1)

        out["clust"] = cluster
        out["clust_index"] = np.argmax(cluster, axis=1)
        out["clust_top3"] = clust_top3
        out["clust_max"] = np.max(cluster, axis=1)
    
    if detailed_info is True:
        out["nhits"] = torch.cat(nhits).numpy()
        out["sumQ"] = torch.cat(sumQ).numpy()
        out["maxQ"] = torch.cat(maxQ).numpy()
        out["yrange"] = torch.cat(y_range).numpy()
        out["xrange"] = torch.cat(x_range).numpy()
    
    ## Add the filename and event id if applicable
    if filenames: out["filename"] = filenames
    if event_ids: out["event_id"] = event_ids
    return out

## Function to reorder the order of clusters in the processed data
def reorder_clusters(data_processed, sim_processed):

    ## Rage quit if someone tries to call this without cluster labels
    if "clust" not in data_processed: return
        
    ## How frequently is each cluster selected in data
    unique, counts = np.unique(data_processed['clust_index'], return_counts=True)

    ## Order from most common to least common
    order = np.argsort(-counts)

    mapping_arr = np.zeros(unique.max() + 1, dtype=np.int64)
    mapping_arr[unique[order]] = np.arange(len(unique))

    data_processed['clust_index'] = mapping_arr[data_processed['clust_index']]
    sim_processed['clust_index']  = mapping_arr[sim_processed['clust_index']]
    
    data_processed['clust'] = data_processed['clust'][:,order]
    sim_processed['clust'] = sim_processed['clust'][:,order]


@torch.no_grad()
def argmax_consistency(c_cat, device=None):
    batch_size = c_cat.shape[0] // 2
    c_i, c_j = c_cat[:batch_size], c_cat[batch_size:]
    
    argmax_i = torch.argmax(c_i, dim=1)
    argmax_j = torch.argmax(c_j, dim=1)
    
    same = (argmax_i == argmax_j).float()
    mean_same = same.mean()
    if device is not None: mean_same = mean_same.to(device)
    return mean_same

@torch.no_grad()
def topk_consistency(c_cat, k=2):
    batch_size = c_cat.shape[0] // 2
    c_i, c_j = c_cat[:batch_size], c_cat[batch_size:]
    
    # Top-k indices for each view
    topk_i = torch.topk(c_i, k, dim=1).indices
    topk_j = torch.topk(c_j, k, dim=1).indices
    
    # For each sample, check if there's an overlap in the sets
    overlap = (topk_i.unsqueeze(2) == topk_j.unsqueeze(1))
    same = overlap.any(dim=(1,2)).float()
    
    return same.mean().item()


def compute_cluster_overlap(c_probs, topk=2):

    N, K = c_probs.shape
    topk_indices = np.argpartition(-c_probs, kth=topk-1, axis=1)[:, :topk]  # (N, topk)

    overlap_matrix = np.zeros((K, K), dtype=float)

    for i in range(K):
        idx_i = np.any(topk_indices == i, axis=1)  # samples where cluster i is in top-k
        for j in range(i, K):
            idx_j = np.any(topk_indices == j, axis=1)
            denom = np.logical_or(idx_i, idx_j).sum()
            if denom > 0:
                overlap = np.logical_and(idx_i, idx_j).sum() / denom
            else:
                overlap = 0.0
            overlap_matrix[i, j] = overlap
            overlap_matrix[j, i] = overlap  # symmetric

    return overlap_matrix

    
def plot_overlap_matrix(overlap_matrix, merged_labels=None, min_val=0, max_val=1.0):
    plt.figure(figsize=(8,6))
    im = plt.imshow(overlap_matrix, cmap="viridis", vmin=min_val, vmax=max_val) #, norm='log')
    plt.colorbar(im, label="Top-k overlap")

    plt.title("Cluster Overlap Matrix")
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    
    # Optionally mark merged clusters
    if merged_labels is not None:
        # Sort by merged cluster ID for block structure
        order = np.argsort(merged_labels)
        plt.xticks(range(len(order)), order, rotation=90)
        plt.yticks(range(len(order)), order)
    else:
        plt.xticks(range(overlap_matrix.shape[0]))
        plt.yticks(range(overlap_matrix.shape[0]))

    plt.show()

def parse_binning(x, nbins=None, x_min=None, x_max=None):

    if x_min is None: x_min = x.min()
    if x_max is None: x_max = x.max()

    if np.issubdtype(x.dtype, np.integer):
        if nbins is None: return np.arange(x_min, x_max+2) - 0.5, True
        else: return np.linspace(x_min, x_max+1, nbins+1), True
    else:
        if nbins is None: return 50, False
        else: return np.linspace(x_min, x_max, nbins+1), False

# Make a histogram broken down into simulation and data, for arbitrary x variables
def plot_metric_pass_fail(xvar, mask, nbinsx=None, x_min=None, x_max=None, xtitle="xvar", ytitle="N. images", save_name=None):
   
    ## Deal with binning myself for some reason...
    bins, is_int = parse_binning(xvar, nbinsx, x_min, x_max)
        
    xvar_pass = xvar[mask]
    xvar_fail = xvar[~mask]

    plt.hist([xvar_pass, xvar_fail], bins=bins,
             stacked=True,
             histtype='stepfilled',
             align='mid',
             label=['Pass', 'Fail'],
             color=['lightcoral', 'mediumseagreen'])

    ## More fun with integers
    if is_int:
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        #ax.grid(False)

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(
        ncol=2,
        fontsize="medium",
        loc="lower right",
        bbox_to_anchor=(0.5, 1.),
        frameon=False
    )
    plt.tight_layout()
    # plt.grid(True)
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_metric_by_confidence(xvar, confidence, nbinsx=None, x_min=None, x_max=None, xtitle="xvar", ytitle="N. images", normalize=False, save_name=None):

    bins, is_int = parse_binning(xvar, nbinsx, x_min, x_max)

    nsteps = 6
    labels = ["<0.5", "0.5-0.8", "0.8-0.9", "0.9-0.95", "0.95-0.99", ">0.99"]
    n_steps = len(labels)
    
    ## colormap
    colors = plt.cm.tab20.colors
    cmap = mcolors.ListedColormap(colors[:n_steps])
    colors = [cmap(i) for i in range(n_steps)]
    
    # Collect metric values by label
    data_by_confidence = []
    masks = []
    
    masks.append((confidence < 0.5))
    masks.append((confidence > 0.5)&(confidence < 0.8))
    masks.append((confidence > 0.8)&(confidence < 0.9))
    masks.append((confidence > 0.9)&(confidence < 0.95))
    masks.append((confidence > 0.95)&(confidence < 0.99))
    masks.append((confidence > 0.99))

    for mask in masks:
        if np.any(mask):
            data_by_confidence.append(xvar[mask])
        else:
            # Empty array so it contributes nothing to the histogram
            data_by_confidence.append(np.array([]))

    plt.figure(figsize=(8, 6))
    counts, bin_edges, patches = plt.hist(
        data_by_confidence,
        bins=bins,
        histtype='stepfilled',
        align='mid',
        stacked=True,
        label=labels,
        density=normalize,
        color=colors
    )

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles[::-1],
        labels[::-1],
        ncol=3,
        fontsize="small",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.),
        frameon=False
    )
    plt.tight_layout()  # prevents clipping
    # plt.grid(True)
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


    
# Make a histogram broken down into all possible labels, for arbitrary x variables
def plot_metric_by_label(xvar, labels, nbinsx=None, x_min=None, x_max=None, xtitle="xvar", ytitle="N. images", normalize=False):

    ## Deal with binning myself for some reason...
    bins, is_int = parse_binning(xvar, nbinsx, x_min, x_max)
    
    label_values = [m.value for m in Label]
    label_names  = [m.name for m in Label]

    ## If there are more than 20 labels, this will obviously go a bit funky
    all_colors = (
        plt.cm.tab20.colors +
        plt.cm.tab20b.colors +
        plt.cm.tab20c.colors +
        plt.cm.tab10.colors
    )

    cmap = mcolors.ListedColormap(all_colors)
    colors = [cmap(i) for i in range(len(label_values))]
    
    # Collect metric values by label
    data_by_label = []
    for value in label_values:
        mask = (labels == value)
        if np.any(mask):
            data_by_label.append(xvar[mask])
        else:
            # Empty array so it contributes nothing to the histogram
            data_by_label.append(np.array([]))

    plt.figure(figsize=(8, 6))
    counts, bin_edges, patches = plt.hist(
        data_by_label,
        bins=bins,
        histtype='stepfilled',
        align='mid',
        stacked=True,
        label=label_names,
        density=normalize,
        color=colors
    )

    if is_int:
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(
        ncol=3,
        fontsize="x-small",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.),
        frameon=False
    )
    plt.tight_layout()  # prevents clipping
    # plt.grid(True)
    plt.show()
    plt.close()

def plot_metric_data_vs_alt(data_xvar, alt_xvar, sim_labels, nbinsx=None, x_min=None, x_max=None, xtitle="xvar", ytitle="N. images", normalize=False, save_name=None):

    ## Deal with binning myself for some reason...
    bins, is_int = parse_binning(data_xvar, nbinsx, x_min, x_max)

    ## If there are more than 20 labels, this will obviously go a bit funky
    all_colors = (
        plt.cm.tab20.colors +
        plt.cm.tab20b.colors +
        plt.cm.tab20c.colors +
        plt.cm.tab10.colors
    )

    cmap = mcolors.ListedColormap(all_colors[1:])

    plt.figure(figsize=(8, 6))

    ## Add alternative data
    plt.hist(
        alt_xvar,
        bins=bins,
        histtype="stepfilled",
        stacked=True,
        label="Alt data",
        density=normalize,
        color=all_colors[1],
        alpha=0.7
    )

    # Add data
    plt.hist(
        data_xvar,
        bins=bins,
        histtype="step",
        density=normalize,
        color=all_colors[0],
        linewidth=1.5,
        label="Data"
    )

    if is_int:
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles[::-1],
        labels[::-1],
        ncol=2,
        fontsize="small",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.),
        frameon=False
    )
    plt.tight_layout()  # prevents clipping
    # plt.grid(True)
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    
def plot_metric_data_vs_sim(data_xvar, sim_xvar, sim_labels, nbinsx=None, x_min=None, x_max=None, xtitle="xvar", ytitle="N. images", normalize=False, save_name=None):

    ## Deal with binning myself for some reason...
    bins, is_int = parse_binning(data_xvar, nbinsx, x_min, x_max)
    
    label_values = [m.value for m in Label]
    label_names  = [m.name for m in Label]

    ## Skip the data label because it's being plotted separately here
    label_values = label_values[1:]
    label_names  = label_names[1:] 
    
    ## If there are more than 20 labels, this will obviously go a bit funky
    all_colors = (
        plt.cm.tab20.colors +
        plt.cm.tab20b.colors +
        plt.cm.tab20c.colors +
        plt.cm.tab10.colors
    )

    cmap = mcolors.ListedColormap(all_colors[1:])
    colors = [cmap(i) for i in range(len(label_values))]
    
    # Collect metric values by label
    sim_by_label = []
    for value in label_values:
        mask = (sim_labels == value)
        if np.any(mask):
            sim_by_label.append(sim_xvar[mask])
        else:
            # Empty array so it contributes nothing to the histogram
            sim_by_label.append(np.array([]))

    plt.figure(figsize=(8, 6))

    ## Add MC
    plt.hist(
        sim_by_label,
        bins=bins,
        histtype="stepfilled",
        stacked=True,
        label=label_names,
        density=normalize,
        color=colors,
        alpha=0.7
    )

    # Add data
    plt.hist(
        data_xvar,
        bins=bins,
        histtype="step",
        density=normalize,
        color=all_colors[0],
        linewidth=1.5,
        label="Data"
    )

    if is_int:
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles[::-1],
        labels[::-1],
        ncol=3,
        fontsize="small",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.),
        frameon=False
    )
    plt.tight_layout()  # prevents clipping
    # plt.grid(True)
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_metric_by_cluster(xvar, cluster_vect, nbinsx=None, x_min=None, x_max=None, xtitle="xvar", ytitle="N. images", normalize=False, save_name=None):

    ## Deal with binning myself for some reason...
    bins, is_int = parse_binning(xvar, nbinsx, x_min, x_max)

    unique_values = np.unique(cluster_vect)
    n_clusters = len(unique_values)

    cluster_names = [str(i) for i in unique_values]

    ## If there are more than 20 labels, this will obviously go a bit funky
    all_colors = (
        plt.cm.tab20.colors +
        plt.cm.tab20b.colors +
        plt.cm.tab20c.colors +
        plt.cm.tab10.colors
    )

    cmap = mcolors.ListedColormap(all_colors[:n_clusters])
    colors = [cmap(i) for i in range(len(unique_values))]
    
    # Collect metric values by label
    data_by_cluster = []
    for value in unique_values:
        mask = (cluster_vect == value)
        if np.any(mask):
            data_by_cluster.append(xvar[mask])
        else:
            # Empty array so it contributes nothing to the histogram
            data_by_cluster.append(np.array([]))

    plt.figure(figsize=(8, 6))
    counts, bin_edges, patches = plt.hist(
        data_by_cluster,
        bins=bins,
        histtype='stepfilled',
        align='mid',
        stacked=True,
        label=cluster_names,
        density=normalize,
        color=colors
    )

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles[::-1],
        labels[::-1],
        ncol=10,
        fontsize="small",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.),
        frameon=False
    )
    plt.tight_layout()  # prevents clipping
    # plt.grid(True)
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_cluster_examples(dataset, cluster_ids, index, max_images=8, cluster_probs=None, save_name=None): 

    ## Sort colours
    cmap = cm.turbo.copy()
    cmap.set_under("#F0F0F0")
    
    plt.figure(figsize=(max_images*2,6))

    ## Get a mask of cluster_ids
    indices = np.where(np.array(cluster_ids) == index)[0]

    ## If the probabilities are given, show the top N probabilities
    if cluster_probs is not None:
        indices = indices[np.argsort(np.array(cluster_probs)[indices])][::-1]
    
    ## Grab the first N images (if there are N)
    if len(indices) < max_images: max_images = len(indices)
        
    ## Plot
    for i in range(max_images):
        ax = plt.subplot(1,max_images,i+1)
        
        numpy_coords, numpy_feats, *_ = dataset[indices[i]]

        # Create batched coordinates for the SparseTensor input
        orig_bcoords  = ME.utils.batched_coordinates([numpy_coords])
        orig_bfeats  = torch.from_numpy(np.concatenate([numpy_feats], 0)).float()
        orig = ME.SparseTensor(orig_bfeats, orig_bcoords)
            
        inputs  = make_dense_from_tensor(orig, 0, 768, 256)
        inputs  = inputs .cpu().squeeze().numpy()
        
        plt.imshow(inputs, origin='lower', cmap=cmap, vmin=1e-6)
        ax.axis('off')
    plt.tight_layout()
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_cluster_bigblock(dataset, cluster_ids, index, max_x=10, max_y=10, cluster_probs=None, save_name=None): 

    ## Sort colours
    cmap = cm.turbo.copy()
    cmap.set_under("#F0F0F0")
    
    plt.figure(figsize=(max_y*2.1, max_x*6))
    ## Get a mask of cluster_ids
    indices = np.arange(max_x*max_y) 
    if index != None: 
        indices = np.where(np.array(cluster_ids) == index)[0]
        ## If the probabilities are given, show the top N probabilities
        if cluster_probs is not None:
            indices = indices[np.argsort(np.array(cluster_probs)[indices])][::-1]
    max_images = min(len(indices), max_x*max_y)
    
    ## Plot
    for i in range(max_images):
        ax = plt.subplot(max_x,max_y,i+1)
        
        numpy_coords, numpy_feats, *_ = dataset[indices[i]]
    
        # Create batched coordinates for the SparseTensor input
        orig_bcoords  = ME.utils.batched_coordinates([numpy_coords])
        orig_bfeats  = torch.from_numpy(np.concatenate([numpy_feats], 0)).float()
        orig = ME.SparseTensor(orig_bfeats, orig_bcoords)
            
        inputs  = make_dense_from_tensor(orig, 0, 768, 256)
        inputs  = inputs .cpu().squeeze().numpy()

        nonzero_vals = inputs[inputs > 0]
        vmax = np.percentile(nonzero_vals, 80)
        
        plt.imshow(inputs, origin='lower', cmap=cmap, vmin=1e-6, vmax=vmax)
        ax.axis('off')
        plt.tight_layout()

    plt.tight_layout()
    if save_name: plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()  
    plt.close()


def run_vMF(dataset, n_clusters, init="random-class", n_copies=10, verbose=True):

    X_norm = dataset / np.linalg.norm(dataset, axis=1, keepdims=True)

    ## init: k-means++, spherical-k-means, random, random-class (default), random-orthonormal
    ## max_iter: 300
    ## n_init: 10
    ## n_jobs: 1 (number of CPUs to use)
    
    ## vMF = VonMisesFisherMixture(n_clusters=n_clusters, posterior_type='soft', n_init=n_copies, n_jobs=n_copies, verbose=verbose, max_iter=500)
    ## vMF.fit(X_norm)
    ## 
    ## ## For some reasons labels are floats
    ## labels = vMF.predict(X_norm).astype(int)
    ## weights = vMF.weights_
    ## 
    ## labs = np.unique(labels)
    ## 
    ## metrics = {}
    ## 
    ## if labs.size < 2 or labs.size >= len(labels):
    ##     metrics["silhouette"] = None
    ##     metrics["calinski_harabasz"] = None
    ##     metrics["davies_bouldin"] = None
    ## else:
    ##     metrics["silhouette"] = silhouette_score(X_norm, labels, metric="cosine")
    ##     metrics["calinski_harabasz"] = calinski_harabasz_score(X_norm, labels)
    ##     metrics["davies_bouldin"] = davies_bouldin_score(X_norm, labels)
    ## 
    ## if verbose:
    ##     print("Cluster weights:", weights)
    ##     print("Silhouette score:", metrics["silhouette"])
    ##     print("Calinski-Harabasz =", metrics["calinski_harabasz"])
    ##     print("Davies-Bouldin =", metrics["davies_bouldin"])
    ## 
    ## return labels, metrics
    return 

def run_faiss_spherical_kmeans(dataset, n_clusters, n_iter=20, verbose=True, seed=123):
    # Normalize embeddings (critical for cosine clustering)
    X = dataset.astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    N, d = X.shape

    # FAISS k-means (spherical via normalization)
    kmeans = faiss.Kmeans(
        d=d,
        k=n_clusters,
        niter=n_iter,
        verbose=verbose,
        seed=seed,
        spherical=True  # ensures centroid normalization
    )
    kmeans.train(X)

    # Assign clusters
    _, labels = kmeans.index.search(X, 1)
    labels = labels.flatten()

    # Cluster weights
    counts = np.bincount(labels, minlength=n_clusters)
    weights = counts / N

    # Metrics
    labs = np.unique(labels)
    metrics = {}

    if labs.size < 2 or labs.size >= len(labels):
        metrics["silhouette"] = None
        metrics["calinski_harabasz"] = None
        metrics["davies_bouldin"] = None
    else:
        metrics["silhouette"] = silhouette_score(X, labels, metric="cosine")
        metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
        metrics["davies_bouldin"] = davies_bouldin_score(X, labels)

    if verbose:
        print("Cluster weights:", weights)
        print("Silhouette score:", metrics["silhouette"])
        print("Calinski-Harabasz =", metrics["calinski_harabasz"])
        print("Davies-Bouldin =", metrics["davies_bouldin"])

    return labels, metrics, kmeans.centroids
    
def run_tsne_skl(input_vect=None, zvect=None, alpha_vect=None, perp=30, exag=6,
                 lr=2000.0, n_iter=2000, ztitle="Cluster ID", save_name=None, norm=True, n_samples=None, tsne_results=None):
    
    print("Running scikit-learn t-SNE with: perplexity =", perp, "early exaggeration =", exag)

    # L2 normalize vectors if desired (for cosine similarity)
    if norm:
        norms = np.linalg.norm(input_vect, axis=1, keepdims=True)
        input_vect = input_vect / (norms + 1e-10)

    # Create the TSNE object
    if tsne_results is None:
        tsne = TSNE(n_components=2,
                    perplexity=perp,
                    # n_iter=n_iter,
                    early_exaggeration=exag,
                    learning_rate=lr,
                    init='pca',
                    metric='cosine',
                    method='barnes_hut',
                    verbose=0)
    
        tsne_results = tsne.fit_transform(input_vect)

    # Colors   
    unique_labels = np.unique(zvect)
    n_clusters = len(unique_labels)
    # all_colors = [plt.cm.nipy_spectral(i / n_clusters) for i in range(n_clusters)]
    all_colors = (
        plt.cm.tab20.colors +
        plt.cm.tab20b.colors +
        plt.cm.tab20c.colors +
        plt.cm.tab10.colors
    )
    if n_clusters > 70:
        n_extra = n_clusters - 70
        all_colors += tuple(plt.cm.nipy_spectral(i / n_extra) for i in range(n_extra))
        
    cmap = mcolors.ListedColormap(all_colors[:n_clusters])
    norm_cmap = mcolors.BoundaryNorm(boundaries=np.arange(n_clusters + 1), ncolors=n_clusters)

    # Make alphas more distinct
    if alpha_vect is not None:
        alpha_vect = alpha_vect**3
        rgb_colors = np.array([cmap(i % n_clusters)[:3] for i in zvect])
        rgb_colors = np.concatenate([rgb_colors, alpha_vect[:, None]], axis=1)
    else:
        rgb_colors = [cmap(i % n_clusters) for i in zvect]

    # Plot
    print("Found:", input_vect.shape[0], "points")
    s=0.1
    if input_vect.shape[0]<=25000: s=0.5
    if input_vect.shape[0]<=10000: s=2
    
    fig, ax = plt.subplots()
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], s=s, c=rgb_colors)
    ax.grid(False)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_cmap, cmap=cmap), ax=ax)
    cbar.set_label(ztitle)
    plt.xlabel('t-SNE #0')
    plt.ylabel('t-SNE #1')
    if save_name: plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return tsne_results
    
    
## Define a function for running t-SNE using the cuml version
def run_tsne_cuml(input_vect=None, zvect=None, alpha_vect=None, perp=30, exag=6, lr=2000.0, tsne_results=None, ztitle="Cluster ID", save_name=None, norm=True):

    print("Running cuML t-SNE with: perplexity =", perp, "early exaggeration =", exag)
    input_vect = cp.asarray(input_vect, dtype=cp.float32)

    if norm:
        norms = cp.linalg.norm(input_vect, axis=1, keepdims=True)
        input_vect = input_vect / (norms + 1e-10)

    n_neighbors = 2*perp
    if n_neighbors > 1024: n_neighbors = 1024
    
    ## I haven't played with most of cuml's t-SNE parameters
    ## tsne = cuML_TSNE(n_components=2, perplexity=perp, n_iter=3000, \
    ##                  early_exaggeration=exag, learning_rate=lr, exaggeration_iter=250, \
    ##                  learning_rate_method=None, square_distances=False, init='random', late_exaggeration=1, \
    ##                  metric='cosine', method='barnes_hut', verbose=True, n_neighbors=n_neighbors)

    tsne = cuML_TSNE(n_components=2, perplexity=perp, n_iter=5000, \
                     early_exaggeration=exag, learning_rate=lr, \
                     learning_rate_method=None, n_neighbors=n_neighbors, \
                     metric='cosine', method='barnes_hut', verbose=False)
    
    ## Back to basics
    ## tsne = cuML_TSNE(n_components=2, perplexity=perp, n_iter=5000, \
    ##                  early_exaggeration=exag, learning_rate=lr, method='barnes_hut',\
    ##                  metric='cosine', square_distances=False, verbose=True)
    
    if tsne_results is None:
        tsne_results = tsne.fit_transform(input_vect)
        scaler = cuMLScaler()
        tsne_results = scaler.fit_transform(tsne_results)  # tsne_results still on GPU
        tsne_results = cp.asnumpy(tsne_results)
        
    unique_labels = np.unique(zvect)
    n_clusters = len(unique_labels)

    # Use a qualitative colormap with enough colors
    all_colors = (
        plt.cm.tab20.colors +
        plt.cm.tab20b.colors +
        plt.cm.tab20c.colors +
        plt.cm.tab10.colors
    )

    cmap = mcolors.ListedColormap(all_colors[:n_clusters])
    norm = mcolors.BoundaryNorm(boundaries=np.arange(n_clusters + 1), ncolors=n_clusters)

    ## Make alphas more distinct:
    alpha_vect = alpha_vect**3
    rgb_colors = np.array([cmap(i % n_clusters)[:3] for i in zvect])
    rgb_colors = np.concatenate([rgb_colors, alpha_vect[:, None]], axis=1)

    ## Assemble the figure
    fig, ax = plt.subplots()
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], s=0.02, c=rgb_colors)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label(ztitle)
    plt.xlabel('t-SNE #0')
    plt.ylabel('t-SNE #1')
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    return tsne_results


def run_umap_cuml(input_vect=None, zvect=None, n_neighbors=100, min_distance=0.1, n_epochs=2000, alpha_vect=0.5, ztitle="Cluster ID", save_name=None, norm=True):

    input_vect = cp.asarray(input_vect, dtype=cp.float32)

    if norm:
        norms = cp.linalg.norm(input_vect, axis=1, keepdims=True)
        input_vect = input_vect / (norms + 1e-10)
    
    fit = cuML_UMAP(
        negative_sample_rate=10,
        n_neighbors=n_neighbors, 
        min_dist=min_distance, 
        metric='cosine', 
        #build_algo='nn_descent',
        n_epochs=n_epochs,
        init='random',
        random_state=42, 
        verbose=True
    )
    umap_results = fit.fit_transform(input_vect)    
    umap_results = cp.asnumpy(umap_results)

    x_low, x_high = np.percentile(umap_results[:,0], [0.1, 99.9])
    y_low, y_high = np.percentile(umap_results[:,1], [0.1, 99.9])
    
    unique_labels = np.unique(zvect)
    n_clusters = len(unique_labels)

    # Use a qualitative colormap with enough colors
    all_colors = (
        plt.cm.tab20.colors +
        plt.cm.tab20b.colors +
        plt.cm.tab20c.colors +
        plt.cm.tab10.colors
    )

    cmap = mcolors.ListedColormap(all_colors[:n_clusters])
    norm = mcolors.BoundaryNorm(boundaries=np.arange(n_clusters + 1), ncolors=n_clusters)

    gr = plt.scatter(umap_results[:, 0], umap_results[:, 1], s=0.5, alpha=alpha_vect, c=zvect, cmap=cmap, norm=norm)
    plt.colorbar(gr, label=ztitle)
    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)
    plt.xlabel('UMAP #0')
    plt.ylabel('UMAP #1')
    ax = plt.gca()
    ax.grid(False)
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    return
