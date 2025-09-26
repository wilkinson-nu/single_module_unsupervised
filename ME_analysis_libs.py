import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ME_dataset_libs import make_dense, make_dense_from_tensor, Label

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

    
def plot_overlap_matrix(overlap_matrix, merged_labels=None):
    plt.figure(figsize=(8,6))
    im = plt.imshow(overlap_matrix, cmap="viridis", vmin=0, vmax=1) #, norm='log')
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
        if nbins is None: return np.arange(x_min, x_max+2) - 0.5
        else: return np.linspace(x_min, x_max+1, nbins+1)
    else:
        if nbins is None: return 50
        else: return np.linspace(x_min, x_max, nbins+1)

# Make a histogram broken down into simulation and data, for arbitrary x variables
def plot_metric_sim_data(xvar, labels, nbinsx=None, x_min=None, x_max=None, xtitle="xvar", ytitle="N. images"):
   
    ## Deal with binning myself for some reason...
    bins = parse_binning(xvar, nbinsx, x_min, x_max)
        
    data_mask = labels < 0
    xvar_data = xvar[data_mask]
    xvar_sim  = xvar[~data_mask]
    plt.hist([xvar_data, xvar_sim], bins=bins,
            stacked=True,
            histtype='stepfilled',
            align='right',
            label=['Data', 'Sim'],
            color=['lightcoral', 'mediumseagreen'])
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.grid(True)
    plt.legend(
        ncol=2,
        fontsize="medium",
        loc="lower right",
        bbox_to_anchor=(0.5, 1.),
        frameon=False
    )
    plt.tight_layout()
    plt.show()

# Make a histogram broken down into all possible labels, for arbitrary x variables
def plot_metric_by_label(xvar, labels, nbinsx=None, x_min=None, x_max=None, xtitle="xvar", ytitle="N. images", normalize=False):

    ## Deal with binning myself for some reason...
    bins = parse_binning(xvar, nbinsx, x_min, x_max)
    
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
        align='right',
        stacked=True,
        label=label_names,
        density=normalize,
        color=colors
    )

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
    plt.grid(True)
    plt.show()

def plot_metric_by_cluster(xvar, cluster_vect, nbinsx=None, x_min=None, x_max=None, xtitle="xvar", ytitle="N. images", normalize=False):

    ## Deal with binning myself for some reason...
    bins = parse_binning(xvar, nbinsx, x_min, x_max)

    unique_values = np.unique(cluster_vect)
    n_clusters = len(unique_values)

    cluster_names = [str(i) for i in unique_values]

    print(cluster_names)
    
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
        align='right',
        stacked=True,
        label=cluster_names,
        density=normalize,
        color=colors
    )

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(
        ncol=10,
        fontsize="x-small",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.),
        frameon=False
    )
    plt.tight_layout()  # prevents clipping
    plt.grid(True)
    plt.show()