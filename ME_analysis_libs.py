import torch
import MinkowskiEngine as ME
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ME_dataset_libs import make_dense, make_dense_from_tensor, Label
from cuml.manifold import TSNE as cuML_TSNE
import cupy as cp
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from cuml.preprocessing import StandardScaler as cuMLScaler
from cuml.manifold import UMAP as cuML_UMAP

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
    # plt.grid(True)
    plt.legend(
        ncol=2,
        fontsize="medium",
        loc="lower right",
        bbox_to_anchor=(0.5, 1.),
        frameon=False
    )
    plt.tight_layout()
    plt.show()
    plt.close()

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
    # plt.grid(True)
    plt.show()
    plt.close()

def plot_metric_data_vs_sim(data_xvar, sim_xvar, sim_labels, nbinsx=None, x_min=None, x_max=None, xtitle="xvar", ytitle="N. images", normalize=False, save_name=None):

    ## Deal with binning myself for some reason...
    bins = parse_binning(data_xvar, nbinsx, x_min, x_max)
    
    label_values = [m.value for m in Label]
    label_names  = [m.name for m in Label]

    ## Skip the data label because it's being plotted separately here
    label_values = label_values[:-1]
    label_names  = label_names[:-1] 
    
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
        color="black",
        linewidth=1.5,
        label="Data"
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

def plot_metric_by_cluster(xvar, cluster_vect, nbinsx=None, x_min=None, x_max=None, xtitle="xvar", ytitle="N. images", normalize=False, save_name=None):

    ## Deal with binning myself for some reason...
    bins = parse_binning(xvar, nbinsx, x_min, x_max)

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
        align='right',
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
        
        numpy_coords, numpy_feats, _, _, _ = dataset[indices[i]]

        # Create batched coordinates for the SparseTensor input
        orig_bcoords  = ME.utils.batched_coordinates([numpy_coords])
        orig_bfeats  = torch.from_numpy(np.concatenate([numpy_feats], 0)).float()
        orig = ME.SparseTensor(orig_bfeats, orig_bcoords)
            
        inputs  = make_dense_from_tensor(orig, 0, 768, 256)
        inputs  = inputs .cpu().squeeze().numpy()
        
        plt.imshow(inputs, origin='lower')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)            
    plt.tight_layout()
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_cluster_bigblock(dataset, cluster_ids, index, max_x=10, max_y=10, cluster_probs=None, save_name=None): 
    
    plt.figure(figsize=(max_y*2, max_x*6))
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
        
        plt.imshow(inputs, origin='lower')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)    
    plt.tight_layout()
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()  
    plt.close()
    
## Define a function for running t-SNE using the cuml version
def run_tsne_cuml(input_vect=None, zvect=None, perp=30, exag=6, lr=2000.0, alpha_vect=0.5, tsne_results=None, ztitle="Cluster ID", save_name=None):

    print("Running cuML t-SNE with: perplexity =", perp, "early exaggeration =", exag)

    input_vect = cp.asarray(input_vect, dtype=cp.float32)
    
    ## I haven't played with most of cuml's t-SNE parameters
    tsne = cuML_TSNE(n_components=2, perplexity=perp, n_iter=2000, \
                     early_exaggeration=exag, learning_rate=lr, \
                     learning_rate_method=None, \
                     metric='cosine', method='barnes_hut', verbose=False)
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
    
    gr = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=0.005, alpha=alpha_vect, c=zvect, cmap=cmap, norm=norm)
    plt.colorbar(gr, label=ztitle)
    plt.xlabel('t-SNE #0')
    plt.ylabel('t-SNE #1')
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    return tsne_results


def run_umap_cuml(input_vect=None, zvect=None, n_neighbors=100, min_distance=0.1, n_epochs=2000, alpha_vect=0.5, ztitle="Cluster ID", save_name=None):

    input_vect = cp.asarray(input_vect, dtype=cp.float32)

    fit = cuML_UMAP(
        n_neighbors=n_neighbors, min_dist=min_distance, metric='cosine', random_state=0, n_epochs=n_epochs
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

    gr = plt.scatter(umap_results[:, 0], umap_results[:, 1], s=0.005, alpha=alpha_vect, c=zvect, cmap=cmap, norm=norm)
    plt.colorbar(gr, label=ztitle)
    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)
    plt.xlabel('UMAP #0')
    plt.ylabel('UMAP #1')
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    return
