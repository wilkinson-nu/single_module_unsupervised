import torch
import MinkowskiEngine as ME
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from core.analysis.image_utils import make_dense, make_dense_from_tensor
from datasets.fsd.truth_labels import Label
from matplotlib.ticker import MaxNLocator
import faiss
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


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
def plot_metric_by_label(xvar, labels, nbinsx=None, x_min=None, x_max=None, xtitle="xvar", ytitle="N. images", normalize=False, label_enum=Label):

    ## Deal with binning myself for some reason...
    bins, is_int = parse_binning(xvar, nbinsx, x_min, x_max)
    
    label_values = [m.value for m in label_enum]
    label_names  = [m.name for m in label_enum]

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

    
def plot_metric_data_vs_sim(data_xvar, sim_xvar, sim_labels, nbinsx=None,
                            x_min=None, x_max=None, xtitle="xvar", ytitle="N. images",
                            normalize=True, save_name=None, label_enum=Label, logy=False):

    ## Deal with binning myself for some reason...
    bins, is_int = parse_binning(data_xvar, nbinsx, x_min, x_max)
    
    label_values = [m.value for m in label_enum]
    label_names  = [m.name for m in label_enum]

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
    if logy: plt.yscale("log")
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


def plot_cluster_examples(dataset, cluster_ids, index, max_images=8, cluster_probs=None, save_name=None, image_size=(768, 256)): 

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
            
        inputs  = make_dense_from_tensor(orig, 0, image_size[0], image_size[1])
        inputs  = inputs .cpu().squeeze().numpy()
        
        plt.imshow(inputs, origin='lower', cmap=cmap, vmin=1e-6)
        ax.axis('off')
    plt.tight_layout()
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_cluster_bigblock(dataset, cluster_ids, index, max_x=10, max_y=10, cluster_probs=None, save_name=None, image_size=(768, 256)): 

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
            
        inputs  = make_dense_from_tensor(orig, 0, image_size[0], image_size[1])
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

def run_faiss_spherical_kmeans(dataset, n_clusters, nattempts=20, verbose=False, seed=123):
    # Normalize embeddings (critical for cosine clustering)
    X = dataset.astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    N, d = X.shape

    # FAISS k-means (spherical via normalization)
    kmeans = faiss.Kmeans(
        d=d,
        k=n_clusters,
        niter=20,
        verbose=verbose,
        seed=seed,
        nredo=nattempts,
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

    print("Cluster weights:", weights)
    print("Silhouette score:", metrics["silhouette"])
    print("Calinski-Harabasz =", metrics["calinski_harabasz"])
    print("Davies-Bouldin =", metrics["davies_bouldin"])

    return labels, metrics, kmeans.centroids