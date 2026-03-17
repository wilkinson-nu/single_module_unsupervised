from sklearn.manifold import TSNE as skl_TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from cuml.manifold import TSNE as cuML_TSNE
import cupy as cp
from cuml.preprocessing import StandardScaler as cuMLScaler
from cuml.manifold import UMAP as cuML_UMAP
from sklearn.decomposition import PCA as skl_PCA
from cuml.decomposition import PCA as cuML_PCA

def compute_tsne_skl(input_vect,
                     perp=30,
                     exag=6,
                     lr=2000.0,
                     n_iter=2000,
                     norm=True,
                     metric='cosine',
                     method='barnes_hut',
                     init='pca',
                     pca=None,
                     whiten=False,
                     random_state=None,
                     verbose=0):

    if norm:
        print("L2 normalising tSNE inputs...")
        norms = np.linalg.norm(input_vect, axis=1, keepdims=True)
        input_vect = input_vect / (norms + 1e-10)

    if pca is not None:
        print("Finding", pca, "largest PCA components...")
        input_vect = skl_PCA(n_components=pca, whiten=whiten).fit_transform(input_vect)

    print("Running scikit-learn t-SNE with:",
          "perplexity =", perp,
          "early exaggeration =", exag)
        
    tsne = skl_TSNE(
        n_components=2,
        perplexity=perp,
        early_exaggeration=exag,
        learning_rate=lr,
        init=init,
        metric=metric,
        method=method,
        random_state=random_state,
        verbose=verbose
    )

    
    tsne_results = tsne.fit_transform(input_vect)

    print("Found:", tsne_results.shape[0], "points")
    return tsne_results

def run_tsne_skl(input_vect=None, zvect=None, alpha_vect=None, perp=30, exag=6,
                 lr=2000.0, n_iter=2000, ztitle="Cluster ID", save_name=None, norm=True, n_samples=None, tsne_results=None, pca=None):

    if tsne_results is None:
        tsne_results = compute_tsne_skl(input_vect,
                                        perp=perp,
                                        exag=exag,
                                        lr=lr,
                                        n_iter=n_iter,
                                        norm=norm,
                                        pca=pca)
    plot_tsne(tsne_results,
              zvect=zvect,
              alpha_vect=alpha_vect,
              ztitle=ztitle,
              ax=None,
              add_colorbar=True,
              save_name=save_name)
    return tsne_results


def plot_tsne(tsne_results,
              zvect=None,
              alpha_vect=None,
              ztitle="Cluster ID",
              ax=None,
              add_colorbar=True,
              linear_colorbar=False,
              save_name=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    unique_labels = np.unique(zvect)
    n_clusters = len(unique_labels)

    if linear_colorbar:
        all_colors = tuple(
            plt.cm.nipy_spectral(i / n_clusters) for i in range(n_clusters)
        )
    else:
        all_colors = (
            plt.cm.tab20.colors +
            plt.cm.tab20b.colors +
            plt.cm.tab20c.colors +
            plt.cm.tab10.colors
        )

        if n_clusters > 70:
            n_extra = n_clusters - 70
            all_colors += tuple(
                plt.cm.nipy_spectral(i / n_extra) for i in range(n_extra)
            )

    cmap = mcolors.ListedColormap(all_colors[:n_clusters])
    norm_cmap = mcolors.BoundaryNorm(
        boundaries=np.arange(n_clusters + 1),
        ncolors=n_clusters
    )

    if alpha_vect is not None:
        alpha_vect = alpha_vect**3
        rgb_colors = np.array(
            [cmap(i % n_clusters)[:3] for i in zvect]
        )
        rgb_colors = np.concatenate(
            [rgb_colors, alpha_vect[:, None]],
            axis=1
        )
    else:
        rgb_colors = [cmap(i % n_clusters) for i in zvect]

    npts = tsne_results.shape[0]
    s = 0.1
    if npts <= 25000: s = 0.5
    if npts <= 10000: s = 2
    if npts > 100000: s = 0.01
    
    ax.scatter(tsne_results[:, 0],
               tsne_results[:, 1],
               s=s,
               c=rgb_colors)

    ax.set_xlabel("t-SNE #0")
    ax.set_ylabel("t-SNE #1")
    ax.grid(False)

    if add_colorbar:
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm_cmap, cmap=cmap),
            ax=ax
        )
        cbar.set_label(ztitle, rotation=270, labelpad=20)

    if save_name:
        plt.savefig(save_name,
                    dpi=300,
                    bbox_inches='tight')
    #if ax==None:
    #plt.tight_layout()
    #plt.show()
    #plt.close()
        
    return ax


def plot_tsne_block(tsne_results, processed, apply_alpha_vect=False, save_name=None):
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    ntsne = len(tsne_results)
    n_charged_particles = processed['labels']['nproton'] + \
        processed['labels']['npipm'] + \
        processed['labels']['nkapm']

    alpha_vect = None
    if apply_alpha_vect: alpha_vect = processed['clust_max'][:ntsne]

    plot_tsne(tsne_results, processed.get('clust_index', np.zeros(ntsne))[:ntsne], 
              ax=axes[0][0], alpha_vect=alpha_vect, ztitle="Clust index")
    plot_tsne(tsne_results, processed['labels']['topology'][:ntsne], 
              ax=axes[0][1], alpha_vect=alpha_vect, ztitle="Topology")
    plot_tsne(tsne_results, processed['labels']['mode'][:ntsne], 
              ax=axes[0][2], alpha_vect=alpha_vect, ztitle="Mode")
    plot_tsne(tsne_results, n_charged_particles[:ntsne], 
              ax=axes[1][0], alpha_vect=alpha_vect, ztitle="N. charged particles", linear_colorbar=True)
    plot_tsne(tsne_results, processed['labels']['enu'][:ntsne].astype(int), 
              ax=axes[1][1], alpha_vect=alpha_vect, ztitle=r"$E_{\nu}$ (GeV)", linear_colorbar=True)
    plot_tsne(tsne_results, processed['labels']['q0'][:ntsne].astype(int), 
              ax=axes[1][2], alpha_vect=alpha_vect, ztitle=r"$q_{0}$ (GeV)", linear_colorbar=True)
    plot_tsne(tsne_results, processed['labels']['nproton'][:ntsne], 
              ax=axes[2][0], alpha_vect=alpha_vect, ztitle="N. protons", linear_colorbar=True)
    plot_tsne(tsne_results, processed['labels']['npipm'][:ntsne], 
              ax=axes[2][1], alpha_vect=alpha_vect, ztitle=r"N. $\pi^{\pm}$", linear_colorbar=True)
    plot_tsne(tsne_results, processed['labels']['npi0'][:ntsne], 
              ax=axes[2][2], alpha_vect=alpha_vect, ztitle=r"N. $\pi^{0}$", linear_colorbar=True)

    plt.tight_layout()
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


## Define a function for running t-SNE using the cuml version
def compute_tsne_cuml(input_vect, 
                      perp=30, 
                      exag=6, 
                      lr=2000.0, 
                      n_iter=5000,
                      norm=True,
                      verbose=True,
                      pca=None,
                      whiten=False):
    
    input_vect = cp.asarray(input_vect, dtype=cp.float32)
    
    if norm:
        print("L2 normalising tSNE inputs...")
        norms = cp.linalg.norm(input_vect, axis=1, keepdims=True)
        input_vect = input_vect / (norms + 1e-10)

    if pca is not None:
        print("Finding", pca, "largest PCA components...")
        if whiten is True: print("...including whitening...")
        input_vect = cuML_PCA(n_components=pca, whiten=whiten).fit_transform(input_vect)

    print("Running cuML t-SNE with:",
          "perplexity =", perp,
          "early exaggeration =", exag)
    
    n_neighbors = 2*perp
    if n_neighbors > 1024: n_neighbors = 1024
    
    ## I haven't played with most of cuml's t-SNE parameters
    ## tsne = cuML_TSNE(n_components=2, perplexity=perp, n_iter=3000, \
    ##                  early_exaggeration=exag, learning_rate=lr, exaggeration_iter=250, \
    ##                  learning_rate_method=None, square_distances=False, init='random', late_exaggeration=1, \
    ##                  metric='cosine', method='barnes_hut', verbose=True, n_neighbors=n_neighbors)

    tsne = cuML_TSNE(n_components=2, perplexity=perp, n_iter=n_iter, \
                     early_exaggeration=exag, learning_rate=lr, \
                     learning_rate_method=None, n_neighbors=n_neighbors, \
                     metric='cosine', method='barnes_hut', verbose=verbose)
    
    ## Back to basics
    ## tsne = cuML_TSNE(n_components=2, perplexity=perp, n_iter=5000, \
    ##                  early_exaggeration=exag, learning_rate=lr, method='barnes_hut',\
    ##                  metric='cosine', square_distances=False, verbose=True)
    
    tsne_results = tsne.fit_transform(input_vect)
    scaler = cuMLScaler()
    tsne_results = scaler.fit_transform(tsne_results)  # tsne_results still on GPU
    tsne_results = cp.asnumpy(tsne_results)
    print("Found:", tsne_results.shape[0], "points")
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
