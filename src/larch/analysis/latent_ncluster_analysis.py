import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['axes.grid'] = True          # enable grid globally
matplotlib.rcParams['axes.grid.axis'] = 'x'      # only vertical gridlines
matplotlib.rcParams['grid.linestyle'] = '--'     # dashed lines
matplotlib.rcParams['grid.color'] = 'gray'
matplotlib.rcParams['grid.alpha'] = 0.5

## Make matplotlib do things in batch mode, but not if we're in a jupyter session
if not matplotlib.get_backend().startswith("module://matplotlib_inline"):
    matplotlib.use("Agg")

## Use a GPU if available
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torchvision.transforms.v2 as transforms

## Seeding
SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

## Various shared analysis libraries
from analysis.tsne_utils import compute_tsne_cuml, plot_tsne, plot_tsne_block
from analysis.model_utils import load_checkpoint, get_models_from_checkpoint
from analysis.dataset_utils import get_dataset, image_loop
from analysis.plotting_utils import run_faiss_kmeans
from core.data.augmentations_2d import CenterCrop
from core.data.augmentations_2d import FirstRegionCrop
from datasets.nularbox.augmentations_2d import get_transform, LogAlphaCharge
from analysis.geometry_utils import plot_spectrum, pca_spectrum, cosine_spectrum
from analysis.geometry_utils import plot_similarity_distributions, plot_cumulative_variance
from analysis.geometry_utils import preprocess_embeddings


## For paraellising the ncluster runs
from joblib import Parallel, delayed

def plot_metric(x_vals, y_vals, metric_name, save_name=False):

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals, dtype=object)
    
    mask = [(y is not None) and not (isinstance(y, float) and np.isnan(y)) for y in y_vals]
    x_clean = x_vals[mask]
    y_clean = y_vals[mask].astype(float)
    
    plt.figure(figsize=(6, 4))
    plt.plot(x_clean, y_clean, marker='o')
    plt.xticks(x_vals)

    plt.xlabel("Number of clusters (k)")
    plt.ylabel(metric_name)

    #plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()

## A stopgap measure, return the nominal transform needed for this experiment (the only specific thing)
def get_nom_transform(experiment):

    if experiment == 'nularbox':
        return transforms.Compose([
            CenterCrop((512,512), (256,256)),
            LogAlphaCharge(5)
        ])
    if experiment == 'fsd':
        return FirstRegionCrop((800, 256), (768, 256))

    ## If the experiment name was unrecognised, nope out
    raise ValueError("Unknown experiment name:", experiment)

## A simple wrapper for parallel kmeans processing
def parallel_faiss_kmeans(ncluster, latent, nattempts, spherical):
    print("Processing ncluster =", ncluster)

    labels, metrics, _ = run_faiss_kmeans(latent, 
                                          ncluster,
                                          nattempts=nattempts,
                                          verbose=False,
                                          spherical=spherical
                                          )
    print("Finished ncluster =", ncluster)
    return ncluster, labels, metrics
    
def run_analysis(args):

    ## Setup the encoder
    encoder, heads, training_args = get_models_from_checkpoint(args.input_file)    

    ## Define the nominal transform for this experiment type
    nom_transform = get_nom_transform(args.experiment)
    aug_transform = get_transform('256x256', training_args.aug_type, training_args.aug_prob)
    
    ## Set up the datasets and loaders
    nevents=int(args.nevents)
    nom_dataset, nom_loader = get_dataset(training_args.data_dir, nevents, nom_transform)
    aug_dataset, aug_loader = get_dataset(training_args.data_dir, nevents, aug_transform)
    
    ## Get the processed vectors of interest from the datasets
    nom_processed  = image_loop(encoder, heads, nom_loader, device, return_hidden=True, detailed_info=True)
    aug1_processed = image_loop(encoder, heads, aug_loader, device, return_hidden=True, detailed_info=True)
    aug2_processed = image_loop(encoder, heads, aug_loader, device, return_hidden=True, detailed_info=True)

    ## Loop over the latent spaces
    for latent_name in ["encoder", "proj_final"]:

        ## Cosine similarity comparisons
        plot_similarity_distributions(nom_processed[latent_name],
                                      aug1_processed[latent_name],
                                      aug2_processed[latent_name],
                                      save_name=args.out_name_root+"_cossimcomp_"+latent_name+".png")

        pca_eigvals = pca_spectrum(nom_processed[latent_name])
        plot_spectrum(pca_eigvals, save_name=args.out_name_root+"_pcaeigvals_"+latent_name+".png")
        plot_spectrum(pca_eigvals, save_name=args.out_name_root+"_pcaeigvals_"+latent_name+"_max250.png", xlim=250)
        
        plot_cumulative_variance(pca_eigvals, save_name=args.out_name_root+"_cumpcavar_"+latent_name+".png")
        plot_cumulative_variance(pca_eigvals, xlim=1000, save_name=args.out_name_root+"_cumpcavar_"+latent_name+"_max1000.png")        
        
        cosine_eigvals = cosine_spectrum(nom_processed[latent_name])
        plot_spectrum(cosine_eigvals, save_name=args.out_name_root+"_simeigvals_"+latent_name+".png")
        plot_spectrum(cosine_eigvals, save_name=args.out_name_root+"_simeigvals_"+latent_name+"_max250.png", xlim=250)
        
    ## Get pre-processed embeddings
    X_pca50_spherical = preprocess_embeddings(
        nom_processed['encoder'],
        pca=50,
        drop_first_pca=False,
        whiten=False,
        spherical=True
    )
    
    X_pca50_euclidean = preprocess_embeddings(
        nom_processed['encoder'],
        pca=50,
        drop_first_pca=False,
        whiten=True,
        spherical=False
    )

    X_pca256_spherical = preprocess_embeddings(
        nom_processed['encoder'],
        pca=256,
        drop_first_pca=False,
        whiten=False,
        spherical=True
    )

    X_pca100_euclidean = preprocess_embeddings(
        nom_processed['encoder'],
        pca=100,
        drop_first_pca=False,
        whiten=True,
        spherical=False
    )

    ## t-SNE examples
    print("Starting tSNE...")
    tsne_results_euclidean = compute_tsne_cuml(X_pca50_euclidean,
                                               perp=150, exag=20, lr=500,
                                               metric="euclidean",
                                               verbose=False)

    plot_tsne_block(tsne_results_euclidean, nom_processed, apply_alpha_vect=False,
                    save_name=args.out_name_root+"_euclidean_tsne_block.png")

    tsne_results_spherical = compute_tsne_cuml(X_pca50_spherical,
                                               perp=150, exag=20, lr=500,
                                               metric="cosine",
                                               verbose=False)
    
    plot_tsne_block(tsne_results_spherical, nom_processed, apply_alpha_vect=False,
                    save_name=args.out_name_root+"_spherical_tsne_block.png")


    ## Now run k-means over a variety of different ncluster possibilities
    ncluster_list = [n for n in range(args.clust_min, args.clust_max+1, args.clust_step)]

    ## Process euclidean k-means
    print("Running euclidean k-means...")
    euclidean_results = Parallel(
        n_jobs = args.ngpus,
        prefer="processes"
    )(
        delayed(parallel_faiss_kmeans)(
            ncluster,
            X_pca100_euclidean,
            nattempts=args.nattempts,
            spherical=False
            )
        for ncluster in ncluster_list
    )

    sil_euclidean = []
    ch_euclidean = []
    db_euclidean = []

    ## Process the results
    for ncluster, these_labels, metrics in euclidean_results:
        sil_euclidean.append(metrics["silhouette"])
        ch_euclidean.append(metrics["calinski_harabasz"])
        db_euclidean.append(metrics["davies_bouldin"])

        if ncluster in [50, 100]:
            plot_tsne(tsne_results_euclidean, these_labels, alpha_vect=None, ztitle="Clust index",
                      save_name=args.out_name_root+"_tsne_euclidean"+str(ncluster)+".png")

    ## After the loop over clusters, make some summary plots
    plot_metric(ncluster_list, sil_euclidean, "Silhouette Score", args.out_name_root+"_euclidean_silhouette.png")
    plot_metric(ncluster_list, ch_euclidean, "Calinski–Harabasz Index", args.out_name_root+"_euclidean_ch.png")
    plot_metric(ncluster_list, db_euclidean, "Davies–Bouldin Index", args.out_name_root+"_euclidean_db.png")


    ## Process spherical k-means
    print("Running spherical k-means...")
    spherical_results = Parallel(
        n_jobs = args.ngpus,
        prefer="processes"
    )(
        delayed(parallel_faiss_kmeans)(
            ncluster,
            X_pca256_spherical,
            nattempts=args.nattempts,
            spherical=True
            )
        for ncluster in ncluster_list
    )    

    sil_spherical = []
    ch_spherical = []
    db_spherical = []

    ## Process the results
    for ncluster, these_labels, metrics in spherical_results:
        sil_spherical.append(metrics["silhouette"])
        ch_spherical.append(metrics["calinski_harabasz"])
        db_spherical.append(metrics["davies_bouldin"])

        if ncluster in [50, 100]:
            plot_tsne(tsne_results_spherical, these_labels, alpha_vect=None, ztitle="Clust index",
                      save_name=args.out_name_root+"_tsne_spherical"+str(ncluster)+".png")

    ## After the loop over clusters, make some summary plots
    plot_metric(ncluster_list, sil_spherical, "Silhouette Score", args.out_name_root+"_spherical_silhouette.png")
    plot_metric(ncluster_list, ch_spherical, "Calinski–Harabasz Index", args.out_name_root+"_spherical_ch.png")
    plot_metric(ncluster_list, db_spherical, "Davies–Bouldin Index", args.out_name_root+"_spherical_db.png")
    
        
## Do the business
if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("Model analysis")

    ## Require an input file name and location to dump plots
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--out_name_root', type=str)

    ## This is probably temporary, should switch to an "experiment" model so files know what experiment they're trained on
    ## For now, require an explicit declaration of the type of data used to train this model
    parser.add_argument('--experiment', type=str)
    
    ## Give a sensible default for the number of events to process
    parser.add_argument('--nevents', type=int, default=50000)

    ## Allow use of multiple GPUs
    parser.add_argument('--ngpus', type=int) #, default=1)

    ## Options for stepping through nclusters
    parser.add_argument('--clust_min', type=int, default=10)
    parser.add_argument('--clust_max', type=int, default=60)
    parser.add_argument('--clust_step', type=int, default=10)
    
    ## Options for faiss
    parser.add_argument('--nattempts', type=int, default=10)
    
    ## Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))
    
    run_analysis(args)
