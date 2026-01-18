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

## Import analysis functions
from ME_analysis_libs import run_tsne_skl, plot_cluster_bigblock, run_vMF

## Seeding
SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

## Various shared analysis libraries
from ME_analysis_libs import run_tsne_skl, plot_cluster_bigblock
from ME_analysis_libs import load_checkpoint, get_models_from_checkpoint, get_dataset, image_loop

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

def process_one(ncluster, latent, ncopies):
    print("Processing ncluster =", ncluster)

    labels, metrics = run_vMF(latent,
                              ncluster,
                              init="k-means++",
                              n_copies=ncopies,
                              verbose=True)
    print("Finished ncluster =", ncluster)
    return ncluster, labels, metrics
    
def run_analysis(args):

    ## Setup the encoder
    encoder, heads, training_args = get_models_from_checkpoint(args.input_file)

    encoder.to(device)
    for h in heads.values(): h.to(device)
    
    ## Set up the datasets and loaders
    ntsne=int(args.ntsne)
    data_dataset, data_loader = get_dataset(training_args.data_dir, ntsne)
    
    ## Get the processed vectors of interest from the datasets
    print("Loading inputs...")
    data_processed = image_loop(encoder, heads, data_loader)
    print("...inputs loaded!")
    
    ## t-SNE examples
    print("Starting tSNE...")
    tsne_results = run_tsne_skl(data_processed['latent'][:ntsne].copy(), \
                                np.zeros(ntsne), \
                                perp=150, exag=20, lr=500)

    ncluster_list = [n for n in range(args.clust_min, args.clust_max+1, args.clust_step)]

    ## Can more dynamically pick
    n_jobs = 10
    
    ## Spawn parallel clustering jobs
    results = Parallel(
        n_jobs=n_jobs,
        prefer="processes"
    )(
        delayed(process_one)(
            ncluster, data_processed['latent'][:ntsne].copy(), args.ncopies
        )
        for ncluster in ncluster_list
    )

    print("Making summary plots...")
    silhouette_scores = []
    ch_scores = []
    db_scores = []

    ## Process the results
    for ncluster, these_labels, metrics in results:
        silhouette_scores.append(metrics["silhouette"])
        ch_scores.append(metrics["calinski_harabasz"])
        db_scores.append(metrics["davies_bouldin"])

        _ = run_tsne_skl(data_processed['latent'][:ntsne].copy(), \
                         these_labels[:ntsne].copy(), tsne_results=tsne_results, \
                         save_name=args.out_name_root+"_tSNE_"+str(ncluster)+".png")
        
    ## Loop over a range of clusters
    ## for ncluster in ncluster_list:
    ##     print("Processing ncluster =", ncluster)
    ##     these_labels, metrics = run_vMF(data_processed['latent'][:ntsne].copy(), ncluster, init="k-means++", n_copies=args.ncopies, verbose=True)
    ##     _ = run_tsne_skl(data_processed['latent'][:ntsne].copy(), \
    ##                      these_labels[:ntsne].copy(), tsne_results=tsne_results, \
    ##                      save_name=args.out_name_root+"_tSNE_"+str(ncluster)+".png")
    ## 
    ##     silhouette_scores.append(metrics["silhouette"])
    ##     ch_scores.append(metrics["calinski_harabasz"])
    ##     db_scores.append(metrics["davies_bouldin"])
    ## 
    ##     ## Plot some examples for each cluster:
    ##     if args.example_cluster_images:
    ##         for n in range(training_args.nclusters):
    ##             plot_cluster_bigblock(data_dataset, data_processed['clust_index'], n, 1, 10, \
    ##                                   cluster_probs=data_processed['clust_max'], \
    ##                                   save_name=args.out_name_root+"_data_example"+str(n)+"_top.png")
    ##             plot_cluster_bigblock(data_dataset, data_processed['clust_index'], n, 1, 10, \
    ##                                   save_name=args.out_name_root+"_data_example"+str(n)+"_all.png")

    ## After the loop over clusters, make some summary plots
    plot_metric(ncluster_list, silhouette_scores, "Silhouette Score", args.out_name_root+"_silhouette.png")
    plot_metric(ncluster_list, ch_scores, "Calinski–Harabasz Index", args.out_name_root+"_ch.png")
    plot_metric(ncluster_list, db_scores, "Davies–Bouldin Index", args.out_name_root+"_db.png")

        
## Do the business
if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("Model analysis")

    # Require an input file name and location to dump plots
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--out_name_root', type=str)

    ## Give a sensible default for the number of events to process
    parser.add_argument('--ntsne', type=int, default=20000, nargs='?')

    ## Options for stepping through nclusters
    parser.add_argument('--clust_min', type=int, default=10, nargs='?')
    parser.add_argument('--clust_max', type=int, default=60, nargs='?')
    parser.add_argument('--clust_step', type=int, default=10, nargs='?')  

    ## Options for vMF
    parser.add_argument('--ncopies', type=int, default=10, nargs='?')    
    
    ## Options for controlling the plots to make
    parser.add_argument('--example_cluster_images', type=int, choices=[0,1], default=0, nargs='?')
    
    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))
    
    run_analysis(args)
