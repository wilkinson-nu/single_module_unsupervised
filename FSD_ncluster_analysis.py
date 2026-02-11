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

## Seeding
SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

## Various shared analysis libraries
from analysis.plotting_utils import run_tsne_skl
from analysis.model_utils import load_checkpoint, get_models_from_checkpoint
from analysis.dataset_utils import get_dataset, image_loop
from analysis.plotting_utils import run_faiss_spherical_kmeans
from core.data.augmentations_2d import CenterCrop
from core.data.augmentations_2d import FirstRegionCrop

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
        return CenterCrop((512,512), (256,256))
    if experiment == 'fsd':
        return FirstRegionCrop((800, 256), (768, 256))

    ## If the experiment name was unrecognised, nope out
    raise ValueError("Unknown experiment name:", experiment)

def process_one(ncluster, latent, nattempts):
    print("Processing ncluster =", ncluster)

    labels, metrics, _ = run_faiss_spherical_kmeans(latent, 
                                                    ncluster,
                                                    nattempts=nattempts,
                                                    verbose=True
                                                   )
    print("Finished ncluster =", ncluster)
    return ncluster, labels, metrics
    
def run_analysis(args):

    ## Define the nominal transform for this experiment type
    nom_transform = get_nom_transform(args.experiment)
    
    ## Setup the encoder
    encoder, heads, training_args = get_models_from_checkpoint(args.input_file)
    
    ## Set up the datasets and loaders
    ntsne=int(args.ntsne)
    data_dataset, data_loader = get_dataset(training_args.data_dir, ntsne, nom_transform)
    
    ## Get the processed vectors of interest from the datasets
    data_processed = image_loop(encoder, heads, data_loader, device)
    
    ## t-SNE examples
    print("Starting tSNE...")
    tsne_results = run_tsne_skl(data_processed['latent'][:ntsne].copy(), \
                                np.zeros(ntsne), \
                                perp=150, exag=20, lr=500)

    ncluster_list = [n for n in range(args.clust_min, args.clust_max+1, args.clust_step)]

    ## Can more dynamically pick
    n_jobs = 4
    
    ## Spawn parallel clustering jobs
    results = Parallel(
        n_jobs=n_jobs,
        prefer="processes"
    )(
        delayed(process_one)(
            ncluster, data_processed['latent'][:ntsne].copy(), args.nattempts
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

        _ = run_tsne_skl(data_processed['enc_latent'][:ntsne].copy(), \
                         these_labels[:ntsne].copy(), tsne_results=tsne_results, \
                         save_name=args.out_name_root+"_tSNE_"+str(ncluster)+".png")

    ## After the loop over clusters, make some summary plots
    plot_metric(ncluster_list, silhouette_scores, "Silhouette Score", args.out_name_root+"_silhouette.png")
    plot_metric(ncluster_list, ch_scores, "Calinski–Harabasz Index", args.out_name_root+"_ch.png")
    plot_metric(ncluster_list, db_scores, "Davies–Bouldin Index", args.out_name_root+"_db.png")

        
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
    parser.add_argument('--ntsne', type=int, default=20000, nargs='?')

    ## Options for stepping through nclusters
    parser.add_argument('--clust_min', type=int, default=10, nargs='?')
    parser.add_argument('--clust_max', type=int, default=60, nargs='?')
    parser.add_argument('--clust_step', type=int, default=10, nargs='?')  

    ## Options for faiss
    parser.add_argument('--nattempts', type=int, default=10, nargs='?')    
    
    ## Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))
    
    run_analysis(args)
