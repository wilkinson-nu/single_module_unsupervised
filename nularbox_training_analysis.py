#import bootstrap
import numpy as np
import argparse
import matplotlib

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
from analysis.plotting_utils import plot_metric_data_vs_sim, plot_metric_by_cluster, plot_metric_by_confidence, plot_cluster_bigblock, plot_multiplicity_matrix_grid
from analysis.tsne_utils import compute_tsne_cuml, compute_tsne_skl, plot_tsne, plot_tsne_block
from datasets.nularbox.truth_labels import Mode, Topology

## Seeding
SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

## Various shared analysis libraries
from analysis.model_utils import load_checkpoint, get_models_from_checkpoint
from analysis.dataset_utils import get_dataset, image_loop, reorder_clusters
from core.data.augmentations_2d import CenterCrop
    
def run_analysis(args):

    ## Setup the encoder
    encoder, heads, training_args = get_models_from_checkpoint(args.input_file)

    encoder.to(device)
    for h in heads.values(): h.to(device)

    ## Nominal transformation
    transform = CenterCrop((512,512), (256,256))
    
    ## Set up the datasets and loaders
    nom_dataset, nom_loader = get_dataset(args.nom_data_dir, args.nnom, transform)
    alt_dataset, alt_loader = get_dataset(args.alt_data_dir, args.nalt, transform)

    ## Get the processed vectors of interest from the datasets
    print("Loading inputs...")
    ## Need to modify these to also save intermediate layers from the heads
    nom_processed = image_loop(encoder, heads, nom_loader, device, detailed_info=True, return_hidden=True)
    alt_processed = image_loop(encoder, heads, alt_loader, device, detailed_info=True, return_hidden=True)

    ## Do some magic to re-order the clusters for presentation purposes
    reorder_clusters(nom_processed, alt_processed)
    print("...inputs loaded!")

    ## Make some basic high-level plots
    plot_metric_data_vs_sim(nom_processed['clust_index'], 
                            alt_processed['clust_index'], 
                            alt_processed['labels']['topology'],
                            label_enum=Topology,
                            xtitle="Max. cluster index", 
                            save_name=args.out_name_root+"_clust_index.png")
    
    plot_metric_data_vs_sim(nom_processed['clust_max'], 
                            alt_processed['clust_max'], 
                            alt_processed['labels']['topology'],
                            label_enum=Topology,
                            xtitle="Max. cluster value",
                            save_name=args.out_name_root+"_clust_max.png")
    
    plot_metric_data_vs_sim(nom_processed['nhits'],
                            alt_processed['nhits'],
                            alt_processed['clust_index'],
                            nbinsx=70, x_max=1400,
                            xtitle="N. hits",
                            save_name=args.out_name_root+"_nhits.png")
    
    plot_metric_data_vs_sim(nom_processed['sumQ'],
                            alt_processed['sumQ'],
                            alt_processed['clust_index'],
                            nbinsx=70, x_max=1400,
                            xtitle="Sum Q",
                            save_name=args.out_name_root+"_sumQ.png")
    
    plot_metric_data_vs_sim(nom_processed['maxQ'],
                            alt_processed['maxQ'],
                            alt_processed['clust_index'],
                            nbinsx=100, x_min=1.5, x_max=2.5,
                            xtitle="Max. Q",
                            save_name=args.out_name_root+"_maxQ.png")

    ## Make a list of all representations we might have
    layer_list = ["encoder", "proj_final", "clust_final"]
    #              "proj_layer1", "proj_layer2", "proj_final",
    #              "clust_layer1", "clust_layer2", "clust_final"]

    ## Remove any that don't exist for this file (probably a better way to do all of this)
    for x in layer_list:
        if x not in nom_processed:
            layer_list.remove(x)


    ## Now run tSNE on every representation we have...
    for x in layer_list:

        tsne_results = compute_tsne_cuml(nom_processed[x], perp=100, exag=20, lr=500, verbose=False)

        plot_tsne(tsne_results, nom_processed['clust_index'], alpha_vect=nom_processed['clust_max'], ztitle="Clust index",
                  save_name=args.out_name_root+"_tsne_"+x+"_clust_index_alpha.png")
        plot_tsne(tsne_results, nom_processed['clust_index'], alpha_vect=None, ztitle="Clust index",
                  save_name=args.out_name_root+"_tsne_"+x+"_clust_index.png")

        plot_tsne_block(tsne_results, nom_processed, apply_alpha_vect=False,
                        save_name=args.out_name_root+"_tsne_"+x+"_block.png")

    ## Make some nice blocks. Can I do large 2x5 image blocks?
    nclusters = training_args.nclusters
    chunk_size = 10

    for i in range(0, nclusters, chunk_size):
        chunk = list(range(i, min(i + chunk_size, nclusters)))
        plot_multiplicity_matrix_grid(nom_processed, chunk, max_particles=5, ncols=2, nrows=5,
                                      save_name=args.out_name_root+"_npart_cluster"+str(chunk[0])+"-"+str(chunk[-1])+".png")
    
        
    ## Plot some examples for each cluster:
    ## Can I modify this to put 10 in a block, and just bunch them?
    ## TODO: modify, not sure how hard-coded image size is in these functions
    if args.example_cluster_images:
        for n in range(training_args.nclusters):
            plot_cluster_bigblock(data_dataset, nom_processed['clust_index'], n, 1, 10, \
                                  cluster_probs=nom_processed['clust_max'], \
                                  save_name=args.out_name_root+"_data_example"+str(n)+"_top.png")
            plot_cluster_bigblock(data_dataset, nom_processed['clust_index'], n, 1, 10, \
                                  save_name=args.out_name_root+"_data_example"+str(n)+"_all.png")

        
## Do the business
if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("nularbox_training_analysis")

    # Require an input file name and location to dump plots
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--out_name_root', type=str)
    parser.add_argument('--nom_data_dir', type=str)
    parser.add_argument('--alt_data_dir', type=str)
    
    ## Give a sensible default for the number of events to process
    parser.add_argument('--nnom', type=int, default=100000)
    parser.add_argument('--nalt', type=int, default=100000)

    ## Options for controlling the plots to make
    parser.add_argument('--example_cluster_images', type=int, choices=[0,1], default=0)
    
    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))
    
    run_analysis(args)
