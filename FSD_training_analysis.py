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
from ME_analysis_libs import plot_metric_data_vs_sim, plot_metric_by_cluster, plot_metric_by_confidence, plot_cluster_bigblock
from ME_analysis_libs import run_tsne_cuml, run_umap_cuml, run_tsne_skl

## Seeding
SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

## Various shared analysis libraries
from ME_analysis_libs import load_checkpoint, get_models_from_checkpoint, get_dataset, image_loop, reorder_clusters
    
def run_analysis(args):

    ## Setup the encoder
    encoder, heads, training_args = get_models_from_checkpoint(args.input_file)

    encoder.to(device)
    for h in heads.values(): h.to(device)
    
    ## Set up the datasets and loaders
    data_dataset, data_loader = get_dataset(training_args.data_dir, args.ndata)
    # sim_dataset, sim_loader = get_dataset(training_args.sim_dir, args.nsim)

    ## Temporarily overriding the sim_dir
    sim_dir = training_args.sim_dir
    if sim_dir == "/pscratch/sd/c/cwilk/FSD/SIMULATION":
        sim_dir = "/pscratch/sd/c/cwilk/FSD/SIMULATIONv2"
    sim_dataset, sim_loader = get_dataset(sim_dir, args.nsim)
    
    ## Get the processed vectors of interest from the datasets
    print("Loading inputs...")
    data_processed = image_loop(encoder, heads, data_loader)
    sim_processed = image_loop(encoder, heads, sim_loader)

    ## Do some magic to re-order the clusters for presentation purposes
    reorder_clusters(data_processed, sim_processed)
    print("...inputs loaded!")
    
    ## Make histograms
    plot_metric_data_vs_sim(data_processed['clust_index'], \
                            sim_processed['clust_index'], \
                            sim_processed['labels'],\
                            xtitle="Max. cluster index", \
                            save_name=args.out_name_root+"_clust_index.png")
    plot_metric_by_confidence(data_processed['clust_index'], 
                              data_processed['clust_max'],
                              xtitle="Max. cluster index",
                              save_name=args.out_name_root+"_clust_index_confidence_data.png")
    plot_metric_by_confidence(sim_processed['clust_index'],
                              sim_processed['clust_max'],
                              xtitle="Max. cluster index",
                              save_name=args.out_name_root+"_clust_index_confidence_sim.png")
    
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

    
    plot_metric_data_vs_sim(data_processed['yrange'],\
                            sim_processed['yrange'], \
                            sim_processed['labels'],\
                            nbinsx=80, x_min=0, x_max=800,\
                            xtitle="Range y", \
                            save_name=args.out_name_root+"_yrange.png")
    plot_metric_by_cluster(data_processed['yrange'],\
                           data_processed['clust_index'],\
                           nbinsx=80, x_min=0, x_max=800,\
                           xtitle="Range y", \
                           save_name=args.out_name_root+"_yrange_dataclust.png")
    plot_metric_by_cluster(sim_processed['yrange'],\
                           sim_processed['clust_index'],\
                           nbinsx=80, x_min=0, x_max=800,\
                           xtitle="Range y", \
                           save_name=args.out_name_root+"_yrange_simclust.png")   
    
    
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
    if args.tsne > 0:
        print("Starting tSNE...")
        ntsne=int(args.tsne)
        _ = run_tsne_skl(data_processed['latent'][:ntsne].copy(), \
                         data_processed['clust_index'][:ntsne].copy(), \
                         alpha_vect=data_processed['clust_max'][:ntsne].copy(), \
                         perp=150, exag=20, lr=500, \
                         save_name=args.out_name_root+"_tSNE_data.png")

        ## _ = run_tsne_skl(data_processed['enc_latent'][:ntsne].copy(), \
        ##                  data_processed['clust_index'][:ntsne].copy(), \
        ##                  alpha_vect=data_processed['clust_max'][:ntsne].copy(), \
        ##                  perp=150, exag=20, lr=500, \
        ##                  save_name=args.out_name_root+"_tSNE_ENC_data.png")
        
    ## UMAP
    #run_umap_cuml(data_processed['clust'], data_processed['clust_index'], alpha_vect=data_processed['clust_max'], \
    #              n_neighbors=100, min_distance=0.05, n_epochs=500,\
    #              save_name=args.out_name_root+"_UMAP_data.png")
    
    ## Plot some examples for each cluster:
    if args.example_cluster_images:
        for n in range(training_args.nclusters):
            plot_cluster_bigblock(data_dataset, data_processed['clust_index'], n, 1, 10, \
                                  cluster_probs=data_processed['clust_max'], \
                                  save_name=args.out_name_root+"_data_example"+str(n)+"_top.png")
            plot_cluster_bigblock(data_dataset, data_processed['clust_index'], n, 1, 10, \
                                  save_name=args.out_name_root+"_data_example"+str(n)+"_all.png")
            plot_cluster_bigblock(sim_dataset, sim_processed['clust_index'], n, 1, 10, \
                                  cluster_probs=sim_processed['clust_max'], \
                                  save_name=args.out_name_root+"_sim_example"+str(n)+"_top.png")
            plot_cluster_bigblock(sim_dataset, sim_processed['clust_index'], n, 1, 10, \
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
    parser.add_argument('--tsne', type=int, default=20000, nargs='?')
    
    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))
    
    run_analysis(args)
