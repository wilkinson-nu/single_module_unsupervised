import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import coo_matrix
import joblib
from glob import glob

def make_dataset_summary_plots(input_file_names):

    ## Get some high-level summary information
    maxN = 0
    maxSumQ = 0
    total_images = 0

    ndodgy = 0
    
    ## Binning
    N_bin_edges = np.linspace(0, 5000, 500) 
    lowN_bin_edges = np.linspace(0, 200, 100) 
    Q_bin_edges = np.linspace(0, 200, 200)    
    SumQ_bin_edges = np.logspace(1, 6, 100)
    
    ## Bin counts for 1D plots
    Q_arr,_ = np.histogram([], bins=Q_bin_edges)
    SumQ_arr,_ = np.histogram([], bins=SumQ_bin_edges)    
    maxQ_arr,_ = np.histogram([], bins=Q_bin_edges)
    N_arr,_ = np.histogram([], bins=N_bin_edges)
    lowN_arr,_ = np.histogram([], bins=lowN_bin_edges)

    ## Now for a 2D plot
    N_vs_sumQ_arr,_,_ = np.histogram2d([], [], bins=(lowN_bin_edges,SumQ_bin_edges))

    
    ## Loop over all of the files
    for f in glob(input_file_names):
        input_list = joblib.load(f)

        nimages = len(input_list)
        print("Found", nimages, "images in", f)

        total_images += nimages
        
        these_N = []
        these_sumQ = []
        these_maxQ = []
        these_Q = []
        
        ## Loop over the images
        for i in range(nimages):

            ## Make a dense array for ease of use
            this_image = input_list[i].data
            these_Q    += list(this_image)
            these_N    .append(np.count_nonzero(this_image))
            these_sumQ .append(np.sum(this_image))
            these_maxQ .append(np.max(this_image))

            if np.sum(this_image) > 1e5 or \
               np.sum(this_image) < 1e3 or \
               np.count_nonzero(this_image) < 200:
                ndodgy += 1
            ## If np.sum(this_image) > 1e5, kind of dodgy, whole tile triggering
            ## If np.sum(this_image) < 1e3, very dull event, not worth throwing in
            ## If np.count_nonzero(this_image) < 200: also very dull
            # if np.sum(this_image) < 1e3:
            #if np.count_nonzero(this_image) > 1000:
            #    ## Show image temporarily
            #     this_dense = input_list[i].toarray()
            #     plt.imshow(this_dense, origin='lower')
            #     plt.show() #block=False)
            #     # input("Continue...")
            
        ## Now fill the histograms
        this_Q_arr,_ = np.histogram(these_Q, bins=Q_bin_edges)
        Q_arr += this_Q_arr

        this_maxQ_arr,_ = np.histogram(these_maxQ, bins=Q_bin_edges)
        maxQ_arr += this_maxQ_arr       
        
        this_SumQ_arr,_ = np.histogram(these_sumQ, bins=SumQ_bin_edges)
        SumQ_arr += this_SumQ_arr

        this_N_arr,_ = np.histogram(these_N, bins=N_bin_edges)
        N_arr += this_N_arr

        this_lowN_arr,_ = np.histogram(these_N, bins=lowN_bin_edges)
        lowN_arr += this_lowN_arr

        this_N_vs_sumQ_arr,_,_ = np.histogram2d(these_N, these_sumQ, bins=(lowN_bin_edges,SumQ_bin_edges))
        N_vs_sumQ_arr += this_N_vs_sumQ_arr

        if max(these_N) > maxN:
            maxN = max(these_N)
        if max(these_sumQ) > maxSumQ:
            maxSumQ = max(these_sumQ)
        
    ## Draw the final histograms
    plt.hist(N_bin_edges[:-1], bins=N_bin_edges, weights=N_arr, log=True)
    plt.xlabel('N. hits')
    plt.ylabel('N. events')
    plt.savefig("nhits_distribution_logy.png")
    plt.close()
    
    plt.hist(lowN_bin_edges[:-1], bins=lowN_bin_edges, weights=lowN_arr, log=False)
    plt.xlabel('N. hits')
    plt.ylabel('N. events')
    plt.savefig("nhits_low_distribution.png")    
    plt.close()
    
    plt.hist(Q_bin_edges[:-1], bins=Q_bin_edges, weights=Q_arr, log=False)
    plt.xlabel('Q')
    plt.ylabel('N. hits')
    plt.savefig("Q_distribution.png")
    plt.close()
    
    plt.hist(Q_bin_edges[:-1], bins=Q_bin_edges, weights=maxQ_arr, log=False)
    plt.xlabel('Max. Q')
    plt.ylabel('N. events')
    plt.savefig("maxQ_distribution.png")
    plt.close()
    
    plt.hist(SumQ_bin_edges[:-1], bins=SumQ_bin_edges, weights=SumQ_arr, log=True)
    plt.xlabel('Sum Q')
    plt.xscale('log')
    plt.ylabel('N. events')
    plt.savefig("sumQ_distribution.png")
    plt.close()
    
    ## plt.figure()
    ## plt.imshow(N_vs_sumQ_arr, interpolation='nearest', origin='lower', norm='log') #, extent=[N_bin_edges[0], N_bin_edges[-1], SumQ_bin_edges[0], SumQ_bin_edges[-1]])
    ## plt.xlabel('N. hits')
    ## plt.ylabel('Sum Q')
    ## plt.colorbar(label='N. events')
    ## plt.show()

    print("Total", total_images, "images")
    print("Maximum number of hits:", maxN)
    print("Maximum sum of charge:", maxSumQ)
    print("Found", total_images - ndodgy, "good events")
    
    
if __name__ == '__main__':

    ## Take an input file and convert it to an h5 file of images
    if len(sys.argv) < 2:
        print("An input file and output file name must be provided as arguments!")
        sys.exit()

    input_file_names = sys.argv[1]
    make_dataset_summary_plots(input_file_names)
