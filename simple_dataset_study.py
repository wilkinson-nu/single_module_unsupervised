import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import coo_matrix
from glob import glob

# plt.rcParams['text.usetex'] = True

def get_n_images(input_file_names):

    images = 0
    for file in glob(input_file_names):
        f = h5py.File(file, 'r', libver='latest')
        images += f.attrs['N']
        f.close()

    print("Found a total of", images, "images")


def make_dataset_summary_plots(input_file_names, plotDir="plots_fixdupes_pluscuts"):

    max_images = 1e6
    sum_images = 0
    
    ## Get some high-level summary information
    maxN = 0
    maxSumQ = 0
    total_images = 0

    ndodgy = 0
    
    ## Binning
    N_lin_bin_edges = np.linspace(0, 4000, 200)
    N_log_bin_edges = np.logspace(0, 3.7, 100)
    Q_bin_edges = np.linspace(0, 5, 100)    
    SumQ_bin_edges = np.logspace(2, 4, 100)
    
    ## Bin counts for 1D plots
    Q_arr,_ = np.histogram([], bins=Q_bin_edges)
    SumQ_arr,_ = np.histogram([], bins=SumQ_bin_edges)    
    maxQ_arr,_ = np.histogram([], bins=Q_bin_edges)
    N_lin_arr,_ = np.histogram([], bins=N_lin_bin_edges)
    N_log_arr,_ = np.histogram([], bins=N_log_bin_edges)
    
    ## Loop over all of the files
    for file in glob(input_file_names):

        if sum_images > max_images: break

        print("Reading", file)
        f = h5py.File(file, 'r', libver='latest')

        nimages = f.attrs['N']
        print("Found", nimages, "images")

        total_images += nimages
        
        these_N = []
        these_sumQ = []
        these_maxQ = []
        these_Q = []
        
        ## Loop over the images
        for i in range(nimages):

            if sum_images > max_images: break
            
            ## Make a dense array for ease of use
            group = f[str(i)]
            data = group['data'][:]
            these_Q   += list(data)
            these_N    .append(np.count_nonzero(data))
            these_sumQ .append(np.sum(data))
            these_maxQ .append(np.max(data))
        
            if np.sum(data) > 1e5 or \
               np.sum(data) < 1e3 or \
               np.count_nonzero(data) < 200:
                ndodgy += 1
            ## If np.sum(data) > 1e5, kind of dodgy, whole tile triggering
            ## If np.sum(data) < 1e3, very dull event, not worth throwing in
            ## If np.count_nonzero(data) < 200: also very dull
            if np.sum(data) > 1e5:
                print("Found event with sumQ =", np.sum(data))
            # if np.sum(data) < 1e3:
            #    ## Show image temporarily
            #     this_dense = input_list[i].toarray()
            #     plt.imshow(this_dense, origin='lower')
            #     plt.show() #block=False)
            #     # input("Continue...")

            sum_images += 1
            
        ## Now fill the histograms
        this_Q_arr,_ = np.histogram(these_Q, bins=Q_bin_edges)
        Q_arr += this_Q_arr
        
        this_maxQ_arr,_ = np.histogram(these_maxQ, bins=Q_bin_edges)
        maxQ_arr += this_maxQ_arr       
        
        this_SumQ_arr,_ = np.histogram(these_sumQ, bins=SumQ_bin_edges)
        SumQ_arr += this_SumQ_arr
        
        this_lin_N_arr,_ = np.histogram(these_N, bins=N_lin_bin_edges)
        N_lin_arr += this_lin_N_arr

        this_log_N_arr,_ = np.histogram(these_N, bins=N_log_bin_edges)
        N_log_arr += this_log_N_arr
        
        if max(these_N) > maxN:
            maxN = max(these_N)
        if max(these_sumQ) > maxSumQ:
            maxSumQ = max(these_sumQ)

        ## End of this file
        f.close()
        
    ## Draw the final histograms
    plt.hist(N_lin_bin_edges[:-1], bins=N_lin_bin_edges, weights=N_lin_arr, log=True)
    plt.xlabel('N. hits')
    plt.ylabel('N. events')
    plt.savefig(plotDir+"/nhits_distribution_linx.png")
    plt.close()

    plt.hist(N_log_bin_edges[:-1], bins=N_log_bin_edges, weights=N_log_arr, log=True)
    plt.xlabel('N. hits')
    plt.ylabel('N. events')
    plt.savefig(plotDir+"/nhits_distribution_logx.png")
    plt.close()
    
    plt.hist(Q_bin_edges[:-1], bins=Q_bin_edges, weights=Q_arr, log=False)
    plt.xlabel(r'log$_{10}$(1 + Q)')
    plt.ylabel('N. hits')
    plt.savefig(plotDir+"/Q_distribution.png")
    plt.close()
    
    plt.hist(Q_bin_edges[:-1], bins=Q_bin_edges, weights=maxQ_arr, log=False)
    plt.xlabel(r'Max. log$_{10}$(1 + Q)')
    plt.ylabel('N. events')
    plt.savefig(plotDir+"/maxQ_distribution.png")
    plt.close()
    
    plt.hist(SumQ_bin_edges[:-1], bins=SumQ_bin_edges, weights=SumQ_arr, log=True)
    plt.xlabel(r'$\sum$log$_{10}$(1 + Q)')
    plt.xscale('log')
    plt.ylabel('N. events')
    plt.savefig(plotDir+"/sumQ_distribution.png")
    plt.close()
    
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
    get_n_images(input_file_names)
    # make_dataset_summary_plots(input_file_names)
