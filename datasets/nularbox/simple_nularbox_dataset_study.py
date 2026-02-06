import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import coo_matrix
from glob import glob

def make_dataset_summary_plots(input_file_names, plotDir="plots"):

    max_images = 1e7
    sum_images = 0
    
    ## Get some high-level summary information
    maxN = 0
    maxSumQ = 0
    total_images = 0
    maxQ = 0
    ndodgy = 0
    maxLogQ = 0
    maxSumLogQ = 0
    nEmpty = 0
    
    ## Binning
    N_lin_bin_edges = np.linspace(0, 4000, 200)
    N_log_bin_edges = np.logspace(0, 3.7, 100)
    LogQ_bin_edges = np.linspace(0, 2, 100)    
    SumLogQ_bin_edges = np.logspace(0, 3, 100)
    Q_bin_edges = np.linspace(0, 2, 100)
    SumQ_bin_edges = np.linspace(0, 500, 100)
    
    ## Bin counts for 1D plots
    Q_arr,_ = np.histogram([], bins=Q_bin_edges)
    LogQ_arr,_ = np.histogram([], bins=LogQ_bin_edges)
    SumQ_arr,_ = np.histogram([], bins=SumQ_bin_edges)    
    maxLogQ_arr,_ = np.histogram([], bins=LogQ_bin_edges)
    SumLogQ_arr,_ = np.histogram([], bins=SumLogQ_bin_edges)
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
        these_logQ = []
        these_sumLogQ = []
        these_maxLogQ = []
        
        ## Loop over the images
        for i in range(nimages):

            if sum_images > max_images: break
            
            ## Make a dense array for ease of use
            group = f[str(i)]
            data = group['data'][:]
            if len(data) < 1:
                nEmpty += 1
                continue
            
            these_Q   += list(data)
            these_N    .append(np.count_nonzero(data))
            these_sumQ .append(np.sum(data))
            these_maxQ .append(np.max(data))

            log_data = np.log10(1 + data)
            these_logQ += list(log_data)
            these_sumLogQ .append(np.sum(log_data))
            these_maxLogQ .append(np.max(log_data))
            
            sum_images += 1
            
        ## Now fill the histograms
        this_Q_arr,_ = np.histogram(these_Q, bins=Q_bin_edges)
        Q_arr += this_Q_arr

        this_LogQ_arr,_ = np.histogram(these_logQ, bins=LogQ_bin_edges)
        LogQ_arr += this_LogQ_arr
        
        this_maxQ_arr,_ = np.histogram(these_maxQ, bins=Q_bin_edges)
        maxQ_arr += this_maxQ_arr       

        this_maxLogQ_arr,_ = np.histogram(these_maxLogQ, bins=LogQ_bin_edges)
        maxLogQ_arr += this_maxLogQ_arr
        
        this_SumQ_arr,_ = np.histogram(these_sumQ, bins=SumQ_bin_edges)
        SumQ_arr += this_SumQ_arr

        this_SumLogQ_arr,_ = np.histogram(these_sumLogQ, bins=SumLogQ_bin_edges)
        SumLogQ_arr += this_SumLogQ_arr
        
        this_lin_N_arr,_ = np.histogram(these_N, bins=N_lin_bin_edges)
        N_lin_arr += this_lin_N_arr

        this_log_N_arr,_ = np.histogram(these_N, bins=N_log_bin_edges)
        N_log_arr += this_log_N_arr
        
        if max(these_N) > maxN:
            maxN = max(these_N)
        if max(these_sumQ) > maxSumQ:
            maxSumQ = max(these_sumQ)
        if max(these_sumLogQ) > maxSumLogQ:
            maxSumLogQ = max(these_sumLogQ)
            
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
    plt.xscale('log')
    plt.ylabel('N. events')
    plt.savefig(plotDir+"/nhits_distribution_logx.png")
    plt.close()
    
    plt.hist(Q_bin_edges[:-1], bins=Q_bin_edges, weights=Q_arr, log=False)
    plt.xlabel(r'Raw Q')
    plt.ylabel('N. hits')
    plt.savefig(plotDir+"/Q_distribution.png")
    plt.close()

    plt.hist(LogQ_bin_edges[:-1], bins=LogQ_bin_edges, weights=LogQ_arr, log=False)
    plt.xlabel(r'log$_{10}$(1 + Q)')
    plt.ylabel('N. hits')
    plt.savefig(plotDir+"/LogQ_distribution.png")
    plt.close()
    
    plt.hist(Q_bin_edges[:-1], bins=Q_bin_edges, weights=maxQ_arr, log=False)
    plt.xlabel(r'Max. raw Q')
    plt.ylabel('N. events')
    plt.savefig(plotDir+"/maxQ_distribution.png")
    plt.close()

    plt.hist(LogQ_bin_edges[:-1], bins=LogQ_bin_edges, weights=maxLogQ_arr, log=False)
    plt.xlabel(r'Max. log$_{10}$(1 + Q)')
    plt.ylabel('N. events')
    plt.savefig(plotDir+"/maxLogQ_distribution.png")
    plt.close()
    
    plt.hist(SumQ_bin_edges[:-1], bins=SumQ_bin_edges, weights=SumQ_arr, log=True)
    plt.xlabel(r'$\sum$ raw Q')
    plt.xscale('linear')
    plt.ylabel('N. events')
    plt.savefig(plotDir+"/sumQ_distribution.png")
    plt.close()

    plt.hist(SumLogQ_bin_edges[:-1], bins=SumLogQ_bin_edges, weights=SumLogQ_arr, log=True)
    plt.xlabel(r'$\sum$log$_{10}$(1 + Q)')
    plt.xscale('log')
    plt.ylabel('N. events')
    plt.savefig(plotDir+"/sumLogQ_distribution.png")
    plt.close()
    
    print("Total", total_images, "images")
    print("Maximum number of hits:", maxN)
    print("Maximum sum of charge:", maxSumQ)
    print("Maximum sum of log charge:", maxSumLogQ)
    print("N. empty:", nEmpty)
    
    
if __name__ == '__main__':

    ## Take an input file and convert it to an h5 file of images
    if len(sys.argv) < 2:
        print("An input file and output file name must be provided as arguments!")
        sys.exit()

    input_file_names = sys.argv[1]
    make_dataset_summary_plots(input_file_names)
