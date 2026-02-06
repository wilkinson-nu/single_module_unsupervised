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
    maxSumE = 0
    total_images = 0
    maxE = 0
    ndodgy = 0
    maxLogE = 0
    maxSumLogE = 0
    nEmpty = 0
    
    ## Binning
    N_lin_bin_edges = np.linspace(0, 4000, 200)
    N_log_bin_edges = np.logspace(0, 3.7, 100)
    LogE_bin_edges = np.linspace(0, 2.5, 100)    
    SumLogE_bin_edges = np.logspace(0, 3, 100)
    E_bin_edges = np.logspace(0, 2.4, 125)
    SumE_bin_edges = np.linspace(0, 5000, 100)
    
    ## Bin counts for 1D plots
    E_arr,_ = np.histogram([], bins=E_bin_edges)
    LogE_arr,_ = np.histogram([], bins=LogE_bin_edges)
    SumE_arr,_ = np.histogram([], bins=SumE_bin_edges)    
    maxLogE_arr,_ = np.histogram([], bins=LogE_bin_edges)
    SumLogE_arr,_ = np.histogram([], bins=SumLogE_bin_edges)
    maxE_arr,_ = np.histogram([], bins=E_bin_edges)

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
        these_sumE = []
        these_maxE = []
        these_E = []
        these_logE = []
        these_sumLogE = []
        these_maxLogE = []
        
        ## Loop over the images
        for i in range(nimages):

            if sum_images > max_images: break
            
            ## Make a dense array for ease of use
            group = f[str(i)]
            data = group['data'][:]
            if len(data) < 1:
                nEmpty += 1
                continue
            
            these_E   += list(data)
            these_N    .append(np.count_nonzero(data))
            these_sumE .append(np.sum(data))
            these_maxE .append(np.max(data))

            log_data = np.log10(1 + data)
            these_logE += list(log_data)
            these_sumLogE .append(np.sum(log_data))
            these_maxLogE .append(np.max(log_data))
            
            sum_images += 1
            
        ## Now fill the histograms
        this_E_arr,_ = np.histogram(these_E, bins=E_bin_edges)
        E_arr += this_E_arr

        this_LogE_arr,_ = np.histogram(these_logE, bins=LogE_bin_edges)
        LogE_arr += this_LogE_arr
        
        this_maxE_arr,_ = np.histogram(these_maxE, bins=E_bin_edges)
        maxE_arr += this_maxE_arr       

        this_maxLogE_arr,_ = np.histogram(these_maxLogE, bins=LogE_bin_edges)
        maxLogE_arr += this_maxLogE_arr
        
        this_SumE_arr,_ = np.histogram(these_sumE, bins=SumE_bin_edges)
        SumE_arr += this_SumE_arr

        this_SumLogE_arr,_ = np.histogram(these_sumLogE, bins=SumLogE_bin_edges)
        SumLogE_arr += this_SumLogE_arr
        
        this_lin_N_arr,_ = np.histogram(these_N, bins=N_lin_bin_edges)
        N_lin_arr += this_lin_N_arr

        this_log_N_arr,_ = np.histogram(these_N, bins=N_log_bin_edges)
        N_log_arr += this_log_N_arr
        
        if max(these_N) > maxN:
            maxN = max(these_N)
        if max(these_sumE) > maxSumE:
            maxSumE = max(these_sumE)
        if max(these_sumLogE) > maxSumLogE:
            maxSumLogE = max(these_sumLogE)
            
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
    
    plt.hist(E_bin_edges[:-1], bins=E_bin_edges, weights=E_arr, log=True)
    plt.xlabel(r'Raw E (MeV)')
    plt.xscale('log')
    plt.ylabel('N. hits')
    plt.savefig(plotDir+"/E_distribution.png")
    plt.close()

    plt.hist(LogE_bin_edges[:-1], bins=LogE_bin_edges, weights=LogE_arr, log=True)
    plt.xlabel(r'log$_{10}$(1 + E)')
    plt.ylabel('N. hits')
    plt.savefig(plotDir+"/LogE_distribution.png")
    plt.close()
    
    plt.hist(E_bin_edges[:-1], bins=E_bin_edges, weights=maxE_arr, log=False)
    plt.xlabel(r'Max. raw E (MeV)')
    plt.ylabel('N. events')
    plt.savefig(plotDir+"/maxE_distribution.png")
    plt.close()

    plt.hist(LogE_bin_edges[:-1], bins=LogE_bin_edges, weights=maxLogE_arr, log=False)
    plt.xlabel(r'Max. log$_{10}$(1 + E)')
    plt.ylabel('N. events')
    plt.savefig(plotDir+"/maxLogE_distribution.png")
    plt.close()
    
    plt.hist(SumE_bin_edges[:-1], bins=SumE_bin_edges, weights=SumE_arr, log=True)
    plt.xlabel(r'$\sum$ raw E (MeV)')
    plt.xscale('linear')
    plt.ylabel('N. events')
    plt.savefig(plotDir+"/sumE_distribution.png")
    plt.close()

    plt.hist(SumLogE_bin_edges[:-1], bins=SumLogE_bin_edges, weights=SumLogE_arr, log=True)
    plt.xlabel(r'$\sum$log$_{10}$(1 + E)')
    plt.xscale('log')
    plt.ylabel('N. events')
    plt.savefig(plotDir+"/sumLogE_distribution.png")
    plt.close()
    
    print("Total", total_images, "images")
    print("Maximum number of hits:", maxN)
    print("Maximum sum of E:", maxSumE)
    print("Maximum sum of log E:", maxSumLogE)
    print("N. empty:", nEmpty)
    
    
if __name__ == '__main__':

    ## Take an input file and convert it to an h5 file of images
    if len(sys.argv) < 2:
        print("An input file and output file name must be provided as arguments!")
        sys.exit()

    input_file_names = sys.argv[1]
    make_dataset_summary_plots(input_file_names)
