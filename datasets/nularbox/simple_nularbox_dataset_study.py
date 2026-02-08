import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import coo_matrix
from glob import glob
from truth_labels import Topology, Mode


def clip_fill_array(array, value):
    if 0 <= value < len(array):
        array[value] += 1
    
def make_dataset_summary_plots(input_file_names, output_name_root="plots/"):
    
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

    ## Label histograms
    cc_arr        = np.zeros(2, dtype=int)
    nneutron_arr  = np.zeros(21, dtype=int)
    nproton_arr   = np.zeros(21, dtype=int)
    nantineut_arr = np.zeros(6, dtype=int)
    nantiprot_arr = np.zeros(6, dtype=int)    
    npipm_arr     = np.zeros(6, dtype=int)
    npi0_arr      = np.zeros(6, dtype=int)
    nkapm_arr     = np.zeros(6, dtype=int)
    nka0_arr      = np.zeros(6, dtype=int)
    nem_arr       = np.zeros(6, dtype=int)
    nmuon_arr     = np.zeros(6, dtype=int)
    nstrange_arr  = np.zeros(6, dtype=int)
    ncharm_arr    = np.zeros(6, dtype=int)

    enu_bin_edges = np.linspace(0, 50, 100)
    enu_arr,_     = np.histogram([], bins=enu_bin_edges)
    q0_bin_edges  = np.linspace(0, 50, 100)
    q0_arr,_      = np.histogram([], bins=q0_bin_edges)
    
    topo_values = [e.value for e in Topology]
    topo_min_val, topo_max_val = min(topo_values), max(topo_values)
    topo_num_bins = topo_max_val - topo_min_val + 1
    topo_arr = np.zeros(topo_num_bins, dtype=int)

    mode_values = [e.value for e in Mode]
    mode_min_val, mode_max_val = min(mode_values), max(mode_values)
    mode_num_bins = mode_max_val - mode_min_val + 1
    mode_arr = np.zeros(mode_num_bins, dtype=int)    
    
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

        these_enu = []
        these_q0 = []
        
        ## Loop over the images
        for i in range(nimages):

            if sum_images > max_images: break
            
            ## Make a dense array for ease of use
            group = f[str(i)]
            data = group['data'][:]
            if len(data) < 1:
                nEmpty += 1
                continue

            ## Sort out data histograms
            these_E   += list(data)
            these_N    .append(np.count_nonzero(data))
            these_sumE .append(np.sum(data))
            these_maxE .append(np.max(data))

            log_data = np.log10(1 + data)
            these_logE += list(log_data)
            these_sumLogE .append(np.sum(log_data))
            these_maxLogE .append(np.max(log_data))

            ## Sort out label histograms
            ## [('cc', '?'), ('topology', 'i1'), ('mode', 'i1'), ('nneutron', 'i1'), ('nantineut', 'i1'), ('nproton', 'i1'), ('nantiprot', 'i1'), ('npipm', 'i1'), ('npi0', 'i1'), ('nkapm', 'i1'), ('nka0', 'i1'), ('nem', 'i1'), ('nmuon', 'i1'), ('nstrange', 'i1'), ('ncharm', 'i1'), ('enu', '<f4'), ('q0', '<f4')]
            label = group['label'][()]
            cc_arr[int(label['cc'])] += 1
            clip_fill_array(nneutron_arr, label['nneutron'])
            clip_fill_array(nproton_arr, label['nproton'])
            clip_fill_array(nantineut_arr, label['nantineut'])
            clip_fill_array(nantiprot_arr, label['nantiprot'])
            clip_fill_array(npipm_arr, label['npipm'])
            clip_fill_array(npi0_arr, label['npi0'])
            clip_fill_array(nkapm_arr, label['nkapm'])    
            clip_fill_array(nka0_arr, label['nka0'])     
            clip_fill_array(nem_arr, label['nem'])      
            clip_fill_array(nmuon_arr, label['nmuon'])    
            clip_fill_array(nstrange_arr, label['nstrange'])
            clip_fill_array(ncharm_arr, label['ncharm'])               

            topo_arr[label['topology']-topo_min_val] += 1
            mode_arr[label['mode']-mode_min_val] += 1
            
            these_enu .append(label['enu'])
            these_q0  .append(label['q0'])
            
            ## Increment counter
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

        this_enu_arr,_ = np.histogram(these_enu, bins=enu_bin_edges)
        enu_arr += this_enu_arr

        this_q0_arr,_ = np.histogram(these_q0, bins=q0_bin_edges)
        q0_arr += this_q0_arr        
        
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
    plt.tight_layout()
    plt.savefig(output_name_root+"nhits_distribution_linx.png")
    plt.close()

    plt.hist(N_log_bin_edges[:-1], bins=N_log_bin_edges, weights=N_log_arr, log=True)
    plt.xlabel('N. hits')
    plt.xscale('log')
    plt.ylabel('N. events')
    plt.tight_layout()
    plt.savefig(output_name_root+"nhits_distribution_logx.png")
    plt.close()
    
    plt.hist(E_bin_edges[:-1], bins=E_bin_edges, weights=E_arr, log=True)
    plt.xlabel(r'Raw E (MeV)')
    plt.xscale('log')
    plt.ylabel('N. hits')
    plt.tight_layout()
    plt.savefig(output_name_root+"E_distribution.png")
    plt.close()

    plt.hist(LogE_bin_edges[:-1], bins=LogE_bin_edges, weights=LogE_arr, log=True)
    plt.xlabel(r'log$_{10}$(1 + E)')
    plt.ylabel('N. hits')
    plt.tight_layout()
    plt.savefig(output_name_root+"LogE_distribution.png")
    plt.close()
    
    plt.hist(E_bin_edges[:-1], bins=E_bin_edges, weights=maxE_arr, log=False)
    plt.xlabel(r'Max. raw E (MeV)')
    plt.ylabel('N. events')
    plt.tight_layout()
    plt.savefig(output_name_root+"maxE_distribution.png")
    plt.close()

    plt.hist(LogE_bin_edges[:-1], bins=LogE_bin_edges, weights=maxLogE_arr, log=False)
    plt.xlabel(r'Max. log$_{10}$(1 + E)')
    plt.ylabel('N. events')
    plt.tight_layout()
    plt.savefig(output_name_root+"maxLogE_distribution.png")
    plt.close()
    
    plt.hist(SumE_bin_edges[:-1], bins=SumE_bin_edges, weights=SumE_arr, log=True)
    plt.xlabel(r'$\sum$ raw E (MeV)')
    plt.xscale('linear')
    plt.ylabel('N. events')
    plt.tight_layout()
    plt.savefig(output_name_root+"sumE_distribution.png")
    plt.close()

    plt.hist(SumLogE_bin_edges[:-1], bins=SumLogE_bin_edges, weights=SumLogE_arr, log=True)
    plt.xlabel(r'$\sum$log$_{10}$(1 + E)')
    plt.xscale('log')
    plt.ylabel('N. events')
    plt.tight_layout()
    plt.savefig(output_name_root+"sumLogE_distribution.png")
    plt.close()

    ## Sort out label histograms
    plt.bar([0, 1], cc_arr, tick_label=['NC', 'CC'])
    plt.ylabel('N. events')
    plt.tight_layout()
    plt.savefig(output_name_root+"cc.png")
    plt.close()

    plt.bar(range(21), nneutron_arr)
    plt.ylabel('N. events')
    plt.xlabel('N. 2112')
    plt.tight_layout()
    plt.savefig(output_name_root+"nneutron.png")
    plt.close()

    plt.bar(range(21), nproton_arr)
    plt.ylabel('N. events')
    plt.xlabel('N. 2212')
    plt.tight_layout()
    plt.savefig(output_name_root+"nproton.png")
    plt.close()    
    
    plt.bar(range(6), nantineut_arr, log=True)
    plt.ylabel('N. events')
    plt.xlabel('N. -2112')
    plt.tight_layout()
    plt.savefig(output_name_root+"nantineut.png")
    plt.close()
    
    plt.bar(range(6), nantiprot_arr, log=True)
    plt.ylabel('N. events')
    plt.xlabel('N. -2212')
    plt.tight_layout()
    plt.savefig(output_name_root+"nantiprot.png")
    plt.close()
    
    plt.bar(range(6), npipm_arr, log=True)    
    plt.ylabel('N. events')
    plt.xlabel(r'N. $\pi^{\pm}$')
    plt.tight_layout()
    plt.savefig(output_name_root+"npipm.png")
    plt.close()
    
    plt.bar(range(6), npi0_arr, log=True)     
    plt.ylabel('N. events')
    plt.xlabel(r'N. $\pi^{0}$')
    plt.tight_layout()
    plt.savefig(output_name_root+"npi0.png")
    plt.close()
    
    plt.bar(range(6), nkapm_arr, log=True)    
    plt.ylabel('N. events')
    plt.xlabel(r'N. $K^{\pm}$')
    plt.tight_layout()
    plt.savefig(output_name_root+"nkapm.png")
    plt.close()
    
    plt.bar(range(6), nka0_arr, log=True)     
    plt.ylabel('N. events')
    plt.xlabel(r'N. $K^{0}$')
    plt.tight_layout()
    plt.savefig(output_name_root+"nka0.png")
    plt.close()
    
    plt.bar(range(6), nem_arr, log=True)      
    plt.ylabel('N. events')
    plt.xlabel('N. EM')
    plt.tight_layout()
    plt.savefig(output_name_root+"nem.png")
    plt.close()
    
    plt.bar(range(6), nmuon_arr, log=True)    
    plt.ylabel('N. events')
    plt.xlabel(r'N. $\mu^{\pm}$')
    plt.tight_layout()
    plt.savefig(output_name_root+"nmuon.png")
    plt.close()
    
    plt.bar(range(6), nstrange_arr, log=True) 
    plt.ylabel('N. events')
    plt.xlabel('N. Strange (not kaon)')
    plt.tight_layout()
    plt.savefig(output_name_root+"nstrange.png")
    plt.close()
    
    plt.bar(range(6), ncharm_arr, log=True)   
    plt.ylabel('N. events')
    plt.xlabel('N. Charm')
    plt.tight_layout()
    plt.savefig(output_name_root+"ncharm.png")
    plt.close()

    plt.hist(enu_bin_edges[:-1], bins=enu_bin_edges, weights=enu_arr, log=False)
    plt.xlabel(r'$E_{\nu}$ (GeV)')
    plt.ylabel('N. events')
    plt.tight_layout()
    plt.savefig(output_name_root+"enu.png")
    plt.close()

    plt.hist(q0_bin_edges[:-1], bins=q0_bin_edges, weights=q0_arr, log=False)
    plt.xlabel(r'$q_{0}$ (GeV)')
    plt.ylabel('N. events')
    plt.tight_layout()
    plt.savefig(output_name_root+"q0.png")
    plt.close()

    plt.bar([e.name for e in Topology], topo_arr, align='center', log=False)   
    plt.ylabel('N. events')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_name_root+"topology.png")
    plt.close()    
    
    plt.bar([e.name for e in Mode], mode_arr, align='center', log=False)   
    plt.ylabel('N. events')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_name_root+"mode.png")
    plt.close()

    print("Total", total_images, "images")
    print("Maximum number of hits:", maxN)
    print("Maximum sum of E:", maxSumE)
    print("Maximum sum of log E:", maxSumLogE)
    print("N. empty:", nEmpty)
    
    
if __name__ == '__main__':

    ## Take an input file and convert it to an h5 file of images
    if len(sys.argv) < 3:
        print("An input file and output file name must be provided as arguments!")
        sys.exit()

    input_file_names = sys.argv[1]
    output_name_root = sys.argv[2]
    make_dataset_summary_plots(input_file_names, output_name_root)
