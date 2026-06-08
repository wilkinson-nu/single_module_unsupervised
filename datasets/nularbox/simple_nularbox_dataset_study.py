import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import coo_matrix
from glob import glob
from truth_labels import Topology, Mode
from enum import Enum

## Damn I miss ROOT
class TH1Dish:
    def __init__(self, bin_edges, dtype=np.int64):
        self.bin_edges = np.asarray(bin_edges, dtype=float)
        self.counts = np.zeros(len(self.bin_edges) - 1, dtype=dtype)

        ## Keep track of the min/max
        self.min_seen = None
        self.max_seen = None

        ## Add a buffer 
        self.buffer = []

    def Fill(self, values):
        arr = np.atleast_1d(np.asarray(values))
        self.buffer.append(arr)

    def FlushBuffer(self):
        
        if not self.buffer: return
        values = np.concatenate(self.buffer)
        self.buffer.clear()
        
        if values.size:
            vmin = values.min()
            vmax = values.max()

            self.min_seen = vmin if self.min_seen is None else min(self.min_seen, vmin)
            self.max_seen = vmax if self.max_seen is None else max(self.max_seen, vmax)

        hist, _ = np.histogram(values, bins=self.bin_edges)
        self.counts += hist
        
    def Reset(self):
        self.counts.fill(0)
        self.buffer.clear()
        self.min_seen = None
        self.max_seen = None
        
    def GetArray(self):
        return self.counts.copy(), self.bin_edges.copy()

    def GetMinMax(self):
        return self.min_seen, self.max_seen

    def Draw(self, filename=None, xtitle=None, ytitle="N. Entries",
        logx=False, logy=False, show=False):

        ## Flush to make sure nothing is cached
        self.FlushBuffer()
        
        plt.hist(self.bin_edges[:-1],
                 bins=self.bin_edges,
                 weights=self.counts)

        if logx: plt.xscale("log")
        if logy: plt.yscale("log")
        if xtitle: plt.xlabel(xtitle)
        if ytitle: plt.ylabel(ytitle)
        plt.tight_layout()

        if filename:
            plt.savefig(filename)
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()

            
class TH1Iish:
    def __init__(self, nbins, dtype=np.int64):
        self.nbins = int(nbins)
        if self.nbins <= 0:
            raise ValueError("nbins must be positive")

        self.counts = np.zeros(self.nbins, dtype=dtype)

        ## Keep track of the min/max
        self.min_seen = None
        self.max_seen = None

        ## Add a buffer 
        self.buffer = []
        
    def Fill(self, values):
        arr = np.atleast_1d(np.asarray(values))
        self.buffer.append(arr)

    def FlushBuffer(self):

        if not self.buffer: return
        values = np.concatenate(self.buffer)
        self.buffer.clear()
        
        if values.size:
            vmin = values.min()
            vmax = values.max()

            self.min_seen = vmin if self.min_seen is None else min(self.min_seen, vmin)
            self.max_seen = vmax if self.max_seen is None else max(self.max_seen, vmax)

        ## Check there are values to put inside the histogram range
        mask = (values >= 0) & (values < self.nbins)
        if not np.any(mask): return

        idx, cnt = np.unique(values[mask], return_counts=True)
        self.counts[idx] += cnt
        
    def Reset(self):
        self.counts.fill(0)
        self.buffer.clear()
        self.min_seen = None
        self.max_seen = None
        
    def GetArray(self):
        return self.counts.copy(), self.bin_edges.copy()

    def GetMinMax(self):
        return self.min_seen, self.max_seen

    def Draw(self, filename=None, xlabels=None, xtitle=None, ytitle="N. Entries",
             logy=False, show=False, rotate_labels=0):

        ## Flush to make sure nothing is cached
        self.FlushBuffer()
        
        x = np.arange(self.nbins)
        plt.bar(x, self.counts, align="center", width=0.8)

        if xlabels is not None:
            if len(xlabels) != self.nbins:
                raise ValueError("labels length must match number of bins")
            
            plt.xticks(x, xlabels, rotation=rotate_labels)
        else:
            plt.xticks(x)
        
        if logy: plt.yscale("log")

        if xtitle: plt.xlabel(xtitle)
        if ytitle: plt.ylabel(ytitle)
        plt.tight_layout()

        if filename:
            plt.savefig(filename)
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()            


class TH1Enum:
    def __init__(self, enum_class):

        if not issubclass(enum_class, Enum):
            raise TypeError("enum_class must be an Enum")

        self.enum_class = enum_class
        self.bin_labels = [e.name for e in enum_class]

        values = np.array([e.value for e in enum_class], dtype=int)
        self.min_val = values.min()
        self.max_val = values.max()
        
        self.nbins = len(enum_class)
        self.counts = np.zeros(self.nbins, dtype=int)

        ## Add a buffer 
        self.buffer = []

    def Fill(self, values):

        arr = np.atleast_1d(np.asarray(values))

        ## Turn the enum to an int if it is passed as an enum
        if issubclass(arr.dtype.type, Enum):
            arr = np.array([v.value for v in arr], dtype=int)
        else:
            arr = arr.astype(int, copy=False)
            
        self.buffer.append(arr)

    ## TODO
    def FlushBuffer(self):

        if not self.buffer: return
        values = np.concatenate(self.buffer)
        self.buffer.clear()

        ## Map values to the enum
        idx = values - self.min_val
        idx, cnt = np.unique(idx, return_counts=True)
        self.counts[idx] += cnt
        
    def Reset(self):
        self.counts.fill(0)
        self.buffer.clear()
        
    def GetArray(self):
        return self.counts.copy(), self.bin_labels.copy()

    ## Not implemented, because who cares?
    def GetMinMax(self):
        return None, None

    def Draw(self, filename=None, xtitle=None, ytitle="N. Entries",
             logy=False, show=False, rotate_labels=90):

        ## Flush to make sure nothing is cached
        self.FlushBuffer()
        
        x = np.arange(self.nbins)
        plt.bar(x, self.counts, align="center", width=0.8)
        plt.xticks(x, self.bin_labels, rotation=rotate_labels)
        
        if logy: plt.yscale("log")

        if xtitle: plt.xlabel(xtitle)
        if ytitle: plt.ylabel(ytitle)
        plt.tight_layout()

        if filename:
            plt.savefig(filename)
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()            


def make_dataset_summary_plots(input_file_names, output_name_root="plots/"):
    
    max_images = 1e7
    sum_images = 0
    
    ## Get some high-level summary information
    total_images = 0
    nEmpty = 0
    
    ## Setup histograms
    nhits_lin_hist = TH1Dish(np.linspace(0, 4000, 200))
    nhits_log_hist = TH1Dish(np.logspace(0, 3.7, 100))
    E_hist         = TH1Dish(np.logspace(-1, 2.4, 125))
    SumE_hist      = TH1Dish(np.linspace(0, 5000, 100))
    maxE_hist      = TH1Dish(np.logspace(-1, 2.4, 125))
    enu_hist       = TH1Dish(np.linspace(0, 50, 100))
    q0_hist        = TH1Dish(np.linspace(0, 50, 100))

    ## Investigate the distribution of hit positions
    row_hist       = TH1Dish(np.linspace(0, 511, 256))
    col_hist       = TH1Dish(np.linspace(0, 511, 256))
    
    ## Test out a range of transforms
    alpha_min=1
    alpha_max=10
    alpha_hist_list = [TH1Dish(np.linspace(0, 5, 100)) for x in range(alpha_min, alpha_max+1)]

    ## Label histograms
    cc_hist        = TH1Iish(nbins=2)
    nneutron_hist  = TH1Iish(nbins=21)
    nproton_hist   = TH1Iish(nbins=21)
    nantineut_hist = TH1Iish(nbins=6)
    nantiprot_hist = TH1Iish(nbins=6)    
    npipm_hist     = TH1Iish(nbins=6)
    npi0_hist      = TH1Iish(nbins=6)
    nkapm_hist     = TH1Iish(nbins=6)
    nka0_hist      = TH1Iish(nbins=6)
    nem_hist       = TH1Iish(nbins=6)
    nmuon_hist     = TH1Iish(nbins=6)
    nstrange_hist  = TH1Iish(nbins=6)
    ncharm_hist    = TH1Iish(nbins=6)
    ndeuteron_hist = TH1Iish(nbins=6)
    ntritium_hist  = TH1Iish(nbins=6)
    nalpha_hist    = TH1Iish(nbins=6)
    nhelium3_hist  = TH1Iish(nbins=6)
    nnuclfrag_hist = TH1Iish(nbins=6)

    ## Special hists
    ncluster_hist  = TH1Iish(nbins=11)
    ncharged_hist  = TH1Iish(nbins=21)
    
    ## Special enum histograms
    topo_hist      = TH1Enum(Topology)
    mode_hist      = TH1Enum(Mode)
    
    ## Loop over all of the files
    for file in glob(input_file_names):

        if sum_images > max_images: break

        print("Reading", file)
        f = h5py.File(file, 'r', libver='latest')

        nimages = f.attrs['N']
        print("Found", nimages, "images")

        total_images += nimages

        ## Loop over the images
        for i in range(nimages):

            if sum_images > max_images: break
            
            ## Make a dense array for ease of use
            group = f[str(i)]
            data = group['data'][:]
            row  = group['row'][:]
            col  = group['col'][:]
            if len(data) < 1:
                nEmpty += 1
                continue

            ## Fill histograms
            E_hist         .Fill(data)
            nhits_lin_hist .Fill(np.count_nonzero(data))
            nhits_log_hist .Fill(np.count_nonzero(data))
            SumE_hist      .Fill(np.sum(data))
            maxE_hist      .Fill(np.max(data))

            ## Fill position histograms
            row_hist       .Fill(row)
            col_hist       .Fill(col)
            
            ## Alpha histograms
            for a in range(alpha_min, alpha_max+1):
                a_data = np.log10(1 + a*data)/np.log10(1+a)
                alpha_hist_list[a-alpha_min] .Fill(a_data)
            
            ## Sort out label histograms
            label = group['label'][()]
            cc_hist       .Fill(int(label['cc']))            
            nneutron_hist .Fill(label['nneutron'])
            nproton_hist  .Fill(label['nproton'])
            nantineut_hist.Fill(label['nantineut'])
            nantiprot_hist.Fill(label['nantiprot'])
            npipm_hist    .Fill(label['npipm'])
            npi0_hist     .Fill(label['npi0'])
            nkapm_hist    .Fill(label['nkapm'])    
            nka0_hist     .Fill(label['nka0'])     
            nem_hist      .Fill(label['nem'])      
            nmuon_hist    .Fill(label['nmuon'])    
            nstrange_hist .Fill(label['nstrange'])
            ncharm_hist   .Fill(label['ncharm'])               
            ndeuteron_hist.Fill(label['ndeuteron'])
            ntritium_hist .Fill(label['ntritium'])
            nalpha_hist   .Fill(label['nalpha'])
            nhelium3_hist .Fill(label['nhelium3'])
            nnuclfrag_hist.Fill(label['nnuclfrag'])
            topo_hist     .Fill(label['topology'])
            mode_hist     .Fill(label['mode'])
            enu_hist      .Fill(label['enu'])
            q0_hist       .Fill(label['q0'])

            ncharged = label['nproton'] + label['npipm'] + label['nkapm']
            ncluster = label['ndeuteron'] + label['nalpha'] + label['nhelium3'] + label['ntritium'] + label['nnuclfrag']

            ncharged_hist .Fill(ncharged)
            ncluster_hist .Fill(ncluster)
            
            ## Increment counter
            sum_images += 1
            
        ## Flush the histograms which get filled per hit
        E_hist        .FlushBuffer()
        row_hist      .FlushBuffer()
        col_hist      .FlushBuffer()
        for a in range(alpha_min, alpha_max+1): alpha_hist_list[a-alpha_min] .FlushBuffer()

        ## End of this file
        f.close()
        
    ## Draw the final histograms
    nhits_lin_hist .Draw(output_name_root+"nhits_distribution_linx.png", xtitle='N. hits', logy=True)
    nhits_log_hist .Draw(output_name_root+"nhits_distribution_logx.png", xtitle='N. hits', logx=True, logy=True)
    E_hist         .Draw(output_name_root+"E_distribution.png", xtitle=r'Raw E (MeV)', logx=True, logy=True)
    SumE_hist      .Draw(output_name_root+"sumE_distribution.png", xtitle=r'$\sum$ raw E (MeV)', logy=True)  
    maxE_hist      .Draw(output_name_root+"maxE_distribution.png", xtitle=r'Max. raw E (MeV)')
    enu_hist       .Draw(output_name_root+"enu.png", xtitle=r'$E_{\nu}$ (GeV)')   
    q0_hist        .Draw(output_name_root+"q0.png", xtitle=r'$q_{0}$ (GeV)')

    row_hist       .Draw(output_name_root+"row_logy.png", xtitle='Row coord.', logy=True)
    col_hist       .Draw(output_name_root+"col_logy.png", xtitle='Column coord.', logy=True)
    row_hist       .Draw(output_name_root+"row_liny.png", xtitle='Row coord.', logy=False)
    col_hist       .Draw(output_name_root+"col_liny.png", xtitle='Column coord.', logy=False)
    
    for a in range(alpha_min, alpha_max+1):
        alpha_hist_list[a-alpha_min] .Draw(output_name_root+"LogAlphaE"+str(a)+"_lin_distribution.png", xtitle=r'log$_{10}$(1 + '+str(a)+'E)/log$_{10}$(1 + '+str(a)+')', logy=False)
        alpha_hist_list[a-alpha_min] .Draw(output_name_root+"LogAlphaE"+str(a)+"_log_distribution.png", xtitle=r'log$_{10}$(1 + '+str(a)+'E)/log$_{10}$(1 + '+str(a)+')', logy=True)       
    
    ## Sort out label histograms
    cc_hist        .Draw(output_name_root+"cc.png", xlabels=['NC', 'CC'])
    nneutron_hist  .Draw(output_name_root+"nneutron.png", xtitle='N. 2112')
    nproton_hist   .Draw(output_name_root+"nproton.png", xtitle='N. 2212')
    nantineut_hist .Draw(output_name_root+"nantineut.png", xtitle='N. -2112', logy=True)
    nantiprot_hist .Draw(output_name_root+"nantiprot.png", xtitle='N. -2212', logy=True)
    npipm_hist     .Draw(output_name_root+"npipm.png", xtitle=r'N. $\pi^{\pm}$', logy=True)
    npi0_hist      .Draw(output_name_root+"npi0.png", xtitle=r'N. $\pi^{0}$', logy=True)
    nkapm_hist     .Draw(output_name_root+"nkapm.png", xtitle=r'N. $K^{\pm}$', logy=True)
    nka0_hist      .Draw(output_name_root+"nka0.png", xtitle=r'N. $K^{0}$', logy=True)    
    nem_hist       .Draw(output_name_root+"nem.png", xtitle='N. EM', logy=True)
    nmuon_hist     .Draw(output_name_root+"nmuon.png", xtitle=r'N. $\mu^{\pm}$', logy=True)
    nstrange_hist  .Draw(output_name_root+"nstrange.png", xtitle='N. Strange (not kaon)', logy=True)
    ncharm_hist    .Draw(output_name_root+"ncharm.png", xtitle='N. Charm', logy=True)
    ndeuteron_hist .Draw(output_name_root+"ndeuteron.png", xtitle='N. deuteron', logy=True)
    ntritium_hist  .Draw(output_name_root+"ntritium.png", xtitle='N. tritium', logy=True)
    nalpha_hist    .Draw(output_name_root+"nalpha.png", xtitle='N. alpha', logy=True)
    nhelium3_hist  .Draw(output_name_root+"nhelium3.png", xtitle=r'N. $^{3}$He', logy=True)
    nnuclfrag_hist .Draw(output_name_root+"nnuclfrag.png", xtitle='N. nuclear fragments', logy=True)
    ncluster_hist  .Draw(output_name_root+"ncluster.png", xtitle='N. cluster', logy=True)
    ncharged_hist  .Draw(output_name_root+"ncharged.png", xtitle='N. charged', logy=True)    
    topo_hist      .Draw(output_name_root+"topology.png")
    topo_hist      .Draw(output_name_root+"topology_logy.png", logy=True)
    mode_hist      .Draw(output_name_root+"mode.png")
    mode_hist      .Draw(output_name_root+"mode_logy.png", logy=True)
    
    minN, maxN = nhits_lin_hist.GetMinMax()
    minSumE, maxSumE = SumE_hist.GetMinMax()
    minE, maxE = E_hist.GetMinMax()
    print("Total", total_images, "images")
    print("Maximum number of hits:", maxN)
    print("Sum of E:", minSumE, "--", maxSumE)
    print("E:", minE, "--", maxE)    
    print("N. empty:", nEmpty)
    
    
if __name__ == '__main__':

    ## Take an input file and convert it to an h5 file of images
    if len(sys.argv) < 3:
        print("An input file and output file name must be provided as arguments!")
        sys.exit()

    input_file_names = sys.argv[1]
    output_name_root = sys.argv[2]
    make_dataset_summary_plots(input_file_names, output_name_root)
