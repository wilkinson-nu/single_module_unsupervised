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

class TH2Dish:
    def __init__(self, xbin_edges, ybin_edges, dtype=np.int64):
        self.xbin_edges = np.asarray(xbin_edges, dtype=float)
        self.ybin_edges = np.asarray(ybin_edges, dtype=float)
        self.counts = np.zeros((len(self.xbin_edges) - 1, len(self.ybin_edges) - 1), dtype=dtype)

        self.min_seen = None
        self.max_seen = None

        self.buffer = []

    def Fill(self, xvalues, yvalues):

        assert len(xvalues) == len(yvalues)
        
        xarr = np.atleast_1d(np.asarray(xvalues))
        yarr = np.atleast_1d(np.asarray(yvalues))
        self.buffer.append((xarr, yarr))

    def FlushBuffer(self):

        if not self.buffer: return
        xvalues = np.concatenate([b[0] for b in self.buffer])
        yvalues = np.concatenate([b[1] for b in self.buffer])
        self.buffer.clear()
        
        hist, _, _ = np.histogram2d(xvalues, yvalues, bins=[self.xbin_edges, self.ybin_edges])
        self.counts += hist.astype(self.counts.dtype)
        
        if self.counts.size:
            zmin = self.counts.min()
            zmax = self.counts.max()
            self.min_seen = zmin if self.min_seen is None else min(self.min_seen, zmin)
            self.max_seen = zmax if self.max_seen is None else max(self.max_seen, zmax)

    def Reset(self):
        self.counts.fill(0)
        self.buffer.clear()
        self.min_seen = None
        self.max_seen = None

    def GetArray(self):
        return self.counts.copy(), self.xbin_edges.copy(), self.ybin_edges.copy()

    def GetMinMax(self):
        return self.min_seen, self.max_seen

    def Draw(self, filename=None, xtitle=None, ytitle=None,
             logx=False, logy=False, logz=False, show=False):

        self.FlushBuffer()

        fig, ax = plt.subplots()
        norm = matplotlib.colors.LogNorm() if logz else None
        arr = self.counts.T  ## transpose so x is horizontal and y is vertical
        im = ax.imshow(arr, origin='lower', aspect='auto', norm=norm,
                       extent=[self.xbin_edges[0], self.xbin_edges[-1],
                                self.ybin_edges[0], self.ybin_edges[-1]])
        plt.colorbar(im, ax=ax)

        if logx: ax.set_xscale("log")
        if logy: ax.set_yscale("log")
        if xtitle: ax.set_xlabel(xtitle)
        if ytitle: ax.set_ylabel(ytitle)
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

            
def setup_truth_histograms():

    hists = {}    

    # High level truth histograms
    hists['enu']       = TH1Dish(np.linspace(0, 50, 100))
    hists['q0']        = TH1Dish(np.linspace(0, 50, 100))

    # Particle count histograms
    for name, nbins in [('cc', 2), ('nneutron', 21), ('nproton', 21),
                        ('nantineut', 6), ('nantiprot', 6), ('npipm', 6),
                        ('npi0', 6), ('nkapm', 6), ('nka0', 6), ('nem', 6),
                        ('nmuon', 6), ('nstrange', 6), ('ncharm', 6),
                        ('ndeuteron', 6), ('ntritium', 6), ('nalpha', 6),
                        ('nhelium3', 6), ('nnuclfrag', 6),
                        ('ncluster', 11), ('ncharged', 21)]:
        hists[name] = TH1Iish(nbins=nbins)

    # Enum histograms
    hists['topology'] = TH1Enum(Topology)
    hists['mode']     = TH1Enum(Mode)

    return hists


def fill_truth_histograms(hists, label):
    
    for key in ['nneutron', 'nproton', 'nantineut', 'nantiprot', 'npipm', 'npi0',
                'nkapm', 'nka0', 'nem', 'nmuon', 'nstrange', 'ncharm',
                'ndeuteron', 'ntritium', 'nalpha', 'nhelium3', 'nnuclfrag',
                'enu', 'q0']:
        hists[key].Fill(label[key])

    hists['cc']      .Fill(int(label['cc']))
    hists['topology'].Fill(label['topology'])
    hists['mode']    .Fill(label['mode'])
    hists['ncharged'].Fill(label['nproton'] + label['npipm'] + label['nkapm'])
    hists['ncluster'].Fill(label['ndeuteron'] + label['nalpha'] + label['nhelium3'] 
                           + label['ntritium'] + label['nnuclfrag'])
    return True


def draw_truth_histograms(hists, output_name_root):

    r = output_name_root

    hists['enu']      .Draw(r+"enu.png", xtitle=r'$E_{\nu}$ (GeV)')
    hists['q0']       .Draw(r+"q0.png", xtitle=r'$q_{0}$ (GeV)')

    for name, xtitle, kwargs in [
        ('cc',         None,                       {'xlabels': ['NC', 'CC']}),
        ('nneutron',   'N. 2112',                  {}),
        ('nproton',    'N. 2212',                  {}),
        ('nantineut',  'N. -2112',                 {'logy': True}),
        ('nantiprot',  'N. -2212',                 {'logy': True}),
        ('npipm',      r'N. $\pi^{\pm}$',          {'logy': True}),
        ('npi0',       r'N. $\pi^{0}$',            {'logy': True}),
        ('nkapm',      r'N. $K^{\pm}$',            {'logy': True}),
        ('nka0',       r'N. $K^{0}$',              {'logy': True}),
        ('nem',        'N. EM',                    {'logy': True}),
        ('nmuon',      r'N. $\mu^{\pm}$',          {'logy': True}),
        ('nstrange',   'N. Strange (not kaon)',     {'logy': True}),
        ('ncharm',     'N. Charm',                  {'logy': True}),
        ('ndeuteron',  'N. deuteron',               {'logy': True}),
        ('ntritium',   'N. tritium',                {'logy': True}),
        ('nalpha',     'N. alpha',                  {'logy': True}),
        ('nhelium3',   r'N. $^{3}$He',             {'logy': True}),
        ('nnuclfrag',  'N. nuclear fragments',      {'logy': True}),
        ('ncluster',   'N. cluster',                {'logy': True}),
        ('ncharged',   'N. charged',                {'logy': True}),
    ]:
        hists[name].Draw(r+f"{name}.png", xtitle=xtitle, **kwargs)

    hists['topology'].Draw(r+"topology.png")
    hists['topology'].Draw(r+"topology_logy.png", logy=True)
    hists['mode']    .Draw(r+"mode.png")
    hists['mode']    .Draw(r+"mode_logy.png", logy=True)


def setup_data_histograms(alpha_min, alpha_max, p="xz"):
    
    hists = {}
    
    # Image-level histograms
    hists['nhits_lin'] = TH1Dish(np.linspace(0, 4000, 200))
    hists['nhits_log'] = TH1Dish(np.logspace(0, 3.7, 100))
    hists['E']         = TH1Dish(np.logspace(-1, 2.4, 125))
    hists['SumE']      = TH1Dish(np.linspace(0, 5000, 100))
    hists['MaxE']      = TH1Dish(np.logspace(-1, 2.4, 125))

    # Position histograms
    hists[p[0]]             = TH1Dish(np.linspace(0, 512, 257))
    hists[p[1]]             = TH1Dish(np.linspace(0, 512, 257))
    hists[p[0]+'_vs_E']     = TH2Dish(np.linspace(0, 512, 257), np.logspace(-1, 2.4, 125))
    hists[p[1]+'_vs_E']     = TH2Dish(np.linspace(0, 512, 257), np.logspace(-1, 2.4, 125))
    hists[p[0]+'_vs_'+p[1]] = TH2Dish(np.linspace(0, 512, 257), np.linspace(0, 512, 257))

    # Add options for 3D
    if len(p) == 3:
        hists[p[2]]             = TH1Dish(np.linspace(0, 512, 257))
        hists[p[2]+'_vs_E']     = TH2Dish(np.linspace(0, 512, 257), np.logspace(-1, 2.4, 125))
        hists[p[0]+'_vs_'+p[2]] = TH2Dish(np.linspace(0, 512, 257), np.linspace(0, 512, 257))
        hists[p[1]+'_vs_'+p[2]] = TH2Dish(np.linspace(0, 512, 257), np.linspace(0, 512, 257))
        
    # Alpha transform histograms
    hists['alpha'] = [TH1Dish(np.linspace(0, 5, 100)) for _ in range(alpha_min, alpha_max+1)]

    return hists


def fill_data_histograms(hists, group, alpha_min, alpha_max, p="xz"):

    data = group['data_'+p][:]

    if len(p) == 2:
        ax0  = group['row_'+p][:]
        ax1  = group['col_'+p][:]

    if len(p) == 3:
        coords = group['coords_xyz']
        ax0 = coords[:,0]
        ax1 = coords[:,1]
        ax2 = coords[:,2]
        
    if len(data) < 1:
        return False  # empty image

    # Image-level histograms
    hists['nhits_lin'] .Fill(np.count_nonzero(data))
    hists['nhits_log'] .Fill(np.count_nonzero(data))
    hists['E']         .Fill(data)
    hists['SumE']      .Fill(np.sum(data))
    hists['MaxE']      .Fill(np.max(data))

    # Position histograms
    hists[p[0]]             .Fill(ax0)
    hists[p[1]]             .Fill(ax1)
    hists[p[0]+'_vs_E']     .Fill(ax0, data)
    hists[p[1]+'_vs_E']     .Fill(ax1, data)
    hists[p[0]+'_vs_'+p[1]] .Fill(ax0, ax1)

    # Add options for 3D
    if len(p) == 3:
        hists[p[2]]             .Fill(ax2)
        hists[p[2]+'_vs_E']     .Fill(ax2, data)
        hists[p[0]+'_vs_'+p[2]] .Fill(ax0, ax2)
        hists[p[1]+'_vs_'+p[2]] .Fill(ax1, ax2)

    for a in range(alpha_min, alpha_max+1):
        a_data = np.log10(1 + a*data) / np.log10(1 + a)
        hists['alpha'][a-alpha_min].Fill(a_data)

    return True


def flush_per_hit_histograms(hists, alpha_min, alpha_max, p="xz"):
    for key in ['E', p[0], p[1], p[0]+'_vs_'+p[1], p[0]+'_vs_E', p[1]+'_vs_E']:
        hists[key].FlushBuffer()

    if len(p)==3:
        for key in [p[2], p[2]+'_vs_E', p[0]+'_vs_'+p[2], p[1]+'_vs_'+p[2]]:
            hists[key].FlushBuffer()

    for a in range(alpha_min, alpha_max+1):
        hists['alpha'][a-alpha_min].FlushBuffer()


def draw_data_histograms(hists, output_name_root, alpha_min, alpha_max, p="xz"):

    r = output_name_root

    hists['nhits_lin'].Draw(r+"nhits_distribution_linx_"+p+".png", xtitle='N. hits', logy=True)
    hists['nhits_log'].Draw(r+"nhits_distribution_logx_"+p+".png", xtitle='N. hits', logx=True, logy=True)
    hists['E']        .Draw(r+"E_distribution_"+p+".png", xtitle=r'Raw E (MeV)', logx=True, logy=True)
    hists['SumE']     .Draw(r+"sumE_distribution_"+p+".png", xtitle=r'$\sum$ raw E (MeV)', logy=True)
    hists['MaxE']     .Draw(r+"maxE_distribution_"+p+".png", xtitle=r'Max. raw E (MeV)')

    hists[p[0]].Draw(r+p[0]+"_logy_"+p+".png", xtitle=p[0]+' coord.', logy=True)
    hists[p[1]].Draw(r+p[1]+"_logy_"+p+".png", xtitle=p[1]+' coord.', logy=True)
    hists[p[0]].Draw(r+p[0]+"_liny_"+p+".png", xtitle=p[0]+' coord.', logy=False)
    hists[p[1]].Draw(r+p[1]+"_liny_"+p+".png", xtitle=p[1]+' coord.', logy=False)

    hists[p[0]+'_vs_'+p[1]].Draw(r+p[0]+'_vs_'+p[1]+"_linz_"+p+".png", xtitle=p[0]+" coord.", ytitle=p[1]+" coord.", logz=False)
    hists[p[0]+'_vs_'+p[1]].Draw(r+p[0]+'_vs_'+p[1]+"_logz_"+p+".png", xtitle=p[0]+" coord.", ytitle=p[1]+" coord.", logz=True)
    hists[p[0]+'_vs_E']    .Draw(r+p[0]+"_vs_E_linz_"+p+".png",  xtitle=p[0]+" coord.", ytitle="Raw E(MeV)", logz=False)
    hists[p[0]+'_vs_E']    .Draw(r+p[0]+"_vs_E_logz_"+p+".png",  xtitle=p[0]+" coord.", ytitle="Raw E(MeV)", logz=True)
    hists[p[1]+'_vs_E']    .Draw(r+p[1]+"_vs_E_linz_"+p+".png",  xtitle=p[1]+" coord.", ytitle="Raw E(MeV)", logz=False)
    hists[p[1]+'_vs_E']    .Draw(r+p[1]+"_vs_E_logz_"+p+".png",  xtitle=p[1]+" coord.", ytitle="Raw E(MeV)", logz=True)

    for a in range(alpha_min, alpha_max+1):
        xtitle = r'log$_{10}$(1 + '+str(a)+'E)/log$_{10}$(1 + '+str(a)+')'
        hists['alpha'][a-alpha_min].Draw(r+f"LogAlphaE{a}_liny_"+p+".png", xtitle=xtitle, logy=False)
        hists['alpha'][a-alpha_min].Draw(r+f"LogAlphaE{a}_logy_"+p+".png", xtitle=xtitle, logy=True)

    if len(p) == 3:
        hists[p[2]].Draw(r+p[2]+"_logy_"+p+".png", xtitle=p[2]+' coord.', logy=True)
        hists[p[2]].Draw(r+p[2]+"_liny_"+p+".png", xtitle=p[2]+' coord.', logy=False)
        hists[p[2]+'_vs_E']    .Draw(r+p[2]+"_vs_E_linz_"+p+".png",  xtitle=p[2]+" coord.", ytitle="Raw E(MeV)", logz=False)
        hists[p[2]+'_vs_E']    .Draw(r+p[2]+"_vs_E_logz_"+p+".png",  xtitle=p[2]+" coord.", ytitle="Raw E(MeV)", logz=True)        

        hists[p[0]+'_vs_'+p[2]].Draw(r+p[0]+'_vs_'+p[2]+"_linz_"+p+".png", xtitle=p[0]+" coord.", ytitle=p[2]+" coord.", logz=False)
        hists[p[0]+'_vs_'+p[2]].Draw(r+p[0]+'_vs_'+p[2]+"_logz_"+p+".png", xtitle=p[0]+" coord.", ytitle=p[2]+" coord.", logz=True)
        hists[p[1]+'_vs_'+p[2]].Draw(r+p[1]+'_vs_'+p[2]+"_linz_"+p+".png", xtitle=p[1]+" coord.", ytitle=p[2]+" coord.", logz=False)
        hists[p[1]+'_vs_'+p[2]].Draw(r+p[1]+'_vs_'+p[2]+"_logz_"+p+".png", xtitle=p[1]+" coord.", ytitle=p[2]+" coord.", logz=True)
        
## Do the business
def make_dataset_summary_plots(input_file_names, output_name_root="plots/"):

    alpha_min, alpha_max = 5, 5
    max_images  = 1e7
    sum_images  = 0
    total_images = 0
    nEmpty       = 0

    truth_hists = setup_truth_histograms()
    xz_hists    = setup_data_histograms(alpha_min, alpha_max, "xz")
    xy_hists    = setup_data_histograms(alpha_min, alpha_max, "xy")
    xyz_hists   = setup_data_histograms(alpha_min, alpha_max, "xyz")
   
    for file in glob(input_file_names):
        if sum_images > max_images: break

        print("Reading", file)
        with h5py.File(file, 'r', libver='latest') as f:
            nimages = f.attrs['N']
            print("Found", nimages, "images")
            total_images += nimages

            for i in range(nimages):
                if sum_images > max_images: break
                group = f[str(i)]
                filled = fill_truth_histograms(truth_hists, group['label'][()])
                filled = fill_data_histograms(xz_hists, group, alpha_min, alpha_max, "xz")
                filled = fill_data_histograms(xy_hists, group, alpha_min, alpha_max, "xy")
                filled = fill_data_histograms(xyz_hists, group, alpha_min, alpha_max, "xyz")
                
                if not filled:
                    nEmpty += 1
                    continue
                sum_images += 1

            flush_per_hit_histograms(xz_hists, alpha_min, alpha_max, "xz")
            flush_per_hit_histograms(xy_hists, alpha_min, alpha_max, "xy")
            flush_per_hit_histograms(xyz_hists, alpha_min, alpha_max, "xyz")            
            
    draw_truth_histograms(truth_hists, output_name_root)
    draw_data_histograms(xz_hists, output_name_root, alpha_min, alpha_max, "xz")
    draw_data_histograms(xy_hists, output_name_root, alpha_min, alpha_max, "xy")
    draw_data_histograms(xyz_hists, output_name_root, alpha_min, alpha_max, "xyz")    
    
if __name__ == '__main__':

    ## Take an input file and convert it to an h5 file of images
    if len(sys.argv) < 3:
        print("An input file and output file name must be provided as arguments!")
        sys.exit()

    input_file_names = sys.argv[1]
    output_name_root = sys.argv[2]
    make_dataset_summary_plots(input_file_names, output_name_root)
