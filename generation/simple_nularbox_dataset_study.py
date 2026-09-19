import sys
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from glob import glob
from larch.datasets.nularbox.truth_labels import (
    PARTICLE_STACK_DTYPE,
    EVENT_LABEL_DTYPE,
    CCTopology,
    Topology,
    Mode,
)
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
        values = np.concatenate(self.buffer).astype(np.int64, copy=False)
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
        return self.counts.copy(), np.arange(self.nbins + 1)

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
        self.members = list(enum_class)
        self.bin_labels = [member.name for member in self.members]
        self.value_to_bin = {
            member.value: i
            for i, member in enumerate(self.members)
        }

        self.nbins = len(self.members)
        self.counts = np.zeros(self.nbins, dtype=np.int64)
        self.buffer = []

    def Fill(self, values):
        arr = np.atleast_1d(np.asarray(values)).astype(int, copy=False)
        self.buffer.append(arr)

    def FlushBuffer(self):
        
        if not self.buffer: return

        values = np.concatenate(self.buffer)
        self.buffer.clear()

        for value, count in zip(*np.unique(values, return_counts=True)):
            index = self.value_to_bin.get(int(value))
            if index is None:
                print(f"Warning: unknown {self.enum_class.__name__} "
                      f"value {value}, seen {count} times")
                continue
            self.counts[index] += count
        
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



ENUM_FIELDS = {
    "mode": Mode,
    "topology_truth": Topology,
    "topology_visible": Topology,
    "cctopology_truth": CCTopology,
    "cctopology_visible": CCTopology,
}

FLOAT_BINS = {
    "enu": np.linspace(0, 50, 101),
    "q0": np.linspace(0, 50, 101),
    "edep_5mm": np.linspace(0, 1000, 201),
    "edep_10mm": np.linspace(0, 1000, 201),
    "edep_20mm": np.linspace(0, 1000, 201),
    "edep_50mm": np.linspace(0, 1000, 201),
}

INTEGER_BINS = {
    "nneutron": 21,
    "nproton": 21,
    "ncharged": 21,
    "ncluster": 11,
}


def setup_record_histograms(dtype):
    hists = {}

    for name in dtype.names:
        field_dtype = dtype.fields[name][0]

        if name in ENUM_FIELDS:
            hists[name] = TH1Enum(ENUM_FIELDS[name])

        elif np.issubdtype(field_dtype, np.bool_):
            hists[name] = TH1Iish(2)

        elif np.issubdtype(field_dtype, np.integer):
            hists[name] = TH1Iish(INTEGER_BINS.get(name, 6))

        elif np.issubdtype(field_dtype, np.floating):
            hists[name] = TH1Dish(FLOAT_BINS.get(name, np.linspace(0, 100, 101)))

    return hists            


def setup_label_histograms():
    return {
        "particle_truth": setup_record_histograms(PARTICLE_STACK_DTYPE),
        "particle_visible": setup_record_histograms(PARTICLE_STACK_DTYPE),
        "event": setup_record_histograms(EVENT_LABEL_DTYPE),
    }


def fill_record_histograms(hists, records):
    for name, hist in hists.items():
        hist.Fill(records[name])


def fill_label_histograms(hists, h5file, start, end):
    fill_record_histograms(hists["particle_truth"],
                           h5file["particle_truth"][start:end])
    fill_record_histograms(hists["particle_visible"],
                           h5file["particle_visible"][start:end])
    fill_record_histograms(hists["event"],
                           h5file["event_labels"][start:end])

    
def draw_record_histograms(hists, output_root):
    for name, hist in hists.items():
        kwargs = {}

        if isinstance(hist, TH1Iish) and name == "cc":
            kwargs["xlabels"] = ["NC", "CC"]

        if isinstance(hist, (TH1Iish, TH1Enum)):
            hist.Draw(f"{output_root}{name}.png",
                      xtitle=name,
                      **kwargs)
            hist.Draw(f"{output_root}{name}_logy.png",
                      xtitle=name,
                      logy=True,
                      **kwargs)
        else:
            hist.Draw(f"{output_root}{name}.png",
                      xtitle=name)
            hist.Draw(f"{output_root}{name}_logy.png",
                      xtitle=name,
                      logy=True)


def draw_label_histograms(hists, output_root):
    for group_name, group_hists in hists.items():
        draw_record_histograms(group_hists,
                               f"{output_root}{group_name}_")

def setup_data_histograms(alpha_min, alpha_max, p="xz"):
    
    hists = {}
    
    ## Image-level histograms
    hists['nhits_lin'] = TH1Dish(np.linspace(0, 4000, 200))
    hists['nhits_log'] = TH1Dish(np.logspace(0, 3.7, 100))
    hists['E']         = TH1Dish(np.logspace(-1, 2.4, 125))
    hists['SumE']      = TH1Dish(np.linspace(0, 5000, 100))
    hists['MaxE']      = TH1Dish(np.logspace(-1, 2.4, 125))

    ## Position histograms
    hists[p[0]]             = TH1Dish(np.linspace(0, 512, 257))
    hists[p[1]]             = TH1Dish(np.linspace(0, 512, 257))
    hists[p[0]+'_vs_E']     = TH2Dish(np.linspace(0, 512, 257), np.logspace(-1, 2.4, 125))
    hists[p[1]+'_vs_E']     = TH2Dish(np.linspace(0, 512, 257), np.logspace(-1, 2.4, 125))
    hists[p[0]+'_vs_'+p[1]] = TH2Dish(np.linspace(0, 512, 257), np.linspace(0, 512, 257))

    if p == "xyz":
        hists[p[2]]             = TH1Dish(np.linspace(0, 512, 257))
        hists[p[2]+'_vs_E']     = TH2Dish(np.linspace(0, 512, 257), np.logspace(-1, 2.4, 125))
        hists[p[0]+'_vs_'+p[2]] = TH2Dish(np.linspace(0, 512, 257), np.linspace(0, 512, 257))
        hists[p[1]+'_vs_'+p[2]] = TH2Dish(np.linspace(0, 512, 257), np.linspace(0, 512, 257))
        
    ## Alpha transform histograms
    hists['alpha'] = [TH1Dish(np.linspace(0, 5, 100)) for _ in range(alpha_min, alpha_max+1)]

    return hists

def fill_data_histograms(hists,
                         h5file,
                         event_index,
                         alpha_min,
                         alpha_max,
                         p="xz"):
    
    offsets = h5file[f"{p}_offsets"]
    start = int(offsets[event_index])
    end = int(offsets[event_index + 1])

    data = h5file[f"{p}_data"][start:end]

    if p == "xyz":
        coords = h5file["xyz_coords"][start:end]
        ax0 = coords[:, 0]
        ax1 = coords[:, 1]
        ax2 = coords[:, 2]
    else:
        ax0 = h5file[f"{p}_row"][start:end]
        ax1 = h5file[f"{p}_col"][start:end]

    if len(data) < 1:
        return False

    hists["nhits_lin"].Fill(np.count_nonzero(data))
    hists["nhits_log"].Fill(np.count_nonzero(data))
    hists["E"].Fill(data)
    hists["SumE"].Fill(np.sum(data))
    hists["MaxE"].Fill(np.max(data))

    hists[p[0]].Fill(ax0)
    hists[p[1]].Fill(ax1)
    hists[p[0] + "_vs_E"].Fill(ax0, data)
    hists[p[1] + "_vs_E"].Fill(ax1, data)
    hists[p[0] + "_vs_" + p[1]].Fill(ax0, ax1)

    if p == "xyz":
        hists[p[2]].Fill(ax2)
        hists[p[2] + "_vs_E"].Fill(ax2, data)
        hists[p[0] + "_vs_" + p[2]].Fill(ax0, ax2)
        hists[p[1] + "_vs_" + p[2]].Fill(ax1, ax2)

    for alpha in range(alpha_min, alpha_max + 1):
        transformed = np.log10(1 + alpha * data)/ np.log10(1 + alpha)
        hists["alpha"][alpha - alpha_min].Fill(transformed)

    return True

def flush_per_hit_histograms(hists, alpha_min, alpha_max, p="xz"):
    for key in ['E', p[0], p[1], p[0]+'_vs_'+p[1], p[0]+'_vs_E', p[1]+'_vs_E']:
        hists[key].FlushBuffer()

    if p=="xyz":
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

    if p == "xyz":
        hists[p[2]].Draw(r+p[2]+"_logy_"+p+".png", xtitle=p[2]+' coord.', logy=True)
        hists[p[2]].Draw(r+p[2]+"_liny_"+p+".png", xtitle=p[2]+' coord.', logy=False)
        hists[p[2]+'_vs_E']    .Draw(r+p[2]+"_vs_E_linz_"+p+".png",  xtitle=p[2]+" coord.", ytitle="Raw E(MeV)", logz=False)
        hists[p[2]+'_vs_E']    .Draw(r+p[2]+"_vs_E_logz_"+p+".png",  xtitle=p[2]+" coord.", ytitle="Raw E(MeV)", logz=True)        

        hists[p[0]+'_vs_'+p[2]].Draw(r+p[0]+'_vs_'+p[2]+"_linz_"+p+".png", xtitle=p[0]+" coord.", ytitle=p[2]+" coord.", logz=False)
        hists[p[0]+'_vs_'+p[2]].Draw(r+p[0]+'_vs_'+p[2]+"_logz_"+p+".png", xtitle=p[0]+" coord.", ytitle=p[2]+" coord.", logz=True)
        hists[p[1]+'_vs_'+p[2]].Draw(r+p[1]+'_vs_'+p[2]+"_linz_"+p+".png", xtitle=p[1]+" coord.", ytitle=p[2]+" coord.", logz=False)
        hists[p[1]+'_vs_'+p[2]].Draw(r+p[1]+'_vs_'+p[2]+"_logz_"+p+".png", xtitle=p[1]+" coord.", ytitle=p[2]+" coord.", logz=True)
        

def make_dataset_summary_plots(input_file_names,
                               output_name_root="plots/"):

    alpha_min, alpha_max = 5, 5
    max_images = 100_000
    sum_images = 0
    total_images = 0
    n_empty = 0

    os.makedirs(output_name_root, exist_ok=True)

    label_hists = setup_label_histograms()
    xz_hists    = setup_data_histograms(alpha_min, alpha_max, "xz")
    xy_hists    = setup_data_histograms(alpha_min, alpha_max, "xy")
    xyz_hists   = setup_data_histograms(alpha_min, alpha_max, "xyz")

    for filename in sorted(glob(input_file_names)):
        if sum_images >= max_images: break
        print("Reading", filename)

        with h5py.File(filename, "r", libver="latest") as h5file:
            nimages = int(h5file.attrs["N"])
            print("Found", nimages, "images")
            total_images += nimages

            required = {
                "particle_truth",
                "particle_visible",
                "event_labels",
            }
            missing = required - set(h5file.keys())

            if missing:
                raise RuntimeError(f"{filename} is missing new-schema datasets: "
                                   f"{sorted(missing)}")

            ## How many events to include from this file
            n_to_process = min(nimages, max_images - sum_images)

            # Labels can be filled efficiently as arrays.
            fill_label_histograms(label_hists, h5file, 0, n_to_process)

            ## Fill data histograms in an event loop
            for event_index in range(n_to_process):
                filled_xz  = fill_data_histograms(xz_hists, h5file, event_index, alpha_min, alpha_max, "xz")
                filled_xy  = fill_data_histograms(xy_hists, h5file, event_index, alpha_min, alpha_max, "xy")
                filled_xyz = fill_data_histograms(xyz_hists, h5file, event_index, alpha_min, alpha_max, "xyz")

                if not (filled_xz and filled_xy and filled_xyz): n_empty += 1
                sum_images += 1

            flush_per_hit_histograms(xz_hists, alpha_min, alpha_max, "xz")
            flush_per_hit_histograms(xy_hists, alpha_min, alpha_max, "xy")
            flush_per_hit_histograms(xyz_hists, alpha_min, alpha_max, "xyz")

    draw_label_histograms(label_hists, output_name_root)
    draw_data_histograms(xz_hists, output_name_root, alpha_min, alpha_max, "xz")
    draw_data_histograms(xy_hists, output_name_root, alpha_min, alpha_max, "xy")
    draw_data_histograms(xyz_hists, output_name_root, alpha_min, alpha_max, "xyz")  
    
    print("Processed:", sum_images)
    print("Available:", total_images)
    print("Events with at least one empty representation:", n_empty)
    
if __name__ == '__main__':

    ## Take an input file and convert it to an h5 file of images
    if len(sys.argv) < 3:
        print("An input file and output file name must be provided as arguments!")
        sys.exit()

    input_file_names = sys.argv[1]
    output_name_root = sys.argv[2]
    make_dataset_summary_plots(input_file_names, output_name_root)
