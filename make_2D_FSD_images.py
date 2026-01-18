import sys
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
from scipy.sparse import coo_matrix
from enum import Enum, auto
import json

show_plots = False

## FSD data, reflow v3
y_limits = (-148.61399841308594, 148.61399841308594)
z_limits = (-47.43000030517578, 47.43000030517578)

## FSD images are 256x800
## FSD pixel pitch: 0.372
nz=256
ny=800

## What are the FSD dimensions for truth?
y_truth_limit = (-149.7995, 149.7995)
x_truth_limit = (-47.1085, 47.1085)
z_truth_limit = (-47.8318, 47.8318)

FSD_min = np.asarray([-47.1085,-149.7995,-47.8318], dtype=np.float64)
FSD_max = np.asarray([ 47.1085, 149.7995, 47.8318], dtype=np.float64)

def make_image(these_hits):

    ## coo_matrix sums any duplicated values, so don't sum them
    ## These could be extracted from the hits without a loop... I think
    y_list = []
    z_list = []
    q_list = []

    for hit in these_hits:
        
        ## Check for NaNs
        if np.isnan(hit['Q']) or \
           np.isnan(hit['y']) or \
           np.isnan(hit['z']):
            continue

        ## Remove negative charges
        if hit['Q'] < 0: continue
        
        ## Get pixel numbers and charge values
        pixel_pitch = 0.372
        this_z = int((hit['z'] - z_limits[0])/pixel_pitch + pixel_pitch/10.)
        this_y = int((hit['y'] - y_limits[0])/pixel_pitch + pixel_pitch/10.)
        
        this_q = hit['Q']

        y_list.append(this_y)
        z_list.append(this_z)
        q_list.append(this_q)
        
    y_arr = np.array(y_list)
    z_arr = np.array(z_list)
    q_arr = np.array(q_list)

    this_sparse = coo_matrix((q_arr, (y_arr, z_arr)), dtype=np.float32, shape=(ny, nz))

    ## Slightly confusing as this isn't necessary for csr or csc matrices
    this_sparse.sum_duplicates()

    # Apply the log10(1+x) transformation to the data array
    trans_data = np.log10(1 + this_sparse.data)

    # Reconstruct the coo_matrix with the transformed data
    trans_sparse = coo_matrix((trans_data, (this_sparse.row, this_sparse.col)), shape=this_sparse.shape)
    
    return trans_sparse

def in_volume(point):
    point = np.asarray(point, dtype=np.float64)
    ## Add a little padding
    return np.all((point >= FSD_min-1e-5) & (point <= FSD_max+1e-5))


## Define labels
class Label(Enum):

    ## Default
    NOLABEL = -1
    
    ## e+/- or photon induced
    EM = auto()

    ## Neutron induced
    NEUTRON = auto()

    ## Proton induced
    PROTON = auto()

    ## Charged pion induced
    PION = auto()
    
    ## Multiple muons deposit energy into the active volume
    MULTIMUON = auto()

    ## The muon interacted outside the active volume
    EXTERNAL = auto()

    ## Stopping muon which is captured by a nucleus and then decays not through a Michel
    STOPPINGCAPTURE = auto()

    ## Muon which decays into a Michel inside the volume
    STOPPINGMICHEL = auto()

    ## This is a catch-all category for stopping events that aren't the other two...
    STOPPINGOTHER = auto()
    
    ## This is meant to be for through-going muons without much colinear activity
    THROUGHCLEAN = auto()

    ## This is to try and get at through-going muons with colinear showers
    THROUGHMESSY = auto()

    ## A method to dump the list
    @classmethod
    def print_members(cls):
        for member in cls:
            print(f"{member.name}: {member.value}")
    
    
def get_truth_label(trajs, segments):

    ntraj  = len(trajs)
    nsegs  = len(segments)
    
    ## Grab the primaries
    prims = trajs[trajs['parent_id']==-1]
    daughters = trajs[trajs['parent_id']==0]
    nprim = len(prims)

    seg_mask = np.isin(trajs['traj_id'], segments['traj_id'])
    masked_trajs = trajs[seg_mask]
    masked_daughters = masked_trajs[masked_trajs['parent_id']==0]
    
    ## If there's more than one primary... kind of tough
    nmuon = np.count_nonzero(np.abs(prims['pdg_id'])==13)
    nEM   = np.count_nonzero((np.abs(prims['pdg_id'])==11)|(prims['pdg_id']==22))
    nNeut = np.count_nonzero(np.abs(prims['pdg_id'])==2112)
    nProt = np.count_nonzero(np.abs(prims['pdg_id'])==2212)
    nPion = np.count_nonzero(np.abs(prims['pdg_id'])==211)

    ## Simple categories if there is no muon
    if nmuon == 0:
        if nEM > 0: return Label.EM
        if nProt > 0: return Label.PROTON
        if nNeut > 0: return Label.NEUTRON
        if nPion > 0: return Label.PION
        print("Unhandled primary particle PDG:", prims['pdg_id'])

    ## Do the primaries make it into the detector?
    prim_segments = segments[segments['traj_id'] == 0]
    nprims_entering = len(np.unique(prim_segments['event_id']))
        
    ## This category can include multiple clean tracks, or events in which one is very much dominant
    if nmuon > 1 and nprims_entering > 1: return Label.MULTIMUON

    ## If a muon interacts outside the detector volume but daughters make it in
    if nprims_entering == 0: return Label.EXTERNAL

    ## Does the primary end in the detector?
    ends = in_volume(prims['xyz_end'])
    
    ## Deal with stopping tracks
    if ends:

        ## Use G4 process codes to classify events
        nhaddecay = np.count_nonzero((masked_daughters['start_process'] == 4) & (masked_daughters['start_subprocess'] == 151))
        nmichel = np.count_nonzero((masked_daughters['start_process'] == 6) & (masked_daughters['start_subprocess'] == 201))

        ## These are rare, but included as printouts to flag unusual events (maybe better categories can be defined later)
        if nhaddecay >0 and nmichel > 0:
            print("BOTH DECAYS")
            print([(pdg,p,s) for pdg,p,s in zip(masked_daughters['pdg_id'], masked_daughters['start_process'], masked_daughters['start_subprocess'])])
        if nhaddecay == 0 and nmichel == 0:
            print("SOMETHING ELSE")
            print([(pdg,p,s) for pdg,p,s in zip(masked_daughters['pdg_id'], masked_daughters['start_process'], masked_daughters['start_subprocess'])])
        
        ## Return the relevant category
        if nhaddecay > 0: return Label.STOPPINGCAPTURE
        if nmichel > 0: return Label.STOPPINGMICHEL
        return Label.STOPPINGOTHER

    ## Deal with through-going tracks
    else:
        ## Not a great means of separation, but events with pair production tend to be messier
        ## This distinction isn't amazing though... clean tracks tend to be very clean, but messy tracks are a mix...
        npos = len(masked_trajs[masked_trajs['pdg_id']==-11])
        if npos > 0: return Label.THROUGHMESSY
        return Label.THROUGHCLEAN
    
    return Label.NOLABEL


def make_images(input_file_name, output_file_name, min_hits=1):

    f = h5py.File(input_file_name, "r")

    raw_events = f['charge/events/data']
    ## Allows a minimum number of hits to be an interesting event
    events = raw_events[raw_events['nhit'] > min_hits]

    ## How many events do we have?
    nevts = len(events)
    print("Found:", len(events), "events")
    
    hits = f['charge/calib_prompt_hits/data']
    hits_ref = f['charge/events/ref/charge/calib_prompt_hits/ref']
    hits_region = f['charge/events/ref/charge/calib_prompt_hits/ref_region']
    print("With total:", len(hits), "hits")

    ## Check if this is a simulation file and set up datasets if so
    mc_ints = None
    mc_int_refs = None
    mc_int_region = None
    mc_trajs = None
    mc_traj_refs = None
    mc_traj_region = None
    mc_segs = None
    mc_seg_refs = None
    mc_seg_region = None
    if 'mc_truth' in f:
        mc_ints = f['mc_truth/interactions/data']
        mc_int_refs = f['charge/raw_events/ref/mc_truth/interactions/ref']
        mc_int_region = f['charge/raw_events/ref/mc_truth/interactions/ref_region']
        mc_trajs = f['mc_truth/trajectories/data']
        mc_traj_refs = f['mc_truth/interactions/ref/mc_truth/trajectories/ref']
        mc_traj_region = f['mc_truth/interactions/ref/mc_truth/trajectories/ref_region']
        mc_segs = f['mc_truth/segments/data']
        mc_seg_refs = f['mc_truth/interactions/ref/mc_truth/segments/ref']
        mc_seg_region = f['mc_truth/interactions/ref/mc_truth/segments/ref_region']
    
    if show_plots:
        plt.ion()
        plt.figure(figsize=(3, 7))

    sparse_image_list = []
    event_id_list = []

    nhits_list = []
    label_list = []
    
    ## Loop over events
    for evt in range(nevts):

        ## Check on the progress
        if evt % int(nevts/10) == 0 and evt != 0: print("Processed evt", evt, "/", nevts)
        
        ## Now grab all the hits associated with the event
        ev_id = events[evt]['id']

        hit_ref = hits_ref[hits_region[ev_id,'start']:hits_region[ev_id,'stop']]
        hit_ref = np.sort(hit_ref[hit_ref[:,0] == ev_id, 1])
        these_hits = hits[hit_ref]

        ## Use a default label if this is not MC
        this_label = Label.NOLABEL

        ## If this is MC, figure out the appropriate label
        if mc_ints:
            ## Get the mc interaction (there's only one index)
            mc_int_ref = mc_int_refs[mc_int_region[ev_id,'start']:mc_int_region[ev_id,'stop']]
            mc_int_ref = np.sort(mc_int_ref[mc_int_ref[:,0] == ev_id, 1])
            these_ints = mc_ints[mc_int_ref]
            
            ## Now get trajectories, but... there can be more than one interaction in the image...
            traj_starts = mc_traj_region[mc_int_ref, 'start']
            traj_stops  = mc_traj_region[mc_int_ref, 'stop']
            seg_starts = mc_seg_region[mc_int_ref, 'start']
            seg_stops  = mc_seg_region[mc_int_ref, 'stop']
            
            print("traj_starts:", traj_starts)
            print("traj_stops:", traj_stops)
            print("seg_starts:", seg_starts)
            print("seg_stops:", seg_stops)

            print("N. hits =", len(these_hits))
            
            ## Extract final trajectories
            traj_ref_chunks = [mc_traj_refs[start:stop] for start, stop in zip(traj_starts, traj_stops)]
            all_traj_refs = np.vstack(traj_ref_chunks)
            all_traj_refs = np.sort(all_traj_refs[np.isin(all_traj_refs[:, 0], mc_int_ref), 1])
            these_trajs = mc_trajs[all_traj_refs]

            ## Now do the same for segments
            seg_starts = mc_seg_region[mc_int_ref, 'start']
            seg_stops  = mc_seg_region[mc_int_ref, 'stop']
            seg_ref_chunks = [mc_seg_refs[start:stop] for start, stop in zip(seg_starts, seg_stops)]
            all_seg_refs = np.vstack(seg_ref_chunks)
            all_seg_refs = np.sort(all_seg_refs[np.isin(all_seg_refs[:, 0], mc_int_ref), 1])
            these_segs = mc_segs[all_seg_refs]

            ## Finally able to figure out the category
            this_label = get_truth_label(these_trajs, these_segs)
    
        ## Get an image with 256x800 pixels
        this_sparse = make_image(these_hits)

        ## Check whether this is a "good image" (very arbitrary for now)
        ## Really this is removing the large number of uninteresting images with few hits
        if np.count_nonzero(this_sparse.data) < 100: continue

        ## If we've passed all cuts, fill
        sparse_image_list .append(this_sparse)
        event_id_list     .append(ev_id)
        nhits_list        .append(np.count_nonzero(this_sparse.data))
        label_list        .append(this_label.value)

        ## For interactive label debugging
        if show_plots and this_label in [Label.NOLABEL]:
            print("Category:", this_label.name)
            this_image = this_sparse.toarray()
            gr = plt.imshow(this_image, origin='lower')
            plt.show()
            input("Pause...")

    ## Make a nice 2D image of the labels
    if show_plots:

        Label.print_members()
        nhits_list = np.asarray(nhits_list)
        label_list = np.asarray(label_list)

        # Define bins
        num_nhits_bins = 50  # or whatever you choose
        nhits_bins = np.linspace(0, 1000, num_nhits_bins + 1)
        
        label_values = np.unique(label_list)
        label_bins = np.append(label_values, label_values[-1] + 1)  # one bin per label
        
        # Create 2D histogram
        plt.hist2d(nhits_list, label_list, bins=[nhits_bins, label_bins], cmap='viridis', norm=LogNorm())
        
        # Axis labels
        plt.xlabel('nhits')
        plt.ylabel('label')
        plt.colorbar(label='Count')
        plt.title('2D Histogram: nhits vs label')
        plt.yticks(label_values)  # Set tick marks at label values
        
        plt.show()
        input("Pause...")
        
    ## Write the images to an hdf5 file
    with h5py.File(output_file_name, 'w') as fout:

        ## Save the number of images in the file
        fout.attrs['N'] = len(sparse_image_list)

        ## Save the labels defined when this file was produced
        label_dict = {member.name: member.value for member in Label}
        fout.attrs['label_enum'] = json.dumps(label_dict)
        
        for i, (sparse_image, event_id, label) in enumerate(zip(sparse_image_list, event_id_list, label_list)):
            group = fout.create_group(str(i))
            group.create_dataset('data', data=sparse_image.data)
            group.create_dataset('row', data=sparse_image.row.astype(np.uint16))
            group.create_dataset('col', data=sparse_image.col.astype(np.uint16))
            group.create_dataset('label', data=np.int8(label))
            group.attrs['shape'] = np.array(sparse_image.shape, dtype=np.uint16)
            group.attrs['event_id'] = np.uint32(event_id)
            
    ## Close the input file
    f.close()

if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("Image maker")

    # Require an input file name and location to dump plots
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)

    ## Allow a minimum number of hits cut
    parser.add_argument('--min_hits', type=int, default=1, nargs='?')

    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))

    make_images(args.input, args.output, args.min_hits)
