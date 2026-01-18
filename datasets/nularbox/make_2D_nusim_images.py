import sys
import ROOT
from glob import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import coo_matrix
from enum import Enum, auto
from collections import defaultdict

make_plots = True


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
        nhaddecay=0
        #nhaddecay = np.count_nonzero((masked_daughters['start_process'] == 4) & ((masked_daughters['start_subprocess'] == 151)|(masked_daughters['start_subprocess'] == 131))
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

    
## How do we deal with events where nothing happens...?
def get_3D_image_from_event(event, origin, voxel_size):
    
    ## This is accumulating all of the contributions
    acc = defaultdict(float)
    
    ## Loop over the detector volumes
    ## Note that for the simple geometry this is length 1.
    for seg in event.SegmentDetectors:
        
        ## Loop over the segments in that volume
        nChunks = len(seg[1])
        for n in range(nChunks):
            
            ## Get the start point, end point and deposited energy
            p0_tlv = seg[1][n].GetStart()
            p1_tlv = seg[1][n].GetStop()
            E      = seg[1][n].GetEnergyDeposit()
            
            p0 = np.array([p0_tlv.X(), p0_tlv.Y(), p0_tlv.Z()], dtype=np.float64)
            p1 = np.array([p1_tlv.X(), p1_tlv.Y(), p1_tlv.Z()], dtype=np.float64)
            delta = p1 - p0
            length = np.linalg.norm(delta)
            
            ## Deal with fringe cases that the length is 0
            if length == 0:
                print("Found a zero-length segment")
                ix, iy, iz = np.floor((p0 - origin) / voxel_size).astype(int)
                acc[(ix, iy, iz)] += E
                continue

            ## Find the step direction along each axis
            step = np.sign(delta).astype(int)

            ## Distance to cross one voxel along each axis
            t_delta = np.empty(3, dtype=np.float64)

            ## Distance to the first voxel boundary along each axis
            t_max = np.empty(3, dtype=np.float64)

            ## Find start and final voxel indices
            voxel = np.floor((p0 - origin) / voxel_size).astype(int)
            voxel_end = np.floor((p1 - origin) / voxel_size).astype(int)

            ## Figure out t_delta and t_max
            for i, d in enumerate(delta):
                if d != 0:
                    ## Find the next voxel boundary along axis i
                    if step[i] > 0:
                        next_boundary = (voxel[i] + 1) * voxel_size[i] + origin[i]
                    else:
                        next_boundary = voxel[i] * voxel_size[i] + origin[i]
                    t_max[i] = (next_boundary - p0[i]) / d
                    t_delta[i] = voxel_size[i] / abs(d)
                else:
                    ## If parallel to an axis, it will never cross a boundary
                    t_max[i] = np.inf
                    t_delta[i] = np.inf

            ## Start of segment
            t = 0.0
            while t < 1.0:
                ## The next value at which a boundary is crossed
                t_next = min(min(t_max), 1.0)
                ## Length of segment inside the current voxel
                l_voxel = (t_next - t) * length
                ## Add fraction of charge to the accumulator 
                acc[tuple(voxel)] += E * (l_voxel / length)
                
                ## Check for edge case if this is the last voxel:
                if t_next >= 1.0: break
                
                # Advance along the axis that has the next crossing point
                axis = np.argmin(t_max)
                voxel[axis] += step[axis]
                t = t_next
                t_max[axis] += t_delta[axis]

    ## Prepare for COO coordinates
    coords = np.array(list(acc.keys()), dtype=np.int32)
    values = np.array(list(acc.values()), dtype=np.float32)
    return coords, values


def collapse_3d_to_2d(coords_3d, values_3d, keep_axes=(0, 1)):

    # Select the two axes we want to keep
    kept = coords_3d[:, keep_axes]   # shape (N,2)

    # Make a structured array so numpy can do row-wise unique
    dtype = np.dtype([('a', np.int32), ('b', np.int32)])
    structured = np.empty(len(kept), dtype=dtype)
    structured['a'] = kept[:, 0]
    structured['b'] = kept[:, 1]

    # Find unique 2D coordinates and mapping
    uniq, inverse = np.unique(structured, return_inverse=True)

    # Sum values for identical (a,b)
    values_2d = np.zeros(len(uniq), dtype=values_3d.dtype)
    np.add.at(values_2d, inverse, values_3d)

    # Convert back to plain (M,2) array
    coords_2d = np.vstack([uniq['a'], uniq['b']]).T

    return coords_2d, values_2d

    
def read_edepsim_output(infilelist, output_file_name):

    output_size = np.array([256, 256, 256])
    
    ## For debugging
    if make_plots:
        plt.ion()
        plt.figure(figsize=(7, 7))
        
    ## Uniform and small pixel pitch
    dx, dy, dz = 0.372, 0.372, 0.372
    voxel_size = np.array([dx, dy, dz])

    ## Origin for the grid, offset to avoid the vertex being at a bin edge, maybe better to jitter?
    origin = voxel_size/2
    
    ## Get the file(s)
    edep_tree = ROOT.TChain("EDepSimEvents")
    groo_tree = ROOT.TChain("DetSimPassThru/gRooTracker")

    ## Allow for escaped wildcards in the input...
    for f in glob(infilelist):
        edep_tree.Add(f)
        groo_tree.Add(f)

    ## Ensure ROOT doesn't manage the lifetime
    event = ROOT.TG4Event()
    edep_tree.SetBranchAddress("Event", event)

    ## lists of the objects we want to keep
    sparse_image_list = []
    event_id_list = []
    label_list = []
    
    ## Loop over events
    nevts  = edep_tree.GetEntries()
    for evt in range(nevts):
        edep_tree.GetEntry(evt)
        #groo_tree.GetEntry(evt)

        ## Add a check for empty images
        if len(event.Trajectories) <=1: continue
        
        coords_3d_raw, values_3d_raw = get_3D_image_from_event(event, origin, voxel_size)

        x = coords_3d_raw[:, 0]
        y = coords_3d_raw[:, 1]
        z = coords_3d_raw[:, 2]
        
        ## Restrict to an area around the vertex and mask out the image
        ## Apply the restriction in 3D to avoid integrating over an arbitrary z region...
        mask = ((x >= -output_size[0]/2) & (x < output_size[0]/2) &
                (y >= -output_size[1]/2) & (y < output_size[1]/2) &
                (z >= -output_size[2]/2) & (z < output_size[2]/2))
        values_3d = values_3d_raw[mask]
        coords_3d = coords_3d_raw[mask] + output_size/2

        ## Which axes to project onto
        keep_axes = [0, 2]
        keep_shape = (output_size[keep_axes[0]], output_size[keep_axes[1]])
        row = coords_3d[:, keep_axes[0]]
        col = coords_3d[:, keep_axes[1]]

        ## Make a 2D projection by summing duplicates
        this_sparse_2d = coo_matrix((values_3d, (row, col)), shape=keep_shape)
        this_sparse_2d .sum_duplicates() 

        ## Save a billion and 1 truth labels... Probably need to keep revisiting
        ## Topology
        ## N. charged particles
        ## N. neutrons
        ## N. protons
        ## N. charged pions
        ## N. EM showers
        ## N. exotic
        ## CC/NC
        ## Mode?
        ## Enu
        ## q0
        
            
        ## At this point, save
        sparse_image_list .append(this_sparse_2d)
        event_id_list     .append(evt)
        ## Need to improve this
        label_list        .append(1)

        ## Optionally dump out some files to have a look at
        if make_plots:
            plt.imshow(this_sparse_2d.toarray(), origin='lower')
            plt.savefig("plots/image_"+str(evt)+".png")
            
        
    ## Write the images to an hdf5 file
    with h5py.File(output_file_name, 'w') as fout:

        ## Save the number of images in the file
        fout.attrs['N'] = len(sparse_image_list)

        ## Save the labels defined when this file was produced
        # label_dict = {member.name: member.value for member in Label}
        # fout.attrs['label_enum'] = json.dumps(label_dict)
        
        for i, (sparse_image, event_id, label) in enumerate(zip(sparse_image_list, event_id_list, label_list)):
            group = fout.create_group(str(i))
            group.create_dataset('data', data=sparse_image.data)
            group.create_dataset('row', data=sparse_image.row.astype(np.uint16))
            group.create_dataset('col', data=sparse_image.col.astype(np.uint16))
            # group.create_dataset('label', data=np.int8(label))
            group.attrs['shape'] = np.array(sparse_image.shape, dtype=np.uint16)
            group.attrs['event_id'] = np.uint32(event_id)
    ## Done
    
if __name__ == '__main__':

    ## Take an input file and convert it to an h5 file of images
    if len(sys.argv) < 3:
        print("An input file and output file name must be provided as arguments!")
        sys.exit()

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    print(input_file_name)
                                     
    read_edepsim_output(input_file_name, output_file_name)
