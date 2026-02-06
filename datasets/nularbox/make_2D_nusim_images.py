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

## This is not something to be taken lightly as it will dump out an image for every event...
make_plots = False

## Initial label types to store, to be clarified and then will need to be versioned (probably)
LABEL_DTYPE_EXP = np.dtype([
    ("cc",        np.bool_),
    ("topology",  np.uint8),
    ("mode",      np.uint8),
    ("ncharged",  np.int8),
    ("nneutrons", np.int8),
    ("npipm",     np.int8),
    ("npi0",      np.int8),
    ("nkpm",      np.int8),
    ("nexotic",   np.int8),
    ("enu",       np.float32),
    ("q0",        np.float32),
])


def get_neutrino_4mom(groo_event):
    ## Topology << Enum class
    ## N. charged particles
    ## N. neutrons
    ## N. protons
    ## N. charged pions
    ## N. EM showers
    ## N. exotic
    ## CC/NC
    ## Mode? << Enum class
    ## Enu
    ## q0
    
    ## Loop over the particles in GENIE's stack
    for p in range(groo_event.StdHepN):

        ## Look for the particle status
        ## 0 is initial state, 1 is final, check the GENIE docs for others
        if groo_event.StdHepStatus[p] != 0: continue

        ## Check for a neutrino (any flavor)
        if abs(groo_event.StdHepPdg[p]) not in [12, 14, 16]: continue

        ## Kindly redirect any complaints about this line to /dev/null
        ## edep-sim uses MeV, gRooTracker uses GeV...
        return TLorentzVector(groo_event.StdHepP4[p*4 + 0]*1000,
                              groo_event.StdHepP4[p*4 + 1]*1000,
                              groo_event.StdHepP4[p*4 + 2]*1000,
                              groo_event.StdHepP4[p*4 + 3]*1000)
    ## Should never happen...
    return None

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

    ## Add protection against 0 and 1 hit events...
    if coords.size == 0:
        coords = coords.reshape(0, 3)
        values = values.reshape(0)
    else:
        coords = coords.reshape(-1, 3)

    return coords, values


def read_edepsim_output(infilelist, output_file_name):

    output_size = np.array([256, 256, 256])
    
    ## For debugging
    if make_plots:
        plt.ion()
        plt.figure(figsize=(7, 7))
        
    ## Uniform and small pixel pitch
    ## Uses mm, the default output unit for edep-sim
    dx, dy, dz = 3.72, 3.72, 3.72
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
        
        labels = np.zeros((), dtype=LABEL_DTYPE_EXP)
        ## Fill like:
        ## labels["energy"] = enu
        
        ## At this point, save
        sparse_image_list .append(this_sparse_2d)
        event_id_list     .append(evt)
        ## Need to improve this
        label_list        .append(labels)

        ## Optionally dump out some files to have a look at
        if make_plots:
            plt.imshow(this_sparse_2d.toarray(), origin='lower')
            plt.savefig("plots/image_"+str(evt)+".png")

            
    ## Write the images to an hdf5 file
    with h5py.File(output_file_name, 'w') as fout:

        ## Save the number of images in the file
        fout.attrs['N'] = len(sparse_image_list)

        ## Store label_struct schema
        fout.attrs['label_dtype'] = LABEL_DTYPE_EXP.descr
        ## Save the labels defined when this file was produced
        # label_dict = {member.name: member.value for member in Label}
        # fout.attrs['label_enum'] = json.dumps(label_dict)
        
        for i, (sparse_image, event_id, label_struct) in enumerate(zip(sparse_image_list, event_id_list, label_list)):
            group = fout.create_group(str(i))
            group.create_dataset('data', data=sparse_image.data)
            group.create_dataset('row', data=sparse_image.row.astype(np.uint16))
            group.create_dataset('col', data=sparse_image.col.astype(np.uint16))
            group.create_dataset('labels', data=label_struct, dtype=LABEL_DTYPE_EXP)
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

    read_edepsim_output(input_file_name, output_file_name)
