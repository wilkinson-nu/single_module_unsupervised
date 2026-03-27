import sys
import ROOT
from glob import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import coo_matrix
from collections import defaultdict
import json
from truth_labels import LABEL_DTYPE_EXP, Topology, Mode
import argparse

## This is not something to be taken lightly as it will dump out an image for every event...
make_plots = False

def get_mode(code):

    is_cc = "[CC]" in code
    is_dis = "DIS" in code
    is_res = "RES" in code
    is_2p2h = "MEC" in code
    is_qe = "QES" in code
    is_coh = "COH" in code
    is_imd = "IMD" in code
    is_nuee = "NuEEL" in code
    
    if is_dis:
        if is_cc: return Mode.CCDIS
        else: return Mode.NCDIS
    elif is_res:
        if is_cc: return Mode.CCRES
        else: return Mode.NCRES
    elif is_2p2h:
        if is_cc: return Mode.CC2p2h
        else: return Mode.NC2p2h
    elif is_qe:
        if is_cc: return Mode.CCQE
        else: return Mode.NCQE
    elif is_coh:
        if is_cc: return Mode.CCCOH
        else: return Mode.NCCOH
    elif is_imd:
        return Mode.IMD
    elif is_nuee:
        return Mode.NUEE

    print("Found unparseable code:", code)
    return Mode.NONE 

def get_topology(labels, vertex):

    if labels["nstrange"]+labels["ncharm"]+labels["nkapm"]+labels["nka0"] > 0:
        if labels["cc"]: return Topology.CCOther
        else: return Topology.NCOther
    if labels["npipm"]+labels["npi0"]>2:
        if labels["cc"]: return Topology.CCNpi
        else: return Topology.NCNpi        
    if labels["npipm"]+labels["npi0"]>1:
        if labels["cc"]: return Topology.CC2pi
        else: return Topology.NC2pi
    if labels["npipm"]+labels["npi0"]==0:
        if labels["cc"]: return Topology.CC0pi
        else: return Topology.NC0pi
    if labels["npipm"] == 1 and labels["npi0"]==0:
        if labels["cc"]: return Topology.CC1pipm
        else: return Topology.NC1pipm
    if labels["npipm"] == 0 and labels["npi0"]==1:
        if labels["cc"]: return Topology.CC1pi0
        else: return Topology.NC1pi0

    print("Unknown topology:", [x.GetPDGCode() for x in vertex.Particles])
    return Topology.NONE

def get_neutrino_4mom(groo_event):
    
    ## Loop over the particles in GENIE's stack
    ## I think the neutrino is always position 0...
    for p in range(groo_event.StdHepN):

        ## Look for the particle status
        ## 0 is initial state, 1 is final, check the GENIE docs for others
        if groo_event.StdHepStatus[p] != 0: continue

        ## Check for a neutrino (any flavor)
        if abs(groo_event.StdHepPdg[p]) not in [12, 14, 16]: continue

        return ROOT.TLorentzVector(groo_event.StdHepP4[p*4 + 0]*1000,
                                   groo_event.StdHepP4[p*4 + 1]*1000,
                                   groo_event.StdHepP4[p*4 + 2]*1000,
                                   groo_event.StdHepP4[p*4 + 3]*1000)
    ## Should never happen...
    return None

## Assuming a well ordered stack... check this is the case for other GENIE versions
def is_ccinc(pdg_list):
    if abs(pdg_list[0]) in [12, 14, 16]: return False
    return True

def get_truth_labels(vertex, groo):

    labels = np.zeros((), dtype=LABEL_DTYPE_EXP)

    ## Get all of the primary particles coming out of the event
    pdg_list = [x.GetPDGCode() for x in vertex.Particles]
    
    ## Get the neutrino and outgoing lepton
    nu_4mom = get_neutrino_4mom(groo)
    lep_4mom = vertex.Particles[0].GetMomentum()

    labels["cc"] = is_ccinc(pdg_list)
    labels["enu"] = nu_4mom.E()/1000.
    labels["q0"] = (nu_4mom.E() - lep_4mom.E())/1000.

    ## Remove the leading lepton from the list (strong assumption about the order)
    pdg_list = pdg_list[1:]

    ## Strip any neutrinos
    pdg_list = [x for x in pdg_list if abs(x) not in [12, 14, 16]]
    
    ## Now count particles in the list (and modify the list)
    labels["nproton"] = sum(1 for x in pdg_list if x == 2212)
    pdg_list = [x for x in pdg_list if x != 2212]
    labels["nantiprot"] = sum(1 for x in pdg_list if x == -2212)
    pdg_list = [x for x in pdg_list if x != -2212]    
    labels["nneutron"] = sum(1 for x in pdg_list if x == 2112)
    pdg_list = [x for x in pdg_list if x != 2112]
    labels["nantineut"] = sum(1 for x in pdg_list if x == -2112)
    pdg_list = [x for x in pdg_list if x != -2112]    
    labels["npipm"] = sum(1 for x in pdg_list if abs(x) == 211)
    pdg_list = [x for x in pdg_list if abs(x) != 211]
    labels["npi0"] = sum(1 for x in pdg_list if x == 111)
    pdg_list = [x for x in pdg_list if x != 111]
    labels["nkapm"] = sum(1 for x in pdg_list if abs(x) == 321)
    pdg_list = [x for x in pdg_list if abs(x) != 321]
    labels["nka0"] = sum(1 for x in pdg_list if abs(x) in [310, 311])
    pdg_list = [x for x in pdg_list if abs(x) not in [310, 311]]
    labels["nem"] = sum(1 for x in pdg_list if abs(x) not in [22, 11])
    pdg_list = [x for x in pdg_list if abs(x) not in [22, 11]]
    labels["nlambda0"] = sum(1 for x in pdg_list if abs(x) == 3122)
    pdg_list = [x for x in pdg_list if abs(x) != 3122]    
    labels["nstrange"] = sum(1 for x in pdg_list if abs(x) in [3222, 3122, 3112, 3212])
    pdg_list = [x for x in pdg_list if abs(x) not in [3222, 3122, 3112, 3212]]    
    labels["ncharm"] = sum(1 for x in pdg_list if abs(x) in [411, 4122, 421, 4212, 4222, 431])
    pdg_list = [x for x in pdg_list if abs(x) not in [411, 4122, 421, 4212, 4222, 431]]       
    labels["nmuon"] = sum(1 for x in pdg_list if abs(x) == 13)
    pdg_list = [x for x in pdg_list if abs(x) != 13]

    ## Add some fragmentation categories for INCL
    labels["ndeuteron"] = sum(1 for x in pdg_list if x == 1000010020)
    pdg_list = [x for x in pdg_list if x != 1000010020]
    labels["nalpha"] = sum(1 for x in pdg_list if x == 1000020040)
    pdg_list = [x for x in pdg_list if x != 1000020040]    
    labels["nhelium3"] = sum(1 for x in pdg_list if x == 1000020030)
    pdg_list = [x for x in pdg_list if x != 1000020030]
    labels["ntritium"] = sum(1 for x in pdg_list if x == 1000010030)
    pdg_list = [x for x in pdg_list if x != 1000010030] 
    labels["nnuclfrag"] = sum(1 for x in pdg_list if (x > 1000030060 and x < 1000180400))
    pdg_list = [x for x in pdg_list if not (x >= 1000020060 and x < 1000180400)]
    
    ## Also remove remnant nuclei (coherent events)
    pdg_list = [x for x in pdg_list if x not in [1000180400]]

    ## Sanity check during testing
    if len(pdg_list)>0: print("Remaining list:", pdg_list)

    labels["topology"] = np.int8(get_topology(labels, vertex).value)
    labels["mode"] = np.int8(get_mode(str(groo.EvtCode)).value)
        
    return labels

## We want to ignore all hits produced by neutrons or their daughters
## So, make a set of all true trajectories that are neutrons or their descendants 
def get_neutron_and_daughter_ids(event):
    
    neutrons  = set()
    daughters = set()
    
    for traj in event.Trajectories:
        
        if traj.GetPDGCode() == 2112:
            neutrons .add(traj.GetTrackId())
            continue
        par_id = traj.GetParentId()
        if par_id in neutrons or par_id in daughters:
            daughters .add(traj.GetTrackId())

    return neutrons.union(daughters)

## Also allow hits produced by K0L to escape
def get_k0l_ids(event):
    
    k0ls  = set()
    daughters = set()
    
    for traj in event.Trajectories:
        
        if traj.GetPDGCode() == 130:
            k0ls .add(traj.GetTrackId())
            continue
        par_id = traj.GetParentId()
        if par_id in k0ls or par_id in daughters:
            daughters .add(traj.GetTrackId())

    return k0ls.union(daughters)

## Get a set of trajectory IDs with total energy < 10 MeV
## This is a semi-arbitrary cut-off to ignore delta rays and
## other low-energy stuff that leaks out of the detector
def get_low_energy_ids(event, low_E_cut=10):
    return set(x.GetTrackId() for x in event.Trajectories if x.GetInitialMomentum().E() < low_E_cut)


def exit_downstream(point, box_size):

    px = point[0]
    py = point[1]
    pz = point[2]

    ## Shortcut negative z values
    if pz <= 0: return False
    
    hx = box_size[0]/2
    hy = box_size[1]/2
    hz = box_size[2]/2

    x_hit = px * hz / pz
    y_hit = py * hz / pz

    return abs(x_hit) <= hx and abs(y_hit) <= hy


## Check whether the muon exits
def exiting_muon(event, muon_id, box_size, downstream=False):

    ## Loop over detector segments
    for seg in event.SegmentDetectors:
        nChunks = len(seg[1])
        for n in range(nChunks):
            
            ## Get the primary id that is associated with this segment
            key_contrib = seg[1][n].GetContributors()[0]

            ## Only consider contributions that can be tracked back to the primary muon
            if key_contrib != muon_id: continue

            pos = seg[1][n].GetStop()

            ## If it exits out of z, treat in a special way
            if abs(pos[2]) > box_size[2]/2.:

                ## If we require a downstream muon, check
                if downstream: return exit_downstream(pos, box_size)
                else: return True

            ## If not, consider x and y
            if abs(pos[0]) > box_size[0]/2.:
                if downstream: return False
                else: return True
            if abs(pos[1]) > box_size[1]/2.:
                if downstream: return False
                else: return True

    return False


## This is designed to select a set of events in which:
## - The muon exits the volume of interest
## - (Optionally) the muon exits downstream, aka in the +z direction
## - No other activity escapes the volume of interest except for neutrons or low energy junk, or neutrinos
## - Where the volume of interest can be a defined cube of voxels
## - It simply returns true or false
def cc_contained_cut(event, box_size, downstream=True):
    
    ## Get the primary lepton (assumes a well ordered stack)
    out_lep = event.Primaries[0].Particles[0]

    ## Check this is a numuCC event
    if abs(out_lep.GetPDGCode()) != 13: return False

    ## Check if the muon exits
    if not exiting_muon(event, out_lep.GetTrackId(), box_size, downstream): return False
    
    ## Get all neutrons and neutron descendents in the event
    neutron_ids = get_neutron_and_daughter_ids(event)
    
    ## Get a list of low energy truth trajectories (may be quite long)
    low_energy_ids = get_low_energy_ids(event)

    ## Get a list of k0l daughters
    k0l_ids = get_k0l_ids(event)
    
    ## Loop over detector segments
    for seg in event.SegmentDetectors:        
        ## seg[0] is the detector volume (named according to the gdml file tag)
        ## seg[1] is an array of segments in the volume
        
        ## Loop over the segments in the volume
        nChunks = len(seg[1])
        for n in range(nChunks):
            
            ## Get the truth trajectory ID that is the primary contributor to this segment
            ## (Multiple particles can deposit energy at the same point in space, hence the ambiguity)
            key_contrib = seg[1][n].GetContributors()[0]
            par_contrib = seg[1][n].GetPrimaryId()

            ## Take primary muon out
            if par_contrib == out_lep.GetTrackId(): continue
            
            ## Did this segment come (mostly) from a neutron or a descendant from a neutron?
            if key_contrib in neutron_ids: continue

            ## Also ignore k0L for this
            if key_contrib in k0l_ids: continue
            
            ## Skip anything which is very low energy (delta rays often escape the volume and distort the containment numbers)
            if key_contrib in low_energy_ids: continue
            
            ## See if this is outside my "contained" box
            pos = seg[1][n].GetStop()

            if abs(pos[0]) > box_size[0]/2.: return False
            if abs(pos[1]) > box_size[1]/2.: return False
            if abs(pos[2]) > box_size[2]/2.: return False
            
    ## If we got here, it's good!
    return True


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


def make_images(infilelist,
                output_file_name,
                image_size,
                box_size,
                exit_downstream,
                min_hits,
                threshold):

    output_size = np.array([image_size, image_size, image_size])
    
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

    ## Change box_size to a real physical size
    if box_size is not None: box_size *= voxel_size
    
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

    nrejected = 0
    nselected = 0
    
    ## Loop over events
    nevts  = edep_tree.GetEntries()
    for evt in range(nevts):
        edep_tree.GetEntry(evt)
        groo_tree.GetEntry(evt)

        ## Add a check for empty images
        if len(event.Trajectories) <=1: continue

        ## Apply pre-selection
        if box_size is not None:
            if not cc_contained_cut(event, box_size, exit_downstream): continue
        nselected += 1 
        
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

        ## Shift so the masked coordinates start at (0,0,0)
        coords_3d = coords_3d_raw[mask] + output_size/2

        ## Which axes to project onto
        keep_axes = [0, 2]
        keep_shape = (output_size[keep_axes[0]], output_size[keep_axes[1]])
        row = coords_3d[:, keep_axes[0]]
        col = coords_3d[:, keep_axes[1]]

        ## Make a 2D projection by summing duplicates
        this_sparse_2d = coo_matrix((values_3d, (row, col)), shape=keep_shape)
        this_sparse_2d .sum_duplicates() 

        ## Modify the array by removing below threshold hits
        if threshold > 0:
            mask = this_sparse_2d.data >= threshold
            this_sparse_2d.data = this_sparse_2d.data[mask]
            this_sparse_2d.row  = this_sparse_2d.row[mask]
            this_sparse_2d.col  = this_sparse_2d.col[mask]
        
        vertex = edep_tree.Event.Primaries[0]
        labels = get_truth_labels(vertex, groo_tree)

        ## Decide whether to proceed given the number of hits in the central region
        central_mask = (this_sparse_2d.row > (output_size[0]/4)-1) & (this_sparse_2d.row < (output_size[0]*3/4)-1) \
            & (this_sparse_2d.col > (output_size[1]/4)-1) & (this_sparse_2d.col < (output_size[1]*3/4)-1)
        
        if np.count_nonzero(this_sparse_2d.data[central_mask]) < min_hits:
            print("Rejected event with labels:", labels)
            print("Topology =", Topology.name_from_index(labels['topology']))
            print("Mode =", Mode.name_from_index(labels['mode']))
            print("N. hits (central) =", np.count_nonzero(this_sparse_2d.data), "(", np.count_nonzero(this_sparse_2d.data[central_mask]), ")")
            nrejected += 1
            continue
        
        ## At this point, save
        sparse_image_list .append(this_sparse_2d)
        event_id_list     .append(evt)
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

        ## Save enums defined when making the file
        fout.attrs['Topology_enum'] = json.dumps({m.name: m.value for m in Topology})
        fout.attrs['Mode_enum'] = json.dumps({m.name: m.value for m in Mode})
        
        for i, (sparse_image, event_id, label_struct) in enumerate(zip(sparse_image_list, event_id_list, label_list)):
            group = fout.create_group(str(i))
            group.create_dataset('data', data=sparse_image.data)
            group.create_dataset('row', data=sparse_image.row.astype(np.uint16))
            group.create_dataset('col', data=sparse_image.col.astype(np.uint16))
            group.create_dataset('label', data=label_struct, dtype=LABEL_DTYPE_EXP)
            group.attrs['shape'] = np.array(sparse_image.shape, dtype=np.uint16)
            group.attrs['event_id'] = np.uint32(event_id)

    ## Report summary
    print("Selected", nselected, "/", nevts, "events")
    print("Of which", nrejected, "cut with N. hits <", min_hits)
    print("Saved:", nselected - nrejected)
    ## Done
    
    
if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("Image maker")

    # Require an input file name and location to dump plots
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)

    ## Image size option
    parser.add_argument('--image_size', type=int, default=512)    

    ## Box size for containment (not necessarily the same as image size)
    parser.add_argument('--box_size', type=int, default=256)

    ## Do we require the muon to exit downstream of the box (if it exists)
    parser.add_argument('--exit_downstream', type=int, default=1)
    
    ## Allow a minimum number of hits cut
    parser.add_argument('--min_hits', type=int, default=1)

    ## Add a threshold option
    parser.add_argument('--threshold', type=float, default=0)
    
    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))

    make_images(args.input,
                args.output,
                args.image_size,
                args.box_size,
                args.exit_downstream,
                args.min_hits,
                args.threshold)
