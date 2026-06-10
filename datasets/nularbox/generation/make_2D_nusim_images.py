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
from PIL import Image

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
    labels["nem"] = sum(1 for x in pdg_list if abs(x) in [22, 11])
    pdg_list = [x for x in pdg_list if abs(x) not in [22, 11]]
    labels["nlambda0"] = sum(1 for x in pdg_list if abs(x) == 3122)
    pdg_list = [x for x in pdg_list if abs(x) != 3122]    
    labels["nstrange"] = sum(1 for x in pdg_list if abs(x) in [3222, 3112, 3212])
    pdg_list = [x for x in pdg_list if abs(x) not in [3222, 3112, 3212]]    
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
    labels["nnuclfrag"] = sum(1 for x in pdg_list if (x >= 1000020060 and x < 1000180400))
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


def muon_exits_downstream(point, bbox):

    px = point[0]
    py = point[1]
    pz = point[2]

    ## Shortcut negative z values
    if pz <= bbox[1][2]: return False
    
    t = bbox[1][2] / pz
    x_hit = px * t
    y_hit = py * t
    return bbox[0][0] <= x_hit <= bbox[1][0] and bbox[0][1] <= y_hit <= bbox[1][1]

## Check whether:
## - The muon exits the volume of interest
## - (Optionally) the muon exits downstream, aka in the +z direction
def exiting_muon(event, muon_id, bbox, downstream=False):

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
            if pos[2] > bbox[1][2] or pos[2] < bbox[0][2]:

                ## If we require a downstream muon, check
                if downstream: return muon_exits_downstream(pos, bbox)
                else: return True

            ## If not, consider x and y
            if pos[0] > bbox[1][0] or pos[0] < bbox[0][0]:
                if downstream: return False
                else: return True
            if pos[1] > bbox[1][1] or pos[1] < bbox[0][1]:
                if downstream: return False
                else: return True

    return False


## This is designed to select a set of events in which:
## - No other activity escapes the volume of interest except for neutrons or low energy junk, or neutrinos
## - Where the volume of interest can be a defined cube of voxels
def hadron_contained_cut(event, bbox):
    
    ## Get the primary lepton (assumes a well ordered stack)
    out_lep = event.Primaries[0].Particles[0]
    
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
            
            ## See if this is outside my bounding box
            pos = seg[1][n].GetStop()
            if np.any(pos.Vect() < bbox[0]) or np.any(pos.Vect() > bbox[1]): return False

            ## Just be really sure...
            pos = seg[1][n].GetStart()
            if np.any(pos.Vect() < bbox[0]) or np.any(pos.Vect() > bbox[1]):
                print("Removed hadron that started outside my bounding box")
                return False
            
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
                if t_next >= 1.0 or np.all(voxel==voxel_end): break
                
                # Step along all axes with a crossing at t_next
                axes = np.where(np.abs(t_max - t_next) < 1E-10)[0]
                for axis in axes:
                    voxel[axis] += step[axis]
                    t_max[axis] += t_delta[axis]
                t = t_next

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
                offset,
                box_size,
                box_offset,
                exit_downstream,
                min_hits,
                threshold,
                hadron_cont):

    output_full_size = np.array([image_size, image_size, image_size])
    output_half_size = output_full_size//2
    offset = np.array(offset)

    
    ## Uniform and small pixel pitch
    ## Uses mm, the default output unit for edep-sim
    dx, dy, dz = 3.72, 3.72, 3.72
    voxel_size = np.array([dx, dy, dz])

    ## Origin for the grid, offset to avoid the vertex being at a bin edge, maybe better to jitter?
    origin = voxel_size/2

    ## Set the bounding box for defining containment
    bbox_size = output_half_size
    bbox_offset = offset
    
    ## Allow for explicit bbox setting
    if box_size is not None: bbox_size = np.array([box_size, box_size, box_size])//2
    if box_offset is not None: bbox_offset = np.array(box_offset)
    
    bbox = np.array([(-bbox_size - bbox_offset) * voxel_size + origin,
                     (bbox_size - bbox_offset) * voxel_size + origin])
   
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
    event_data_list = []

    nnc       = 0
    nmuonfail = 0
    nhadfail  = 0
    nminhits  = 0
    nselected = 0
    
    ## Loop over events
    nevts  = edep_tree.GetEntries()
    for evt in range(nevts):
        edep_tree.GetEntry(evt)
        groo_tree.GetEntry(evt)

        ## Add a check for empty images
        if len(event.Trajectories) <=1: continue

        ## Get the primary lepton (assumes a well ordered stack)
        out_lep = event.Primaries[0].Particles[0]

        ## Check this is a numuCC event
        if abs(out_lep.GetPDGCode()) != 13:
            nnc += 1
            continue
        
        ## Check if the muon exits
        if not exiting_muon(event, out_lep.GetTrackId(), bbox, exit_downstream):
            nmuonfail += 1
            continue
    
        if hadron_cont and not hadron_contained_cut(event, bbox):
            nhadfail += 1
            continue

        ## If we pass the main selection cuts, get truth info
        vertex = edep_tree.Event.Primaries[0]
        labels = get_truth_labels(vertex, groo_tree)

        ## Get voxelised 3D hits
        coords_3d_raw, values_3d_raw = get_3D_image_from_event(event, origin, voxel_size)
        x_raw = coords_3d_raw[:, 0]
        y_raw = coords_3d_raw[:, 1]
        z_raw = coords_3d_raw[:, 2]
        
        ## Restrict to an area around the vertex and mask out the image
        mask = ((x_raw >= -output_half_size[0] - offset[0]) & (x_raw < output_half_size[0] - offset[0]) &
                (y_raw >= -output_half_size[1] - offset[1]) & (y_raw < output_half_size[1] - offset[1]) &
                (z_raw >= -output_half_size[2] - offset[2]) & (z_raw < output_half_size[2] - offset[2]))
        values_3d = values_3d_raw[mask]

        ## Shift so the masked coordinates start at (offset))
        coords_3d = coords_3d_raw[mask] + offset + output_half_size 

        ## Apply threshold to 3D hits only
        if threshold > 0:
            mask = values_3d >= threshold
            values_3d = values_3d[mask]
            coords_3d = coords_3d[mask]

        ## Check we're above the minimum number of hits (in 3D)
        if np.count_nonzero(values_3d) < min_hits:
            print("Rejected event with labels:", labels)
            print("Topology =", Topology.name_from_index(labels['topology']))
            print("Mode =", Mode.name_from_index(labels['mode']))
            print("N. hits =", np.count_nonzero(values_3d))
            nminhits += 1
            continue        
        nselected += 1
        
        if coords_3d.size > 0:
            assert coords_3d.min() >= 0, f"Negative coordinate: {coords_3d.min()}"
            assert coords_3d.max() < image_size, f"Coordinate too large: {coords_3d.max()}"
        
        ## Project onto XZ (and sum duplicates)
        shape_xz = (output_full_size[0], output_full_size[2])
        row_xz = coords_3d[:, 0]
        col_xz = coords_3d[:, 2]
        this_xz = coo_matrix((values_3d, (row_xz, col_xz)), shape=shape_xz)
        this_xz .sum_duplicates()
        #img = Image.fromarray((arr * 255).astype(np.uint8))
        #img.save("plots/image_"+str(evt)+".png")

        ## Project onto XY (and sum duplicates)
        shape_xy = (output_full_size[0], output_full_size[1])
        row_xy = coords_3d[:, 0]
        col_xy = coords_3d[:, 1]
        this_xy = coo_matrix((values_3d, (row_xy, col_xy)), shape=shape_xy)
        this_xy .sum_duplicates()        
        
        ## Keep track of events that get this far
        event_data_list.append({
            'image_xz':  this_xz,
            'image_xy':  this_xy,
            'coords_3d':  coords_3d,
            'values_3d':  values_3d,
            'event_id':   evt,
            'label':      labels,
        })

        ## Optionally dump out some files to have a look at
        if make_plots:
            plt.figure(figsize=(7,7))
            plt.imshow(this_xz.toarray(), origin='lower')
            plt.savefig("plots/image_"+str(evt)+"_xz.png")
            plt.close()
            plt.figure(figsize=(7,7))
            plt.imshow(this_xy.toarray(), origin='lower')
            plt.savefig("plots/image_"+str(evt)+"_xy.png")
            plt.close()            
            
    ## Write the images to an hdf5 file
    with h5py.File(output_file_name, 'w') as fout:
        
        ## Save the number of images in the file
        fout.attrs['N'] = len(event_data_list)

        ## Store label_struct schema
        fout.attrs['label_dtype'] = LABEL_DTYPE_EXP.descr

        ## Save enums defined when making the file
        fout.attrs['Topology_enum'] = json.dumps({m.name: m.value for m in Topology})
        fout.attrs['Mode_enum'] = json.dumps({m.name: m.value for m in Mode})

        for i, ev in enumerate(event_data_list):
            group = fout.create_group(str(i))
            
            # 3D sparse
            group.create_dataset('data_xyz',   data=ev['values_3d'], dtype=np.float32)
            group.create_dataset('coords_xyz', data=ev['coords_3d'].astype(np.uint16))
            group.attrs['shape_3d'] = np.array(output_full_size, dtype=np.uint16)
            
            # XZ projection
            group.create_dataset('data_xz', data=ev['image_xz'].data, dtype=np.float32)
            group.create_dataset('row_xz',  data=ev['image_xz'].row.astype(np.uint16))
            group.create_dataset('col_xz',  data=ev['image_xz'].col.astype(np.uint16))
            group.attrs['shape_xz'] = np.array(ev['image_xz'].shape, dtype=np.uint16)
            
            # XY projection
            group.create_dataset('data_xy', data=ev['image_xy'].data, dtype=np.float32)
            group.create_dataset('row_xy',  data=ev['image_xy'].row.astype(np.uint16))
            group.create_dataset('col_xy',  data=ev['image_xy'].col.astype(np.uint16))
            group.attrs['shape_xy'] = np.array(ev['image_xy'].shape, dtype=np.uint16)
            
            ## Shared info
            group.create_dataset('label', data=ev['label'], dtype=LABEL_DTYPE_EXP)
            group.attrs['event_id'] = np.uint32(ev['event_id'])

    ## Report summary
    print("Selected", nselected, "/", nevts, "events")
    print("Rejected", nnc, "/", nevts, "as NC")
    print("Rejected", nmuonfail, "/", nevts, "for muon kinematics")
    print("Rejected", nhadfail, "/", nevts, "for uncontained hadrons")
    print("Rejected", nminhits, "/", nevts, "which had N. hits <", min_hits)
    ## Done
    
    
if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("Image maker")

    # Require an input file name and location to dump plots
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)

    ## Image size option
    parser.add_argument('--image_size', type=int, default=512)

    ## Add vertex offset option    
    parser.add_argument('--offset', type=int, nargs=3, default=[0, 0, 0], metavar=('OX', 'OY', 'OZ'))

    ## Box size and offset for containment (otherwise image size and offset will be used)
    parser.add_argument('--box_size', type=int, default=None)
    parser.add_argument('--box_offset', type=int, nargs=3, default=None, metavar=('OX', 'OY', 'OZ'))

    ## Do we require the muon to exit downstream of the box (if it exists)
    parser.add_argument('--exit_downstream', type=int, choices=[0,1], default=1)
    
    ## Allow a minimum number of hits cut
    parser.add_argument('--min_hits', type=int, default=1)

    ## Add a threshold option
    parser.add_argument('--threshold', type=float, default=0)

    ## Add containment option
    parser.add_argument('--hadron_cont', type=int, choices=[0,1], default=1)
    
    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))

    make_images(args.input,
                args.output,
                args.image_size,
                args.offset,
                args.box_size,
                args.box_offset,
                bool(args.exit_downstream),
                args.min_hits,
                args.threshold,
                bool(args.hadron_cont))
