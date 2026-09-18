import sys
import ROOT
from glob import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import coo_matrix
from collections import defaultdict
import json
from truth_labels import PARTICLE_STACK_DTYPE, EVENT_LABEL_DTYPE, CCTopology, Topology, Mode
import argparse
import matplotlib.patches as patches
from collections import Counter

## TODO: look at more "event shape" variables --> probably only interesting for 3D
## TODO: Do the k0L decay in the detector?
## TODO: Count detached vertices? --> neutron producing a detached vertex, or lambda? Or k0s in detector?

## This is not something to be taken lightly as it will dump out an image for every event...
make_plots = False

def position_to_numpy(position):
    return np.array(
        [position.X(), position.Y(), position.Z()],
        dtype=np.float64,
    )

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

def get_topology(part, cc):

    if part["nlambda0"]+part["nkapm"]+part["nka0"]+part['nantiprot']+part['nantineut'] > 0:
        if cc: return Topology.CCOther
        else: return Topology.NCOther
    if part["npipm"]+part["npi0"]>2:
        if cc: return Topology.CCNpi
        else: return Topology.NCNpi        
    if part["npipm"]+part["npi0"]>1:
        if cc: return Topology.CC2pi
        else: return Topology.NC2pi
    if part["npipm"]+part["npi0"]==0:
        if cc: return Topology.CC0pi
        else: return Topology.NC0pi
    if part["npipm"] == 1 and part["npi0"]==0:
        if cc: return Topology.CC1pipm
        else: return Topology.NC1pipm
    if part["npipm"] == 0 and part["npi0"]==1:
        if cc: return Topology.CC1pi0
        else: return Topology.NC1pi0

    return Topology.NONE

def get_cctopology(part, cc):

    ## Shortcut NC events
    if not cc: return CCTopology.NC

    ## Remove any complex events
    if part["nlambda0"]+part["nkapm"]+part["nka0"]+part['nantiprot']+part['nantineut'] > 0:
        return CCTopology.CCOther

    ## Everything else should be some combination of pions and protons
    if part["npipm"] == 0 and part["npi0"] == 0:
        if part["nproton"] == 0:
            return CCTopology.CC0pi0p
        elif  part["nproton"] == 1:
            return CCTopology.CC0pi1p
        else:
            return CCTopology.CC0piNp
    elif part["npipm"] == 1 and part["npi0"] == 0:
        if part["nproton"] == 0:
            return CCTopology.CC1pipm0pi0_0p
        elif  part["nproton"] == 1:
            return CCTopology.CC1pipm0pi0_1p
        else:
            return CCTopology.CC1pipm0pi0_Np
    elif part["npipm"] > 1 and part["npi0"] == 0:
        if part["nproton"] == 0:
            return CCTopology.CCNpipm0pi0_0p
        elif  part["nproton"] == 1:
            return CCTopology.CCNpipm0pi0_1p
        else:
            return CCTopology.CCNpipm0pi0_Np
    elif part["npipm"] == 0 and part["npi0"] == 1:
        if part["nproton"] == 0:
            return CCTopology.CC0pipm1pi0_0p
        elif  part["nproton"] == 1:
            return CCTopology.CC0pipm1pi0_1p
        else:
            return CCTopology.CC0pipm1pi0_Np        
    elif part["npipm"] == 0 and part["npi0"] > 1:
        if part["nproton"] == 0:
            return CCTopology.CC0pipmNpi0_0p
        elif  part["nproton"] == 1:
            return CCTopology.CC0pipmNpi0_1p
        else:
            return CCTopology.CC0pipmNpi0_Np
    else:
        if part["nproton"] == 0:
            return CCTopology.CCmixedpi_0p
        elif  part["nproton"] == 1:
            return CCTopology.CCmixedpi_1p
        else:
            return CCTopology.CCmixedpi_Np

    return CCTopology.NONE


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

def check_unrecognized_pdgs(pdgs):
    recognized = {
        12, 14, 16,
        -12, -14, -16,
        2212, -2212,
        2112, -2112,
        211, -211, 111,
        321, -321,
        130, 310, 311, -311,
        22, 11, -11, 13, -13,
        3122, -3122,
        1000010020,  # deuteron
        1000010030,  # tritium
        1000020030,  # helium-3
        1000020040,  # alpha
        1000180400,  # argon remnant
    }

    remaining = [
        pdg for pdg in pdgs
        if pdg not in recognized
        and not (1000020060 <= pdg < 1000180400)
    ]

    if remaining:
        print("Remaining PDG list:", remaining)
        

## Create a PARTICLE_STACK_DTYPE record
def get_particle_counts(pdg_list):

    result = np.zeros((), dtype=PARTICLE_STACK_DTYPE)
    counts = Counter(pdg_list)

    result["nproton"]   = counts[2212]
    result["nantiprot"] = counts[-2212]
    result["nneutron"]  = counts[2112]
    result["nantineut"] = counts[-2112]
    result["npip"] = counts[211]
    result["npim"] = counts[-211]
    result["npi0"] = counts[111]
    result["nkap"] = counts[321]
    result["nkam"] = counts[-321]
    result["nka0"] = sum(counts[pdg] for pdg in (130, 310, 311, -311))
    result["ngamma"] = counts[22]
    result["nepm"] = counts[11] + counts[-11]
    result["nmuon"] = counts[13] + counts[-13]
    result["nlambda0"] = counts[3122] + counts[-3122]
    result["ndeuteron"] = counts[1000010020]
    result["ntritium"]  = counts[1000010030]
    result["nhelium3"]  = counts[1000020030]
    result["nalpha"]    = counts[1000020040]
    result["nnuclfrag"] = sum(count for pdg, count in counts.items() if 1000020060 <= pdg < 1000180400)

    ## Compound counts
    result["npipm"] = result["npip"] + result["npim"]
    result["nkapm"] = result["nkap"] + result["nkam"]
    result["nem"] = result["nepm"] + result["ngamma"]
    result["ncharged"] = result["nproton"] + result["npipm"] + result["nkapm"] + result["nmuon"]
    result["ncluster"] = result["ndeuteron"] + result["ntritium"] + result["nhelium3"] + result["nalpha"] + result["nnuclfrag"]

    ## Scream if any particles we weren't expecting made it through
    check_unrecognized_pdgs(pdg_list)

    return result


def get_event_features(vertex,
                       groo,
                       particle_truth,
                       particle_visible,
                       deposition_features):
    event_labels = np.zeros((), dtype=EVENT_LABEL_DTYPE)

    particles = list(vertex.Particles)
    pdg_list = [int(particle.GetPDGCode()) for particle in particles]

    nu_4mom = get_neutrino_4mom(groo)
    outgoing_4mom = particles[0].GetMomentum()


    cc = is_ccinc(pdg_list)
    event_labels["cc"] = cc
    event_labels["enu"] = nu_4mom.E() / 1000.0
    event_labels["q0"] = (nu_4mom.E() - outgoing_4mom.E()) / 1000.0
    event_labels["mode"] = np.int8(get_mode(str(groo.EvtCode)).value)

    # Derived particle-topology labels
    event_labels["topology_truth"] = np.int8(get_topology(particle_truth, cc).value)
    event_labels["topology_visible"] = np.int8(get_topology(particle_visible, cc).value)
    event_labels["cctopology_truth"] = np.int8(get_cctopology(particle_truth, cc).value)
    event_labels["cctopology_visible"] = np.int8(get_cctopology(particle_visible, cc).value)

    # Energy deposited in spheres around the true vertex
    edep = deposition_features["edep_inside"]
    event_labels["edep_5mm"] = np.float32(edep[5.0])
    event_labels["edep_10mm"] = np.float32(edep[10.0])
    event_labels["edep_20mm"] = np.float32(edep[20.0])   
    event_labels["edep_50mm"] = np.float32(edep[50.0])
    
    return event_labels

def get_excluded_ids(event,
                     low_energy_cut=10.0):

    neutron_lineage = set()
    k0l_lineage = set()
    lowe_lineage = set()

    for traj in event.Trajectories:
        track_id = int(traj.GetTrackId())
        parent_id = int(traj.GetParentId())
        pdg = int(traj.GetPDGCode())

        ## Neutrons and descendents
        if abs(pdg) == 2112 or parent_id in neutron_lineage:
            neutron_lineage .add(track_id)

        ## Same treatment for K0L
        if pdg == 130 or parent_id in k0l_lineage:
            k0l_lineage .add(track_id)

        ## Remove very low energy fluff
        energy =  traj.GetInitialMomentum().E()
        if energy < low_energy_cut or parent_id in lowe_lineage:
            lowe_lineage .add(track_id)

    return {
        "all": (neutron_lineage | k0l_lineage | lowe_lineage),
        "neutron": neutron_lineage,
        "k0l": k0l_lineage,
        "lowe": lowe_lineage,
    }

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
def hadron_contained_cut(event, excluded, bbox):
    
    ## Get the primary lepton (assumes a well ordered stack)
    out_lep = event.Primaries[0].Particles[0]
    
    ## Get all neutrons and neutron descendents in the event
    neutron_ids = excluded["neutron"]
    
    ## Get a list of low energy truth trajectories (may be quite long)
    low_energy_ids = excluded["lowe"]

    ## Get a list of k0l daughters
    k0l_ids = excluded["k0l"]
    
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


## Energy deposited within a spherical volume (for vertex activity)
## Defaults to assuming the vertex is always centered on (0,0,0)
def segment_fraction_inside_sphere(p0, p1, radius, center=[0,0,0]):

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    d = p1 - p0
    q = p0 - center

    a = np.dot(d, d)

    # Zero-length segment
    if a <= 0.0: return float(np.dot(q, q) <= radius * radius)

    b = 2.0 * np.dot(q, d)
    c = np.dot(q, q) - radius * radius

    discriminant = b * b - 4.0 * a * c

    if discriminant < 0.0:
        # No boundary crossing. The segment is either wholly inside
        # or wholly outside.
        midpoint = 0.5 * (p0 + p1)
        return float(
            np.dot(midpoint - center, midpoint - center)
            <= radius * radius
        )

    sqrt_discriminant = np.sqrt(max(discriminant, 0.0))

    t0 = (-b - sqrt_discriminant) / (2.0 * a)
    t1 = (-b + sqrt_discriminant) / (2.0 * a)

    if t0 > t1:
        t0, t1 = t1, t0

    # The part inside the sphere lies between the two roots.
    t_enter = max(0.0, t0)
    t_exit = min(1.0, t1)

    return max(0.0, t_exit - t_enter)


def get_deposition_features(event,
                            excluded,
                            vertex_radii=(5.0, 10.0, 20.0, 50.0),
                            visible_radius=20.0,
                            visible_min_edep=0.1,
                            vertex_position=(0,0,0)):
    """
    Calculate:
      * total deposited energy inside each vertex-centered sphere;
      * energy deposited outside visible_radius by each primary ID;
      * the set of primary IDs satisfying the visibility requirement.

    GetPrimaryId() attributes descendant deposits to their original primary.

    Energies are returned in the units of TG4HitSegment::GetEnergyDeposit(),
    normally MeV.
    """
    vertex_position = np.asarray(vertex_position, dtype=np.float64)
    vertex_radii = tuple(float(r) for r in vertex_radii)

    edep_inside = {radius: 0.0 for radius in vertex_radii}
    primary_edep_outside = defaultdict(float)

    ## Vertices to ignore
    excluded_ids = excluded["all"]

    for detector_name, segments in event.SegmentDetectors:
        for segment in segments:
            
            energy = float(segment.GetEnergyDeposit())
            if energy <= 0.0: continue

            p0 = position_to_numpy(segment.GetStart())
            p1 = position_to_numpy(segment.GetStop())

            ## Keep track of energy inside each vertex box (sphere) independent of contributor
            for radius in vertex_radii:
                fraction_inside = segment_fraction_inside_sphere(p0, p1, radius, vertex_position)
                edep_inside[radius] += energy * fraction_inside

            ## What produced this segment
            contributors = segment.GetContributors()
            if len(contributors) == 0: continue
            contributor_id = contributors[0]

            ## If this is in the ignore list... ignore
            if contributor_id in excluded_ids: continue

            ## What primary produced this?
            primary_id = segment.GetPrimaryId()

            fraction_inside = segment_fraction_inside_sphere(p0, p1, visible_radius, vertex_position)
            fraction_outside = max(0.0, 1.0 - fraction_inside)

            primary_edep_outside[primary_id] += (energy * fraction_outside)

    visible_primary_ids = {
        primary_id
        for primary_id, energy in primary_edep_outside.items()
        if energy > visible_min_edep
    }

    return {
        "edep_inside": edep_inside,
        "primary_edep_outside": dict(primary_edep_outside),
        "visible_primary_ids": visible_primary_ids,
    }


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
            
            p0 = position_to_numpy(p0_tlv)
            p1 = position_to_numpy(p1_tlv)
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
    bbox_pixel = np.array([-bbox_size - bbox_offset,
                           bbox_size - bbox_offset])
    
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

        ## Get a dictionary with a description of all of the IDs which we exclude
        excluded_ids = get_excluded_ids(event)
        
        if hadron_cont and not hadron_contained_cut(event, excluded_ids, bbox):
            nhadfail += 1
            continue

        ## If we pass the main selection cuts, get truth info for labels
        vertex = event.Primaries[0]
        vertex_particles = list(vertex.Particles)

        ## Check that the first outgoing particle is indeed a muon...
        if abs(int(vertex_particles[0].GetPDGCode())) != 13:
            raise RuntimeError("Expected first vertex particle to be the outgoing muon")

        ## Skip the muon
        vertex_particles = vertex_particles[1:]
        
        pdg_list_truth = [int(particle.GetPDGCode()) for particle in vertex_particles]

        particle_truth = get_particle_counts(pdg_list_truth)

        ## Get info for visible labels
        deposition_features = get_deposition_features(event, excluded_ids)

        visible_pdg_list = [
            p.GetPDGCode()
            for p in vertex_particles
            if int(p.GetTrackId()) in deposition_features["visible_primary_ids"]
        ]
        particle_visible = get_particle_counts(visible_pdg_list)

        event_labels = get_event_features(vertex,
                                          groo_tree,
                                          particle_truth,
                                          particle_visible,
                                          deposition_features)
        
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
            print("Rejected event with:")
            print("--- Mode =", Mode.name_from_index(event_labels['mode']))
            print("--- True CC topology =", CCTopology.name_from_index(event_labels['cctopology_truth']))
            print("--- Vis. CC topology =", CCTopology.name_from_index(event_labels['cctopology_visible']))
            print("--- N. hits =", np.count_nonzero(values_3d))
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
            'event_labels':  event_labels,
            'particle_truth': particle_truth,
            'particle_visible': particle_visible,
        })

        ## Optionally dump out some files to have a look at
        if make_plots:
            cmap = cm.turbo.copy()
            cmap.set_under("#F0F0F0")
            
            ## Rectangle is [x, y]
            xz_rect = patches.Rectangle((bbox_pixel[0][2]+256+offset[2], bbox_pixel[0][0]+256+offset[0]),
                                        bbox_pixel[1][2]-bbox_pixel[0][2],
                                        bbox_pixel[1][0]-bbox_pixel[0][0],
                                        linewidth=1, edgecolor='red', facecolor='none',
                                        linestyle='dashed')
            
            plt.figure(figsize=(7,7))
            ## This plots as [row, col] --> [x, y]
            plt.imshow(this_xz.toarray(), origin='lower', vmin=1e-6, cmap=cmap)
            plt.gca().add_patch(xz_rect)
            plt.xlabel('Z')
            plt.ylabel('X')
            plt.tight_layout()
            plt.savefig("plots/image_"+str(evt)+"_xz.png")
            plt.close()

            xy_rect = patches.Rectangle((bbox_pixel[0][1]+256+offset[1], bbox_pixel[0][0]+256+offset[0]),
                                        bbox_pixel[1][1]-bbox_pixel[0][1],
                                        bbox_pixel[1][0]-bbox_pixel[0][0],
                                        linewidth=1, edgecolor='red', facecolor='none',
                                        linestyle='dashed')
            plt.figure(figsize=(7,7))
            plt.imshow(this_xy.toarray(), origin='lower', vmin=1e-6, cmap=cmap)
            plt.gca().add_patch(xy_rect)
            plt.xlabel('Y')
            plt.ylabel('X')
            plt.tight_layout()
            plt.savefig("plots/image_"+str(evt)+"_xy.png")
            plt.close()
            
    ## Write the images to an hdf5 file
    with h5py.File(output_file_name, 'w', libver='latest') as fout:
        N = len(event_data_list)
        fout.attrs['N'] = N
        fout.attrs['particle_dtype'] = PARTICLE_STACK_DTYPE.descr
        fout.attrs['event_label_dtype'] = EVENT_LABEL_DTYPE.descr
        fout.attrs['Topology_enum'] = json.dumps({m.name: m.value for m in Topology})
        fout.attrs['CCTopology_enum'] = json.dumps({m.name: m.value for m in CCTopology})
        fout.attrs['Mode_enum'] = json.dumps({m.name: m.value for m in Mode})
        fout.attrs['shape_3d'] = np.array(output_full_size, dtype=np.uint16)
        fout.attrs['shape_xz'] = np.array((output_full_size[0], output_full_size[2]), dtype=np.uint16)
        fout.attrs['shape_xy'] = np.array((output_full_size[0], output_full_size[1]), dtype=np.uint16)

        xyz_data, xyz_coords = [], []
        xz_data, xz_row, xz_col = [], [], []
        xy_data, xy_row, xy_col = [], [], []
        
        particle_truth_array = np.empty(N, dtype=PARTICLE_STACK_DTYPE)
        particle_visible_array = np.empty(N, dtype=PARTICLE_STACK_DTYPE)
        event_labels_array = np.empty(N, dtype=EVENT_LABEL_DTYPE)
        event_ids = np.empty(N, dtype=np.uint32)
        
        xyz_off = np.zeros(N + 1, dtype=np.int64)
        xz_off  = np.zeros(N + 1, dtype=np.int64)
        xy_off  = np.zeros(N + 1, dtype=np.int64)

        for i, ev in enumerate(event_data_list):
            xyz_data.append(ev['values_3d'].astype(np.float32))
            xyz_coords.append(ev['coords_3d'].astype(np.uint16))
            xz_data.append(ev['image_xz'].data.astype(np.float32))
            xz_row.append(ev['image_xz'].row.astype(np.uint16))
            xz_col.append(ev['image_xz'].col.astype(np.uint16))
            xy_data.append(ev['image_xy'].data.astype(np.float32))
            xy_row.append(ev['image_xy'].row.astype(np.uint16))
            xy_col.append(ev['image_xy'].col.astype(np.uint16))

            ## Event records
            particle_truth_array[i] = ev["particle_truth"]
            particle_visible_array[i] = ev["particle_visible"]
            event_labels_array[i] = ev["event_labels"]
            event_ids[i] = np.uint32(ev["event_id"])

            xyz_off[i+1] = xyz_off[i] + len(ev['values_3d'])
            xz_off[i+1]  = xz_off[i]  + len(ev['image_xz'].data)
            xy_off[i+1]  = xy_off[i]  + len(ev['image_xy'].data)

        def _cat1(chunks, dt): return np.concatenate(chunks).astype(dt) if chunks else np.zeros(0, dt)
        def _cat2(chunks, dt, c):
            ne = [x for x in chunks if x.size]
            return np.concatenate(ne).astype(dt) if ne else np.zeros((0, c), dt)

        ## Optionally include compression (none by default)
        cargs = {} #dict(compression='gzip', compression_opts=1, shuffle=True)
        fout.create_dataset('xyz_data',   data=_cat1(xyz_data, np.float32), **cargs)
        fout.create_dataset('xyz_coords', data=_cat2(xyz_coords, np.uint16, 3), **cargs)
        fout.create_dataset('xyz_offsets', data=xyz_off)
        fout.create_dataset('xz_data', data=_cat1(xz_data, np.float32), **cargs)
        fout.create_dataset('xz_row',  data=_cat1(xz_row, np.uint16), **cargs)
        fout.create_dataset('xz_col',  data=_cat1(xz_col, np.uint16), **cargs)
        fout.create_dataset('xz_offsets', data=xz_off)
        fout.create_dataset('xy_data', data=_cat1(xy_data, np.float32), **cargs)
        fout.create_dataset('xy_row',  data=_cat1(xy_row, np.uint16), **cargs)
        fout.create_dataset('xy_col',  data=_cat1(xy_col, np.uint16), **cargs)
        fout.create_dataset('xy_offsets', data=xy_off)
        fout.create_dataset("particle_truth", data=particle_truth_array)
        fout.create_dataset("particle_visible", data=particle_visible_array)
        fout.create_dataset("event_labels", data=event_labels_array)
        fout.create_dataset("event_id", data=event_ids)
            
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
