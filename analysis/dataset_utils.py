from core.data.datasets import single_2d_dataset_ME, solo_ME_collate_fn, solo_ME_collate_fn_with_meta
from core.data.augmentations_2d import FirstRegionCrop
import torch
import MinkowskiEngine as ME
import numpy as np
import time
from collections import defaultdict


def get_dataset(input_dir, nevents, nom_transform=False, return_metadata=False):

    print("Loading", nevents," events from", input_dir)

    ## This is a relic as this was initially done for the FSD dataset
    if nom_transform == False:
        nom_transform = transforms.Compose([
            FirstRegionCrop((800, 256), (768, 256)),
        ])
    
    dataset = single_2d_dataset_ME(input_dir, \
                                   transform=nom_transform, \
                                   max_events=nevents,\
                                   return_metadata=return_metadata)
    this_collate = solo_ME_collate_fn
    if return_metadata: this_collate = solo_ME_collate_fn_with_meta
    
    loader = torch.utils.data.DataLoader(dataset,
                                         collate_fn=this_collate,
                                         batch_size=2048,
                                         shuffle=False,
                                         num_workers=8)

    print("Loaded", dataset.__len__(), "events")
    return dataset, loader


def image_loop(encoder, heads, loader, device, detailed_info=False, return_hidden=False):

    ## Record some timing info
    start = time.time()

    representations = defaultdict(list)     
    latent = []    ## This is the instance clustering space
    enc_latent = []    ## This is after the encoder (as passed to the clustering head) 
    hidden_latent = []
    cluster = []
    nhits = []
    maxQ = []
    sumQ = []
    labels = []
    filenames = []
    event_ids = []

    encoder.eval()
    encoder.to(device)
    for h in heads.values():
        h.eval()
        h.to(device)
    
    ## Loop over the images (discard any extra info returned by loader)
    for batch in loader:

        ## Start with the items that are always there
        batch_coords, batch_feats, batch_labels = batch[:3]
        ## If the loader returns the file names and event ids, collect those too
        batch_filenames = batch[3] if len(batch) > 3 else None
        batch_eventids = batch[4] if len(batch) > 4 else None
        
        batch_size = len(batch_labels)
        batch_coords = batch_coords.to(device)
        batch_feats = batch_feats.to(device)
        orig_batch = ME.SparseTensor(batch_feats, batch_coords, device=device)            

        dec_coords = orig_batch.decomposed_coordinates
        dec_feats  = orig_batch.decomposed_features
        
        ## Now do the forward passes            
        with torch.no_grad(): 
            encoded_instance_batch, encoded_cluster_batch = encoder(orig_batch, batch_size)
            if "clust" in heads: clust_batch = heads["clust"](encoded_cluster_batch, return_hidden=return_hidden)
            proj_batch = heads["proj"](encoded_instance_batch, return_hidden=return_hidden)

        ## Normalize the output (this is a bit fragile, but backwards compatible)
        if not return_hidden:
            proj_batch = {"proj_final": proj_batch}
            if "clust" in heads: clust_batch = {"clust_final": clust_batch}

        ## Get the representations all in one place
        for k, v in proj_batch.items():
            representations[k].append(v.detach().cpu())
        if "clust" in heads:
            for k, v in clust_batch.items():
                representations[k].append(v.detach().cpu())
        representations["encoder"].append(encoded_cluster_batch.detach().cpu())

        labels.extend(batch_labels)
        
        ## If desired, add a load more info, but this slows things down a lot...
        if detailed_info is True:
            nhits_batch = torch.tensor([f.shape[0] for f in dec_feats], device=device)
            sumQ_batch  = torch.stack([f.sum() for f in dec_feats])
            maxQ_batch  = torch.stack([f.max() for f in dec_feats])
        
            # Move everything to the CPU
            nhits.append(nhits_batch.cpu())
            sumQ.append(sumQ_batch.cpu())
            maxQ.append(maxQ_batch.cpu())

        if batch_filenames is not None:
            filenames.extend(batch_filenames)
        if batch_eventids is not None:
            event_ids.extend(batch_eventids)

    ## Turn into numpy arrays 
    representations = {k: torch.cat(v).numpy() for k, v in representations.items()}

    ## Return a dictionary to make my life easier
    out = {
        "labels": np.array(labels),
    }

    ## Merge in the encoded features
    out .update(representations)
    
    if "clust" in heads:
        cluster = out["clust_final"]
        out["clust"] = cluster
        out["clust_index"] = np.argmax(cluster, axis=1)
        out["clust_max"] = np.max(cluster, axis=1)
    
    if detailed_info is True:
        out["nhits"] = torch.cat(nhits).numpy()
        out["sumQ"] = torch.cat(sumQ).numpy()
        out["maxQ"] = torch.cat(maxQ).numpy()
    
    ## Add the filename and event id if applicable
    if filenames: out["filename"] = filenames
    if event_ids: out["event_id"] = event_ids
    
    print("Time to process events with image_loop:", time.time() - start)
    return out


## Function to reorder the order of clusters in the processed data
def reorder_clusters(data_processed, sim_processed):

    ## Rage quit if someone tries to call this without cluster labels
    if "clust" not in data_processed: return
        
    ## How frequently is each cluster selected in data
    unique, counts = np.unique(data_processed['clust_index'], return_counts=True)

    ## Order from most common to least common
    order = np.argsort(-counts)

    mapping_arr = np.zeros(unique.max() + 1, dtype=np.int64)
    mapping_arr[unique[order]] = np.arange(len(unique))

    data_processed['clust_index'] = mapping_arr[data_processed['clust_index']]
    sim_processed['clust_index']  = mapping_arr[sim_processed['clust_index']]
    
    data_processed['clust'] = data_processed['clust'][:,order]
    sim_processed['clust'] = sim_processed['clust'][:,order]
