from torch.utils.data import Dataset
import h5py
import numpy as np
import os
from glob import glob
from bisect import bisect
import MinkowskiEngine as ME
import torch
import time

class paired_2d_dataset_ME(Dataset):

    def __init__(self, infile_dir, nom_transform, aug_transform=None, max_events=None):
        self.hdf5_files = sorted(glob(os.path.join(infile_dir, '*.h5')))
        self.file_indices = []
        self.nom_transform = nom_transform
        self.aug_transform = aug_transform
        self.max_events = max_events

        ## Sort out the file map
        self.create_file_indices()

        ## Apply some limitation to the size
        if self.max_events and max_events < self.length:
            self.length = self.max_events

            
    def create_file_indices(self):
        cumulative_size = 0

        for file in self.hdf5_files:
            self.file_indices.append(cumulative_size)
            with h5py.File(file, 'r', libver='latest') as f:            
                cumulative_size += f.attrs['N']

        self.file_indices.append(cumulative_size)
        self.length = cumulative_size

    def apply_aug_with_retry(self, coords, feats, max_retries=100):
        for _ in range(max_retries):
            out_coords, out_feats = self.aug_transform(coords, feats)
            if out_feats.size > 0:
                return out_coords, out_feats
        ## If no valid augmentation has been found, bail
        raise RuntimeError("Augmentation failed on initial image with feats.size =", feats.size)
    
    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        file_index = bisect(self.file_indices, idx)-1
        this_idx = idx - self.file_indices[file_index]

        file_path = self.hdf5_files[file_index]

        with h5py.File(file_path, 'r', libver='latest') as f:
            group = f[str(this_idx)]
            data = group['data_xz'][:]
            row  = group['row_xz'][:]
            col  = group['col_xz'][:]

        ## Use the format that ME requires
        ## Note that we can't build the sparse tensor here because ME uses some sort of global indexing
        ## And this function is replicated * num_workers
        raw_coords = np.vstack((row, col)).T
        raw_feats = data.reshape(-1, 1)  # Reshape data to be of shape (N, 1)
        
        ## Apply transforms to augment the data
        if not self.aug_transform:
            raw_coords, raw_feats = self.nom_transform(raw_coords, raw_feats)
            aug1_coords,aug1_feats = raw_coords,raw_feats
            aug2_coords,aug2_feats = raw_coords,raw_feats
        else:
            ## Make sure the images aren't empty...            
            aug1_coords, aug1_feats = self.apply_aug_with_retry(raw_coords, raw_feats)
            aug2_coords, aug2_feats = self.apply_aug_with_retry(raw_coords, raw_feats)
            raw_coords, raw_feats   = self.nom_transform(raw_coords, raw_feats)

        return aug1_coords, aug1_feats, aug2_coords, aug2_feats, raw_coords, raw_feats

def triple_ME_collate_fn(batch):
    aug1_coords, aug1_feats, aug2_coords, aug2_feats, raw_coords, raw_feats = zip(*batch)

    # Create batched coordinates for the SparseTensor input
    aug1_bcoords = ME.utils.batched_coordinates(aug1_coords)
    aug2_bcoords = ME.utils.batched_coordinates(aug2_coords)
    raw_bcoords  = ME.utils.batched_coordinates(raw_coords)

    # Concatenate all lists
    aug1_bfeats = torch.from_numpy(np.concatenate(aug1_feats, 0)).float()
    aug2_bfeats = torch.from_numpy(np.concatenate(aug2_feats, 0)).float()
    raw_bfeats  = torch.from_numpy(np.concatenate(raw_feats, 0)).float()

    return aug1_bcoords, aug1_bfeats, aug2_bcoords, aug2_bfeats, raw_bcoords, raw_bfeats


def cat_ME_collate_fn(batch):
    aug1_coords, aug1_feats, aug2_coords, aug2_feats, _, _ = zip(*batch)

    coords_list = list(aug1_coords) + list(aug2_coords)
    feats_list  = list(aug1_feats)  + list(aug2_feats)
    
    # Create batched coordinates for the SparseTensor input
    cat_bcoords = ME.utils.batched_coordinates(coords_list)

    # Concatenate all lists
    cat_bfeats = torch.from_numpy(np.concatenate(feats_list, axis=0)).float()

    return cat_bcoords, cat_bfeats, len(batch)*2


class single_2d_dataset_ME(Dataset):

    def __init__(self, infile_dir, transform, max_events=None, return_metadata=False):
        self.hdf5_files = sorted(glob(os.path.join(infile_dir, '*.h5')))
        self.file_indices = []
        self.transform = transform
        self.max_events = max_events
        self.return_metadata = return_metadata
        
        ## Sort out the file map
        self.create_file_indices()

        ## Apply some limitation to the size
        if self.max_events and max_events < self.length:
            self.length = self.max_events
         
    def create_file_indices(self):
        cumulative_size = 0
        
        for file in self.hdf5_files:
            self.file_indices.append(cumulative_size)
            f = h5py.File(file, 'r', libver='latest')
            cumulative_size += f.attrs['N']
            f .close()
        self.file_indices.append(cumulative_size)
        self.length = cumulative_size

    def apply_aug_with_retry(self, coords, feats, max_retries=100):
        for _ in range(max_retries):
            out_coords, out_feats = self.transform(coords, feats)
            if out_feats.size > 0:
                return out_coords, out_feats
        ## If no valid augmentation has been found, bail
        raise RuntimeError("Augmentation failed on initial image with feats.size =", feats.size)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):

        file_index = bisect(self.file_indices, idx)-1
        this_idx = idx - self.file_indices[file_index]

        file_path = self.hdf5_files[file_index]

        with h5py.File(file_path, 'r', libver='latest') as f:
            group = f[str(this_idx)]
            data = group['data_xz'][:]
            row  = group['row_xz'][:]
            col  = group['col_xz'][:]
            
            # Check for 'label' dataset and fall back if missing
            label = -1
            if 'label' in group: label = group['label'][()]

            ## Get the event_id if requested
            event_id = group.attrs.get("event_id", this_idx)
            
        ## Use the format that ME requires
        ## Note that we can't build the sparse tensor here because ME uses some sort of global indexing
        ## And this function is replicated * num_workers
        coords = np.vstack((row, col)).T 
        feats = data.reshape(-1, 1)  # Reshape data to be of shape (N, 1)            
        coords, feats = self.apply_aug_with_retry(coords, feats)

        if self.return_metadata:
            filename = os.path.basename(file_path)
            return coords, feats, label, filename, event_id        
        return coords, feats, label
    
def solo_ME_collate_fn(batch):
    coords, feats, labels = zip(*batch)
    
    # Create batched coordinates for the SparseTensor input
    bcoords  = ME.utils.batched_coordinates(coords)
    
    # Concatenate all lists
    bfeats  = torch.from_numpy(np.concatenate(feats, 0)).float()
    
    return bcoords, bfeats, labels

## label_clamp allows a configurable maximum to be provided
## Derived labels allows raw labels to be added or otherwise manipulated
def solo_labelled_collate_fn(batch,
                             label_clamp=None,
                             derived_labels=None):
    coords, feats, labels = zip(*batch)
    
    ## Create batched coordinates for the SparseTensor input
    bcoords  = ME.utils.batched_coordinates(coords)
    
    ## Concatenate all lists
    bfeats  = torch.from_numpy(np.concatenate(feats, 0)).float()
    
    ## Batch the labels into dict of tensors
    label_names = labels[0].dtype.names
    blabels = {}

    ## Make a batched set of labels
    for name in label_names:
        blabels[name] = torch.from_numpy( np.array([l[name] for l in labels]))
        
    ## Compute derived labels from unclamped values
    if derived_labels:
        for name, fn in derived_labels.items():
            blabels[name] = fn(blabels)

    ## Now clamp everything if required
    if label_clamp:
        for name, clamp_val in label_clamp.items():
            if name in blabels:
                blabels[name] = torch.clamp(blabels[name], 0, clamp_val)
    
    return bcoords, bfeats, blabels, len(batch)


def solo_ME_collate_fn_with_meta(batch):
    coords, feats, labels, filenames, event_ids = zip(*batch)
    bcoords = ME.utils.batched_coordinates(coords)
    bfeats = torch.from_numpy(np.concatenate(feats, 0)).float()
    return bcoords, bfeats, labels, filenames, event_ids
