import MinkowskiEngine as ME
import torch
import numpy as np


def make_dense(coords_batch, feats_batch, device, index=0, max_i=256, max_j=128, device):
    img = ME.SparseTensor(feats_batch.float(), coords_batch.int(), device=device)
    coords, feats = img.decomposed_coordinates_and_features
    batch_size = len(coords)
    img_dense,_,_ = img.dense(torch.Size([batch_size, 1, max_i, max_j]))
    return img_dense[index].squeeze().numpy()

def make_dense_from_tensor(sparse_batch, index=0, max_i=256, max_j=128):
    coords, feats = sparse_batch.decomposed_coordinates_and_features
    batch_size = len(coords)
    img_dense,_,_ = sparse_batch.dense(torch.Size([batch_size, 1, max_i, max_j]), min_coordinate=torch.IntTensor([0,0]))
    return img_dense[index]

def make_dense_array(coords, feats, max_i=256, max_j=128):
    img_dense = np.zeros((max_i, max_j))
    i_coords, j_coords = coords[:, 0], coords[:, 1]
    img_dense[i_coords, j_coords] = feats
    return img_dense
