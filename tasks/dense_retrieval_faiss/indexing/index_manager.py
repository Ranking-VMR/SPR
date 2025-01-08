import torch
import faiss
import numpy as np


class IndexManager():
    def __init__(self, dim):
        self.dim = dim

    def save(self, tensor, path_prefix):
        torch.save(tensor, path_prefix)


def load_index_part(filename, verbose=True):
    part = torch.load(filename)

    if type(part) == list:  # for backward compatibility
        part = torch.cat(part)

    return part

# load numpy array directly.
def load_index_part_np(filename, verbose=True):

    part = np.load(filename)
    #l2_norms = np.linalg.norm(part, axis=1, keepdims=True)
    #normalized_part = part / l2_norms

    #return normalized_part
    return part