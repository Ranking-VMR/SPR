import os
import torch
import h5py

from math import ceil
from itertools import accumulate

from utils.general.basic_utils import load_json


def get_parts(directory):
    extension = '.pt'

    parts = sorted([int(filename[: -1 * len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])

    assert list(range(len(parts))) == parts, parts

    # Integer-sortedness matters.
    parts_paths = [os.path.join(directory, '{}{}'.format(filename, extension)) for filename in parts]
    samples_paths = [os.path.join(directory, '{}.sample'.format(filename)) for filename in parts]
    #samples_paths = parts_paths

    return parts, parts_paths, samples_paths

"""
def load_vidlens(directory, flatten=True):
    parts, _, _ = get_parts(directory)

    doclens_filenames = [os.path.join(directory, 'doclens.{}.json'.format(filename)) for filename in parts]
    all_doclens = [ujson.load(open(filename)) for filename in doclens_filenames]

    if flatten:
        all_doclens = [x for sub_doclens in all_doclens for x in sub_doclens]

    return all_doclens
"""

def load_vidlens(directory):
    vidlen_dict = load_json(directory)
    vid_names = sorted(list(vidlen_dict.keys()))

    all_doclens = [vidlen_dict[vid_name] for vid_name in vid_names]

    return all_doclens

def get_parts_by_config(vid_feat_path, vid_anno_path, eval_split_name="val"):
    """
    Args:
        vid_feat_path: str, root path to the npy files containing the video features.
        vid_anno_path: str, path to the json file containing the video annotations.
        eval_split_name: str, the evaluation split name. We use "val" as default.
    Return:
        parts: list of range(len(parts))
        parts_paths: list of files in the directory (each: n.pt)
        samples_paths: list of samples in the directory (each: n.sample)
    """
    extension = ".npy"

    video_data = load_json(vid_anno_path)[eval_split_name]
    vid_names = sorted(list(video_data.keys()))

    parts_paths = [os.path.join(vid_feat_path, '{}{}'.format(vid_name, extension)) for vid_name in vid_names]
    parts = list(range(len(vid_names)))
    samples_paths = parts_paths 
    
    return parts, parts_paths, samples_paths
