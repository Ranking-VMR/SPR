import os

from utils.general.basic_utils import load_json

# get a list of video lengths
def load_vidlens(directory):
    vidlen_dict = load_json(directory)
    vid_names = sorted(list(vidlen_dict.keys()))

    all_doclens = [vidlen_dict[vid_name] for vid_name in vid_names]

    return all_doclens

def load_vidnames(directory):
    vidlen_dict = load_json(directory)
    vid_names = sorted(list(vidlen_dict.keys()))

    return vid_names

def get_parts_by_config(vid_feat_path, vid_anno_path):
    """
    To generate required video file info for building faiss index.
    Args:
        vid_feat_path (str): root path to the npy files containing the video features.
        vid_anno_path (str): path to the json file containing the video annotations.
            This json file should have the following structure: {vid_name: num_segments / durations ...}.
            Here we only use the keys (vid_name) to get the video names, and the values can be ignored.
        eval_split_name: str, the evaluation split name. We use "val" as default.
    Return:
        parts (list): the list of video names. It is not used for anything except for providing the total video number.
        parts_paths (list): list of file paths of video features (each: n.npy).
        samples_paths: list of file paths of sampled video featuresy (each: n.sample). 
                       This is used for training the faiss index by only part of the video features.
                       But in our implementation, we use all the video features for training, therefore, samples_paths = parts_paths.
    """
    
    extension = ".npy"

    vid_names = load_vidnames(vid_anno_path)
    parts = vid_names
    parts_paths = [os.path.join(vid_feat_path, '{}{}'.format(vid_name, extension)) for vid_name in vid_names]
    samples_paths = parts_paths
    
    return parts, parts_paths, samples_paths