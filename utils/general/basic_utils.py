import json
import os
import glob
import numpy as np
import pickle
import torch
import h5py
from tqdm import tqdm
from loguru import logger
from terminaltables import AsciiTable

"""
File I/O utilities.
"""

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def save_lines(list_of_str, filepath):
    with open(filepath, "w") as f:
        f.write("\n".join(list_of_str))


def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]

def save_hdf5(data_dict, h5_file_path):
    """
    Args:
        data_dict (dict): {key: np.ndarray}
        h5_file_path (str): path to the hdf5 file.
    """

    with h5py.File(h5_file_path, 'w') as f:
        for name, data in tqdm(data_dict.items(), desc=f"Saving to hdf5 path {h5_file_path}"):
            f.create_dataset(name, data=data)

def save_ckpt(ckpt_root_path, model, optimizer, scheduler, model_info, epoch=None, iter=None):
    model_to_save = model.module if hasattr(model, 'module') else model
    suffix_epoch = "" if epoch is None else f"_epoch{epoch}"
    suffix_iter = "" if iter is None else f"_iter{iter}"
    suffix = suffix_epoch + suffix_iter
    ckpt_path = os.path.join(ckpt_root_path, f"{model_info}_ckpt{suffix}.bin")
    save_dict = {
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model_to_save.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
    }
    if epoch is not None:
        save_dict["epoch"] = epoch
    if iter is not None:
        save_dict["iter"] = iter
    torch.save(save_dict, ckpt_path)

    logger.info(f"Ckpt saved to {ckpt_path}")


def load_ckpt(ckpt_path, model, optimizer=None, scheduler=None):
    logger.info(f"Ckpt loaded from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(state_dict["scheduler_state_dict"])
    
    return model, optimizer, scheduler

def remove_old_files(folder_path, keep_last=5):
    """
    Remove old files in the folder. It's commonly used to keep the latest ckpt files.
    """
    # Get all files in the folder and sort them by the last modified time.
    files = glob.glob(os.path.join(folder_path, '*'))
    files.sort(key=os.path.getmtime)

    # Remove the files except for the last `keep_last` files.
    if len(files) > keep_last:
        for file_to_remove in files[:-keep_last]:
            os.remove(file_to_remove)

def form_data_table(data, title=None):
    """
    Print a table with the given data.
    Args:
        data (list of list): the data to be printed.
        title (str): the title of the table.
    """
    num_row = len(data)
    num_col = len(data[0])
    table = AsciiTable(data, title)
    for i in range(num_col):
        table.justify_columns[i] = 'center'
    return table.table