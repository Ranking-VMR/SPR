import os
from tqdm import tqdm
import faiss
import numpy as np
import math
import threading
import queue
from loguru import logger

from utils.dense_retrieval_faiss.utils import grouper
from tasks.rvmr_seg_retr_faiss.indexing.loaders import get_parts_by_config
from tasks.dense_retrieval_faiss.indexing.index_manager import load_index_part_np
from tasks.dense_retrieval_faiss.indexing.faiss_index import IVFPQ_Index, IVFFlat_Index, FlatL2_Index, FlaIP_Index

def get_faiss_index_name(args, offset=None, endpos=None):
    
    if args.index_type == 'ivfpq':
        partitions_info = '' if args.partitions is None else f'.{args.partitions}'
        partitions_info = partitions_info + f".{args.m}.{args.nbits}"
    elif args.index_type == 'ivfflat':
        partitions_info = '' if args.partitions is None else f'.{args.partitions}'
    elif args.index_type in ['flatl2', 'flatip']:
        partitions_info = ''
    
    range_info = '' if offset is None else f'.{offset}-{endpos}'

    return f'{args.index_type}{partitions_info}{range_info}.faiss'

def load_sample(samples_paths, sample_fraction=None):
    """
    Func: load a set of embeddings from the samples_paths.
    Args:
        samples_paths: list of files in the directory (each: n.sample)
        sample_fraction: fraction of embeddings to load
    Return:
        sample: a set of embeddings, numpy.array.
    """
    sample = []

    for filename in tqdm(samples_paths,desc="Loading embedding samples ..."):
        #logger.info(f"#> Loading {filename} ...")
        part = load_index_part_np(filename)

        if sample_fraction:
            #part = part[torch.randint(0, high=part.size(0), size=(int(part.size(0) * sample_fraction),))]
            indices = np.random.randint(0, high=part.shape[0], size=int(part.shape[0] * sample_fraction))
            part = part[indices]
        sample.append(part)

    #sample = torch.cat(sample).float().numpy()
    sample = np.concatenate(sample).astype(np.float32)

    logger.info(f"#> Sample has shape {sample.shape}")

    return sample

def prepare_faiss_index(args, slice_samples_paths):
    
    if args.index_type in ['flatl2', 'flatip']:
        sample_head = load_index_part_np(slice_samples_paths[0])
        dim = sample_head.shape[-1]
        if args.index_type == 'flatl2':
            index = FlatL2_Index(dim)
            logger.info(f"#> Flat L2, no need to train...")
            return index
        elif args.index_type == 'flatip':
            index = FlaIP_Index(dim)
            logger.info(f"#> Flat IP, no need to train...")
            return index
    
    training_sample = load_sample(slice_samples_paths, sample_fraction=args.sample)
    dim = training_sample.shape[-1]
    
    if args.index_type == 'ivfflat':
        index = IVFFlat_Index(dim, args.partitions)
    elif args.index_type == 'ivfpq':
        index = IVFPQ_Index(dim, args.partitions, args.m, args.nbits)
    
    logger.info('#> Normalizing the vectors before training...')
    
    faiss.normalize_L2(training_sample)
    
    logger.info("#> Training with the vectors...")
    index.train(training_sample)

    logger.info("Done training!\n")

    return index


def index_faiss(args):
    SPAN = 3

    logger.info("#> Starting..")

    parts, parts_paths, samples_paths = get_parts_by_config(args.vid_feat_path, args.vid_anno_path)
    # samples_paths: features for training the faiss index (here we set samples_paths = parts_paths, i.e., using all the video features for training)
    # parts_paths: features for building full index

    if args.sample is not None:
        # If sample ratio is provided, we will only randomly re-sample a fraction of the embeddings in parts_paths for training.
        assert args.sample, args.sample
        logger.info(f"#> Training with {round(args.sample * 100.0, 1)}% of *all* embeddings (provided --sample).")
        samples_paths = parts_paths

    num_parts_per_slice = math.ceil(len(parts) / args.slices)

    for slice_idx, part_offset in enumerate(range(0, len(parts), num_parts_per_slice)):
        part_endpos = min(part_offset + num_parts_per_slice, len(parts))

        slice_parts_paths = parts_paths[part_offset:part_endpos]
        slice_samples_paths = samples_paths[part_offset:part_endpos]

        if args.slices == 1:
            faiss_index_name = get_faiss_index_name(args)
        else:
            faiss_index_name = get_faiss_index_name(args, offset=part_offset, endpos=part_endpos)

        output_path = os.path.join(args.index_path, faiss_index_name)
        logger.info(f"#> Processing slice #{slice_idx+1} of {args.slices} (range {part_offset}..{part_endpos}).")
        logger.info(f"#> Will write to {output_path}.")

        assert not os.path.exists(output_path), output_path

        index = prepare_faiss_index(args, slice_samples_paths)


        loaded_parts = queue.Queue(maxsize=1)

        def _loader_thread(thread_parts_paths):
            for filenames in grouper(thread_parts_paths, SPAN, fillvalue=None):
                sub_collection = [load_index_part_np(filename) for filename in filenames if filename is not None]
                sub_collection = np.concatenate(sub_collection).astype(np.float32)
                loaded_parts.put(sub_collection)
        
        # Here we create a thread to load the embeddings. The only producer makes sure that the embeddings are loaded in the correct order.
        thread = threading.Thread(target=_loader_thread, args=(slice_parts_paths,))
        thread.start()

        logger.info("#> Indexing the vectors...")

        # The main thread consumes the embeddings and adds them to the index.
        for filenames in grouper(slice_parts_paths, SPAN, fillvalue=None):
            sub_collection = loaded_parts.get()
            # normalize the vectors before adding them to the index            
            faiss.normalize_L2(sub_collection)
            index.add(sub_collection)

        logger.info("Done indexing!")

        index.save(output_path)

        logger.info(f"\n\nDone! All complete (for slice #{slice_idx+1} of {args.slices})!")

        thread.join()



