import os
import time
import faiss
import random
import torch
import numpy as np
from loguru import logger
from multiprocessing import Pool
import time

from utils.dense_retrieval_faiss.utils import flatten
from tasks.rvmr_seg_retr_faiss.indexing.loaders import load_vidlens

"""
FaissIndex is a class that uses the FAISS library to perform efficient similarity search.
Key Features:
    1. Load the trained FAISS index, retrieve top-k segments, build projection from segment ID to video ID (if necessary).
    2. Apply multi-processing to speed up the de-duplication operation, enabling large-scale retrieval tasks.
"""
class FaissIndex():

    def __init__(self, vidlen_path, faiss_index_path, nprobe, part_range=None):
        """_summary_
        
        Args:  
            vidlen_path: str, path to the video_length file directory.
            faiss_index_path: str, path to the FAISS index file.
            nprobe: int, number of visited cells during retrieving, only applicable to IVF indexes.
            part_range: range, the range of the index parts to be used (this is different from the video- or segment-level range).
        """
        logger.info(f"#> Loading the FAISS index from {faiss_index_path} ..")

        faiss_part_range = os.path.basename(faiss_index_path).split('.')[-2].split('-')

        if len(faiss_part_range) == 2: # when the index only correspnds to a part of the data
            faiss_part_range = range(*map(int, faiss_part_range))
            assert part_range[0] in faiss_part_range, (part_range, faiss_part_range)
            assert part_range[-1] in faiss_part_range, (part_range, faiss_part_range)
        else:
            faiss_part_range = None

        self.part_range = part_range
        self.faiss_part_range = faiss_part_range

        self.faiss_index = faiss.read_index(faiss_index_path)
        if hasattr(self.faiss_index, 'nprobe'):
            self.faiss_index.nprobe = nprobe

        logger.info("#> Building the emb2vid mapping..")
        all_vidlens = load_vidlens(vidlen_path)

        vid_offset = 0 # number of videos before the current partition.
        if faiss_part_range is not None:
            logger.info(f"#> Restricting all_vidlens to the range {faiss_part_range}.")
            vid_offset = len(flatten(all_vidlens[:faiss_part_range.start]))
            all_vidlens = all_vidlens[faiss_part_range.start:faiss_part_range.stop]

        self.relative_range = None
        if self.part_range is not None:
            start = self.faiss_part_range.start if self.faiss_part_range is not None else 0
            a = len(flatten(all_vidlens[:self.part_range.start - start]))
            b = len(flatten(all_vidlens[:self.part_range.stop - start]))
            self.relative_range = range(a, b)
            logger.info(f"self.relative_range = {self.relative_range}")

        all_vidlens = flatten(all_vidlens)

        total_num_embeddings = sum(all_vidlens)
        self.emb2vid = torch.zeros(total_num_embeddings, dtype=torch.int)
        self.seg_offsets = torch.zeros(total_num_embeddings, dtype=torch.int)

        offset_doclens = 0 # accumulated number of embeddings so far.
        for vid, dlength in enumerate(all_vidlens):
            self.emb2vid[offset_doclens: offset_doclens + dlength] = vid_offset + vid # (vid_offset + vid) is the video ID in the original corpus.
            self.seg_offsets[offset_doclens: offset_doclens + dlength] = torch.arange(0, dlength, dtype=torch.int)
            offset_doclens += dlength

        logger.info(f"len(self.emb2vid) = {len(self.emb2vid)}")

    # search via faiss index, return the top segment ids and video ids.
    def retrieve(self, faiss_depth, Q, verbose=False):
        """_summary_

        Args:
            faiss_depth (int): number of retrieved segments.
            Q (torch.Tensor): Input query tensor. [M, D] or [M, N, D]
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            vids (list): List of de-duplicated video IDs. [M, num_vids (different for each query)]
            seg_ids (torch.Tensor): Tensor of retrieved segment IDs. [M, faiss_depth * embeddings_per_query]
            seg_vids (torch.Tensor): Tensor of retrieved video IDs. [M, faiss_depth * embeddings_per_query]
            seg_offsets (torch.Tensor): Tensor of segment offsets. [M, faiss_depth * embeddings_per_query]
        """
        #s = time.time()
        seg_ids = self.queries_to_seg_ids(faiss_depth, Q, verbose=verbose)
        #e1 = time.time()
        
        seg_vids = self.emb2vid[seg_ids] # video ID for each segment.
        seg_offsets = self.seg_offsets[seg_ids] # offset for each segment.
        
        """"
        vids = self.seg_ids_to_vids(seg_ids, verbose=verbose) # List of unique video IDs for each query.

        if self.relative_range is not None:
            vids = [[vid for vid in vids_ if vid in self.relative_range] for vids_ in vids]
        """
        #e2 = time.time()
        
        #logger.info(f"#> Time: {e1 - s:.2f} sec for searching, {e2 - e1:.2f} sec for segid_to_vid.")
        
        retrieval_results = {
            'seg_ids': seg_ids,
            'seg_vids': seg_vids,
            'seg_offsets': seg_offsets
        }
        
        return retrieval_results

    # search via faiss index, return the top segment ids.
    def queries_to_seg_ids(self, faiss_depth, Q, verbose=True):
        """_summary_

        Args:
            faiss_depth (int): number of retrieved segments.
            Q (torch.Tensor): Input query tensor. [M, D] or [M, N, D]
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            seg_ids (torch.Tensor): Tensor of retrieved segment IDs. [M, faiss_depth * embeddings_per_query]
        """
        # Flatten into a matrix for the faiss search.
        if Q.dim() == 2: # when the input only contains CLS tokens.
            Q = Q.unsqueeze(1)
        num_queries, embeddings_per_query, dim = Q.size() # Here we allow multiple embddings per query.
        Q_faiss = Q.view(num_queries * embeddings_per_query, dim).cpu().contiguous()

        # Search in large batches with faiss.
        if verbose:
            logger.info(f"#> Search in batches with faiss. \t Q.size() = {Q.size()}, Q_faiss.size() = {Q_faiss.size()}")

        embeddings_ids = []
        faiss_bsize = embeddings_per_query * 5000
        for offset in range(0, Q_faiss.size(0), faiss_bsize):
            endpos = min(offset + faiss_bsize, Q_faiss.size(0))

            if verbose:
                logger.info(f"#> Searching from {offset} to {endpos}...")

            some_Q_faiss = Q_faiss[offset:endpos].float().numpy()
            
            # normalize the query before searching (maybe should be changed to faiss.normalize_L2(some_Q_faiss) in the future)
            if verbose:
                logger.info("#> Normalizing the query before searching...")

            normalized_some_Q_faiss = some_Q_faiss
            faiss.normalize_L2(normalized_some_Q_faiss)
            
            _, some_seg_ids = self.faiss_index.search(normalized_some_Q_faiss, faiss_depth)
            embeddings_ids.append(torch.from_numpy(some_seg_ids))

        seg_ids = torch.cat(embeddings_ids)

        # Reshape to (number of queries, non-unique embedding IDs per query)
        seg_ids = seg_ids.view(num_queries, embeddings_per_query * seg_ids.size(1))

        return seg_ids

    # segment ids to video ids.
    def seg_ids_to_vids(self, seg_ids, verbose=True):
        """_summary_

        Args:
            seg_ids (torch.Tensor): Tensor of retrieved segment IDs. [M, faiss_depth * embeddings_per_query]
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            all_vids: List of de-duplicated video IDs. [M, num_vids (different for each query)]
        """
        # Find unique vids per query.
        if verbose: 
            logger.info("#> Lookup the VIDs..")
        
        all_vids = self.emb2vid[seg_ids]

        if verbose:
            logger.info(f"#> Converting to a list [shape = {all_vids.size()}]..")

        all_vids = all_vids.tolist()

        if verbose:
            logger.info("#> Removing duplicates (in parallel if large enough)..")

        if len(all_vids) > 5000:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            pool = Pool(16)
            all_vids = list(pool.map(uniq_keep_order, all_vids))
            pool.close()
            pool.join()
        else:
            all_vids = list(map(uniq_keep_order, all_vids))

        if verbose:
            logger.info("#> Done with seg_ids_to_vids().")

        return all_vids # list of lists

def uniq(seq):
    """
    Remove duplicates from a list. Please note that this cannot guarantee the order of the elements.
    """
    return list(set(seq))

def uniq_keep_order(seq):
    """
    Remove duplicates from a list while keeping the order of the elements.
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]