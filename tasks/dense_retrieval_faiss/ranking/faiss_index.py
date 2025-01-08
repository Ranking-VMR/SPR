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
from tasks.dense_retrieval_faiss.indexing.loaders import load_vidlens


"""
这段代码定义了一个 FaissIndex 类，使用 FAISS 库进行高效的相似度搜索。
它能够加载 FAISS 索引、构建嵌入到文档 ID 的映射，并提供检索函数用于查询。
通过多进程加速去重操作，代码能够处理大规模的查询任务。
这在需要高效检索的大规模文档集合中非常有用。
"""

class FaissIndex():

    def __init__(self, vidlen_path, faiss_index_path, nprobe, part_range=None):
        """
        Args:  
            vidlen_path: str, path to the video_length file directory.
            faiss_index_path: str, path to the FAISS index file.
            nprobe: int, number of visited cells during retrieving.
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
        self.faiss_index.nprobe = nprobe

        logger.info("#> Building the emb2pid mapping..")
        all_doclens = load_vidlens(vidlen_path)

        pid_offset = 0 # number of videos before the current partition.
        if faiss_part_range is not None:
            print(f"#> Restricting all_doclens to the range {faiss_part_range}.")
            pid_offset = len(flatten(all_doclens[:faiss_part_range.start]))
            all_doclens = all_doclens[faiss_part_range.start:faiss_part_range.stop]

        self.relative_range = None
        if self.part_range is not None:
            start = self.faiss_part_range.start if self.faiss_part_range is not None else 0
            a = len(flatten(all_doclens[:self.part_range.start - start]))
            b = len(flatten(all_doclens[:self.part_range.stop - start]))
            self.relative_range = range(a, b)
            print(f"self.relative_range = {self.relative_range}")

        all_doclens = flatten(all_doclens)

        total_num_embeddings = sum(all_doclens)
        self.emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

        offset_doclens = 0 # accumulated number of embeddings so far.
        for pid, dlength in enumerate(all_doclens):
            self.emb2pid[offset_doclens: offset_doclens + dlength] = pid_offset + pid # (pid_offset + pid) is the video ID. Currently, it assumes the id starts from 0.
            offset_doclens += dlength

        logger.info(f"len(self.emb2pid) = {len(self.emb2pid)}")

        self.parallel_pool = Pool(16)

    # search via faiss index, return the top video ids.
    def retrieve(self, faiss_depth, Q, verbose=False):
        s = time.time()
        embedding_ids = self.queries_to_embedding_ids(faiss_depth, Q, verbose=verbose)
        e1 = time.time()
        pids = self.embedding_ids_to_pids(embedding_ids, verbose=verbose)

        if self.relative_range is not None:
            pids = [[pid for pid in pids_ if pid in self.relative_range] for pids_ in pids]
        e2 = time.time()
        
        logger.info(f"#> Time: {e1 - s:.2f} sec for searching, {e2 - e1:.2f} sec for seg2vid.")
        
        return pids

    # search via faiss index, return the top segment ids.
    def queries_to_embedding_ids(self, faiss_depth, Q, verbose=True):
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
            #l2_norms = np.linalg.norm(some_Q_faiss, axis=-1, keepdims=True)
            #normalized_some_Q_faiss = some_Q_faiss / l2_norms
            normalized_some_Q_faiss = some_Q_faiss
            faiss.normalize_L2(normalized_some_Q_faiss)
            
            _, some_embedding_ids = self.faiss_index.search(normalized_some_Q_faiss, faiss_depth)
            embeddings_ids.append(torch.from_numpy(some_embedding_ids))

        embedding_ids = torch.cat(embeddings_ids)

        # Reshape to (number of queries, non-unique embedding IDs per query)
        embedding_ids = embedding_ids.view(num_queries, embeddings_per_query * embedding_ids.size(1))

        return embedding_ids

    # segment ids to video ids.
    def embedding_ids_to_pids(self, embedding_ids, verbose=True):
        # Find unique PIDs per query.
        if verbose: 
            logger.info("#> Lookup the PIDs..")
        
        all_pids = self.emb2pid[embedding_ids]

        if verbose:
            logger.info(f"#> Converting to a list [shape = {all_pids.size()}]..")

        all_pids = all_pids.tolist()

        if verbose:
            logger.info("#> Removing duplicates (in parallel if large enough)..")

        if len(all_pids) > 5000:
            all_pids = list(self.parallel_pool.map(uniq_keep_order, all_pids))
            #all_pids = list(self.parallel_pool.map(uniq, all_pids))
        else:
            all_pids = list(map(uniq_keep_order, all_pids))
            #all_pids = list(map(uniq, all_pids))

        if verbose:
            logger.info("#> Done with embedding_ids_to_pids().")

        return all_pids # list of lists

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