import numpy as np
from functools import partial
from loguru import logger
from collections import defaultdict
from tasks.rvmr_seg_retr_faiss.ranking.faiss_index import FaissIndex

import time

class SegmentRetrieval():
    """
    Faiss segment-level retrieval.
    """
    # Initialize FaissIndex and retrieve function.
    def __init__(self, args, query_gen_func=None):
        """
        Args:
            args (argparse.Namespace): arguments.
            query_gen_func (func: [str] -> torch.Tensor): query feature generation function.
        """
        
        self.query_gen_func = query_gen_func
        if  self.query_gen_func is None:
            logger.warning("No query feature generation function is provided. Can only accept query features.")
        
        # build faiss-index retrieval system.
        self.faiss_depth = args.faiss_depth
        self.faiss_index = FaissIndex(args.vidlen_path, args.faiss_index_path, args.nprobe, part_range=args.part_range)
        self.retrieve = partial(self.faiss_index.retrieve, faiss_depth = self.faiss_depth)
    
    # top-k segment retrieval for each query.
    def topk_seg_retrieval(self, vid2name, qid_list, queries=None, q_feat=None, verbose=False):
        """_summary_

        Args:
            queries (list): List of query texts. [M]
            q_feat (torch.Tensor): Input query tensor. [M, D] or [M, N, D]
            vid2name (list): List of video names.
            all_vidlens (list of int): List of video lengths (sec). Here we use array seg_recall_np and seg_rank_np (shape of [vid_len]) to represent the recall results of a video.
            seg_duration (int): Duration of each segment (sec).
            qid_list (list): List of query IDs.
        
        Returns:
            retrieved_segments (dict): dict of videos including top-n recalled segments for each query. {query_id: {video_name: {seg_recall: np.array, seg_rank: np.array}}}
        """
        
        start_time = time.time()
        
        assert queries is not None or q_feat is not None, "Either queries or q_feat should be provided."
        if q_feat is None:
            assert self.query_gen_func is not None, "query_gen_func should be provided if q_feat is None."
            q_feat = self.query_gen_func(queries)
        
        assert q_feat.size(0) == len(qid_list), f"q_feat.size(0) = {q_feat.size(0)}, len(qid_list) = {len(qid_list)}"
        
        end_time1 = time.time()
        
        retrieval_results = self.retrieve(Q=q_feat, verbose=verbose)
        # seg_vids (torch.Tensor): Tensor of retrieved video IDs. [M, faiss_depth * embeddings_per_query]
        # seg_offsets (torch.Tensor): Tensor of segment offsets. [M, faiss_depth * embeddings_per_query]
        seg_vids, seg_offsets = retrieval_results["seg_vids"].tolist(), retrieval_results["seg_offsets"].tolist()
        
        retrieved_segments = {} 
        retrieved_vids = {}
        for i, query_id in enumerate(qid_list):
            # find topn video_name and init the recalled array
            video_seg_recall = defaultdict(list)
            
            pred_vid_id = seg_vids[i] # video IDs of the top-k retrieved segments for query i.
            pred_seg_offset = seg_offsets[i] # segment offsets in videos of the top-k retrieved segments for query i.
            
            for j in range(len(pred_vid_id)):
                video_name = vid2name[pred_vid_id[j]]
                seg_info = {
                    "seg_offset": pred_seg_offset[j],
                    "seg_rank": j
                }
                video_seg_recall[video_name].append(seg_info)
            
            video_name_list = list(video_seg_recall.keys())
            for k in video_name_list:
                video_seg_recall[k].sort(key=lambda x: x["seg_offset"])

            retrieved_segments[query_id] = video_seg_recall
            retrieved_vids[query_id] = video_name_list
        end_time2 = time.time()
        logger.info(f"Time for generating query features: {end_time1 - start_time}")
        logger.info(f"Time for segment retrieval: {end_time2 - end_time1}")
        return retrieved_segments, retrieved_vids