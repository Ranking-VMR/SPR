import torch
import time
import numpy as np
import pprint
from functools import partial
from types import SimpleNamespace
from tqdm import tqdm
from tasks.dense_retrieval_faiss.ranking.faiss_index import FaissIndex
from datasets.vcmr.retrieval_dataset import build_dataset, build_loader
from loguru import logger

from tasks.vcmr_CLIP.eval_vr import get_submission_top_n
from tasks.vcmr_CLIP.eval import eval_retrieval

"""
Faiss retrieval and evaluation.
"""
class TVR_VR_Evaluator():
    """
    初始化 Ranker 对象，接收参数 args、推理对象 inference 和 FAISS 深度 faiss_depth。
    如果设置了 faiss_depth，初始化 FaissIndex 并定义 retrieve 方法。
    """
    def __init__(self, args, inference, faiss_depth=1024):
        
        # build CLIP inference.
        self.inference = inference
        
        # build faiss-index retrieval system.
        self.faiss_depth = faiss_depth
        if faiss_depth is not None:
            self.faiss_index = FaissIndex(args.vidlen_path, args.faiss_index_path, args.nprobe, part_range=args.part_range)
            self.retrieve = partial(self.faiss_index.retrieve, self.faiss_depth)

        # build dataset and loader.
        dataset_args = SimpleNamespace() # temporally used for dataset building.
        dataset_args.VID_FEAT_PATH = args.vid_feat_path
        dataset_args.QUERY_ANNO_PATH = args.query_anno_path
        dataset_args.VID_ANNO_PATH = args.vid_anno_path
        test_dataset = build_dataset(dataset_args)
        self.test_dataset = test_dataset
        #self.test_loader = build_loader(test_dataset, args.bsize, 4)
        
        # build related data.
        self.video2idx = test_dataset.video2idx
        self.video_data = test_dataset.video_data
        self.vid_names = test_dataset.vid_names
        self.query_data = test_dataset.query_data
        
        # other settings
        self.device = args.device
        self.n_total_query = len(test_dataset)
        self.n_vid = len(self.video_data)
        self.max_n_videos = 100
        self.bsize = args.bsize
    
    # Encode queries into vectors.
    def encode(self, queries):
        assert type(queries) in [list, tuple], type(queries)

        Q = self.inference.queryFromText(queries, bsize=self.bsize if len(queries) > self.bsize else None)

        return Q

    def pid2vid(self, pids):
        """
        pid returned by self.retrieve ranges from 0 to n_vids.
        Here we use the dict to project pid to TVR video annotation id.
        """
        """
        def trans_pid(pids, video2idx, vid_names):
            return [video2idx[vid_names[pid]] for pid in pids]
        
        if len(pids) > 5000:
            return list(self.faiss_index.parallel_pool.map(lambda x: trans_pid(x, self.video2idx, self.vid_names), pids))
        else:
            return list(map(lambda x: trans_pid(x, self.video2idx, self.vid_names), pids))
        """
        vids = [[self.video2idx[self.vid_names[pid]] for pid in sublist] for sublist in pids]
        return vids
    
    def form_results(self, vids):
        """
        Given a list of video ids, we need to truncate or repeat the list to make it of length max_n_videos.
        """
        def comp_n_trunc(list):
            if len(list) >= self.max_n_videos:
                return list[:self.max_n_videos]
            else:
                return list + [list[-1]] * (self.max_n_videos - len(list))

        """
        if len(vids) > 5000:
            return list(self.faiss_index.parallel_pool.map(comp_n_trunc, vids))
        else:
            return list(map(comp_n_trunc, vids))
        """
        return [comp_n_trunc(sublist) for sublist in vids]


    def generate_vid_vr(self):
        #Encode all queries in the dataset.
        s = time.time()
        queries = [x["desc"] for x in self.query_data]
        Q = self.encode(queries)
        e1 = time.time()
        logger.info(f"#> Time: {e1 - s:.2f} sec for query encoding.")
        pids = self.retrieve(Q, verbose=True)
        #vids = self.pid2vid(pids) # need to check
        vids = pids
        length = np.array([len(x) for x in vids], dtype=np.int32)
        mean = np.mean(length)
        median = np.median(length)
        q1 = np.percentile(length, 25)
        q3 = np.percentile(length, 75)
        print(f"mean: {mean}, median: {median}, q1: {q1}, q3: {q3}")
        sorted_q2c_indices = np.array(self.form_results(vids), dtype=np.int32)
        assert sorted_q2c_indices.shape == (self.n_total_query, self.max_n_videos), sorted_q2c_indices.shape
        return sorted_q2c_indices
    
    def prepare_eval_submission_vr(self, sorted_q2c_scores, sorted_q2c_indices):
        vr_res = []
        for i, (_sorted_q2c_scores_row, _sorted_q2c_indices_row) in tqdm(
                enumerate(zip(sorted_q2c_scores[:, :100], sorted_q2c_indices[:, :100])),
                desc="[VR] Loop over queries to generate predictions", total=self.n_total_query):
            cur_vr_redictions = []
            for j, (v_score, v_meta_idx) in enumerate(zip(_sorted_q2c_scores_row, _sorted_q2c_indices_row)):
                video_idx = self.video2idx[self.vid_names[v_meta_idx]]
                #video_idx = v_meta_idx
                cur_vr_redictions.append([video_idx, 0, 0, float(v_score)])
            cur_query_pred = dict(desc_id=self.query_data[i]['desc_id'], desc=self.query_data [i]["desc"],
                                    predictions=cur_vr_redictions)
            vr_res.append(cur_query_pred)
        eval_res = dict(VR=vr_res, video2idx=self.video2idx)

        # compute the evaluation metric, here (line 116-137) we follow the codes in ReLoCLNet/method_tvr/inference.py  eval_epoch()
        # The iou_thds are used for computing moment retrieval metrics
        self.IOU_THDS = (0.5, 0.7)  # (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        max_after_nms = 100
        eval_submission = get_submission_top_n(eval_res, top_n=max_after_nms)
        
        return eval_submission
    
    def random_eval_vr(self):
        sorted_q2c_scores = np.ones((self.n_total_query, self.max_n_videos), dtype=np.float32)
        num_vids = len(self.vid_names)
        sorted_q2c_indices =  np.random.randint(0, num_vids, size=sorted_q2c_scores.shape)
        
        # prepare submission for evaluation.
        eval_submission = self.prepare_eval_submission_vr(sorted_q2c_scores, sorted_q2c_indices)
        
        # assume eval on val set
        metrics = eval_retrieval(eval_submission, self.query_data, iou_thds=self.IOU_THDS,
                                match_number=False, verbose=True, use_desc_type=True)
        
        # return metrics, metrics_nms,
        metrics_no_nms = metrics
        logger.info("metrics_no_nms \n{}".format(pprint.pformat(metrics_no_nms, indent=4)))
        return
    
    def eval_vr(self):

        # generate video retrieval results.
        sorted_q2c_indices = self.generate_vid_vr()
        sorted_q2c_scores = np.ones((self.n_total_query, self.max_n_videos), dtype=np.float32)
        
        # prepare submission for evaluation.
        eval_submission = self.prepare_eval_submission_vr(sorted_q2c_scores, sorted_q2c_indices)
        
        # assume eval on val set
        metrics = eval_retrieval(eval_submission, self.query_data, iou_thds=self.IOU_THDS,
                                match_number=False, verbose=True, use_desc_type=True)
        
        # return metrics, metrics_nms,
        metrics_no_nms = metrics
        logger.info("metrics_no_nms \n{}".format(pprint.pformat(metrics_no_nms, indent=4)))
        return 
    