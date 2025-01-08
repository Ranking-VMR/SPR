from tqdm import tqdm
import torch
import numpy as np
import os
import math
from utils.general.basic_utils import load_json
#from functools import partial
from loguru import logger
import time

#from tasks.rvmr_seg_retr_faiss.ranking.faiss_index import FaissIndex
from tasks.rvmr_seg_retr_faiss.ranking.segment_retrieval import SegmentRetrieval
from tasks.rvmr_seg_retr_faiss.indexing.loaders import load_vidnames
from tasks.rvmr_seg_retr_faiss.coarse_proposal.rule_based_proposal import rule_based_proposal
from tasks.rvmr_seg_retr_faiss.evaluation.inference_model import ModelInferenceFrozenCLIP4Video
from utils.evaluation_metrics.rvmr import calculate_ndcg_iou, calculate_video_recall
from utils.general.setting_utils import init_rand
from utils.general.basic_utils import save_json
from utils.dense_retrieval_faiss.parser import Arguments
from utils.general.config import load_config
from tasks.dense_retrieval_faiss.indexing.faiss import get_faiss_index_name

def setup_args():
    init_rand(2024)

    parser = Arguments(description='End-to-end retrieval and ranking with ColBERT.')

    #parser.add_model_parameters()
    #parser.add_model_inference_parameters()
    #parser.add_ranking_input()
    parser.add_retrieval_input()

    # for faiss index
    parser.add_argument('--faiss_name', dest='faiss_name', default=None, type=str)
    parser.add_argument('--part-range', dest='part_range', default=None, type=str)

    # for TVR dataset annoations
    parser.add_argument('--vid_dur_path', help='path of video duration (sec) file', default=None, type=str)
    parser.add_argument('--vidlen_path', help='path of video segment feature length file', default=None, type=str)
    parser.add_argument('--test_anno_path', help='path of test set annotation file', default=None, type=str)
    parser.add_argument('--query_feat_path', help='path of query features', default=None, type=str)
    
    # model-related args
    parser.add_argument('--model_type', help='type of model', choices=["CLIP", "CLIP4Video", "FrozenCLIP4Video"], default=None, type=str)
    parser.add_argument('--architecture', help='architecture of the model', choices=["CLIP-ViT-B-32", "CLIP-ViT-B-16", "CLIP-ViT-L-14", "CLIP-ViT-L-14-336"], default=None, type=str)
    parser.add_argument('--ckpt_path', help='path of model checkpoints', default=None, type=str)
    parser.add_argument("--cfg_dir", help="Path to the configuration file", default=None, type=str)
    

    # for moment proposal
    parser.add_argument('--proposal_type', help='ways of linking segments into moment proposals', default=None, type=str)
    parser.add_argument('--seg_duration', help='duration (sec) of each segment.', default=None, type=int)
    args = parser.parse()
    
    #args.depth = args.depth if args.depth > 0 else None

    if args.part_range:
        part_offset, part_endpos = map(int, args.part_range.split('..'))
        args.part_range = range(part_offset, part_endpos)

    
    args.index_path = os.path.join(args.index_root, args.index_name)

    if args.faiss_name is not None:
        args.faiss_index_path = os.path.join(args.index_path, args.faiss_name)
    else:
        args.faiss_index_path = os.path.join(args.index_path, get_faiss_index_name(args))

    
    # other settings
    def check_cuda_installed():
        return torch.cuda.is_available()
    args.device = 'cuda' if check_cuda_installed() else 'cpu'
    
    return args

def process_annoations(anno_path):
    """
    Load and process annotations from anno_path.

    Args:
        anno_path: str, path of the annotation file.

    Returns:
        gt_moments_all: dict, key: query_id, value: list of gt moments.
        qid_list: list of query ids.
        query_list: list of queries.
    """
    logger.info(f"#> Loading and processing annotations from {anno_path}")
    gt_moments_all = {}
    annos = load_json(anno_path)
    #import ipdb; ipdb.set_trace()
    qid_list = [anno["query_id"] for anno in annos]
    query_list = [anno["query"] for anno in annos]
    gt_moment_list = [anno["relevant_moment"] for anno in annos]
    
    for qid, gt_moments in zip(qid_list, gt_moment_list):
        if "relevance" in gt_moments[0].keys():
            gt_moments = [gt for gt in gt_moments if gt["relevance"] > 0] # only keep relevant moments
        gt_moments_all[qid] = gt_moments
    return gt_moments_all, qid_list, query_list

if __name__ == "__main__":
    #SEG_DURATION = 4
    KS = [10, 20, 40]
    TS = [0.3, 0.5, 0.7]

    # args
    args = setup_args()
    
    # load related data
    vid2name = load_vidnames(args.vid_dur_path) # list of video names. The order is the same as the order of videos in faiss index.
    all_durations = load_json(args.vid_dur_path) # dict of video durations. {video_name: duration (float)}
    # load annotations
    gt_moments_all, qid_list, query_list = process_annoations(args.test_anno_path)
    
    # load query features or build model to generate query features
    if args.query_feat_path is not None:
        logger.info(f"#> Loading query features from {args.query_feat_path}")
        q_emb_np = np.load(args.query_feat_path) # directly load query features from existing file. [num_query, d]
        q_emb_tensor = torch.from_numpy(q_emb_np)
        query_gen_func=None
    else:
        assert args.model_type is not None, "Model type should be provided to generate query features."
        assert args.architecture is not None, "Architecture should be provided to generate query features."
        
        logger.info(f"#> Generating query features from text queries using model {args.model_type}-{args.architecture}...")
        # build model
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        if args.model_type == "FrozenCLIP4Video":
            cfg = load_config(args.cfg_dir)
            logger.info("Config loaded: {}".format(cfg))
            model_infer = ModelInferenceFrozenCLIP4Video(cfg, args.ckpt_path, device)
            assert args.architecture == cfg.MODEL.CLIP.TYPE
            assert args.seg_duration == cfg.DATASETS.SEG_DURATION
        else:
            raise NotImplementedError(f"Model type {args.model_type} not implemented.")
        
        
        query_gen_func = lambda x: model_infer.generate_qemb_from_text(x, bsize=128, to_cpu=True)
        logger.info("#> Model loaded.")
        
        

    # Build faiss-index retrieval system.
    seg_retrieval = SegmentRetrieval(args, query_gen_func=query_gen_func)
    

    start_time = time.time()
    # top-k segment retrieval for each query.
    retrieved_segments, retrieved_vids = seg_retrieval.topk_seg_retrieval(vid2name, qid_list, queries=query_list)
    
    # Generate moments based on the top-k retrieved segments for each query.
    rule_info = {"rule_type": args.proposal_type,}
    if "const_gap" in args.proposal_type:
        rule_info["rule_type"] = "const_gap"
        rule_info["gap"] = int(args.proposal_type.split("_")[-1])
    elif "adp_gap" in args.proposal_type:
        rule_info["rule_type"] = "adp_gap"
        rule_info["alpha"] = float(args.proposal_type.split("_")[-1])
    pred_moments_all = rule_based_proposal(retrieved_segments, all_durations, args.seg_duration, rule_info, args.faiss_depth)
    end_time = time.time()
    print(f"Time cost: {end_time - start_time} s")


    # save pred result to json
    #import ipdb; ipdb.set_trace()
    #save_json(pred_moments_all, "val_coarse_proposals.json", save_pretty=True)
    
     
    average_ndcg = calculate_ndcg_iou(gt_moments_all, pred_moments_all , TS, KS)
    print(f"{args.index_type} and {args.proposal_type}:")
    for K, vs in average_ndcg.items():
        for T, v in vs.items():
            print(f"VAL NDCG@{K}, IoU={T}: {v:.4f}")

    # for temporary use
    average_vid_recall = calculate_video_recall(gt_moments_all, pred_moments_all)
    print(f"VAL Video Recall when Segment Recall@{args.faiss_depth}: {average_vid_recall:.4f}")
    
  