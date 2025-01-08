import torch
from tqdm import tqdm, trange
from collections import defaultdict
import numpy as np
import os
from utils.evaluation_metrics.common_metrics import calculate_recall_rate

@torch.no_grad()
def compute_corpus_feature(model, corpus_dataloader, device, np_save_path=None):
    """
    Compute the model feature for all segments in the corpus.
    Returns:
        all_seg_feats (torch.Tensor): The visual features for all segments, shape: [all_num_segment, d].
    Or:
        save all segment features to numpy files.
    """
    all_seg_feats = []
    model.eval()
    for batch_data in tqdm(corpus_dataloader, desc="Extracting segment features for corpus..."):
        bs = len(batch_data["video_name"])
        for i in range(bs):
            seg_frame_feats = batch_data["seg_frame_feats"][i].to(device)
            visual_mask =batch_data["seg_frame_mask"][i].to(device)
            seg_feats = model.get_segment_feature(visual_feats=seg_frame_feats, visual_mask=visual_mask)
            if np_save_path is not None:
                save_path = os.path.join(np_save_path, batch_data["video_name"][i] + ".npy")
                np.save(save_path, seg_feats.cpu().numpy())
            else:
                all_seg_feats.append(seg_feats)
    if np_save_path is not None:
        return
    else:
        # Concatenate all video features and masks
        all_seg_feats = torch.cat(all_seg_feats, dim=0)
        return all_seg_feats

@torch.no_grad()
def compute_text_feature(model, test_dataloader, device):
    q_feats_list = []
    q_gt_list = []
    qid_list = []
    model.eval()
    for batch_data in tqdm(test_dataloader, desc="Get query features..."):
        query_feat = batch_data["query_feat"].to(device)
        query_id = batch_data["query_id"]
        relevant_segment = batch_data["relevant_segment"]
        query_feat = model.get_query_feature(query_feat)
        #bs = query_feat.shape[0]

        qid_list.extend(query_id)
        q_feats_list.append(query_feat)
        q_gt_list.extend(relevant_segment)
    q_feats = torch.cat(q_feats_list, dim=0)
    return q_feats, qid_list, q_gt_list 
    """
    q_feats_dict = {}
    q_gt_dict = {}
    model.eval()
    for batch_data in tqdm(test_dataloader, desc="Get query features..."):
        query_feat = batch_data["query_feat"].to(device)
        query_id = batch_data["query_id"]
        relevant_segment = batch_data["relevant_segment"]
        query_feat = model.get_query_feature(query_feat)
        bs = query_feat.shape[0]
        for i in range(bs):
            q_feats_dict[query_id[i]] = query_feat[i]
            q_gt_dict[query_id[i]] = relevant_segment[i]
    return q_feats_dict, q_gt_dict
    """

@torch.no_grad()
def compute_segment_recall(model, all_seg_feats, corpus_seg_list, all_query_feats, all_query_id, all_q_gt, topks):
    
    model.eval()
    all_sim_matrix = model.compute_similarity_matrix(all_query_feats, all_seg_feats)

    ALL_RESULTS = {}
    gt_video_recall = defaultdict(list) # List of video recall rates for each query at each top-k.
    gt_seg_recall = defaultdict(list) # List of segment recall rates for each query at each top-k.
    gt_seg_weighted_recall = defaultdict(list) # List of segment weighted recall rates for each query at each top-k.
    unique_video_nums = defaultdict(list) # List of unique video numbers for each query at each top-k.
    for i in trange(len(all_query_id), desc="Calculate the Recall"):
        simi = all_sim_matrix[i]
        query_id = all_query_id[i]
        one_gt = [gt for gt in all_q_gt[i] if gt["relevance"] > 0]
        if len(one_gt) == 0:
            #print(f"Query {query_id} has no relevant segment.")
            continue

        one_gt_vid = [gt["video_name"] for gt in one_gt]
        one_gt_seg = [f"{gt['video_name']}_{gt['segment_idx']}" for gt in one_gt]
        one_gt_seg_scores = [gt["score"] for gt in one_gt]

        for topk in topks:
            _, top_k_indices = torch.topk(simi, topk)
            top_k_seg_names = [corpus_seg_list[idx] for idx in top_k_indices]
            top_k_video_names =  [seg_name.rsplit('_', 1)[0] for seg_name in top_k_seg_names]
            
            # ========= gt_video_recall =================
            video_recall_rate = calculate_recall_rate(one_gt_vid, top_k_video_names)
            gt_video_recall[topk].append(video_recall_rate)

            top_k_video_names_unique = set(top_k_video_names)
            unique_video_nums[topk].append(len(top_k_video_names_unique))

            # ========= gt_seg_recall_seg ====================
            seg_recall_rate = calculate_recall_rate(one_gt_seg, top_k_seg_names)
            gt_seg_recall[topk].append(seg_recall_rate)
            
            # ========= gt_seg_weighted_recall_seg ==========
            retrieved_seg_score = []
            for j in range(len(one_gt_seg)):
                if one_gt_seg[j] in top_k_seg_names:
                    retrieved_seg_score.append(one_gt_seg_scores[j])
                else:
                    retrieved_seg_score.append(0)
            seg_weighted_recall_rate = np.sum(retrieved_seg_score) / np.sum(one_gt_seg_scores)
            gt_seg_weighted_recall[topk].append(seg_weighted_recall_rate)
    
    ALL_RESULTS["gt_video_recall"] = {topk: np.mean(recall) for topk, recall in gt_video_recall.items()}
    ALL_RESULTS["gt_seg_recall"] = {topk: np.mean(recall) for topk, recall in gt_seg_recall.items()}
    ALL_RESULTS["unique_video_nums"] = {topk: np.mean(n) for topk, n in unique_video_nums.items()}
    ALL_RESULTS["gt_seg_weighted_recall"] = {topk: np.mean(recall) for topk, recall in gt_seg_weighted_recall.items()}
    
    flag_result = ALL_RESULTS["gt_seg_weighted_recall"][100]
    return ALL_RESULTS, flag_result
            
def evaluate_recall(model, all_seg_feats, corpus_seg_list, test_dataloader, device, topks):
    all_query_feats, all_query_id, all_q_gt = compute_text_feature(model, test_dataloader, device)
    ALL_RESULTS, flag_result = compute_segment_recall(model, all_seg_feats, corpus_seg_list, all_query_feats, all_query_id, all_q_gt, topks)
    return ALL_RESULTS, flag_result