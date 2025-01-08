from tqdm import tqdm
import numpy as np
from collections import defaultdict
import copy
from loguru import logger
import math

from utils.evaluation_metrics.common_metrics import calculate_iou_1d, calculate_recall_rate

# Function to calculate DCG
def calculate_dcg(scores):
    scores_np = np.asarray(scores)
    dcg = np.sum(np.divide(np.power(2, scores) - 1, np.log2(np.arange(2, len(scores_np) + 2)))) # sum((2**score - 1) / np.log2(idx + 2) for idx, score in enumerate(scores))
    return dcg

# Function to calculate NDCG
def calculate_ndcg(pred_scores, true_scores):
    """
    Calculate NDCG
    Args:
        pred_scores (list): Predicted relevance scores.
        true_scores (list): Ground Truth relevance scores.
    Returns:
        ndcg: Normalized Discounted Cumulative Gain.
    """
    assert true_scores == sorted(true_scores, reverse=True), "True scores must be in descending order"
    dcg = calculate_dcg(pred_scores)
    idcg = calculate_dcg(true_scores)
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return ndcg

def calculate_ndcg_iou(all_gt, all_pred, TS, KS):
    """
    Args:
        all_gt (dict): Ground truth annotations. {query_id: [{relevence: rel, video_name: vid_name, timestamp: [start, end]},...]}
        all_pred (dict): Predictions. {query_id: [{video_name: vid_name, timestamp: [start, end]},...]}
        TS (list): List of IoU thresholds. E.g., [0.3, 0.5, 0.7]
        KS (list): List of K values. E.g., [10, 20, 40]
    Returns:
        performance_avg (dict): Averaged NDCG scores for each K and IoU threshold T.
    """
    performance = defaultdict(lambda: defaultdict(list))
    performance_avg = defaultdict(lambda: defaultdict(float))

    max_K = max(KS)

    for query_id in tqdm(all_pred.keys(), desc="Calculate NDCG,IoU: "):
        one_pred = all_pred[query_id]
        if len(one_pred) < max_K:
            logger.warning(f"Query {query_id} has less than {max_K} predictions ({len(one_pred)}).")
        one_gt = [gt for gt in all_gt[query_id] if gt["relevance"] > 0]
        if len(one_gt) == 0:
            logger.info(f"Query {query_id} has no relevant moments.")
            continue
        one_gt = sorted(one_gt, key=lambda x: x["relevance"], reverse=True)

        for T in TS:
            one_gt_drop = copy.deepcopy(one_gt)
            #predictions_with_scores = []
            pred_scores = []
            
            for pred in one_pred:
                pred_video_name, pred_time = pred["video_name"], pred["timestamp"]
                matched_rows = [gt for gt in one_gt_drop if gt["video_name"] == pred_video_name]
                if not matched_rows:
                    pred_scores.append(0)
                else:
                    ious = [calculate_iou_1d(pred_time[0], pred_time[1], gt["timestamp"][0], gt["timestamp"][1]) for gt in matched_rows]
                    max_iou_idx = np.argmax(ious) # This function only returns the first index of largest values. However, they are already sorted by relevance, so the most relevant segment is returned.
                    max_iou_row = matched_rows[max_iou_idx]
                    
                    if ious[max_iou_idx] >= T:
                        pred_scores.append(max_iou_row["relevance"])
                        # Remove the matched ground truth row
                        original_idx = one_gt_drop.index(max_iou_row)
                        one_gt_drop.pop(original_idx)
                    else:
                        pred_scores.append(0)
            
            true_scores = [gt["relevance"] for gt in one_gt]
            
            for K in KS:
                ndcg_score = calculate_ndcg(pred_scores[:K], true_scores[:K])
                performance[K][T].append(ndcg_score)
        
    #np.save("performance_tvr.npy", dict(performance))
    for K, vs in performance.items():
        for T, v in vs.items():
            performance_avg[K][T] = np.mean(v)
    
    return performance_avg

# for temporary use
def calculate_video_recall(all_gt, all_pred):
    """
    Args:
        all_gt (dict): Ground truth annotations. {query_id: [{relevence: rel, video_name: vid_name, timestamp: [start, end]},...]}
        all_pred (dict): Predictions. {query_id: [{video_name: vid_name, timestamp: [start, end]},...]}
        TOP-K (int): Top-K predictions to consider.
    Returns:
        performance_avg (dict): Averaged NDCG scores for each K and IoU threshold T.
    """
    video_recall = list()
    for query_id in tqdm(all_pred.keys(), desc="Calculate Video Recall: "):
        one_pred = all_pred[query_id]
        one_gt = [gt for gt in all_gt[query_id] if gt["relevance"] > 0]
        if len(one_gt) == 0:
            continue
        one_gt = sorted(one_gt, key=lambda x: x["relevance"], reverse=True)
        #if one_gt[0]["relevance"] <= 2:
        #    print(f"Query {query_id} has low relevance {one_gt[0]['relevance']}.")
        gt_vids = [gt["video_name"] for gt in one_gt]
        pred_vids = [pred["video_name"] for pred in one_pred]
        #pred_vids = one_pred

        video_recall.append(calculate_recall_rate(gt_vids, pred_vids))
    
    return np.mean(video_recall)