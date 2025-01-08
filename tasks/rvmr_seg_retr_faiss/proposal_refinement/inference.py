from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from SPR.datasets.rvmr.i3d_feat_prop_refine_dataset import prepare_batch_inputs
from tasks.rvmr_seg_retr_faiss.proposal_refinement.run_utils import topk_3d, generate_min_max_length_mask
from utils.evaluation_metrics.rvmr import calculate_ndcg_iou

def index2moments(start_idx, end_idx, seg_duration):
    """
    Args:
        start_idx (int): start index of the segment.
        end_idx (int): end index of the segment.
        seg_duration (int): duration of each segment.
    Returns:
        start (int): start timestamp of the moment.
        end (int): end timestamp of the moment.
    Given st_idx = 2, ed_idx = 5 (i.e. clip [2,3,4,5] are included), the translated back ts should be [3:9].
    """
    start = start_idx * seg_duration
    end = (end_idx + 1) * seg_duration
    return start, end

def calculate_relative_recall_rate(gt_vids, pred_vids, k):
    #relative_num = min(k, len(gt_vids))
    gt_set = set(gt_vids)
    pred_set = set(pred_vids[:k])
    true_positives = len(gt_set.intersection(pred_set))
    relative_num = min(k, len(gt_set))
    recall_rate = true_positives / relative_num # recall_rate ranges from 0 to 1
    return recall_rate

def calculate_relative_video_recall(all_gt, all_pred, KS):
    video_recall = defaultdict(list)
    video_recall_avg = defaultdict(float)
    for query_id in tqdm(all_pred.keys(), desc="Calculate Video Recall: "):
        one_pred = all_pred[query_id]
        one_gt = [gt for gt in all_gt[query_id] if gt["relevance"] > 0]
        if len(one_gt) == 0:
            continue
        gt_vids = [gt["video_name"] for gt in one_gt]
        pred_vids = [pred["video_name"] for pred in one_pred]
        #pred_vids = one_pred
        for k in KS:
            video_recall[k].append(calculate_relative_recall_rate(gt_vids, pred_vids, k))
    
    for k in KS:
        video_recall_avg[k] = np.mean(video_recall[k])
    
    return video_recall_avg

@torch.no_grad()
def generate_refined_moment_prediction(model, eval_loader, video_lens, opt):
    model.eval()
    all_pred = {}
    for batch in tqdm(eval_loader, desc="Refine coarse predictions and evalution", total=len(eval_loader)):
        model_inputs = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory, task="eval")
        
        bsz = model_inputs["query_feat"].shape[0]
        for batch_idx in range(bsz):
            query_feat = model_inputs["query_feat"][batch_idx].unsqueeze(0)
            query_mask = model_inputs["query_mask"][batch_idx].unsqueeze(0)
            video_feat = model_inputs["video_feat"][batch_idx]
            video_mask = model_inputs["video_mask"][batch_idx]
            sub_feat = model_inputs["sub_feat"][batch_idx]
            sub_mask = model_inputs["sub_mask"][batch_idx]
            
            video_feat, sub_feat = model.encode_context(video_feat, video_mask, sub_feat, sub_mask)
            
            query_scores, start_probs, end_probs = model.get_pred_from_raw_query(
                query_feat = query_feat,
                query_mask = query_mask, 
                video_feat = video_feat,
                video_mask = video_mask,
                sub_feat = sub_feat,
                sub_mask = sub_mask,
                cross=True)
            
            # final ranking score: theta = start_probs * end_probs * exp(alpha * query_scores)
            query_scores = torch.exp(opt.q2c_alpha * query_scores) # [1, Nv]
            start_probs = F.softmax(start_probs, dim=-1) # [1, Nv, L]
            end_probs = F.softmax(end_probs, dim=-1) # [1, Nv, L]
            
            # compute moment-level ranking scores and generate refined proposal for each coarse moemnt proposal
            all_2D_map = torch.einsum("qvm,qv,qvn->qvmn", start_probs, query_scores, end_probs) # [1, Nv, L, L]
            map_mask = generate_min_max_length_mask(all_2D_map.shape, min_l=opt.min_pred_l, max_l=opt.max_pred_l)
            all_2D_map = all_2D_map * torch.from_numpy(map_mask).to(all_2D_map.device)
            
            score_map = all_2D_map.squeeze(0) # [Nv, L, L]
            n_moments = score_map.shape[0]
            query_id = batch[0][batch_idx]["query_id"]
            coarse_moment_proposals  = batch[0][batch_idx]["coarse_moment_proposals"]
            new_video_timestamp = batch[0][batch_idx]["new_video_timestamp"]
            clip_length = batch[0][batch_idx]["clip_length"]
            pred_result = []
            for moment_idx in range(n_moments):
                video_name = coarse_moment_proposals[moment_idx]["video_name"]
                pre_proposal_len = new_video_timestamp[moment_idx][0]
                moment_score_map = score_map[moment_idx:moment_idx+1]
                top_score, top_idx = topk_3d(moment_score_map, 1) # for each moment, only keep the top-1 refined proposal
                start_idx, end_idx = top_idx[0][1].item(), top_idx[0][2].item()
                pred_start_time, pred_end_time = index2moments(start_idx, end_idx, clip_length[moment_idx])
                pred_start_time += pre_proposal_len
                pred_end_time += pre_proposal_len
                
                pred_result.append({
                    "video_name": video_name,
                    "timestamp": [pred_start_time, min(pred_end_time,video_lens[video_name])],
                    "model_scores": top_score[0].item(),
                })
                # print(pred_result)
            
            pred_result.sort(key=lambda x: x["model_scores"], reverse=True)
            all_pred[str(query_id)] = pred_result
    
    return all_pred

def eval_epoch(model, eval_loader, eval_gt, video_lens, opt):
    all_pred = generate_refined_moment_prediction(model, eval_loader, video_lens, opt)
    average_ndcg = calculate_ndcg_iou(eval_gt, all_pred, opt.iou_threshold, opt.ndcg_topk)
    average_video_recall = calculate_relative_video_recall(eval_gt, all_pred, opt.ndcg_topk)
    
    return average_ndcg, average_video_recall

@torch.no_grad()
def generate_refined_moment_prediction_debug(model, eval_loader, video_lens, opt):
    model.eval()
    all_pred = {}
    for batch in tqdm(eval_loader, desc="Refine coarse predictions and evalution", total=len(eval_loader)):
        model_inputs = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory, task="eval")
        
        bsz = model_inputs["query_feat"].shape[0]
        for batch_idx in range(bsz):
            query_feat = model_inputs["query_feat"][batch_idx].unsqueeze(0)
            query_mask = model_inputs["query_mask"][batch_idx].unsqueeze(0)
            video_feat = model_inputs["video_feat"][batch_idx]
            video_mask = model_inputs["video_mask"][batch_idx]
            sub_feat = model_inputs["sub_feat"][batch_idx]
            sub_mask = model_inputs["sub_mask"][batch_idx]
            
            video_feat, sub_feat = model.encode_context(video_feat, video_mask, sub_feat, sub_mask)
            
            query_scores, start_probs, end_probs = model.get_pred_from_raw_query(
                query_feat = query_feat,
                query_mask = query_mask, 
                video_feat = video_feat,
                video_mask = video_mask,
                sub_feat = sub_feat,
                sub_mask = sub_mask,
                cross=True)
            
            # final ranking score: theta = start_probs * end_probs * exp(alpha * query_scores)
            query_scores = torch.exp(opt.q2c_alpha * query_scores) # [1, Nv]
            start_probs = F.softmax(start_probs, dim=-1) # [1, Nv, L]
            end_probs = F.softmax(end_probs, dim=-1) # [1, Nv, L]
            
            # compute moment-level ranking scores and generate refined proposal for each coarse moemnt proposal
            all_2D_map = torch.einsum("qvm,qv,qvn->qvmn", start_probs, query_scores, end_probs) # [1, Nv, L, L]
            #all_2D_map = torch.einsum("qvm, qvn->qvmn", start_probs, end_probs) # [1, Nv, L, L]
            map_mask = generate_min_max_length_mask(all_2D_map.shape, min_l=opt.min_pred_l, max_l=opt.max_pred_l)
            all_2D_map = all_2D_map * torch.from_numpy(map_mask).to(all_2D_map.device)
            
            score_map = all_2D_map.squeeze(0) # [Nv, L, L]
            n_moments = score_map.shape[0]
            query_id = batch[0][batch_idx]["query_id"]
            coarse_moment_proposals  = batch[0][batch_idx]["coarse_moment_proposals"]
            new_video_timestamp = batch[0][batch_idx]["new_video_timestamp"]
            clip_length = batch[0][batch_idx]["clip_length"]
            pred_result = []
            for moment_idx in range(n_moments):
                video_name = coarse_moment_proposals[moment_idx]["video_name"]
                pre_proposal_len = new_video_timestamp[moment_idx][0]
                moment_score_map = score_map[moment_idx:moment_idx+1]
                top_score, top_idx = topk_3d(moment_score_map, 1) # for each moment, only keep the top-1 refined proposal
                start_idx, end_idx = top_idx[0][1].item(), top_idx[0][2].item()
                pred_start_time, pred_end_time = index2moments(start_idx, end_idx, clip_length[moment_idx])
                pred_start_time += pre_proposal_len
                pred_end_time += pre_proposal_len
                
                pred_result.append({
                    "video_name": video_name,
                    "timestamp": [pred_start_time, min(pred_end_time,video_lens[video_name])],
                    #"model_scores": coarse_moment_proposals[moment_idx]["moment_rank"],
                    #"model_scores": query_scores[0, moment_idx].item(),
                    "model_scores": top_score[0].item(),
                })
                # print(pred_result)
            
            pred_result.sort(key=lambda x: x["model_scores"], reverse=True)
            all_pred[str(query_id)] = pred_result
    
    return all_pred