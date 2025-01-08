from tqdm import tqdm

def index2moments(start_idx, end_idx, seg_duration):
    """
    Args:
        start_idx (int): start index of the segment.
        end_idx (int): end index of the segment.
        seg_duration (int): duration of each segment.
    Returns:
        start (int): start timestamp of the moment.
        end (int): end timestamp of the moment.
    """
    start = start_idx * seg_duration
    end = (end_idx + 1) * seg_duration
    return start, end

# Generate moments based on the top-k retrieved segments for each query.
def rule_based_proposal(retrieved_segments, all_durations, seg_duration, rule_info, topk):
    """_summary_
    
    Args:
        retrieved_segments (dict): dict of queries including top-n recalled segments for each query. 
                            {query_id: {video_name: {seg_recall: np.array (each element corresponds to 1 sec), seg_rank: np.array}}}
        all_durations: (dict): dict of video durations. {video_name: duration (float)}
        seg_duration (int): Duration of each segment (sec).
        rule_info (dict): dict of rule-based proposal parameters. 
                        {rule_type: str, gap: int (optional), alpha: float (optional)}
        topk (int): number of top-k retrieved segments for each query.
    Returns:
        pred_moments_all (dict) : dict of queries including top-n predicted moments for each query.
                            {query_id: [{timestamp: [start, end], video_name: str, moment_rank: int}]}
    """
    rule_type = rule_info["rule_type"]
    if rule_type == "const_gap":
        gap = rule_info["gap"]
        assert gap >= 1
    elif rule_type == "adp_gap":
        alpha = rule_info["alpha"]
        assert alpha >= 0 and alpha <= 1
    else:
        assert rule_type == "original", f"rule_type should be one of ['const_gap', 'adp_gap', 'original'], but got {rule_info['rule_type']}"
        
    
    pred_moments_all = {}
    for query_id, video_seg_recall in tqdm(retrieved_segments.items(), desc=f"Proposing moments for all queries based on top-{topk} retrieved segments with rule {rule_type}:"):
        one_query_preds = []
        
        for video_name, vid_seg_r in video_seg_recall.items():
            # Merge consecutive segments into initial moments
            moments = []
            current_moment = None # [start_seg_idx, end_seg_idx] of the current moment
            moment_rank = -1 # Rank of the current moment. We use the minimum rank of the segments in the moment as the rank of the moment.
            for seg_info in vid_seg_r:
                seg_offset, seg_rank = seg_info["seg_offset"], seg_info["seg_rank"]
                if current_moment is None:
                    current_moment = [seg_offset, seg_offset]
                    moment_rank = seg_rank
                    continue
                if seg_offset == current_moment[1] + 1: # if the current segment is consecutive to the current moment
                    current_moment[1] = seg_offset
                    moment_rank = min(moment_rank, seg_rank)
                else:
                    assert moment_rank != -1
                    moments.append([current_moment, moment_rank])
                    current_moment = [seg_offset, seg_offset]
                    moment_rank = seg_rank
            if current_moment is not None:
                assert moment_rank != -1
                moments.append([current_moment, moment_rank])
            

            # Rule-based moment merging process if specified.
            if rule_type in ["const_gap", "adp_gap"]:
                # Iterate through all recalled continuous moments in the video.
                i = 0
                while i < (len(moments)):
                    start_idx1, end_idx1 = moments[i][0]
                    j = i + 1
                    while j < len(moments):
                        start_idx2, end_idx2 = moments[j][0]
                        
                        if rule_type == "adp_gap":
                            length1 = end_idx1 - start_idx1 + 1
                            length2 = end_idx2 - start_idx2 + 1
                            gap = (length1 + length2) * alpha
                        
                        if start_idx2 - end_idx1 - 1 <= gap:
                            end_idx1 = end_idx2
                            moments[i][0] = [start_idx1, end_idx1]
                            moments[i][1] = min(moments[i][1], moments[j][1])
                            assert moments[i][1] != -1
                            moments.pop(j)
                        else:
                            break
                    i += 1
            
            # Create moment info for each identified moment
            for moment in moments:
                start_idx, end_idx = moment[0]
                start_t, end_t = index2moments(start_idx, end_idx, seg_duration) # previous [start, end] correspond to the start and end index of 1 sec segment. To change to timestamp, we use [start, end + 1].
                
                start_t = max(0, start_t)
                end_t = min(all_durations[video_name], end_t) # Ensure the end time is within the video duration.
                moment_rank = moment[1]
                moment_info = {
                    "timestamp": [start_t, end_t], 
                    "video_name": video_name,
                    "moment_rank": moment_rank
                }
                one_query_preds.append(moment_info)
        
        one_query_preds.sort(key=lambda x: x["moment_rank"])
        assert one_query_preds[0]["moment_rank"] == 0, f"Expected the first moment to have rank 0, but got {one_query_preds[0]['moment_rank']}"
        
        pred_moments_all[query_id] = one_query_preds
    
    return pred_moments_all 