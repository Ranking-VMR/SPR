from easydict import EasyDict as EDict
from tqdm import tqdm
from loguru import logger
import math
import h5py
import concurrent.futures
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel
import torch
import torch.nn.functional as F

from tasks.rvmr_seg_retr_faiss.proposal_refinement.run_utils import prepare_model, load_model
from tasks.rvmr_seg_retr_faiss.proposal_refinement.run_utils import topk_3d, generate_min_max_length_mask
from tasks.rvmr_seg_retr_faiss.proposal_refinement.tensor_utils import pad_sequences_1d
from tasks.rvmr_seg_retr_faiss.proposal_refinement.basic_utils import load_json, l2_normalize_np_array, uniform_feature_sampling


def prepare_refine_model(model_type, hidden_size, ckpt_pth, device):
    """
    Prepare model for refine stage.

    model_type = ""
    hidden_size = 768
    ckpt_pth = ""
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """
    model_config = EDict(
            visual_input_size=768 if "clip" in model_type else 1024,
            sub_input_size=768,  # for both desc and subtitles
            query_input_size=768,  # for both desc and subtitles
            hidden_size=hidden_size,  # hidden dimension
            conv_kernel_size=5,
            conv_stride=1,
            max_ctx_l=128,
            max_desc_l=30,
            input_drop=0.1,
            drop=0.1,
            n_heads=8,  # self-att heads
            initializer_range=0.02,  # for linear layer
            device=device,
            model_type=model_type)

    model = prepare_model(model_config)
    model, _, _ = load_model(ckpt_pth, device, model)
    return model

class TextFeatureExtractor:
    def __init__(self, model_type, device):
        
        model_source_dict = {
            "CLIP-ViT-L-14": "openai/clip-vit-large-patch14",
            "BERT": "bert-base-uncased"
        }
        
        if "clip" in model_type:
            text_model_type = "CLIP-ViT-L-14"
            model_source = model_source_dict[text_model_type]
            self.model = CLIPModel.from_pretrained(model_source)
            self.processor = CLIPProcessor.from_pretrained(model_source)
        else:
            text_model_type = "BERT"
            model_source = model_source_dict[text_model_type]
            self.model = BertModel.from_pretrained(model_source)
            self.processor = BertTokenizer.from_pretrained(model_source)
        
        self.model.to(device)
        self.model.eval()
        

        self.text_processing = lambda query: self.processor(text=query, return_tensors="pt", padding=True, truncation=True)
        
        self.model_type = model_type
        self.device = device
    
    @torch.no_grad()
    def query2emb(self, text_input):
        """
        Args:
            text_input (dict): The text input dictionary, generated from CLIPProcessor.
        Returns:
            q_emb (torch.Tensor): The query embeddings from the model, shape of [bs, d].
        """
        self.model.eval()
        text_input = text_input.to(self.device)
        if "clip" in self.model_type:
            q_emb = self.model.get_text_features(**text_input) # [bs, d]
        else:
            outputs = self.model(**text_input)
            q_emb = outputs.last_hidden_state # [bs, L, d]
        
        return q_emb
    
    @torch.no_grad()
    def generate_qemb_from_text(self, queries, bsize=None, to_cpu=False):
        """
        Directly generate query embeddings from text queries, this is used for small-scale query embeddings generation, i.e, real-time query input.
        Args:
            queries (list or tuple): The list of queries, lenth = num_query.
            bsize (int): The batch size of queries.
            to_cpu (bool): Whether to move the tensor to CPU.
        Returns:
            q_emb (torch.Tensor): The query embeddings from the model, shape of [num_query, d].
        """
        self.model.eval()
        logger.info(f"#> Generating query embeddings from {len(queries)} queries.")
        if bsize:
            def split_list(lst, n):
                lst_len = len(lst)
                return [lst[i:min(i + n, lst_len)] for i in range(0, lst_len, n)]
            batch_query = split_list(queries, bsize)
            batches = [self.text_processing(query) for query in batch_query]
            batch_qemb = [self.query2emb(text_input) for text_input in batches]
            q_emb = torch.cat(batch_qemb, dim=0) # [num_query,d] or [num_query,L,d]
            if "clip" not in self.model_type:
                batch_mask = [text_input["attention_mask"] for text_input in batches]
                q_mask = torch.cat(batch_mask, dim=0).to(self.device) # [num_query,L]
                q_emb = q_emb[:,:30,:]
                q_mask = q_mask[:,:30]
        else:
            text_input = self.text_processing(queries)
            q_emb = self.query2emb(text_input) # [num_query,d] or [num_query,L,d]
            if "clip" not in self.model_type:
                q_mask = text_input["attention_mask"].to(self.device)
                q_emb = q_emb[:,:30,:]
                q_mask = q_mask[:,:30]
        
        logger.info(f"#> Query embeddings generated, shape: {q_emb.shape}.")
        
        if "clip" not in self.model_type:
            return (q_emb.cpu(), q_mask.cpu()) if to_cpu else (q_emb, q_mask)
        else:
            return q_emb.cpu() if to_cpu else q_emb

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

class VideoFeatureExtractor:
    def __init__(self, model_type, video_len_path, video_feat_path_or_handler, clip_length, sub_bert_path_or_handler=None, normalize_vfeat=True, normalize_tfeat=True, moment_ctx_len=None, max_ctx_len=128):
        # load required data
        self.video_lens = load_json(video_len_path)
        if isinstance(video_feat_path_or_handler, h5py.File):
            self.vid_feat_h5 = video_feat_path_or_handler
        else:  # str path
            self.vid_feat_h5 = h5py.File(video_feat_path_or_handler, "r") # video feature
        
        if "clip" not in model_type:
            if isinstance(sub_bert_path_or_handler, h5py.File):
                self.sub_bert_h5 = sub_bert_path_or_handler
            else:
                self.sub_bert_h5 = h5py.File(sub_bert_path_or_handler, "r") # subtitle feature
        
        # set related hyper-parameters
        self.model_type = model_type
        self.max_ctx_len = max_ctx_len # max number of tokens for video and subtitle features
        self.clip_length = clip_length # length of each clip (seconds)
        self.moment_ctx_len = moment_ctx_len # context length for the moment
        
        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat
    
    def get_slice_indices(self, ts, max_idx):
        """
        Args:
            ts: [st (float), ed (float)] in seconds, ed > st
            max_idx: length of the video
        Returns:
            [st_idx, ed_idx]: int,
        Given ts = [3.2, 7.6], st_idx = 2, ed_idx = 6
        clips should be indexed as [2:6], i.e. ori_video[2:6] should be sampled.
        """
        st_idx = min(math.floor(ts[0] / self.clip_length), max_idx)
        ed_idx = min(math.ceil(ts[1] / self.clip_length), max_idx)

        return [st_idx, ed_idx]

    def generate_video_feature(self, one_coarse_moment_proposals):
        raw_data = {}
        raw_data["coarse_moment_proposals"] = one_coarse_moment_proposals
        
        model_inputs =  {}
        
        # prepare video and subtitle features of coarse proposals (may be padded with context) 
        raw_data["new_video_indices_in_old_video"] = []
        raw_data["new_video_timestamp"] = []
        raw_data["clip_length"] = []
        model_inputs["video_feat"] = []
        if "clip" not in self.model_type:
            model_inputs["sub_feat"] = []
        for moment in one_coarse_moment_proposals:
            video_name = moment["video_name"]
            video_clip_len = self.vid_feat_h5[video_name][:].shape[0]
            new_video_timestamp = moment["timestamp"]
            
            # pad the moment with context if specified
            if self.moment_ctx_len and self.moment_ctx_len > 0:
                new_video_timestamp = [new_video_timestamp[0] - self.moment_ctx_len, new_video_timestamp[1] + self.moment_ctx_len]
            # clip the moment to the video duration
            new_video_timestamp = [max(0, new_video_timestamp[0]), min(new_video_timestamp[1], self.video_lens[video_name])]
            
            new_video_indices_in_old_video = self.get_slice_indices(new_video_timestamp, max_idx=video_clip_len)
            expanded_new_video_timestamp_in_old_video = [new_video_indices_in_old_video[0] * self.clip_length, new_video_indices_in_old_video[1] * self.clip_length]
            new_video_clip_len = new_video_indices_in_old_video[1] - new_video_indices_in_old_video[0]
            if new_video_clip_len <= self.max_ctx_len :
                raw_data["clip_length"].append(self.clip_length)
            else:
                raw_data["clip_length"].append(new_video_clip_len * self.clip_length / self.max_ctx_len)
            
            # clip the moment to the video duration
            expanded_new_video_timestamp_in_old_video = [max(0, expanded_new_video_timestamp_in_old_video[0]), min(expanded_new_video_timestamp_in_old_video[1], self.video_lens[video_name])]
            
            raw_data["new_video_indices_in_old_video"].append(new_video_indices_in_old_video)
            raw_data["new_video_timestamp"].append(expanded_new_video_timestamp_in_old_video)
            
            video_feat = self.vid_feat_h5[video_name][:][new_video_indices_in_old_video[0]:new_video_indices_in_old_video[1]]
            video_feat = uniform_feature_sampling(video_feat, self.max_ctx_len)
            if self.normalize_vfeat:
                video_feat = l2_normalize_np_array(video_feat)
            model_inputs["video_feat"].append(torch.from_numpy(video_feat))
            

            if "clip" not in self.model_type:
                sub_feat = self.sub_bert_h5[video_name][:][new_video_indices_in_old_video[0]:new_video_indices_in_old_video[1]]
                sub_feat = uniform_feature_sampling(sub_feat, self.max_ctx_len)
                if self.normalize_tfeat:
                    sub_feat = l2_normalize_np_array(sub_feat)
                model_inputs["sub_feat"].append(torch.from_numpy(sub_feat))
        
        model_inputs["video_feat"], model_inputs["video_mask"] = pad_sequences_1d(model_inputs["video_feat"], dtype=torch.float32, fixed_length=self.max_ctx_len)
        if "clip" not in self.model_type:
            model_inputs["sub_feat"], model_inputs["sub_mask"] = pad_sequences_1d(model_inputs["sub_feat"], dtype=torch.float32, fixed_length=self.max_ctx_len)
        
        return dict(meta=raw_data, model_inputs=model_inputs)
    
    def process_one_moment(self, index, one_coarse_moment_proposals):
        moment = one_coarse_moment_proposals[index]
        
        video_name = moment["video_name"]
        video_clip_len = self.vid_feat_h5[video_name][:].shape[0]
        new_video_timestamp = moment["timestamp"]
        
        one_moment_process = {}
        
        # pad the moment with context if specified
        if self.moment_ctx_len and self.moment_ctx_len > 0:
            new_video_timestamp = [new_video_timestamp[0] - self.moment_ctx_len, new_video_timestamp[1] + self.moment_ctx_len]
        # clip the moment to the video duration
        new_video_timestamp = [max(0, new_video_timestamp[0]), min(new_video_timestamp[1], self.video_lens[video_name])]
        
        new_video_indices_in_old_video = self.get_slice_indices(new_video_timestamp, max_idx=video_clip_len)
        expanded_new_video_timestamp_in_old_video = [new_video_indices_in_old_video[0] * self.clip_length, new_video_indices_in_old_video[1] * self.clip_length]
        new_video_clip_len = new_video_indices_in_old_video[1] - new_video_indices_in_old_video[0]
        if new_video_clip_len <= self.max_ctx_len :
            one_moment_process["clip_length"] = self.clip_length
        else:
            one_moment_process["clip_length"] = new_video_clip_len * self.clip_length / self.max_ctx_len
        
        # clip the moment to the video duration
        expanded_new_video_timestamp_in_old_video = [max(0, expanded_new_video_timestamp_in_old_video[0]), min(expanded_new_video_timestamp_in_old_video[1], self.video_lens[video_name])]
        
        one_moment_process["new_video_indices_in_old_video"] = new_video_indices_in_old_video
        one_moment_process["new_video_timestamp"] = expanded_new_video_timestamp_in_old_video
        
        video_feat = self.vid_feat_h5[video_name][:][new_video_indices_in_old_video[0]:new_video_indices_in_old_video[1]]
        video_feat = uniform_feature_sampling(video_feat, self.max_ctx_len)
        if self.normalize_vfeat:
            video_feat = l2_normalize_np_array(video_feat)
        one_moment_process["video_feat"] = torch.from_numpy(video_feat)
        

        if "clip" not in self.model_type:
            sub_feat = self.sub_bert_h5[video_name][:][new_video_indices_in_old_video[0]:new_video_indices_in_old_video[1]]
            sub_feat = uniform_feature_sampling(sub_feat, self.max_ctx_len)
            if self.normalize_tfeat:
                sub_feat = l2_normalize_np_array(sub_feat)
            one_moment_process["sub_feat"] = torch.from_numpy(sub_feat)
        
        return index, one_moment_process
    
        
    
    def generate_video_feature_mt(self, one_coarse_moment_proposals):
        raw_data = {}
        raw_data["coarse_moment_proposals"] = one_coarse_moment_proposals
        
        num_moments = len(one_coarse_moment_proposals)
        results = [None] * num_moments
        
        # prepare video and subtitle features of coarse proposals (may be padded with context) 
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # submit tasks to the executor
            futures = [executor.submit(lambda x: self.process_one_moment(x, one_coarse_moment_proposals), i) for i in range(num_moments)]
            
            # iterate through the futures as they are completed
            for future in concurrent.futures.as_completed(futures):
                index, one_moment_process = future.result()
                results[index] = one_moment_process  # store the result in the results list
        
        raw_data["new_video_indices_in_old_video"] = [result["new_video_indices_in_old_video"] for result in results]
        raw_data["new_video_timestamp"] = [result["new_video_timestamp"] for result in results]
        raw_data["clip_length"] =  [result["clip_length"] for result in results]
        
        model_inputs =  {}
        model_inputs["video_feat"] = [result["video_feat"] for result in results]
        if "clip" not in self.model_type:
            model_inputs["sub_feat"] = [result["sub_feat"] for result in results]
        
        model_inputs["video_feat"], model_inputs["video_mask"] = pad_sequences_1d(model_inputs["video_feat"], dtype=torch.float32, fixed_length=self.max_ctx_len)
        if "clip" not in self.model_type:
            model_inputs["sub_feat"], model_inputs["sub_mask"] = pad_sequences_1d(model_inputs["sub_feat"], dtype=torch.float32, fixed_length=self.max_ctx_len)
        
        return dict(meta=raw_data, model_inputs=model_inputs)

@torch.no_grad()
def generate_refined_moment_prediction(model_type, model, query_feat_extract, queries, video_feat_extract, coarse_moment_proposals, video_lens, device):
    
    model.eval()
    
    
    # prepare query feats
    model_inputs = {}
    if "clip" in model_type:
        model_inputs["query_feat"] = query_feat_extract(queries)
    else:
        model_inputs["query_feat"], model_inputs["query_mask"] = query_feat_extract(queries)
    bsz = model_inputs["query_feat"].shape[0]
    
    # prepare video and sub feats
    visual_input_dict = [video_feat_extract(coarse_moment_proposals[i]) for i in range(bsz)]
    model_inputs["video_feat"] = [visual_input["model_inputs"]["video_feat"] for visual_input in visual_input_dict]
    model_inputs["video_mask"] = [visual_input["model_inputs"]["video_mask"] for visual_input in visual_input_dict]
    if "clip" not in model_type:
        model_inputs["sub_feat"] = [visual_input["model_inputs"]["sub_feat"] for visual_input in visual_input_dict]
        model_inputs["sub_mask"] = [visual_input["model_inputs"]["sub_mask"] for visual_input in visual_input_dict] 
    metas = [visual_input["meta"] for visual_input in visual_input_dict]
    
    all_pred = {}
    for batch_idx in range(bsz):
        if "clip" not in model_type:
            query_feat = model_inputs["query_feat"][batch_idx].unsqueeze(0).to(device)
            query_mask = model_inputs["query_mask"][batch_idx].unsqueeze(0).to(device)
            video_feat = model_inputs["video_feat"][batch_idx].to(device)
            video_mask = model_inputs["video_mask"][batch_idx].to(device)
            sub_feat = model_inputs["sub_feat"][batch_idx].to(device)
            sub_mask = model_inputs["sub_mask"][batch_idx].to(device)
            
            video_feat, sub_feat = model.encode_context(video_feat, video_mask, sub_feat, sub_mask)
            
            query_scores, start_probs, end_probs = model.get_pred_from_raw_query(
                query_feat = query_feat,
                query_mask = query_mask, 
                video_feat = video_feat,
                video_mask = video_mask,
                sub_feat = sub_feat,
                sub_mask = sub_mask,
                cross=True)
        else:
            query_feat = model_inputs["query_feat"][batch_idx].unsqueeze(0).to(device)
            video_feat = model_inputs["video_feat"][batch_idx].to(device)
            video_mask = model_inputs["video_mask"][batch_idx].to(device)
            
            video_feat = model.encode_context(video_feat, video_mask)
            
            query_scores, start_probs, end_probs = model.get_pred_from_raw_query(
                query_feat = query_feat,
                video_feat = video_feat,
                video_mask = video_mask,
                cross=True)
        
        # final ranking score: theta = start_probs * end_probs * exp(alpha * query_scores)
        query_scores = torch.exp(30 * query_scores) # [1, Nv]
        start_probs = F.softmax(start_probs, dim=-1) # [1, Nv, L]
        end_probs = F.softmax(end_probs, dim=-1) # [1, Nv, L]
        
        # compute moment-level ranking scores and generate refined proposal for each coarse moemnt proposal
        all_2D_map = torch.einsum("qvm,qv,qvn->qvmn", start_probs, query_scores, end_probs) # [1, Nv, L, L]
        map_mask = generate_min_max_length_mask(all_2D_map.shape, min_l=1, max_l=16)
        all_2D_map = all_2D_map * torch.from_numpy(map_mask).to(all_2D_map.device)
        
        score_map = all_2D_map.squeeze(0) # [Nv, L, L]
        n_moments = score_map.shape[0]
        query_id = batch_idx
        one_coarse_moment_proposals = coarse_moment_proposals[query_id]
        new_video_timestamp = metas[batch_idx]["new_video_timestamp"]
        clip_length = metas[batch_idx]["clip_length"]
        pred_result = []
        for moment_idx in range(n_moments):
            video_name = one_coarse_moment_proposals[moment_idx]["video_name"]
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