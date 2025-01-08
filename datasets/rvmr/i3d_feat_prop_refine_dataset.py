import h5py
import math
import numpy as np
import random
import math
import torch
from collections import defaultdict
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from tasks.rvmr_seg_retr_faiss.proposal_refinement.basic_utils import load_json, l2_normalize_np_array, uniform_feature_sampling
from tasks.rvmr_seg_retr_faiss.proposal_refinement.tensor_utils import pad_sequences_1d


class TrainDataset(Dataset):

    def __init__(self, data_path, coarse_pred_path, video_len_path, desc_bert_path_or_handler, sub_bert_path_or_handler, max_desc_len,
                 max_ctx_len, video_feat_path_or_handler, clip_length, ori_seg_len, ctx_mode, normalize_vfeat=True,
                 normalize_tfeat=True, moment_ctx_len=None, use_coarse_hard_neg=False):
        """
        Args:
            data_path (str): training annotation file path
            coarse_pred_path (str): coarse prediction file path, including {"pos":{qid: {pair_id: []}}, "neg":{qid: []}}
            desc_bert_path (str): query feature file path
            sub_bert_path (str): subtitle feature file path
            max_desc_len (int): max number of tokens for query feature
            max_ctx_len (int): max number of tokens for video and subtitle features
            video_feat_path (str): video feature file path
            clip_length (float): length of each clip (seconds)
            ori_seg_len (float): length of segment in coarse proposal (seconds)
            ctx_mode (str): modalities to use, e.g., "video_sub" means using both video and subtitle features
            normalize_vfeat (bool, optional): if normalize vfeat. Defaults to True.
            normalize_tfeat (bool, optional): if normalize tfeat. Defaults to True.
        """
        # load annotations
        assert int(moment_ctx_len) in [0,4,8,12] and str(int(moment_ctx_len)) in coarse_pred_path, f"moment_ctx_len should be 0, 4, 8, 12, but got {moment_ctx_len}; coarse_pred_path: {coarse_pred_path}"
        
        train_annos = load_json(data_path)
        self.annotations = self.flat_annotations(train_annos)
        self.coarse_predictions = load_json(coarse_pred_path)
        self.coarse_predictions_pos, self.coarse_predictions_neg = self.coarse_predictions["pos"], self.coarse_predictions["neg"]
        
        self.positive_sample_dict = self.build_positive_sample_dict(self.annotations)
        self.use_coarse_hard_neg = use_coarse_hard_neg
        if use_coarse_hard_neg:
            logger.info("Using hard negatives from coarse predictions.")

        self.video_lens = load_json(video_len_path)

        # set related hyper-parameters
        self.max_desc_len = max_desc_len
        self.max_ctx_len = max_ctx_len # max number of tokens for video and subtitle features
        self.clip_length = clip_length # length of each clip (seconds)
        self.ori_seg_len = ori_seg_len # length of segment in coarse proposal (seconds)
        self.moment_ctx_len = moment_ctx_len # context length for the moment

        # prepare multi-modal feature data
        self.use_video = "video" in ctx_mode
        self.use_sub = "sub" in ctx_mode
        
        if isinstance(desc_bert_path_or_handler, h5py.File):
            self.desc_bert_h5 = desc_bert_path_or_handler
        else:
            self.desc_bert_h5 = h5py.File(desc_bert_path_or_handler, "r") # query feature
        if self.use_video:
            if isinstance(video_feat_path_or_handler, h5py.File):
                self.vid_feat_h5 = video_feat_path_or_handler
            else:  # str path
                self.vid_feat_h5 = h5py.File(video_feat_path_or_handler, "r") # video feature
        if self.use_sub:
            if isinstance(sub_bert_path_or_handler, h5py.File):
                self.sub_bert_h5 = sub_bert_path_or_handler
            else:
                self.sub_bert_h5 = h5py.File(sub_bert_path_or_handler, "r") # subtitle feature

        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        """
        raw_data: {
            "query_id":
            "query":
            "pair_id":
            "video_name":
            "timestamp":
            "duration":
            "similarity":
            "caption":
        }
        return a dictionary:{
            "model_inputs":{
                "simi":
                "query_feat":
                "video_feat":
                "sub_feat":
                "st_ed_indices":
            },
            "meta": raw_data
        }
        
        """

        raw_data = self.annotations[index]
        query_id = str(raw_data["query_id"])
        pair_id = str(raw_data["pair_id"])
        video_name = raw_data["video_name"]
        video_duration = raw_data["duration"]
        
        #================= sample proposal span for training =================
        # we sample from the coarse predictions if the pair_id is in the coarse predictions, otherwise we use create proposal with raw data
        if pair_id in self.coarse_predictions_pos[query_id]:
            moment_for_train = random.choice(self.coarse_predictions_pos[query_id][pair_id])
            new_video_timestamp = moment_for_train["timestamp"]
            # assert moment_for_train["video_name"] == video_name, f"video_name mismatch: {moment_for_train['video_name']} vs {video_name}"
        else:
            new_video_timestamp = raw_data["timestamp"]
            new_video_timestamp = [math.floor(new_video_timestamp[0]/self.ori_seg_len)  * self.ori_seg_len, math.ceil(new_video_timestamp[1]/self.ori_seg_len) * self.ori_seg_len]
            # add some random context with rounded gt.
            new_video_timestamp = [new_video_timestamp[0] - random.randint(0,1) * self.ori_seg_len, new_video_timestamp[1] + random.randint(0,1) * self.ori_seg_len]
        # pad the moment with context if specified   
        if self.moment_ctx_len and self.moment_ctx_len > 0:
            new_video_timestamp = [new_video_timestamp[0] - self.moment_ctx_len, new_video_timestamp[1] + self.moment_ctx_len]
        # clip the moment to the video duration
        new_video_timestamp = [max(0, new_video_timestamp[0]), min(new_video_timestamp[1], video_duration)]
        # get the moment timestamp
        moment_timestamp = raw_data["timestamp"]
                
        # get new_sample_info
        video_clip_len = self.vid_feat_h5[video_name][:].shape[0]
        new_sample_info_dict = self.generate_new_sample_info(new_video_timestamp, moment_timestamp, video_clip_len)
        raw_data.update(new_sample_info_dict)
        #=====================================================================
        
        #================== formulate the model inputs (positive sample) ==================
        
        # prepare query features
        model_inputs = dict()
        model_inputs["simi"] = raw_data["similarity"]
        model_inputs["query_feat"] = self.get_query_feat_by_query_id(query_id)

        # prepare video and subtitle features of new video
        new_video_indices_in_old_video = new_sample_info_dict["new_video_indices_in_old_video"]
        #ctx_l = 0
        if self.use_video:
            video_feat = self.vid_feat_h5[video_name][:][new_video_indices_in_old_video[0]:new_video_indices_in_old_video[1]]
            video_feat = uniform_feature_sampling(video_feat, self.max_ctx_len)
            if self.normalize_vfeat:
                video_feat = l2_normalize_np_array(video_feat)
            model_inputs["video_feat"] = torch.from_numpy(video_feat)
            #ctx_l = len(video_feat)
        else:
            model_inputs["video_feat"] = torch.zeros((2, 2))

        if self.use_sub:  # no need for ctx feature, as the features are already contextualized
            sub_feat = self.sub_bert_h5[video_name][:][new_video_indices_in_old_video[0]:new_video_indices_in_old_video[1]]
            sub_feat = uniform_feature_sampling(sub_feat, self.max_ctx_len)
            if self.normalize_tfeat:
                sub_feat = l2_normalize_np_array(sub_feat)
            model_inputs["sub_feat"] = torch.from_numpy(sub_feat)
            #ctx_l = len(sub_feat)
        else:
            model_inputs["sub_feat"] = torch.zeros((2, 2))
        
        # prepare moment index labels for optimization
        #model_inputs["st_ed_indices"] = new_sample_info_dict["moment_indices_in_new_video"]
        model_inputs["st_ed_indices"] = torch.tensor(new_sample_info_dict["moment_indices_in_new_video"], dtype=torch.long)
        #=====================================================================
        
        
        #================== sample hard negative sample from coarse predictions (if specified) ==================
        if self.use_coarse_hard_neg:
            hard_neg_pool = self.coarse_predictions_neg[query_id]
            hard_neg_info = random.choice(hard_neg_pool)
            raw_data["hard_neg_info"] = hard_neg_info
            hard_neg_video_name = hard_neg_info["video_name"]
            hard_neg_video_timestamp = hard_neg_info["timestamp"]
            
            # pad the moment with context if specified   
            if self.moment_ctx_len and self.moment_ctx_len > 0:
                hard_neg_video_timestamp = [hard_neg_video_timestamp[0] - self.moment_ctx_len, hard_neg_video_timestamp[1] + self.moment_ctx_len]
            # clip the moment to the video duration
            hard_neg_video_timestamp = [max(0, hard_neg_video_timestamp[0]), min(hard_neg_video_timestamp[1], self.video_lens[hard_neg_video_name])]

            # get new_sample_info
            hard_neg_video_clip_len = self.vid_feat_h5[hard_neg_video_name][:].shape[0]
            hard_neg_video_indices_in_old_video = self.get_slice_indices(hard_neg_video_timestamp, hard_neg_video_clip_len)  #【st_idx, ed_idx) in the old video

            # prepare video and subtitle features of hard negative video
            if self.use_video:
                hard_neg_video_feat = self.vid_feat_h5[hard_neg_video_name][:][hard_neg_video_indices_in_old_video[0]:hard_neg_video_indices_in_old_video[1]]
                hard_neg_video_feat = uniform_feature_sampling(hard_neg_video_feat, self.max_ctx_len)
                if self.normalize_vfeat:
                    hard_neg_video_feat = l2_normalize_np_array(hard_neg_video_feat)
                model_inputs["hard_neg_video_feat"] = torch.from_numpy(hard_neg_video_feat)
                #ctx_l = len(video_feat)
            else:
                model_inputs["hard_neg_video_feat"] = torch.zeros((2, 2))

            if self.use_sub:  # no need for ctx feature, as the features are already contextualized
                hard_neg_sub_feat = self.sub_bert_h5[hard_neg_video_name][:][hard_neg_video_indices_in_old_video[0]:hard_neg_video_indices_in_old_video[1]]
                hard_neg_sub_feat = uniform_feature_sampling(hard_neg_sub_feat, self.max_ctx_len)
                if self.normalize_tfeat:
                    hard_neg_sub_feat = l2_normalize_np_array(hard_neg_sub_feat)
                model_inputs["hard_neg_sub_feat"] = torch.from_numpy(hard_neg_sub_feat)
                #ctx_l = len(sub_feat)
            else:
                model_inputs["hard_neg_sub_feat"] = torch.zeros((2, 2))
        
        #=====================================================================
        
        return dict(meta=raw_data, model_inputs=model_inputs)

    def generate_new_sample_info(self, new_video_timestamp, moment_timestamp, video_clip_len):
        """_summary_

        Args:
            new_video_timestamp (list): new video timestamp [st (float), ed (float)] in seconds.
            moment_timestamp (list): original moment timestamp [st (float), ed (float)] in seconds.
            video_clip_len (int): length of the video clip in feature file.
        """
    
        # get the new video indices in the old video, and therefore expand new video timestamp to the nearest boarder.
        new_video_indices_in_old_video = self.get_slice_indices(new_video_timestamp, video_clip_len)  #【st_idx, ed_idx) in the old video
        new_video_clip_len = new_video_indices_in_old_video[1] - new_video_indices_in_old_video[0] # No. of clips in the new video
        expanded_new_video_timestamp_in_old_video = [new_video_indices_in_old_video[0] * self.clip_length, new_video_indices_in_old_video[1] * self.clip_length] # please note the correlation between idx and timestamp, idx -> [idx*clip_length, (idx+1)*clip_length], thus[st_idx, ed_idx) -> [st_idx * clip_length, ed_idx * clip_length]
        
        # if the new sampled video exceed max_ctx_len, we need to adjust the clip length
        if new_video_clip_len <= self.max_ctx_len :
            new_clip_length = self.clip_length
        else:
            new_clip_length = new_video_clip_len * self.clip_length / self.max_ctx_len
        
        # get the indices of the moment in the old & new video
        moment_timestamp_in_old_video = [max(expanded_new_video_timestamp_in_old_video[0], moment_timestamp[0]), min(expanded_new_video_timestamp_in_old_video[1], moment_timestamp[1])] # clip the moment to the new video
        moment_timestamp_in_new_video = [moment_timestamp_in_old_video[0] - expanded_new_video_timestamp_in_old_video[0], moment_timestamp_in_old_video[1] - expanded_new_video_timestamp_in_old_video[0]] # timestamp in the new video
        
        moment_indices_in_new_video = self.get_st_ed_label(moment_timestamp_in_new_video, new_video_clip_len, new_clip_length)
        
        new_sample_info_dict = dict(
                        new_video_timestamp=expanded_new_video_timestamp_in_old_video,
                        new_video_indices_in_old_video=new_video_indices_in_old_video, 
                        new_video_clip_len=new_video_clip_len,
                        moment_timestamp_in_new_video=moment_timestamp_in_new_video,
                        moment_indices_in_new_video=moment_indices_in_new_video,
                    )
        
        return new_sample_info_dict

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
    
    def get_st_ed_label(self, ts, max_idx, clip_length):
        """
        Args:
            ts: [st (float), ed (float)] in seconds, ed > st
            max_idx: length of the video
            clip_length: length of each clip (seconds)
        Returns:
            [st_idx, ed_idx]: int,
        Given ts = [3.2, 7.6], clip_length=1.5: st_idx = 2, ed_idx = 5, (i.e. clip [2,3,4,5] are included)
        clips should be indexed as [2: 5], the translated back ts should be [3:9].
        """
        if max_idx <= self.max_ctx_len:
            st_idx = min(math.floor(ts[0] / clip_length), max_idx)
            ed_idx = min(math.ceil(ts[1] / clip_length), max_idx) - 1
        else:
            st_idx = min(math.floor(ts[0] / clip_length), self.max_ctx_len)
            ed_idx = min(math.ceil(ts[1] / clip_length), self.max_ctx_len) - 1
        return [st_idx, ed_idx]

    def get_query_feat_by_query_id(self, query_id):
        query_feat = self.desc_bert_h5[query_id][:self.max_desc_len]
        if self.normalize_tfeat:
            query_feat = l2_normalize_np_array(query_feat)
        return torch.from_numpy(query_feat)

    def flat_annotations(self, annotations):
        """
        Flat the hierarchical annotations to a list of samples. Each sample contains the query and relevant moment info.
        To facilitate the training process, we neglect the moments with too long duration.
        Returns:
            new_annotations (List[Dict]): The list of samples, each sample contains the query and relevant moment info.
        """
        #logger.info(f"Flatten the hierarchical annotations to a list of samples. Meanwhile, we filter out the moments with duration longer than {max_duration}.")
        logger.info(f"Flatten the hierarchical annotations to a list of samples.")
        new_annotations = []
        for q_anno in annotations:
            query = q_anno["query"]
            query_id = str(q_anno["query_id"])
            relevant_moments = q_anno["relevant_moment"]
            relevant_moments = sorted(relevant_moments, key=lambda x: x["similarity"], reverse=True)
            selected_moments = relevant_moments
            for moment in selected_moments:
                new_anno = {'query': query, 'query_id': query_id}
                new_anno.update(moment)
                new_annotations.append(new_anno)
        return new_annotations
    
    def build_positive_sample_dict(self, train_annos):
        """
        Build the positive sample dictionary. {query_id: [caption1, caption2, ...]}
        Please note that we use caption to label positive samples due to the generation of pos samples are based on captions.
        Args:
            train_annos (List[Dict]): the training annotations.
        Returns:
            positive_sample_dict (Dict): the positive caption dictionary.
        """
        positive_sample_dict = defaultdict(set)
        for anno in train_annos:
            query_id = str(anno["query_id"])
            caption = anno["caption"]
            positive_sample_dict[query_id].add(caption)
        return positive_sample_dict


class EvalDataset(Dataset):
    def __init__(self, data_path, coarse_pred_path, video_len_path, desc_bert_path_or_handler, sub_bert_path_or_handler, max_desc_len,
                 max_ctx_len, video_feat_path_or_handler, clip_length, ctx_mode, normalize_vfeat=True,
                 normalize_tfeat=True, moment_ctx_len=None):


        # load annotations
        self.annotations = load_json(data_path)
        self.coarse_predictions = load_json(coarse_pred_path)
        self.video_lens = load_json(video_len_path)

        # set related hyper-parameters
        self.max_desc_len = max_desc_len
        self.max_ctx_len = max_ctx_len # max number of tokens for video and subtitle features
        self.clip_length = clip_length # length of each clip (seconds)
        self.moment_ctx_len = moment_ctx_len # context length for the moment
        
        # prepare multi-modal feature data
        self.use_video = "video" in ctx_mode
        self.use_sub = "sub" in ctx_mode
        
        if isinstance(desc_bert_path_or_handler, h5py.File):
            self.desc_bert_h5 = desc_bert_path_or_handler
        else:
            self.desc_bert_h5 = h5py.File(desc_bert_path_or_handler, "r") # query feature

        if self.use_video:
            if isinstance(video_feat_path_or_handler, h5py.File):
                self.vid_feat_h5 = video_feat_path_or_handler
            else:  # str path
                self.vid_feat_h5 = h5py.File(video_feat_path_or_handler, "r") # video feature
        if self.use_sub:
            if isinstance(sub_bert_path_or_handler, h5py.File):
                self.sub_bert_h5 = sub_bert_path_or_handler
            else:
                self.sub_bert_h5 = h5py.File(sub_bert_path_or_handler, "r") # subtitle feature
        
        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat
        
        self.ground_truth = self.get_relevant_moment_gt()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        raw_data = self.annotations[index]
        
        # prepare query features
        query_id = str(raw_data["query_id"])
        model_inputs =  {"query_feat": self.get_query_feat_by_query_id(query_id)}
        
        # prepare video and subtitle features of coarse proposals (may be padded with context)
        coarse_moment_proposals = self.coarse_predictions[query_id] # list of moments for query_id from the coarse predictions
        raw_data["coarse_moment_proposals"] = coarse_moment_proposals 
        raw_data["new_video_indices_in_old_video"] = []
        raw_data["new_video_timestamp"] = []
        raw_data["clip_length"] = []
        model_inputs["video_feat"] = []
        model_inputs["sub_feat"] = []
        for moment in coarse_moment_proposals:
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
            
            if self.use_video:
                video_feat = self.vid_feat_h5[video_name][:][new_video_indices_in_old_video[0]:new_video_indices_in_old_video[1]]
                video_feat = uniform_feature_sampling(video_feat, self.max_ctx_len)
                if self.normalize_vfeat:
                    video_feat = l2_normalize_np_array(video_feat)
                model_inputs["video_feat"].append(torch.from_numpy(video_feat))
            else:
                model_inputs["video_feat"].append(torch.zeros((2, 2)))

            if self.use_sub:  # no need for ctx feature, as the features are already contextualized
                sub_feat = self.sub_bert_h5[video_name][:][new_video_indices_in_old_video[0]:new_video_indices_in_old_video[1]]
                sub_feat = uniform_feature_sampling(sub_feat, self.max_ctx_len)
                if self.normalize_tfeat:
                    sub_feat = l2_normalize_np_array(sub_feat)
                model_inputs["sub_feat"].append(torch.from_numpy(sub_feat))
            else:
                model_inputs["sub_feat"].append(torch.zeros((2, 2)))
        
        return dict(meta=raw_data, model_inputs=model_inputs)

    def get_query_feat_by_query_id(self, query_id):
        query_feat = self.desc_bert_h5[query_id][:self.max_desc_len]
        if self.normalize_tfeat:
            query_feat = l2_normalize_np_array(query_feat)
        return torch.from_numpy(query_feat)

    def get_relevant_moment_gt(self):
        """

        Returns:
            gt_all (dict): key: query_id, value: list of relevant moments.
        """
        gt_all = {}
        for data in self.annotations:
            qid = str(data["query_id"])
            relevant_moments = data["relevant_moment"]
            relevant_moments = [moment for moment in relevant_moments if moment["relevance"] > 0]
            gt_all[qid] = relevant_moments
        return gt_all
    
    
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


def collate_fn(batch, task, positive_sample_dict=None):
    batch_meta = [e["meta"] for e in batch]
    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    if task == "train":
        
        # prepare batch-level query-context pos masks
        if positive_sample_dict is not None:
            batch_query_ids = [str(e["meta"]["query_id"]) for e in batch]
            batch_captions = [str(e["meta"]["caption"]) for e in batch]
            batch_q2ctx_pos_mask = [torch.tensor([1 if cap in positive_sample_dict[qid] else 0 for cap in batch_captions], dtype=torch.float32) for qid in batch_query_ids]
            batch_q2ctx_pos_mask = torch.stack(batch_q2ctx_pos_mask, dim=0)
            batched_data["query_context_pos_mask"] = batch_q2ctx_pos_mask
        
        # prepare batched features & masks
        for k in model_inputs_keys:
            if "feat" in k:
                
                if k in ['video_feat', 'sub_feat']:
                    fixed_length = 128
                elif k in ['hard_neg_video_feat', 'hard_neg_sub_feat']:
                    fixed_length = 128
                else:
                    fixed_length = None
                batched_data[k] = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=torch.float32,
                                                fixed_length=fixed_length)
        # prepared batched gt labels
        fixed_length = 128
        if "st_ed_indices" in model_inputs_keys:
            st_ed_indices = [e["model_inputs"]["st_ed_indices"] for e in batch]
            # construct moment localization labels
            batched_data["st_ed_indices"] = torch.stack(st_ed_indices, dim=0)
            # construct moment localization foreground and background labels
            match_labels = np.zeros(shape=(len(st_ed_indices), fixed_length), dtype=np.int32)
            for idx, st_ed_index in enumerate(st_ed_indices):
                st_ed = st_ed_index.cpu().numpy()
                st, ed = st_ed[0], st_ed[1]
                match_labels[idx][st:(ed + 1)] = 1 # include the end
            batched_data['match_labels'] = torch.tensor(match_labels, dtype=torch.long)
        if "simi" in model_inputs_keys:
            simis = [e["model_inputs"]["simi"] for e in batch]
            batched_data["simi"] =  torch.tensor(simis)
    elif task == "eval":
        batched_data["query_feat"] = pad_sequences_1d([e["model_inputs"]["query_feat"] for e in batch], dtype=torch.float32,
                                                fixed_length=None)
        fixed_length = 128
        if batch[0]["model_inputs"]["video_feat"]:
            batched_data["video_feat"] = [pad_sequences_1d(e["model_inputs"]["video_feat"], dtype=torch.float32, fixed_length=fixed_length) for e in batch]
        if batch[0]["model_inputs"]["sub_feat"]:
            batched_data["sub_feat"] = [pad_sequences_1d(e["model_inputs"]["sub_feat"], dtype=torch.float32, fixed_length=fixed_length) for e in batch]

    return  batch_meta, batched_data

def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False, task="train"):
    model_inputs = {}
    if task=="train":
        for k, v in batched_model_inputs.items():
            if "feat" in k:
                model_inputs[k] = v[0].to(device, non_blocking=non_blocking)
                model_inputs[k.replace("feat", "mask")] = v[1].to(device, non_blocking=non_blocking)
            else:
                model_inputs[k] = v.to(device, non_blocking=non_blocking)
    elif task=="eval":
        model_inputs["query_feat"], model_inputs["query_mask"] = batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking), batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking)
        if "video_feat" in batched_model_inputs:
            model_inputs["video_feat"] = [e[0].to(device, non_blocking=non_blocking) for e in batched_model_inputs["video_feat"]]
            model_inputs["video_mask"] = [e[1].to(device, non_blocking=non_blocking) for e in batched_model_inputs["video_feat"]]
        if "sub_feat" in batched_model_inputs:
            model_inputs["sub_feat"] = [e[0].to(device, non_blocking=non_blocking) for e in batched_model_inputs["sub_feat"]]
            model_inputs["sub_mask"] = [e[1].to(device, non_blocking=non_blocking) for e in batched_model_inputs["sub_feat"]]
    return model_inputs

def prepare_dataset(opt):
    train_set = TrainDataset(
        data_path=opt.train_path,
        coarse_pred_path=opt.train_pos_neg_path,
        video_len_path=opt.video_len_path,
        desc_bert_path_or_handler=opt.desc_bert_path,
        sub_bert_path_or_handler=opt.sub_bert_path,
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        video_feat_path_or_handler=opt.video_feat_path,
        clip_length=opt.clip_length,
        ori_seg_len=opt.ori_seg_len,
        ctx_mode=opt.ctx_mode,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat,
        moment_ctx_len=opt.moment_ctx_len,
        use_coarse_hard_neg=opt.use_coarse_hard_neg)
    positive_sample_dict = train_set.positive_sample_dict if opt.model_type in ["reloclnet_rvmr", "reloclnet_rvmr_hard_neg", "reloclnet_rvmr_weighted"] else None
    train_loader = DataLoader(train_set, collate_fn=lambda batch: collate_fn(batch, task='train', positive_sample_dict=positive_sample_dict), batch_size=opt.bsz, num_workers=opt.num_workers, shuffle=True, drop_last=True, pin_memory=opt.pin_memory)


    val_set = EvalDataset(
        data_path=opt.val_path,
        coarse_pred_path=opt.val_coarse_pred_path, 
        video_len_path=opt.video_len_path,
        desc_bert_path_or_handler=train_set.desc_bert_h5,
        sub_bert_path_or_handler=train_set.sub_bert_h5, 
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        video_feat_path_or_handler=train_set.vid_feat_h5,
        clip_length=opt.clip_length,
        ctx_mode=opt.ctx_mode,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat,
        moment_ctx_len=opt.moment_ctx_len,)
    val_loader = DataLoader(val_set, collate_fn=lambda batch: collate_fn(batch, task='eval'), batch_size=opt.bsz_eval, num_workers=opt.num_workers, shuffle=False, drop_last=False, pin_memory=opt.pin_memory)
    
    val_gt = val_set.ground_truth
    video_lens = val_set.video_lens

    test_set = EvalDataset(
        data_path=opt.test_path,
        coarse_pred_path=opt.test_coarse_pred_path, 
        video_len_path=opt.video_len_path,
        desc_bert_path_or_handler=train_set.desc_bert_h5,
        sub_bert_path_or_handler=train_set.sub_bert_h5, 
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        video_feat_path_or_handler=train_set.vid_feat_h5,
        clip_length=opt.clip_length,
        ctx_mode=opt.ctx_mode,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat,
        moment_ctx_len=opt.moment_ctx_len,)
    test_loader = DataLoader(test_set, collate_fn=lambda batch: collate_fn(batch, task='eval'), batch_size=opt.bsz_eval, num_workers=opt.num_workers, shuffle=False, drop_last=False, pin_memory=opt.pin_memory)
    
    test_gt = test_set.ground_truth
    

    return train_loader, val_loader, val_gt, test_loader, test_gt, video_lens

def prepare_dataset_eval(opt):

    val_set = EvalDataset(
        data_path=opt.val_path,
        coarse_pred_path=opt.val_coarse_pred_path, 
        video_len_path=opt.video_len_path,
        desc_bert_path_or_handler=opt.desc_bert_path,
        sub_bert_path_or_handler=opt.sub_bert_path, 
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        video_feat_path_or_handler=opt.video_feat_path,
        clip_length=opt.clip_length,
        ctx_mode=opt.ctx_mode,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat,
        moment_ctx_len=opt.moment_ctx_len,)
    val_loader = DataLoader(val_set, collate_fn=lambda batch: collate_fn(batch, task='eval'), batch_size=opt.bsz_eval, num_workers=opt.num_workers, shuffle=False, drop_last=False, pin_memory=opt.pin_memory)
    
    val_gt = val_set.ground_truth
    video_lens = val_set.video_lens

    test_set = EvalDataset(
        data_path=opt.test_path,
        coarse_pred_path=opt.test_coarse_pred_path, 
        video_len_path=opt.video_len_path,
        desc_bert_path_or_handler=val_set.desc_bert_h5,
        sub_bert_path_or_handler=val_set.sub_bert_h5, 
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        video_feat_path_or_handler=val_set.vid_feat_h5,
        clip_length=opt.clip_length,
        ctx_mode=opt.ctx_mode,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat,
        moment_ctx_len=opt.moment_ctx_len,)
    test_loader = DataLoader(test_set, collate_fn=lambda batch: collate_fn(batch, task='eval'), batch_size=opt.bsz_eval, num_workers=opt.num_workers, shuffle=False, drop_last=False, pin_memory=opt.pin_memory)
    
    test_gt = test_set.ground_truth
    

    return val_loader, val_gt, test_loader, test_gt, video_lens