import h5py
import math
import numpy as np
from collections import defaultdict
import os
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from utils.general.basic_utils import load_json

"""_summary_

This script defines the dataset classes for the RVMR dataset using CLIP features.
"""

class CLIPFeatDataset(Dataset):
    def __init__(self, architecutre, vid_feat_path, query_feat_path):
        """
        A base class for the RVMR dataset using CLIP features.
        Args:
            architecutre (str): the architecture of the video corpus features, video frames are sampled at 1FPS.
            vid_feat_path (str): the path to the video corpus features.
            query_feat_path (str): the path to the query features.
        """
        
        # load video corpus features
        self.architecutre = architecutre
        self.vid_feat_h5 = h5py.File(os.path.join(vid_feat_path, f"{architecutre}.hdf5"), "r")
        self.query_feat_h5 = h5py.File(os.path.join(query_feat_path, f"{architecutre}.hdf5"), "r") if query_feat_path is not None else None
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def _get_query_feat(self, query_id):
        """
        Get the query feature by query_id.
        Args:
            query_id (str): the query id.
        Returns:
            query_feat (np.ndarray): the query feature. dim: (d)
        """
        if self.query_feat_h5 is None:
            return None
        
        query_feat = self.query_feat_h5[query_id][:]
        return query_feat
    
    def _prepare_frame_feat_seq(self, vid_name, seg_idx, seg_dur):
        """
        Prepare the frame feature sequence for a video segment.
        Args:
            vid_name (str): the video name.
            seg_idx (int): the segment index.
            seg_dur (int): the segment duration.
        Returns:
            frame_feat_seq (np.ndarray): the frame feature sequence.
        """

        vid_feat = self.vid_feat_h5[vid_name][:]
        vid_len, _ = vid_feat.shape
        
        start_idx = seg_idx * seg_dur
        end_idx = min((seg_idx + 1) * seg_dur, vid_len)
        frame_feat_seq = vid_feat[start_idx:end_idx]
        
        if len(frame_feat_seq) < seg_dur:
            pad_len = seg_dur - len(frame_feat_seq)
            padded_frame_feat_seq = [frame_feat_seq] + [frame_feat_seq[-1:]] * pad_len
            padded_frame_feat_seq = np.concatenate(padded_frame_feat_seq, axis=0)
        else:
            padded_frame_feat_seq = frame_feat_seq
        
        frame_mask_seq = np.ones((padded_frame_feat_seq.shape[0]), dtype=np.float32)
        
        return padded_frame_feat_seq, frame_mask_seq

class TrainSegDataset(CLIPFeatDataset):
    
    def __init__(self, architecutre, vid_feat_path, query_feat_path, train_anno_path, seg_dur, topk_pos=20, pos_thresh=None):
        """
        Args:
            architecutre (str): the architecture of the video corpus features, video frames are sampled at 1FPS.
            vid_feat_path (str): the path to the video corpus features.
            query_feat_path (str): the path to the query features.
            train_anno_path (str): the path to the training annotations. Each term: {"query_id": qid, "query": query, 
                                                                                    "relevant_segment":[{"video_name": vid_name, 
                                                                                                        "segment_idx": s_idx, 
                                                                                                        "duration": vid_dur, 
                                                                                                        "score": relevance_score}]}
            seg_dur (int): the segment duration, should be multiple of 1s.
            topk_pos (int): the number of positive moments for each query.
        """
        super().__init__(architecutre, vid_feat_path, query_feat_path)
        
        assert type(seg_dur) == int and seg_dur > 0, f"Invalid seg_dur: {seg_dur}"
        self.seg_dur = seg_dur

        self.ori_train_annos = load_json(os.path.join(train_anno_path, f"train_top{topk_pos:02d}_seg_{seg_dur}s.json"))
        logger.info(f"Load training annotations from train_top{topk_pos:02d}_seg_{seg_dur}s.json")
        
        self.pos_thresh = pos_thresh
        if pos_thresh is not None:
            logger.info(f"Filter out the segments with score lower than {pos_thresh}")
        
        self.train_annos = self.flat_annotations(self.ori_train_annos)
        self.score_dict = self.build_score_dict(self.train_annos)
        
    def flat_annotations(self, annotations):
        """
        Flat the hierarchical annotations to a list of samples. Each sample contains the query and relevant segment info.
        Returns:
            new_annotations (List[Dict]): The list of samples, each sample contains the query and relevant segment info.
        """
        new_annotations = []
        for q_anno in annotations:
            query = q_anno["query"]
            query_id = q_anno["query_id"]
            relevant_segments = q_anno["relevant_segment"]
            for segment in relevant_segments:
                if self.pos_thresh is not None and segment["score"] < self.pos_thresh: # we filter out the segments with score lower than pos_thresh if specified
                    continue
                new_anno = {"query": query, "query_id": query_id}
                new_anno.update(segment)
                new_annotations.append(new_anno)
        return new_annotations
    
    def build_score_dict(self, train_annos):
        """
        Build the score dictionary. {query_id: {video_name_segment_idx: score}}
        Args:
            train_annos (List[Dict]): the training annotations.
        Returns:
            score_dict (Dict): the score dictionary.
        """
        score_dict = defaultdict(dict)
        for anno in train_annos:
            query_id = anno["query_id"]
            video_name = anno["video_name"]
            segment_idx = anno["segment_idx"]
            score = anno["score"]
            score_dict[query_id][f"{video_name}_{segment_idx}"] = score
        return score_dict
    
    def __len__(self):
        return len(self.train_annos)
    
    def __getitem__(self, idx):
        anno = self.train_annos[idx]
        video_name = anno["video_name"]
        segment_idx = anno["segment_idx"]
        query_id = anno["query_id"]
        #segment_name = video_name + "_" + str(segment_idx)
        
        # prepare the frame feature sequence
        frame_feat_seq, frame_mask_seq = self._prepare_frame_feat_seq(video_name, segment_idx, self.seg_dur)
        query_feat = self._get_query_feat(str(query_id))
        
        model_inputs = {"frame_feat_seq": frame_feat_seq,
                        "query_feat": query_feat,
                        "query": anno["query"],
                        "query_id": anno["query_id"],
                        "video_name": anno["video_name"],
                        "segment_idx": anno["segment_idx"],
                        "score": anno["score"],
                        "frame_mask_seq": frame_mask_seq}
        
        return model_inputs

class CorpSegDataset(CLIPFeatDataset):
    def __init__(self, architecutre, vid_feat_path, corpus_anno_path, seg_dur):
        """
        Args:
            architecutre (str): the architecture of the video corpus features, video frames are sampled at 1FPS.
            vid_feat_path (str): the path to the video corpus features.
            corpus_anno_path (str): the path to the corpus annotations. Each term: {vid_name: seg_num}
            seg_dur (int): the segment duration, should be multiple of 1s.
        """
        super().__init__(architecutre, vid_feat_path, query_feat_path=None)
        assert type(seg_dur) == int and seg_dur > 0, f"Invalid seg_dur: {seg_dur}"
        
        self.seg_dur = seg_dur
        self.corpus_anno = load_json(os.path.join(corpus_anno_path, f"vidlen_seg_{seg_dur}s.json"))
        self.corpus_anno = sorted(self.corpus_anno.items(), key=lambda x: x[0])
        
        self.corpus_segment_list = []
        for vid_name, seg_num in self.corpus_anno:
            for seg_idx in range(seg_num):
                self.corpus_segment_list.append(f"{vid_name}_{seg_idx}")
        
    
    def __len__(self):
        return len(self.corpus_anno)
    
    def __getitem__(self, idx):
        """
        Returns:
            meta (dict): meta information about the video.
                vid_name (str): the video name.
                vid_feat (np.ndarray): the video feature.
                duration (float): the duration of the video.
        """
        vid_name, num_segment = self.corpus_anno[idx]
        vid_feat = self.vid_feat_h5[vid_name][:]
        
        vid_len, dim = vid_feat.shape
        assert vid_len <= num_segment * self.seg_dur and vid_len > (num_segment-1) * self.seg_dur, f"vid_len: {vid_len}, num_segment: {num_segment}, seg_dur: {self.seg_dur}"
        
        # plan A
        if vid_len % self.seg_dur != 0:
            pad_len = self.seg_dur * num_segment - vid_len
            pad_item = [vid_feat] + [vid_feat[-1:]] * pad_len
            vid_feat = np.concatenate(pad_item, axis=0)
            vid_len = vid_feat.shape[0]
        
        seg_frame_feats = vid_feat.reshape(num_segment, self.seg_dur, dim)
        
        meta = {
            "video_name": vid_name,
            "num_segment": num_segment,
            "seg_frame_feats": seg_frame_feats,
            "seg_frame_mask": np.ones((num_segment, self.seg_dur), dtype=np.float32)
        }

        return meta

    
class TestSegDataset(Dataset):
    def __init__(self, architecutre, query_feat_path, test_anno_path, seg_dur, eval_split_name="val"):
        self.eval_split_name = eval_split_name

        # build annotations
        assert type(seg_dur) == int and seg_dur > 0, f"Invalid seg_dur: {seg_dur}"
        self.seg_dur = seg_dur
        self.test_seg_anno = load_json(os.path.join(test_anno_path, f"{eval_split_name}_seg_{seg_dur}s.json"))
        self.query_feat_h5 = h5py.File(os.path.join(query_feat_path, f"{architecutre}.hdf5"), "r")
    
    def __len__(self):
        return len(self.test_seg_anno)
    
    def _get_query_feat(self, query_id):
        """
        Get the query feature by query_id.
        Args:
            query_id (str): the query id.
        Returns:
            query_feat (np.ndarray): the query feature. dim: (d)
        """
        query_feat = self.query_feat_h5[query_id][:]
        return query_feat
    
    def __getitem__(self, idx):
        anno = self.test_seg_anno[idx]
        query_id = anno["query_id"]
        query_feat = self._get_query_feat(str(query_id))
        model_inputs = {
            "query_id": anno["query_id"],
            "query": anno["query"],
            "relevant_segment": anno["relevant_segment"],
            "query_feat": query_feat
        }
        return model_inputs

def collate_fn_train(score_dict, batch):
    """
    Args:
        score (Dict{Dict}): the score dictionary. {"query_id": {"video_seg_name": score},...}
        batch (List[Dict]): a list of model inputs.
    Returns:
        batch_input (Dict): the model inputs.
    """
    bs = len(batch)
    frame_feat_seq = [item["frame_feat_seq"] for item in batch]
    frame_mask_seq = [item["frame_mask_seq"] for item in batch]
    query_feat = [item["query_feat"] for item in batch]
    query = [item["query"] for item in batch]
    query_id = [item["query_id"] for item in batch]
    video_name = [item["video_name"] for item in batch]
    segment_idx = [item["segment_idx"] for item in batch]
    score = [item["score"] for item in batch]
    
    segment_name = [f"{video_name[i]}_{segment_idx[i]}" for i in range(bs)]
    
    # build the score matrix for infonce loss, shape: [num_query, num_segment], i.e. [bs, bs]
    batch_score_tensors = [torch.tensor([score_dict[q_id][seg] if seg in score_dict[q_id] else 0.0 for seg in segment_name]) for q_id in query_id]
    batch_score_matrix = torch.stack(batch_score_tensors)
    
    batch_input = {
                    "frame_feat_seq": torch.from_numpy(np.stack(frame_feat_seq)),
                    "frame_mask_seq": torch.from_numpy(np.stack(frame_mask_seq)),
                    "query_feat": torch.from_numpy(np.stack(query_feat)),
                    "query": query,
                    "query_id": query_id,
                    "video_name": video_name,
                    "segment_idx": segment_idx,
                    "segment_name": segment_name,
                    "score": score,
                    "batch_score_matrix": batch_score_matrix
                    }
    return batch_input

def collate_fn_corp(batch):
    """
    Args:
        batch (List[Dict]): a list of model inputs.
    Returns:
        batch_input (Dict): the model inputs.
    """
    seg_frame_feats = [torch.from_numpy(item["seg_frame_feats"]) for item in batch]
    seg_frame_mask = [torch.from_numpy(item["seg_frame_mask"]) for item in batch]
    video_name = [item["video_name"] for item in batch]
    num_segment = [item["num_segment"] for item in batch]
    
    batch_input = {
        "seg_frame_feats": seg_frame_feats,
        "seg_frame_mask": seg_frame_mask,
        "video_name": video_name,
        "num_segment": num_segment
    }
    
    return batch_input

def collate_fn_test(batch):
    """
    Args:
        batch (List[Dict]): a list of model inputs.
    Returns:
        batch_input (Dict): the model inputs.
    """
    query_feat = [item["query_feat"] for item in batch]
    query = [item["query"] for item in batch]
    query_id = [item["query_id"] for item in batch]
    relevant_segment = [item["relevant_segment"] for item in batch]
                
    batch_input = {
        "query_feat": torch.from_numpy(np.stack(query_feat)),
        "query": query,
        "query_id": query_id,
        "relevant_segment": relevant_segment
    }

    return batch_input

def build_train_dataloader(cfg):
    dataset = TrainSegDataset(cfg.MODEL.CLIP.TYPE, cfg.DATASETS.VID_FEAT_PATH, cfg.DATASETS.QUERY_FEAT_PATH, cfg.DATASETS.TRAIN_ANNO_PATH, cfg.DATASETS.SEG_DURATION, cfg.DATASETS.TRAIN_TOPK_POS, cfg.DATASETS.TRAIN_POS_THRESH)
    score_dict = dataset.score_dict
    new_collate_fn_train = lambda batch: collate_fn_train(score_dict, batch)
    dataloader = DataLoader(dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, collate_fn=new_collate_fn_train, num_workers=cfg.DATALOADER.NUM_WORKERS)
    return dataloader

def build_corp_dataloader(cfg):
    dataset = CorpSegDataset(cfg.MODEL.CLIP.TYPE, cfg.DATASETS.VID_FEAT_PATH, cfg.DATASETS.CORPUS_ANNO_PATH, cfg.DATASETS.SEG_DURATION)
    dataloader = DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, collate_fn=collate_fn_corp, num_workers=cfg.DATALOADER.NUM_WORKERS)
    corpus_seg_list = dataset.corpus_segment_list
    return dataloader, corpus_seg_list

def build_test_dataloader(cfg):
    val_dataset = TestSegDataset(cfg.MODEL.CLIP.TYPE, cfg.DATASETS.QUERY_FEAT_PATH, cfg.DATASETS.VAL_ANNO_PATH, cfg.DATASETS.SEG_DURATION, eval_split_name="val")
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, collate_fn=collate_fn_test, num_workers=cfg.DATALOADER.NUM_WORKERS)
    test_dataset = TestSegDataset(cfg.MODEL.CLIP.TYPE, cfg.DATASETS.QUERY_FEAT_PATH, cfg.DATASETS.TEST_ANNO_PATH, cfg.DATASETS.SEG_DURATION, eval_split_name="test")
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, collate_fn=collate_fn_test, num_workers=cfg.DATALOADER.NUM_WORKERS)
    return val_dataloader, test_dataloader