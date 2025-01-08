import torch
import h5py
import os
from loguru import logger
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from models.CLIP4Video.frozen_clip_for_video import FrozenCLIP4Video
from utils.general.basic_utils import load_ckpt

class ModelInferenceFrozenCLIP4Video:
    def __init__(self, cfg, checkpoint_path, device):
        
        model_source_dict = {
            "CLIP-ViT-B-32": "openai/clip-vit-base-patch32",
            "CLIP-ViT-B-16": "openai/clip-vit-base-patch16",
            "CLIP-ViT-L-14": "openai/clip-vit-large-patch14",
            "CLIP-ViT-L-14-336": "openai/clip-vit-large-patch14-336",
        }
        
        model_source = model_source_dict[cfg.MODEL.CLIP.TYPE]
        self.clip_model = CLIPModel.from_pretrained(model_source)
        self.clip_model.to(device)
        self.clip_model.eval()
        
        self.model = FrozenCLIP4Video(cfg)
        self.model, _, _ = load_ckpt(checkpoint_path, self.model)

        self.model.to(device)
        self.model.eval()
        
        self.processor = CLIPProcessor.from_pretrained(model_source)
        self.text_processing = lambda query: self.processor(text=query, return_tensors="pt", padding=True, truncation=True)
        self.device = device
    
    @torch.no_grad()
    def query2emb(self, text_input):
        """
        Args:
            text_input (dict): The text input dictionary, generated from CLIPProcessor.
        Returns:
            q_emb (torch.Tensor): The query embeddings from the model, shape of [bs, d].
        """
        self.clip_model.eval()
        self.model.eval()
        text_input = text_input.to(self.device)
        q_emb = self.clip_model.get_text_features(**text_input) # [bs, d]
        q_emb = self.model.get_query_feature(q_emb)
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
        self.clip_model.eval()
        self.model.eval()
        logger.info(f"#> Generating query embeddings from {len(queries)} queries.")
        if bsize:
            def split_list(lst, n):
                lst_len = len(lst)
                return [lst[i:min(i + n, lst_len)] for i in range(0, lst_len, n)]
            batch_query = split_list(queries, bsize)
            batches = [self.text_processing(query) for query in batch_query]
            batch_qemb = [self.query2emb(text_input) for text_input in batches]
            q_emb = torch.cat(batch_qemb, dim=0) # [num_query,d]
        else:
            text_input = self.text_processing(queries)
            q_emb = self.query2emb(text_input) # [num_query,d]
        
        logger.info(f"#> Query embeddings generated, shape: {q_emb.shape}.")
        return q_emb.cpu() if to_cpu else q_emb