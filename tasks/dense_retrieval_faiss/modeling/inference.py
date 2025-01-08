import torch
from loguru import logger

from transformers import CLIPProcessor, CLIPModel, pipeline

class ModelInferenceCLIP():
    
    def __init__(self, architecutre, device):
        model_source_dict = {
            "CLIP-ViT-B-32": "openai/clip-vit-base-patch32",
            "CLIP-ViT-B-16": "openai/clip-vit-base-patch16",
            "CLIP-ViT-L-14": "openai/clip-vit-large-patch14",
            "CLIP-ViT-L-14-336": "openai/clip-vit-large-patch14-336",
        }
        
        logger.info(f"#> Loading {architecutre} model as text encoder.")
        model_source = model_source_dict[architecutre]
        self.model = CLIPModel.from_pretrained(model_source)
        self.model.to(device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_source)
        
        self.device = device

    def query(self, text_input, to_cpu=False):
        with torch.no_grad():
            Q = self.model.get_text_features(**text_input) # [M,D]
            return Q.cpu() if to_cpu else Q

    def pre_processing(self, query):
        # we set "truncation=True" to avoid too long input sequence
        text_input = self.processor(text=query, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        return text_input
    
    def queryFromText(self, queries, bsize=None, to_cpu=False):
        """
        Args:
            queries (list or tuple): The list of queries.
            bsize (int): The batch size of queries.
            to_cpu (bool): Whether to move the tensor to CPU.
        """
        logger.info(f"#> Generating query embeddings from {len(queries)} queries.")
        if bsize:
            def split_list(lst, n):
                lst_len = len(lst)
                return [lst[i:min(i + n, lst_len)] for i in range(0, lst_len, n)]
            batch_query = split_list(queries, bsize)
            batches = [self.pre_processing(query) for query in batch_query]
            batches = [self.query(text_input, to_cpu=to_cpu) for text_input in batches]
            return torch.cat(batches) # [M,D]

        text_input = self.pre_processing(queries)
        return self.query(text_input, to_cpu=to_cpu) # [M,D]