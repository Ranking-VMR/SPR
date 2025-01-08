from transformers import CLIPModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from collections import OrderedDict
from loguru import logger

class InfoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, similarity_matrix):
        """
        Compute the InfoNCE Loss for each rows of the similarity matrix. Here, we assume only the diagonal elements are the positive samples.
        Args:
            similarity_matrix (torch.Tensor): The similarity matrix, shape: [num_row, num_column].
        Returns:
            infonce_loss (torch.Tensor): The average InfoNCE loss of all rows.
        """
        logpt = F.log_softmax(similarity_matrix, dim=-1)
        logpt = torch.diag(logpt)
        infonce_loss = -logpt.mean()
        return infonce_loss

class MILNCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, similarity_matrix, gt_score_matrix):
        """
        Compute the MILNCE Loss for each rows of the similarity matrix. Here, we assume there are multiple positive examples for each row.
        Args:
            similarity_matrix (torch.Tensor): The similarity matrix, shape: [num_row, num_column].
            gt_score_matrix (torch.Tensor): The ground-truth score matrix, shape: [num_row, num_column].
        Returns:
            mil_nce_loss (torch.Tensor): The average MIL-NCE loss of all rows.
        """
        pos_mask = gt_score_matrix > 0
        pos_mask = pos_mask.to(dtype=similarity_matrix.dtype)
        
        prob_all = F.softmax(similarity_matrix, dim=-1)
        prob_pos = torch.sum(pos_mask * prob_all, dim=-1)
        mil_nce_loss = -torch.log(prob_pos).mean()
        
        return mil_nce_loss

class WeightedMILNCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, similarity_matrix, gt_score_matrix):
        """
        Compute the weighted MILNCE Loss for each rows of the similarity matrix. Here, we assume there are multiple positive examples for each row, and they have different weights.
        Args:
            similarity_matrix (torch.Tensor): The similarity matrix, shape: [num_row, num_column].
            gt_score_matrix (torch.Tensor): The ground-truth score matrix (serves as weight matrix for all samples), shape: [num_row, num_column].
        Returns:
            weighted_mil_nce_loss (torch.Tensor): The average weighted MILNCE loss of all rows.
        """
        # Plan A: normalize the positive weight matrix
        #gt_score_matrix = F.normalize(gt_score_matrix, p=1, dim=-1)

        # Plan B: or we set the max of each row to 1
        pos_weight = gt_score_matrix / gt_score_matrix.max(dim=-1, keepdim=True)[0]
        
        prob_all = F.softmax(similarity_matrix, dim=-1)
        prob_pos = torch.sum(pos_weight * prob_all, dim=-1)
        weighted_mil_nce_loss = -torch.log(prob_pos).mean()
        
        return weighted_mil_nce_loss

class MILCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, similarity_matrix, gt_score_matrix):
        """
        Compute the MILCELoss Loss for each rows of the similarity matrix. Here, we assume there are multiple positive examples for each row.
        We set the normalized weights of positive examples as the gt probability distribution.
        Args:
            similarity_matrix (torch.Tensor): The similarity matrix, shape: [num_row, num_column].
            gt_score_matrix (torch.Tensor): The ground-truth score matrix (serves as weight matrix for all samples), shape: [num_row, num_column].
        Returns:
            weighted_ce_loss (torch.Tensor): The average MILCELoss loss of all rows.
        """
        # Plan A: L1 normalize the positive weight matrix
        gt_prob = F.normalize(gt_score_matrix, p=1, dim=-1)
        weighted_ce_loss = F.cross_entropy(similarity_matrix, gt_prob)
        
        return weighted_ce_loss

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]

class FrozenCLIP4Video(nn.Module):
    model_source_dict = {
        "CLIP-ViT-B-32": "openai/clip-vit-base-patch32",
        "CLIP-ViT-B-16": "openai/clip-vit-base-patch16",
        "CLIP-ViT-L-14": "openai/clip-vit-large-patch14",
        "CLIP-ViT-L-14-336": "openai/clip-vit-large-patch14-336",
    }
    model_dim_dict = {
        "CLIP-ViT-B-32": 512,
        "CLIP-ViT-B-16": 512,
        "CLIP-ViT-L-14": 768,
        "CLIP-ViT-L-14-336": 768,
    }
    loss_func_dict = {
        "InfoNCELoss": InfoNCELoss,
        "MILNCELoss": MILNCELoss,
        "WeightedMILNCELoss": WeightedMILNCELoss,
        "MILCELoss": MILCELoss,
    }
    def __init__(self, cfg):
        super().__init__()
        
        # set parameters from CLIP Pretrained model
        logger.info(f"#> Using CLIP model: {cfg.MODEL.CLIP.TYPE}")
        model_source = self.model_source_dict[cfg.MODEL.CLIP.TYPE]
        clip_output_dim = self.model_dim_dict[cfg.MODEL.CLIP.TYPE]
        clip_model = CLIPModel.from_pretrained(model_source)
        self.pt_clip_logit_scale = clip_model.logit_scale
        
        # build temporal modeling module
        self.temporal_modeling = cfg.MODEL.TEMP_MODEL.TYPE
        logger.info(f"#> Using temporal modeling: {self.temporal_modeling}")
        assert self.temporal_modeling in ["meanP", "seqLSTM", "seqTransf"], f"Invalid sim_header: {self.temporal_modeling}"

        if self.temporal_modeling in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cfg.MODEL.TEMP_MODEL.MAX_POS_EMB, clip_output_dim)
        if self.temporal_modeling == "seqTransf":
            transformer_heads = clip_output_dim // 64
            self.seq_transf = Transformer(width=clip_output_dim, layers=cfg.MODEL.TEMP_MODEL.NUM_LAYER,
                                                   heads=transformer_heads)
            logger.info(f"#> Using Transformer Encoder with {cfg.MODEL.TEMP_MODEL.NUM_LAYER} layers and {transformer_heads} heads.")
        elif self.temporal_modeling == "seqLSTM":
            self.seq_lstm = nn.LSTM(input_size=clip_output_dim, hidden_size=clip_output_dim,
                                       batch_first=True, bidirectional=False, num_layers=1)  
            logger.info(f"#> Using LSTM with 1 layers.")
        
        # build dual feature processor module if specified
        if cfg.MODEL.QFEAT_MODEL.PROJ:
            self.qfeat_proj = nn.Linear(clip_output_dim, clip_output_dim)
            logger.info(f"#> Using projection layer for query features.")
        else:
            self.qfeat_proj = nn.Identity()
        
        if cfg.MODEL.TEMP_MODEL.PROJ:
            self.temp_proj = nn.Linear(clip_output_dim, clip_output_dim)
            logger.info(f"#> Using projection layer for segment features.")
        else:
            self.temp_proj = nn.Identity()

        # build loss function
        self.loss_type = cfg.MODEL.LOSS.TYPE
        select_loss_cls = self.loss_func_dict.get(self.loss_type)
        self.loss_func = select_loss_cls()
    
    def forward(self, text_feats, visual_feats, visual_mask, batch_score_matrix):
        """
        Args:
            text_feats (torch.Tensor): The text features, shape: [num_query, d].
            visual_feats (torch.Tensor): The visual features, shape: [num_segment, num_frames, d].
            visual_mask (torch.Tensor): The mask for the video frames, shape: [num_segment, num_frames].
            batch_score_matrix (torch.Tensor): The batch similarity matrix, shape: [num_query, num_segment].
        Returns:
            total_loss (torch.Tensor): The total contrastive loss.
        """
        if self.training:
            logit_matrix = self.generate_sim_matrix(text_feats, visual_feats, visual_mask) # [num_query, num_segment]
            
            if self.loss_type in ["WeightedMILNCELoss", "MILNCELoss", "MILCELoss"]:
                loss_text_to_video = self.loss_func(logit_matrix, batch_score_matrix)
                loss_video_to_text = self.loss_func(logit_matrix.T, batch_score_matrix.T)
            elif self.loss_type == "InfoNCELoss":
                loss_text_to_video = self.loss_func(logit_matrix)
                loss_video_to_text = self.loss_func(logit_matrix.T)
            
            total_loss = (loss_text_to_video + loss_video_to_text) / 2

            return total_loss

        else:
            return None # Set to None to check if the forward pass is only run during the training phase.
        
    def _mean_pooling_for_similarity_visual(self, visual_feats, visual_mask):
        """
        Conduct mean-pooling for the visual features to get the video-level visual features.
        Args:
            visual_feats (torch.Tensor): The visual features, shape: [bs, num_frames, d].
            visual_mask (torch.Tensor): The mask for the video frames, shape: [bs, num_frames].
        Returns:
            video_out (torch.Tensor): The video-level visual features, shape: [bs, d].
        """
        visual_feats= F.normalize(visual_feats, dim=-1)
        visual_mask_un = visual_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_feats = visual_feats * visual_mask_un
        visual_mask_un_sum = torch.sum(visual_mask_un, dim=1, dtype=torch.float)
        visual_mask_un_sum[visual_mask_un_sum == 0.] = 1.  # avoid division by zero
        video_out = torch.sum(visual_feats, dim=1) / visual_mask_un_sum
        return video_out

    def get_query_feature(self, text_feats):
        
        text_feats = self.qfeat_proj(text_feats)
        return text_feats
    
    def get_segment_feature(self, visual_feats, visual_mask):
        """
        Compute the similarity matrix between the text and visual features.
        Args:
            visual_feats (torch.Tensor): The visual features, shape: [num_segment, num_frames, d].
            visual_mask (torch.Tensor): The mask for the video frames, shape: [num_segment, num_frames].
        Returns:
            seg_feats (torch.Tensor): The segment-level visual features, shape: [num_segment, d].
        """
        
        if self.temporal_modeling == "meanP":
            pass
        elif self.temporal_modeling == "seqLSTM":
            # Sequential type: LSTM
            visual_feats_original = visual_feats
            visual_feats = pack_padded_sequence(visual_feats, torch.sum(visual_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_feats, _ = self.seq_lstm(visual_feats)
            if self.training: self.seq_lstm.flatten_parameters()
            visual_feats, _ = pad_packed_sequence(visual_feats, batch_first=True)
            visual_feats = torch.cat((visual_feats, visual_feats_original[:, visual_feats.size(1):, ...].contiguous()), dim=1)
            visual_feats = visual_feats + visual_feats_original
        elif self.temporal_modeling == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_feats_original = visual_feats
            seq_length = visual_feats.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_feats.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_feats.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_feats = visual_feats + frame_position_embeddings
            
            extended_visual_mask = (1.0 - visual_mask.unsqueeze(1)) * -1000000.0
            extended_visual_mask = extended_visual_mask.expand(-1, visual_mask.size(1), -1)
            visual_feats = visual_feats.permute(1, 0, 2)  # NLD -> LND
            
            visual_feats = self.seq_transf(visual_feats, extended_visual_mask)
            visual_feats = visual_feats.permute(1, 0, 2)  # LND -> NLD
            visual_feats = visual_feats + visual_feats_original
        
        #visual_mask = torch.ones(visual_feats.size(0), visual_feats.size(1), dtype=torch.float).to(visual_feats.device)
        seg_feats = self._mean_pooling_for_similarity_visual(visual_feats, visual_mask)
        
        seg_feats = self.temp_proj(seg_feats)
        
        return seg_feats

    def compute_similarity_matrix(self, text_feats, visual_feats):
        """
        Compute the similarity matrix between the query and segment features.
        Args:
            text_feats (torch.Tensor): The text features, shape: [num_query, d].
            visual_feats (torch.Tensor): The segment features, shape: [num_segment, d].
        Returns:
            logit_matrix (torch.Tensor): The logit-scaled similarity matrix, shape: [num_query, num_segment].
        """
        
        seg_feats= F.normalize(visual_feats, dim=-1)
        text_feats= F.normalize(text_feats, dim=-1)
        
        logit_scale = self.pt_clip_logit_scale.exp()
        logit_matrix = logit_scale * torch.einsum('bd,nd->bn', text_feats, seg_feats)
        
        return logit_matrix

    def generate_sim_matrix(self, text_feats, visual_feats, visual_mask):
        """
        Generate the similarity matrix between the input text and visual features.
        Args:
            text_feats (torch.Tensor): The text features, shape: [num_query, d].
            visual_feats (torch.Tensor): The visual features, shape: [num_segment, num_frames, d].
            visual_mask (torch.Tensor): The mask for the video frames, shape: [num_segment, num_frames].
        Returns:
            logit_matrix (torch.Tensor): The logit-scaled similarity matrix, shape: [num_query, num_segment].
        """
        
        text_feats = self.get_query_feature(text_feats)
        seg_feats = self.get_segment_feature(visual_feats, visual_mask)
        
        logit_matrix = self.compute_similarity_matrix(text_feats, seg_feats)
        
        return logit_matrix
    