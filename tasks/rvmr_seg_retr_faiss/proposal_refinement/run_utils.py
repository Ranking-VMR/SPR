import torch
from models.ReLoCLNet.modeling import ReLoCLNet, ReLoCLNet_RVMR, ReLoCLNet_Hard_Neg, ReLoCLNet_RVMR_Hard_Neg, ReLoCLNet_RVMR_Weighted
from tasks.rvmr_seg_retr_faiss.proposal_refinement.optimization import BertAdam
import numpy as np
import copy
from loguru import logger

def count_parameters(model, verbose=True):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        logger.info("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    return n_all, n_trainable

def prepare_model(opt):
    logger.info(f"Prepare model: {opt.model_type}")
    if opt.model_type == "reloclnet":
        model = ReLoCLNet(opt)
    elif opt.model_type == "reloclnet_rvmr":
        model = ReLoCLNet_RVMR(opt)
    elif opt.model_type == "reloclnet_hard_neg":
        model = ReLoCLNet_Hard_Neg(opt)
    elif opt.model_type == "reloclnet_rvmr_hard_neg":
        model = ReLoCLNet_RVMR_Hard_Neg(opt)
    elif opt.model_type == "reloclnet_rvmr_weighted":
        model = ReLoCLNet_RVMR_Weighted(opt)
    else:
        raise ValueError(f"Invalid model_type: {opt.model_type}")
    count_parameters(model)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
    return model

def load_model(ckpt_path, device, model, optimizer=None):
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    start_epoch = checkpoint['epoch']

    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loading model from {ckpt_path} at epoch {start_epoch}")

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loading optimizer from {ckpt_path} at epoch {start_epoch}")
        
    return model, optimizer, start_epoch,

def prepare_optimizer(model, opt, total_train_steps):
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=opt.lr, weight_decay=opt.wd, warmup=opt.lr_warmup_proportion,
                         t_total=total_train_steps, schedule="warmup_linear")
    return optimizer

def save_model(model, optimizer, epoch, path):
    data = {
            'epoch': epoch,
            'model_cfg': model.config,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
    torch.save(data, path)
    logger.info(f"Save checkpoint at {path}")
    logger.info("")
    
    
def topk_3d(tensor, k):
    """
    Find the top k values and their corresponding indices in a 3D tensor.

    Args:
    tensor (torch.Tensor): A 3D tensor of shape [v, m, n].
    k (int): The number of top elements to find.

    Returns:
    topk_values (torch.Tensor): The top k values.
    indices_3d (torch.Tensor): The indices of the top k values in the format [i, j, k].
    """
    def unravel_index(indices, shape):
        
        shape = torch.tensor(shape)
        
        strides = torch.cumprod(shape.flip(0), dim=0).flip(0)
        strides = torch.cat((strides[1:], torch.tensor([1])))

        result = []
        for stride in strides:
            result.append(indices // stride)
            indices = indices % stride
        
        return tuple(result)
    
    # Step 1: Flatten the tensor to 1D
    flat_tensor = tensor.view(-1)

    # Step 2: Find the top k values and their indices in the flattened tensor
    topk_values, topk_indices = torch.topk(flat_tensor, k)

    # Step 3: Convert the flat indices back to the original 3D tensor's indices
    v, m, n = tensor.shape
    indices_3d = torch.stack(unravel_index(topk_indices, (v, m, n)), dim=1)

    return topk_values, indices_3d


def generate_min_max_length_mask(array_shape, min_l, max_l):
    """ The last two dimension denotes matrix of upper-triangle with upper-right corner masked,
    below is the case for 4x4.
    [[0, 1, 1, 0],
     [0, 0, 1, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]
    Args:
        array_shape: np.shape??? The last two dimensions should be the same
        min_l: int, minimum length of predicted span
        max_l: int, maximum length of predicted span
    Returns:
    """
    single_dims = (1, ) * (len(array_shape) - 2)
    mask_shape = single_dims + array_shape[-2:]
    extra_length_mask_array = np.ones(mask_shape, dtype=np.float32)  # (1, ..., 1, L, L)
    mask_triu = np.triu(extra_length_mask_array, k=min_l)
    mask_triu_reversed = 1 - np.triu(extra_length_mask_array, k=max_l)
    final_prob_mask = mask_triu * mask_triu_reversed
    return final_prob_mask  # with valid bit to be 1


def extract_topk_elements(query_scores, start_probs, end_probs, video_names, k):

    # Step 1: Find the top k values and their indices in query_scores
    topk_values, topk_indices = torch.topk(query_scores, k)

    # Step 2: Use these indices to select the corresponding elements from start_probs and end_probs
    selected_start_probs = torch.stack([start_probs[i, indices] for i, indices in enumerate(topk_indices)], dim=0)
    selected_end_probs = torch.stack([end_probs[i, indices] for i, indices in enumerate(topk_indices)], dim=0)
    
    selected_video_name = []
    for i in range(topk_indices.shape[0]):
        vn = copy.deepcopy(video_names)
        tmp = [vn[idx] for idx in topk_indices[i]]
        selected_video_name.append(tmp)

    return topk_values, selected_start_probs, selected_end_probs, selected_video_name

def logger_ndcg_iou(val_ndcg_iou, suffix):
    logger_dict = {}
    for K, vs in val_ndcg_iou.items():
        for T, v in vs.items():
            logger.info(f"{suffix} NDCG@{K}, IoU={T}: {v:.6f}")
            logger_dict[f"{suffix}_NDCG@{K}_IoU_{T}"] = v
    return logger_dict

def logger_video_recall(val_video_recall, suffix):
    logger_dict = {}
    for K, v in val_video_recall.items():
            logger.info(f"{suffix} Video Recall@{K}: {v:.6f}")
            logger_dict[f"{suffix}_Video_Recall@{K}"] = v
    return logger_dict