import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from loguru import logger
import argparse
import json
import os
import time

from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup


from models.CLIP4Video.frozen_clip_for_video import FrozenCLIP4Video
from datasets.rvmr.clip_feat_seg_dataset import build_train_dataloader, build_test_dataloader, build_corp_dataloader
from tasks.rvmr_seg_retr_faiss.evaluation.evaluation_lib import compute_corpus_feature, evaluate_recall
from utils.general.config import load_config
from utils.general.setting_utils import init_rand
from utils.general.basic_utils import save_ckpt, load_ckpt, remove_old_files, form_data_table
from utils.general.loggers import LossTracker, WandbLogger

def logger_recall(all_topk_recall, suffix):
    for recall_name, topk_recall in all_topk_recall.items():
        print(f"For {recall_name}:")
        for topk, recall in topk_recall.items():
            print(f"{suffix} recall top-{topk}: {recall:.4f}")

def get_args():
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument("--cfg_dir", type=str, help="Path to the configuration file")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use wandb for logging")

    # Model and Training parameters
    #parser.add_argument('--warmup_proportion', type=float, default=0.01, help="Proportion of training for warm-up")

    # System settings
    parser.add_argument('--seed', type=int, default=2024, help="Random seed for reproducibility")

    args = parser.parse_args()
    #args.results_path = os.path.join(args.results_path, args.exp_id)
    #os.makedirs(args.results_path, exist_ok=True)
    
    return args

def prep_optimizer_scheduler(cfg, model, total_train_steps):
    if hasattr(model, 'module'):
        model = model.module
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]
    
    # we use prefix "pt_clip" to indicate the parameters that are inherited from pre-trained CLIP model.
    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "pt_clip" in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "pt_clip" not in n]
    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "pt_clip" in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "pt_clip" not in n]

    weight_decay = cfg.SOLVER.OPTIM.WEIGHT_DECAY
    optimizer_grouped_parameters = []
    if len(decay_clip_param_tp) > 0:
        optimizer_grouped_parameters.append({'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': cfg.SOLVER.OPTIM.LR * cfg.SOLVER.OPTIM.COEF_LR})
    if len(decay_noclip_param_tp) > 0:
        optimizer_grouped_parameters.append({'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay})
    if len(no_decay_clip_param_tp) > 0:
        optimizer_grouped_parameters.append({'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': cfg.SOLVER.OPTIM.LR * cfg.SOLVER.OPTIM.COEF_LR})
    if len(no_decay_noclip_param_tp) > 0:
        optimizer_grouped_parameters.append({'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0})

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=cfg.SOLVER.OPTIM.LR, betas=(0.9, 0.99))
    
    #total_train_steps = cfg.SOLVER.OPTIM.MAX_EPOCH * num_iter_per_epoch
    warmup_steps = int(cfg.SOLVER.OPTIM.WARMUP_PROPORTION * total_train_steps)
    
    if cfg.SOLVER.OPTIM.LR_SCHEDULER == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)
    elif cfg.SOLVER.OPTIM.LR_SCHEDULER == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)
    elif cfg.SOLVER.OPTIM.LR_SCHEDULER == "cosine_hard_restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps, num_cycles=2)
    else:
        raise ValueError(f"Invalid LR scheduler: {cfg.SOLVER.OPTIM.LR_SCHEDULER}")
    logger.info(f"{cfg.SOLVER.OPTIM.LR_SCHEDULER} scheduler is prepared.")
    return optimizer, scheduler

def print_recall_results(all_topk_recall, topk_list, data_split):
    title = list(all_topk_recall.keys())
    result_dict = {}
    data = [title]
    for topk in topk_list:
        row = [f"{all_topk_recall[term][topk]:.4f}" for term in title]
        result_dict.update({f"{data_split}_{term}_top-{topk}": all_topk_recall[term][topk] for term in title})
        data.append(row)
    result_table = form_data_table(data, title=f"Recall Results on {data_split}")
    print(result_table)
    return result_dict

def do_train(cfg, device, last_iter, max_iter, model, optimizer, scheduler, save_ckpt_dir, train_dataloader, val_dataloader, test_dataloader, corp_dataloader, corpus_seg_list, loss_tracker, wandb_logger):
    #import ipdb; ipdb.set_trace()
    step_log = cfg.SOLVER.STEP_LOG
    setp_eval = cfg.SOLVER.STEP_EVAL
    max_epoch = cfg.SOLVER.OPTIM.MAX_EPOCH
    num_iter_per_epoch = len(train_dataloader)
    
    best_score = -1.0
    current_iter = last_iter if last_iter is not None else -1
    model.train()

    for epoch in range(max_epoch):
        torch.cuda.empty_cache()
            
        for batch_input in tqdm(train_dataloader, desc="Traing at epoch {}".format(epoch)):
            
            current_iter += 1
            if current_iter >= max_iter:
                return

            # Move the batch data to the device
            for k, v in batch_input.items():
                if isinstance(v, torch.Tensor):
                    batch_input[k] = v.to(device)
            
            # Clear the gradients before each forward pass
            optimizer.zero_grad()

            # Forward pass
            loss = model(batch_input["query_feat"], batch_input["frame_feat_seq"], batch_input["frame_mask_seq"], batch_input["batch_score_matrix"])
            
            
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.SOLVER.OPTIM.GRAD_MAX_NORM)
            optimizer.step()
            scheduler.step()

            # Update the loss tracker
            loss_tracker.update(float(loss))

            # Report the training info at the specified intervals.
            if (current_iter+1) % step_log == 0:
                average_loss = loss_tracker.average_loss()
                print(f"Epoch: {epoch + 1}/{max_epoch}, Total Iter: {current_iter+1}, Loss: {average_loss:.4f}")
                # Reset trackers after reporting
                loss_tracker.reset()
                # Report memory usage
                print("-------------------------")
                for i in range(torch.cuda.device_count()):
                    print(f"Memory Allocated on GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                    print(f"Memory Cached on GPU {i}: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
                print("-------------------------")
                if wandb_logger is not None:
                    lr_base = scheduler.get_last_lr()[1]
                    lr_clip = scheduler.get_last_lr()[0]
                    wandb_logger.log({"train_loss": average_loss,
                                      "lr_base": lr_base,
                                      "lr_clip": lr_clip}, step=(current_iter+1))
            
            
            if (current_iter+1) % setp_eval == 0:
                print(f"Evaluation at Epoch: {epoch + 1}/{max_epoch}, Total Iter: {current_iter+1}")
                flag_recall, val_recalls, test_recalls = do_eval(cfg, device, model, corp_dataloader, corpus_seg_list, val_dataloader, test_dataloader)
                val_result_dict = print_recall_results(val_recalls, cfg.TEST.TOPK, "val")
                test_result_dict = print_recall_results(test_recalls, cfg.TEST.TOPK, "test")
                if wandb_logger is not None:
                    wandb_logger.log(val_result_dict, step=(current_iter+1))
                    wandb_logger.log(test_result_dict, step=(current_iter+1))
                """
                should print the (val_recalls, test_recalls) here, like in a table
                """
                
                # Save the best model if the recall is improved
                if flag_recall > best_score:
                    best_score = flag_recall
                    save_ckpt(save_ckpt_dir, model, optimizer, scheduler, cfg.MODEL.TEMP_MODEL.TYPE, iter=(current_iter+1))
                    remove_old_files(save_ckpt_dir, keep_last=5)
                    logger_recall(val_recalls, "BEST VAL")
                    logger_recall(test_recalls, "BEST TEST") 


@torch.no_grad()
def do_eval(cfg, device, model, corpus_dataloader, corpus_seg_list, val_dataloader, test_dataloader):
    model.eval()

    # Compute the visual feature for all segments in the corpus.
    all_seg_feats = compute_corpus_feature(model, corpus_dataloader, device) 

    # Evaluation of val set.
    val_recalls, flag_recall =  evaluate_recall(model, all_seg_feats, corpus_seg_list, val_dataloader, device, cfg.TEST.TOPK)
    #logger_recall(val_recalls, "VAL")
    
    # Evaluation of val set.
    test_recalls, _ =  evaluate_recall(model, all_seg_feats, corpus_seg_list, test_dataloader, device, cfg.TEST.TOPK)
    #logger_recall(test_recalls, "TEST")        
    
    model.train()

    return flag_recall, val_recalls, test_recalls 

if __name__ == "__main__":
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')

    # parse args
    args = get_args()
    init_rand(args.seed)
    logger.info(f"Arguments:\n{json.dumps(vars(args), indent=4)}")
    
    # load config
    cfg = load_config(args.cfg_dir)
    logger.info("Config loaded: {}".format(cfg))

    # set save_ckpt_dir
    save_ckpt_dir = os.path.join(cfg.SOLVER.CHPK_PATH, time_str)
    os.makedirs(save_ckpt_dir, exist_ok=True)

    # build dataloader
    train_dataloader = build_train_dataloader(cfg)
    val_dataloader, test_dataloader = build_test_dataloader(cfg)
    corp_dataloader, corpus_seg_list = build_corp_dataloader(cfg)

    # build model
    model = FrozenCLIP4Video(cfg)
    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    
    # build optimizer and scheduler
    last_iter = None
    num_iter_per_epoch = len(train_dataloader)
    max_iter = cfg.SOLVER.OPTIM.MAX_EPOCH * num_iter_per_epoch
    print("num_iter_per_epoch: ", num_iter_per_epoch, "max_iter: ", max_iter)
    optimizer, scheduler = prep_optimizer_scheduler(cfg, model, max_iter)

    # load ckpts if specified
    if cfg.MODEL.CHPK_PATH is not None:
        model, optimizer, scheduler = load_ckpt(cfg.MODEL.CHPK_PATH, model, optimizer, scheduler)
        last_iter = scheduler.last_epoch
    

    # build logger
    loss_tracker = LossTracker()
    wandb_logger = WandbLogger(cfg.WANDB, args.cfg_dir) if args.use_wandb else None
    
    do_train(cfg, device, last_iter, max_iter, model, optimizer, scheduler, save_ckpt_dir, train_dataloader, val_dataloader, test_dataloader, corp_dataloader, corpus_seg_list, loss_tracker, wandb_logger)
    wandb_logger.finish()

    