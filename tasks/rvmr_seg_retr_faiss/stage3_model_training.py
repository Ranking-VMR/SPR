import os
import sys
import time
import json
import pprint
import random
import numpy as np
from easydict import EasyDict as EDict
from tqdm import tqdm
#from collections import OrderedDict
import torch
import torch.nn as nn
from loguru import logger


from SPR.datasets.rvmr.i3d_feat_prop_refine_dataset import prepare_batch_inputs, prepare_dataset
from tasks.rvmr_seg_retr_faiss.proposal_refinement.inference import eval_epoch
from tasks.rvmr_seg_retr_faiss.proposal_refinement.setup import get_args
from tasks.rvmr_seg_retr_faiss.proposal_refinement.run_utils import count_parameters, prepare_model, prepare_optimizer, save_model, logger_ndcg_iou, logger_video_recall
from utils.general.setting_utils import init_rand
from utils.general.loggers import WandbLogger


def start_training():
    ############################# Setup Before Training #############################
    logger.info("Setup config, data, optimizers and model...")
    
    # load configs and set basic settings
    opt = get_args()
    logger.add(os.path.join(opt.results_path, "output.log"))
    logger.info(f"Arguments:\n{json.dumps(vars(opt), indent=4)}")
    init_rand(opt.seed)
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {opt.device}")
    
    # settup wandb logger
    wandb_config = EDict(PROJECT="ReloCLNet_stage3", GROUP=opt.wandb_group, JOB_TYPE=opt.wandb_job, NAME=opt.wandb_name)
    wandb_logger = WandbLogger(wandb_config) if opt.use_wandb else None
    
    # prepare dataset and dataloader
    #train_loader, corpus_loader, corpus_video_list, val_loader, test_loader, val_gt, test_gt = prepare_dataset(opt)
    train_loader, val_loader, val_gt, test_loader, test_gt, video_lens = prepare_dataset(opt)
    logger.info("Prepare dataset done.")
    # prepare model and optimizer
    model_config = EDict(
        visual_input_size=opt.visual_input_size,
        sub_input_size=opt.sub_input_size,  # for both desc and subtitles
        query_input_size=opt.query_input_size,  # for both desc and subtitles
        hidden_size=opt.hidden_size,  # hidden dimension
        conv_kernel_size=opt.conv_kernel_size,
        conv_stride=opt.conv_stride,
        max_ctx_l=opt.max_ctx_l,
        max_desc_l=opt.max_desc_l,
        input_drop=opt.input_drop,
        drop=opt.drop,
        n_heads=opt.n_heads,  # self-att heads
        initializer_range=opt.initializer_range,  # for linear layer
        ctx_mode=opt.ctx_mode,  # video, sub or video_sub
        margin=opt.margin,  # margin for ranking loss
        ranking_loss_type=opt.ranking_loss_type,  # loss type, 'hinge' or 'lse'
        lw_neg_q=opt.lw_neg_q,  # loss weight for neg. query and pos. context
        lw_neg_ctx=opt.lw_neg_ctx,  # loss weight for pos. query and neg. context
        lw_hard_neg_ctx=opt.lw_hard_neg_ctx,  # loss weight for hard neg. context
        lw_fcl=opt.lw_fcl,  # loss weight for frame level contrastive learning
        lw_vcl=opt.lw_vcl,  # loss weight for video level contrastive learning
        lw_st_ed=0,  # will be assigned dynamically at training time
        use_hard_negative=False,  # reset at each epoch
        hard_pool_size=opt.hard_pool_size,
        device=opt.device,
        model_type=opt.model_type)

    logger.info("model_config {}".format(model_config))
    model = prepare_model(model_config)
    count_parameters(model)
    num_training_examples = len(train_loader)
    optimizer = prepare_optimizer(model, opt, num_training_examples * opt.n_epoch)
    
    ############################# Start Training #############################
    logger.info("Start Training...")
    
    start_epoch = 0
    prev_best_score = 0
    es_cnt = 0 # for early stopping
    
    """
    # resume model if specified
    if opt.checkpoint is not None:
        model, optimizer, start_epoch = resume_model( opt, model, optimizer, start_epoch)
    """

    for epoch in range(start_epoch, opt.n_epoch):
        torch.cuda.empty_cache()
        logger.info(f"TRAIN EPOCH: {epoch}|{opt.n_epoch}")
        # model setting for each epoch
        if opt.hard_negative_start_epoch != -1 and epoch >= opt.hard_negative_start_epoch:
            model.set_hard_negative(True, opt.hard_pool_size)
        if opt.train_span_start_epoch != -1 and epoch >= opt.train_span_start_epoch:
            model.set_train_st_ed(opt.lw_st_ed)

        
        model.train()
        for step, batch in tqdm(enumerate(train_loader), desc="Training Iteration", total=num_training_examples):
            global_step = epoch * num_training_examples + step + 1
            model_inputs = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

            # model forward and loss backward
            optimizer.zero_grad()
            loss, loss_dict = model(**model_inputs)
            #import ipdb; ipdb.set_trace()
            loss.backward()
            if opt.grad_clip != -1:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip)
            optimizer.step()    
            
            if global_step % opt.log_step == 0:
                logger.info(f"EPOCH {epoch}/{opt.n_epoch} | STEP: {step}|{num_training_examples} | Loss: {loss.item():.6f}")
                loss_info = [f"{k}: {v:.6f}" for k, v in loss_dict.items()]
                logger.info(loss_info)
                lr = optimizer.get_lr()[0]
                logger.info("LR: {:.6f}".format(lr))
                
                if wandb_logger is not None:
                    wandb_logger.log(loss_dict, step=global_step)
                    wandb_logger.log({"lr": lr}, step=global_step)
                
                #for i in range(torch.cuda.device_count()):
                #    print(f"Memory Allocated on GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                #    print(f"Memory Cached on GPU {i}: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
                #print("-------------------------")
            
            if global_step % opt.eval_step == 0:
                val_ndcg_iou, val_video_recall = eval_epoch(model, val_loader, val_gt, video_lens, opt)
                val_logger_dict = logger_ndcg_iou(val_ndcg_iou,  "VAL")
                val_logger_dict2 = logger_video_recall(val_video_recall,  "VAL")

                test_ndcg_iou, test_video_recall = eval_epoch(model, test_loader, test_gt, video_lens, opt)
                test_logger_dict = logger_ndcg_iou(test_ndcg_iou,  "TEST")
                test_logger_dict2 = logger_video_recall(test_video_recall,  "TEST")
                
                if wandb_logger is not None:
                    wandb_logger.log(val_logger_dict, step=global_step)
                    wandb_logger.log(val_logger_dict2, step=global_step)
                    wandb_logger.log(test_logger_dict, step=global_step)
                    wandb_logger.log(test_logger_dict2, step=global_step)

                stop_score = sum([val_ndcg_iou[k][0.7] for k in opt.ndcg_topk])
                if stop_score > prev_best_score:
                    es_cnt = 0
                    prev_best_score = stop_score
                    _ = logger_ndcg_iou(val_ndcg_iou,  "BEST VAL")
                    _ = logger_video_recall(val_video_recall,  "BEST VAL")
                    # save ckpt
                    bestmodel_path = os.path.join(opt.results_path, "best_model.pt")
                    save_model(model, optimizer, epoch, bestmodel_path)
                else:
                    es_cnt += 1
                    if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                        logger.info(f"Early stopping at epoch {epoch}, global step {global_step}")
                        wandb_logger.finish()
                        return
                
                
                
                model.train()

if __name__ == '__main__':
    start_training()