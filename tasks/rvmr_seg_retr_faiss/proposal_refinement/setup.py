import random, torch, os
import numpy as np
import argparse
import time


def get_args():
    parser = argparse.ArgumentParser()
    
    # annoation & data paths
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--train_pos_neg_path", type=str, default=None)
    #parser.add_argument("--corpus_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--val_coarse_pred_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--test_coarse_pred_path", type=str, default=None)
    parser.add_argument("--video_len_path", type=str, default=None)
    parser.add_argument("--video_feat_path", type=str, default=None)
    parser.add_argument("--desc_bert_path", type=str, default=None)
    parser.add_argument("--sub_bert_path", type=str, default=None)
    parser.add_argument("--results_path", type=str, default="results")
    
        
    # setup 
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--exp_id", type=str, default=None, help="id of this run, required at training")
    parser.add_argument("--seed", type=int, default=2024, help="random seed") 
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use wandb for logging")
    parser.add_argument("--wandb_group", type=str, default=None, help="Group name for wandb")
    parser.add_argument("--wandb_job", type=str, default=None, help="Job name for wandb")
    parser.add_argument("--wandb_name", type=str, default=None, help="Try name for wandb")
    
        
    # Training Config
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs to run")
    parser.add_argument("--max_es_cnt", type=int, default=10,
                                 help="number of eval setps to early stop, use -1 to disable early stop")
    parser.add_argument("--eval_step", type=int, default=1000, help="eval interval")
    parser.add_argument("--log_step", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4, help="num subprocesses used to load the data, 0: use main process")
    parser.add_argument("--no_pin_memory", action="store_true", help="No use pin_memory=True for dataloader")
    parser.add_argument("--bsz", type=int, default=128, help="mini-batch size")
    parser.add_argument("--bsz_eval", type=int, default=16, help="mini-batch size") 
    
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_warmup_proportion", type=float, default=0.01, help="Proportion of training to perform linear learning rate warmup.")
    parser.add_argument("--wd", type=float, default=0.001, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=-1, help="perform gradient clip, -1: disable")
        
    # Model loss
    parser.add_argument("--model_type", type=str, default=None, choices=["reloclnet", "reloclnet_rvmr", "reloclnet_hard_neg", "reloclnet_rvmr_hard_neg", "reloclnet_rvmr_weighted", "reloclnet_multirank"],  help="model type")
    parser.add_argument("--margin", type=float, default=0.1, help="margin for hinge loss for general pos-neg loss")
    parser.add_argument("--weak_pos_margin", type=float, default=0.05, help="margin for hinge loss for strong_pos-weak_pos loss")
    parser.add_argument("--hard_neg_margin", type=float, default=0.1, help="margin for hinge loss for weak_pos-hard_neg loss")
    parser.add_argument("--lw_neg_q", type=float, default=1, help="weight for ranking loss with negative query and positive context")
    parser.add_argument("--lw_neg_ctx", type=float, default=1, help="weight for ranking loss with positive query and negative context")
    parser.add_argument("--lw_hard_neg_ctx", type=float, default=0.5, help="weight for ranking loss with weak positive query and hard negative context")
    parser.add_argument("--lw_weak_pos_ctx", type=float, default=0.5, help="weight for ranking loss with strong positive query and weak positive context")
    parser.add_argument("--lw_st_ed", type=float, default=0.01, help="weight for st ed prediction loss")
    parser.add_argument("--lw_fcl", type=float, default=0.03, help="weight for frame CL loss")
    parser.add_argument("--lw_vcl", type=float, default=0.03, help="weight for video CL loss")
    parser.add_argument("--ranking_loss_type", type=str, default="hinge", choices=["hinge", "lse"],  help="att loss type, can be hinge loss or its smooth approximation LogSumExp")
    parser.add_argument("--train_span_start_epoch", type=int, default=0,
                                 help="which epoch to start training span prediction, -1 to disable")
    parser.add_argument("--hard_negative_start_epoch", type=int, default=20, help="which epoch to start hard negative sampling for video-level ranking loss, use -1 to disable")
    parser.add_argument("--hard_pool_size", type=int, default=20, help="hard negatives are still sampled, but from a harder pool.")

    parser.add_argument("--loss_fcl_simi", action="store_true", help="Whether to use simi on frame level contrastive loss")
    parser.add_argument("--loss_vcl_simi", action="store_true", help="Whether to use simi on video level contrastive loss")
    parser.add_argument("--loss_st_ed_simi", action="store_true", help="Whether to use simi on st ed prediction loss")
    parser.add_argument("--loss_neg_ctx_simi", action="store_true", help="Whether to use simi on negative context ranking loss")
    parser.add_argument("--loss_neg_q_simi", action="store_true", help="Whether to use simi on negative query ranking loss")
    
    
    # Model and Data config
    parser.add_argument("--pos_sampling", default=None, help="way of sampling positive scores for video retrieval loss")
    parser.add_argument("--ctx_mode", type=str, default="video_sub", help="which context to use a combination of [video, sub, tef]")
    parser.add_argument("--max_desc_l", type=int, default=30, help="max length of descriptions")
    parser.add_argument("--max_ctx_l", type=int, default=128, help="max number of snippets, 100 for tvr clip_length=1.5, oly 109/21825 > 100")
    parser.add_argument("--clip_length", type=float, default=1.5, help="each video will be uniformly segmented into small clips,  will automatically loaded from ProposalConfigs if None")    
    parser.add_argument("--ori_seg_len", type=float, default=4.0, help="unit length of the original video segment in coarse proposals")
    parser.add_argument("--no_norm_vfeat", action="store_true", help="Do not do normalization on video feat, use it only when using resnet_i3d feat")
    parser.add_argument("--no_norm_tfeat", action="store_true", help="Do not do normalization on text feat")
    parser.add_argument("--moment_ctx_len", type=float, default=8, help="length of context moments padded with coarse proposals (pre & post) during inference")
    parser.add_argument("--moment_max_duration", type=float, default=None, help="max moment duration during training")
    parser.add_argument("--use_coarse_hard_neg", action="store_true", help="whether to use hard negative from coarse proposals")
    
        
    parser.add_argument("--visual_input_size", type=int, default=3072)
    parser.add_argument("--sub_input_size", type=int, default=768)
    parser.add_argument("--query_input_size", type=int, default=768)
    parser.add_argument("--max_position_embeddings", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--input_drop", type=float, default=0.1, help="Applied to all inputs")
    parser.add_argument("--drop", type=float, default=0.1, help="Applied to all other layers")
    parser.add_argument("--conv_kernel_size", type=int, default=5)
    parser.add_argument("--conv_stride", type=int, default=1)
    parser.add_argument("--initializer_range", type=float, default=0.02, help="initializer range for layers")
        
        
    # post processing
    parser.add_argument("--min_pred_l", type=int, default=1, help="constrain the [st, ed] with ed - st >= 2 (2 clips with length 1.5 each, 3 secs  in total this is the min length for proposal-based backup_method)")
    parser.add_argument("--max_pred_l", type=int, default=16, help="constrain the [st, ed] pairs with ed - st <= 16, 24 secs in total (16 clips  with length 1.5 each, this is the max length for proposal-based backup_method)")
    parser.add_argument("--q2c_alpha", type=float, default=30, help="give more importance to top scored videos' spans, the new score will be: s_new = exp(alpha * s),  higher alpha indicates more importance. Note s in [-1, 1]")
    #parser.add_argument("--max_before_nms", type=int, default=200)
    #parser.add_argument("--max_vcmr_video", type=int, default=100, help="re-ranking in top-max_vcmr_video")
    #parser.add_argument("--nms_thd", type=float, default=-1, help="additionally use non-maximum suppression (or non-minimum suppression for distance) to post-processing the predictions. -1: do not use nms. 0.6 for charades_sta, 0.5 for anet_cap")

    # evaluation 
    parser.add_argument("--iou_threshold", type=float, nargs='+', default=[0.3, 0.5, 0.7], help="List of IOU thresholds")
    parser.add_argument("--ndcg_topk", type=int, nargs='+', default=[10, 20, 40], help="List of NDCG top k values")
    args = parser.parse_args()
    
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    args.results_path = os.path.join(args.results_path, f"{args.exp_id}_{time_str}")
    os.makedirs(args.results_path, exist_ok=True)
    
    if args.hard_negative_start_epoch != -1:
        if args.hard_pool_size > args.bsz:
            print("[WARNING] hard_pool_size is larger than bsz")

    if args.use_coarse_hard_neg:
        assert args.model_type in ["reloclnet_hard_neg", "reloclnet_rvmr_hard_neg", "reloclnet_multirank"]

    args.pin_memory = not args.no_pin_memory
    if "video" in args.ctx_mode and args.visual_input_size > 3000:  # 3072, the normalized concatenation of resnet+i3d
            assert args.no_norm_vfeat
    
    return args


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)