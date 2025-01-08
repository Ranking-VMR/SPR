import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from loguru import logger
import argparse
import json
import os


from models.CLIP4Video.frozen_clip_for_video import FrozenCLIP4Video
from datasets.rvmr.clip_feat_seg_dataset import build_corp_dataloader
from tasks.rvmr_seg_retr_faiss.evaluation.evaluation_lib import compute_corpus_feature
from utils.general.config import load_config
from utils.general.setting_utils import init_rand
from utils.general.basic_utils import load_ckpt

def logger_recall(all_topk_recall, suffix):
    for recall_name, topk_recall in all_topk_recall.items():
        print(f"For {recall_name}:")
        for topk, recall in topk_recall.items():
            print(f"{suffix} recall top-{topk}: {recall:.4f}")

def get_args():
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument("--cfg_dir", type=str, help="Path to the configuration file")
    parser.add_argument("--ckpt_dir", type=str, help="Path to the checkpoint file")
    parser.add_argument("--np_save_dir", type=str, help="Root path to save segment features")

    # System settings
    parser.add_argument('--seed', type=int, default=2024, help="Random seed for reproducibility")

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    # parse args
    args = get_args()
    init_rand(args.seed)
    logger.info(f"Arguments:\n{json.dumps(vars(args), indent=4)}")
    
    # load config
    cfg = load_config(args.cfg_dir)
    logger.info("Config loaded: {}".format(cfg))

    # build dataloader
    corp_dataloader, corpus_seg_list = build_corp_dataloader(cfg)

    # build model
    model = FrozenCLIP4Video(cfg)
    model, _, _ = load_ckpt(args.ckpt_dir, model)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        #model = nn.DataParallel(model)
        model.to(device)
    else:
        device = torch.device("cpu")
        model.to(device)
    logger.info(f"#> Model inferenced on {device}.")
    
    compute_corpus_feature(model, corp_dataloader, device, np_save_path=args.np_save_dir)  
    

    

    