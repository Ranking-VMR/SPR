PYTHONPATH="$(dirname $0)/../../..":$PYTHONPATH \
  python ./tasks/rvmr_seg_retr_faiss/stage1_model_training.py \
    --cfg_dir ./exp/rvmr_faiss_index/model_training/cfg_trans.yaml \
    --use_wandb # you can remove --use_wandb if you don't want to use wandb