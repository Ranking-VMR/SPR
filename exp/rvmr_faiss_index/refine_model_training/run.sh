# for training standard version of SPR_{ReLo}-L
PYTHONPATH="$(dirname $0)/../../../..":$PYTHONPATH \
  python ./tasks/rvmr_seg_retr_faiss/stage3_model_training.py \
    --results_path          ./data/reloclnet \
    --train_path            ./data/TVR/TVR-Ranking/ori_anno/train_top40.json \
    --train_pos_neg_path    ./data/TVR/TVR-Ranking/coarse_pred/train_top40_pos_neg_samples_context_8.json \
    --val_path              ./data/TVR/TVR-Ranking/ori_anno/val.json \
    --val_coarse_pred_path  ./data/TVR/TVR-Ranking/coarse_pred/val_coarse_proposals.json \
    --test_path             ./data/TVR/TVR-Ranking/ori_anno/test.json \
    --test_coarse_pred_path ./data/TVR/TVR-Ranking/coarse_pred/test_coarse_proposals.json \
    --video_len_path        ./data/TVR/TVR-Ranking/ori_anno/video_corpus_sorted.json \
    --desc_bert_path        ./data/TVR/TVR_feature/bert_feature/query_tvrr/query_bert.h5 \
    --video_feat_path       ./data/TVR/TVR_feature/video_feature/tvr_i3d_rgb600_avg_cl-1.5.h5 \
    --sub_bert_path         ./data/TVR/TVR_feature/bert_feature/sub_query/tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5 \
    --n_epoch               50 \
    --max_es_cnt            -1 \
    --hard_negative_start_epoch 10 \
    --train_span_start_epoch 0 \
    --lw_st_ed              0.02 \
    --eval_step             2500 \
    --seed                  2024 \
    --bsz                   8 \
    --bsz_eval              16 \
    --exp_id                top40 \
    --num_workers           4 \
    --grad_clip             5.0 \
    --visual_input_size     1024 \
    --hidden_size           768 \
    --moment_ctx_len        8 \
    --ori_seg_len           4 \
    --model_type            reloclnet_rvmr
    #--use_wandb \
    #--wandb_group ori_feature_mil \
    #--wandb_job normal_hard_epoch_10_hard_pool_size_20_e50_hsize768 \
    #--wandb_name lw_st_ed_0.02_top40 \
    #--model_type reloclnet_rvmr
