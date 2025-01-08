# --ckpt_dir corresponds to the checkpoint path for the model, here we use the released checkpoint for example.
# --np_save_dir corresponds to the directory to save segment features for stage2 (buiding index).
PYTHONPATH="$(dirname $0)/../../../..":$PYTHONPATH \
  python ./tasks/rvmr_seg_retr_faiss/stage1_offline_corpus_feat_extract.py \
    --cfg_dir ./exp/rvmr_faiss_index/model_inference/cfg_trans.yaml \
    --ckpt_dir ./data/TVR/Stage1_Model/ckpt.bin \
    --np_save_dir your_directory_to_save_segment_features_for_faiss_index

