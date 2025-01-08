# Flat L2 index
echo "$(dirname $0)/../.."
PYTHONPATH="$(dirname $0)/../../..":$PYTHONPATH \
  python ./tasks/rvmr_seg_retr_faiss/stage2_fast_retrieval_eval.py \
    --index_root ./data/TVR/TVR-Ranking/faiss_index \
    --index_name TVR.383k.CLIP-ViT-L-14.seqTransf.best_model.seg-4s \
    --index_type flatl2 \
    --vid_dur_path ./data/TVR/TVR-Ranking/ori_anno/video_corpus_sorted.json \
    --test_anno_path ./data/TVR/TVR-Ranking/ori_anno/val.json \
    --model_type FrozenCLIP4Video \
    --architecture CLIP-ViT-L-14 \
    --vidlen_path ./data/TVR/TVR-Ranking/seg_anno/vidlen_seg_4s.json \
    --root ./exp/rvmr_faiss_index/experiments \
    --experiment TVR-rvmr \
    --faiss_depth 200 \
    --proposal_type original \
    --seg_duration 4 \
    --cfg_dir ./exp/rvmr_faiss_index/model_inference/cfg_trans.yaml \
    --ckpt_path ./data/TVR/Stage1_Model/ckpt.bin


# IVF index with 8192 partitions & search within top 128 clusters during inference
echo "$(dirname $0)/../.."
PYTHONPATH="$(dirname $0)/../../..":$PYTHONPATH \
  python ./tasks/rvmr_seg_retr_faiss/stage2_fast_retrieval_eval.py \
    --index_root ./data/TVR/TVR-Ranking/faiss_index \
    --index_name TVR.383k.CLIP-ViT-L-14.seqTransf.best_model.seg-4s \
    --index_type ivfflat \
    --vid_dur_path ./data/TVR/TVR-Ranking/ori_anno/video_corpus_sorted.json \
    --test_anno_path ./data/TVR/TVR-Ranking/ori_anno/val.json \
    --model_type FrozenCLIP4Video \
    --architecture CLIP-ViT-L-14 \
    --vidlen_path ./data/TVR/TVR-Ranking/seg_anno/vidlen_seg_4s.json \
    --root ./exp/rvmr_faiss_index/experiments \
    --experiment TVR-rvmr \
    --proposal_type original \
    --seg_duration 4 \
    --cfg_dir ./exp/rvmr_faiss_index/model_inference/cfg_trans.yaml \
    --ckpt_path ./data/TVR/Stage1_Model/ckpt.bin \
    --partitions 8192 \
    --nprobe 128 \
    --faiss_depth 200

# IVFPQ index with 8192 partitions & search within top 128 clusters during inference
echo "$(dirname $0)/../.."
PYTHONPATH="$(dirname $0)/../../..":$PYTHONPATH \
  python ./tasks/rvmr_seg_retr_faiss/stage2_fast_retrieval_eval.py \
    --index_root ./data/TVR/TVR-Ranking/faiss_index \
    --index_name TVR.383k.CLIP-ViT-L-14.seqTransf.best_model.seg-4s \
    --index_type ivfpq \
    --vid_dur_path ./data/TVR/TVR-Ranking/ori_anno/video_corpus_sorted.json \
    --test_anno_path ./data/TVR/TVR-Ranking/ori_anno/val.json \
    --model_type FrozenCLIP4Video \
    --architecture CLIP-ViT-L-14 \
    --vidlen_path ./data/TVR/TVR-Ranking/seg_anno/vidlen_seg_4s.json \
    --root ./exp/rvmr_faiss_index/experiments \
    --experiment TVR-rvmr \
    --proposal_type original \
    --seg_duration 4 \
    --cfg_dir ./exp/rvmr_faiss_index/model_inference/cfg_trans.yaml \
    --ckpt_path ./data/TVR/Stage1_Model/ckpt.bin \
    --partitions 8192 \
    --m 16 \
    --nbits 8 \
    --nprobe 128 \
    --faiss_depth 200
