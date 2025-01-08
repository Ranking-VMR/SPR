# --vid_feat_path corresponds to the directory to the saved segment features by stage1. You can switch to your own directory.

# Flat L2 index
echo "$(dirname $0)/../.."
PYTHONPATH="$(dirname $0)/../../..":$PYTHONPATH \
  python ./tasks/rvmr_seg_retr_faiss/stage2_index_faiss.py \
    --index_root ./data/TVR/TVR-Ranking/faiss_index \
    --index_name TVR.383k.CLIP-ViT-L-14.seqTransf.best_model.seg-4s \
    --index_type flatl2 \
    --vid_feat_path ./data/TVR/TVR-Ranking/seg_feats \
    --vid_anno_path ./data/TVR/TVR-Ranking/seg_anno/vidlen_seg_4s.json \
    --vidlen_path ./data/TVR/TVR-Ranking/seg_anno/vidlen_seg_4s.json \
    --root ./exp/rvmr_faiss_index/experiments \
    --experiment TVR-rvmr

# IVF index with 8192 partitions
echo "$(dirname $0)/../.."
PYTHONPATH="$(dirname $0)/../../..":$PYTHONPATH \
  python ./tasks/rvmr_seg_retr_faiss/stage2_index_faiss.py \
    --index_root ./data/TVR/TVR-Ranking/faiss_index \
    --index_name TVR.383k.CLIP-ViT-L-14.seqTransf.best_model.seg-4s \
    --index_type ivfflat \
    --vid_feat_path ./data/TVR/TVR-Ranking/seg_feats \
    --vid_anno_path ./data/TVR/TVR-Ranking/seg_anno/vidlen_seg_4s.json \
    --vidlen_path ./data/TVR/TVR-Ranking/seg_anno/vidlen_seg_4s.json \
    --root ./exp/rvmr_faiss_index/experiments \
    --experiment TVR-rvmr \
    --partitions 8192


# IVFPQ index with 8192 partitions
echo "$(dirname $0)/../.."
PYTHONPATH="$(dirname $0)/../../..":$PYTHONPATH \
  python ./tasks/rvmr_seg_retr_faiss/stage2_index_faiss.py \
    --index_root ./data/TVR/TVR-Ranking/faiss_index \
    --index_name TVR.383k.CLIP-ViT-L-14.seqTransf.best_model.seg-4s \
    --index_type ivfpq \
    --vid_feat_path ./data/TVR/TVR-Ranking/seg_feats \
    --vid_anno_path ./data/TVR/TVR-Ranking/seg_anno/vidlen_seg_4s.json \
    --vidlen_path ./data/TVR/TVR-Ranking/seg_anno/vidlen_seg_4s.json \
    --root ./exp/rvmr_faiss_index/experiments \
    --experiment TVR-rvmr \
    --partitions 8192 \
    --m 16 \
    --nbits 8
