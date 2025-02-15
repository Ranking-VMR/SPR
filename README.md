# A Flexible and Scalable Framework for Video Moment Search

This is implementation for the paper "A Flexible and Scalable Framework for Video Moment Search": [ArXiv version](https://arxiv.org/abs/2501.05072).

![overview](/figures/Framework.png)
<center>The Segment-Proposal-Ranking (SPR) framework. All videos are divided into non-overlapping, equal-length segments (e.g., 4 seconds) for indexing and searching. The final results are computed based on the relevant segments retrieved.</center>

## Prerequisites
The conda environment of SPR can be built as follow:
```shell script
# preparing environment via conda
conda create --name spr python=3.9
conda activate spr
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install transformers==4.36.1 wandb==0.17.9
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install h5py==3.10.0 loguru==0.7.2 terminaltables==3.1.10 easydict==1.13
```

## Preparation
Please download the related files from [Onedrive](https://entuedu-my.sharepoint.com/:u:/g/personal/chongzhi001_e_ntu_edu_sg/EdI8pzKqoqtKkZhzFutbsZEBP79EXtbv1IzTiWasiNXM8g?e=9Ycsls), and place the unzipped folder `data` to the main directory, resulting in `SPR/data`.

## Quick Start
### Training Dual Feature Projectors for Segment Retrieval
This step is to train the dual feature projectors for generating segment and text embeddings. You can directly run the following command or set configs to adapt to your own settings, e.g., dataset or segment length.
```shell script
# please prepare the required data before training
sh exp/rvmr_faiss_index/model_training/run.sh
```
### (Optional) Generating Segment Embeddings on Your Dataset with Trained Model
This step is necessary only if you want to generate embeddings with your own trained model or process on your own dataset. Please set configs first in the following shell file and then run:
```shell script
sh exp/rvmr_faiss_index/model_inference/run.sh
```

### Build Efficient Index
This step aims to build efficient indexes for online segment retrieval. Our system currently supports *flat L2*, *IVF*, and *IVFPQ* indexes.
```shell script
sh exp/rvmr_faiss_index/build_faiss_index/run.sh
```

### Online Segment Retrieval and Evaluation
We are now able to retrieve the relevant moments among corpus via the build indexes (*flat L2*, *IVF*, and *IVFPQ*):
```shell script
sh exp/rvmr_faiss_index/retrieval_eval/run.sh
```

### Training Moment Refinement and Re-ranking Model
This step trains the model for moment refinement and re-ranking. We take the training of $SPR_{ReLo}-L$ model for example:
```shell script
sh exp/rvmr_faiss_index/refine_model_training/run.sh
```

## Citation
If you feel this project helpful to your research, please cite our work.
```
@misc{zhang2025flexiblescalableframeworkvideo,
      title={A Flexible and Scalable Framework for Video Moment Search}, 
      author={Chongzhi Zhang and Xizhou Zhu and Aixin Sun},
      year={2025},
      eprint={2501.05072},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2501.05072}, 
}
```
