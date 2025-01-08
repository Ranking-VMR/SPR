import os
import random
import math
import numpy as np
import torch

#from utils.dense_retrieval_faiss.runs import Run
from utils.dense_retrieval_faiss.parser import Arguments
from tasks.dense_retrieval_faiss.indexing.faiss import index_faiss
from tasks.dense_retrieval_faiss.indexing.loaders import load_vidlens
from loguru import logger

def init_rand(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    #random.seed(12345)
    init_rand(12345)

    parser = Arguments(description='Faiss indexing for end-to-end retrieval with CLIP.')
    parser.add_index_use_input()

    parser.add_argument('--sample', help='sample rate', default=None, type=float)
    parser.add_argument('--slices', help='slices of index data', default=1, type=int)
    parser.add_argument('--vid_anno_path', help='path of video annotation file', default=None, type=str)
    parser.add_argument('--vid_feat_path', help='path of video feature file', default=None, type=str)
    parser.add_argument('--vidlen_path', help='path of video feature length file', default=None, type=str)

    args = parser.parse()

    assert args.slices >= 1
    assert args.sample is None or (0.0 < args.sample < 1.0), args.sample

    #with Run.context():
    # index_root: /root/to/indexes/.
    # index_name: dataset.quantizer.sub_divide. E.g., MSMARCO.L2.32x200k.
    args.index_path = os.path.join(args.index_root, args.index_name)
    os.makedirs(args.index_path, exist_ok=True)

    num_embeddings = sum(load_vidlens(args.vidlen_path)) # num of all segment embeddings. num_embeddings = 166217 for tvr_vcmr
    #num_embeddings = 166217
    print("#> num_embeddings =", num_embeddings)

    # partitions is num of cells in the IVF index
    if args.partitions is None:
        args.partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
        print('\n\n')
        logger.warning("You did not specify --partitions!")
        logger.warning("Default computation chooses {} partitions (for {} embeddings)".format(args.partitions, num_embeddings))
        print('\n\n')

    index_faiss(args)


if __name__ == "__main__":
    main()