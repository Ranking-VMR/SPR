import os
import math
from loguru import logger

from tasks.rvmr_seg_retr_faiss.indexing.loaders import load_vidlens
from tasks.rvmr_seg_retr_faiss.indexing.faiss import index_faiss
from utils.general.setting_utils import init_rand
from utils.dense_retrieval_faiss.parser import Arguments

def main():
    # parse the arguments
    parser = Arguments(description='Faiss indexing for end-to-end retrieval with CLIP.')
    parser.add_index_use_input()

    parser.add_argument('--sample', help='sample rate', default=None, type=float)
    parser.add_argument('--slices', help='slices of index data', default=1, type=int)
    parser.add_argument('--vid_anno_path', help='path of video annotation file', default=None, type=str)
    parser.add_argument('--vid_feat_path', help='path of video feature file', default=None, type=str)
    parser.add_argument('--vidlen_path', help='path of video segment feature length file', default=None, type=str)
    parser.add_argument('--seed', help='random seed', default=12345, type=int)

    args = parser.parse()

    # set the random seed for reproducibility
    init_rand(args.seed)

    assert args.slices >= 1
    assert args.sample is None or (0.0 < args.sample < 1.0), args.sample

    # set the index path for saving the index
    # index_root: /root/to/indexes/.
    # index_name: dataset.quantizer.sub_divide. E.g., MSMARCO.L2.32x200k.
    args.index_path = os.path.join(args.index_root, args.index_name)
    os.makedirs(args.index_path, exist_ok=True)

    # count the total number of embeddings for indexing
    num_embeddings = sum(load_vidlens(args.vidlen_path)) # num of all segment embeddings. num_embeddings = 383828 for tvr_rvmr (4s segments)
    logger.info(f"#> num_embeddings = {num_embeddings}")

    # set the number of partitions for the IVF index if not specified
    # partitions: num of cells in the IVF index
    if args.partitions is None:
        args.partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
        #print('\n\n')
        logger.warning("\n\n You did not specify --partitions!")
        logger.warning(f"Default computation chooses {args.partitions} partitions (for {num_embeddings} embeddings) \n\n")
        #print('\n\n')

    # index the embeddings using Faiss, and save the index at args.index_path
    index_faiss(args)

if __name__ == "__main__":
    main()

