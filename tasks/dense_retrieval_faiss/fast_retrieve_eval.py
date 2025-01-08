import os
import time
import faiss
import random
import torch
import itertools
from loguru import logger
#from colbert.utils.runs import Run
from multiprocessing import Pool

from utils.dense_retrieval_faiss.parser import Arguments
from tasks.dense_retrieval_faiss.modeling.inference import ModelInferenceCLIP
from tasks.dense_retrieval_faiss.evaluation.tvr_vr_evaluator import TVR_VR_Evaluator
from utils.dense_retrieval_faiss.utils import batch

from tasks.dense_retrieval_faiss.indexing.faiss import get_faiss_index_name

from tasks.dense_retrieval_faiss.index_faiss import init_rand



def eval_vr(args):
    inference = ModelInferenceCLIP(args.architecutre, args.device)
    vr_evaluator = TVR_VR_Evaluator(args, inference, faiss_depth=args.faiss_depth)

    vr_evaluator.eval_vr()
    #vr_evaluator.random_eval_vr()

    return

def main():
    init_rand(12345)

    parser = Arguments(description='End-to-end retrieval and ranking with ColBERT.')

    #parser.add_model_parameters()
    #parser.add_model_inference_parameters()
    #parser.add_ranking_input()
    parser.add_retrieval_input()

    # for faiss index
    parser.add_argument('--faiss_name', dest='faiss_name', default=None, type=str)
    parser.add_argument('--part-range', dest='part_range', default=None, type=str)

    # for TVR annoations
    parser.add_argument('--vid_anno_path', help='path of video annotation file', default=None, type=str)
    parser.add_argument('--vid_feat_path', help='path of video feature file', default=None, type=str)
    parser.add_argument('--vidlen_path', help='path of video feature length file', default=None, type=str)
    parser.add_argument('--query_anno_path', help='path of query annotation file', default=None, type=str)

    # model-related args
    parser.add_argument('--architecutre', dest='architecutre', default=None, type=str)
    parser.add_argument('--bsize', dest='bsize', default=128, type=int)

    args = parser.parse()
    
    #args.depth = args.depth if args.depth > 0 else None

    if args.part_range:
        part_offset, part_endpos = map(int, args.part_range.split('..'))
        args.part_range = range(part_offset, part_endpos)

    args.index_path = os.path.join(args.index_root, args.index_name)

    if args.faiss_name is not None:
        args.faiss_index_path = os.path.join(args.index_path, args.faiss_name)
    else:
        args.faiss_index_path = os.path.join(args.index_path, get_faiss_index_name(args))

    
    # other settings
    def check_cuda_installed():
        return torch.cuda.is_available()
    args.device = 'cuda' if check_cuda_installed() else 'cpu'
    
    eval_vr(args)

"""

def retrieve(args):
    inference = ModelInferenceCLIP(args.architecutre, args.device)
    ranker = Ranker(args, inference, faiss_depth=args.faiss_depth)

    milliseconds = 0

    q_loader = args.q_loader
    
    qids_in_order = list(queries.keys())


    for qoffset, qbatch in batch(qids_in_order, 100, provide_offset=True):
        qbatch_text = [queries[qid] for qid in qbatch]

        rankings = []

        for query_idx, q in enumerate(qbatch_text):
            torch.cuda.synchronize('cuda:0')
            s = time.time()

            Q = ranker.encode([q])
            pids, scores = ranker.rank(Q)

            torch.cuda.synchronize()
            milliseconds += (time.time() - s) * 1000.0

            if len(pids):
                print(qoffset+query_idx, q, len(scores), len(pids), scores[0], pids[0],
                        milliseconds / (qoffset+query_idx+1), 'ms')

            rankings.append(zip(pids, scores))

        for query_idx, (qid, ranking) in enumerate(zip(qbatch, rankings)):
            query_idx = qoffset + query_idx

            if query_idx % 100 == 0:
                logger.info(f"#> Logging query #{query_idx} (qid {qid}) now...")

            ranking = [(score, pid, None) for pid, score in itertools.islice(ranking, args.depth)]
            #rlogger.log(qid, ranking, is_ranked=True)

    print('\n\n')
    #print(ranking_logger.filename)
    print("#> Done.")
    print('\n\n')
"""

if __name__ == "__main__":
    main()