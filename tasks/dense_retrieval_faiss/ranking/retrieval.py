import os
import time
import faiss
import random
import torch
import itertools
from loguru import logger
#from colbert.utils.runs import Run
from multiprocessing import Pool
from tasks.dense_retrieval_faiss.modeling.inference import ModelInferenceCLIP

from colbert.utils.utils import print_message, batch
from colbert.ranking.rankers import Ranker

"""
该 retrieve 函数实现了从输入查询到最终检索结果记录的完整流程。
它使用 ColBERT 模型进行查询编码，通过 Ranker 对象进行查询与文档的相似度计算和排序.
"""
def retrieve(args):
    inference = ModelInferenceCLIP(args.architecutre, args.device)
    ranker = Ranker(args, inference, faiss_depth=args.faiss_depth)

    milliseconds = 0

    q_loader = args.q_loader
    
    qids_in_order = list(queries.keys())

    """
    获取查询并按批次处理，每批次处理 100 个查询。
    对每个查询进行编码和排名，并记录处理时间。
    将排名结果保存到 rankings 列表中。
    将排名结果记录到日志中，并定期打印处理进度。
    """
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