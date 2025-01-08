import torch

from functools import partial

from tasks.dense_retrieval_faiss.ranking.index_part import IndexPart
from tasks.dense_retrieval_faiss.ranking.faiss_index import FaissIndex
#from colbert.utils.utils import flatten, zipstar

"""
Ranker 类结合了 FAISS 索引的快速检索能力和 IndexPart 的精细化评分功能。
通过将查询编码为向量，并对查询与文档的相似度进行评分，Ranker 类实现了高效的查询检索和排序。
这在需要处理大规模文档集合的检索任务中非常有用。
"""
class Ranker():
    """
    初始化 Ranker 对象，接收参数 args、推理对象 inference 和 FAISS 深度 faiss_depth。
    如果设置了 faiss_depth，初始化 FaissIndex 并定义 retrieve 方法。
    初始化 IndexPart 对象，用于精细化评分。
    """
    def __init__(self, args, inference, faiss_depth=1024):
        self.inference = inference
        self.faiss_depth = faiss_depth

        if faiss_depth is not None:
            self.faiss_index = FaissIndex(args.index_path, args.faiss_index_path, args.nprobe, part_range=args.part_range)
            self.retrieve = partial(self.faiss_index.retrieve, self.faiss_depth)

        self.index = IndexPart(args.index_path, dim=inference.colbert.dim, part_range=args.part_range, verbose=True)

    # Encode queries into vectors.
    def encode(self, queries):
        assert type(queries) in [list, tuple], type(queries)

        Q = self.inference.queryFromText(queries, bsize=512 if len(queries) > 512 else None)

        return Q

    """
    对查询向量和文档 ID 进行评分。
    如果未提供文档 ID（pids），使用 retrieve 方法从 FAISS 索引中检索。
    确保文档 ID 是列表或元组类型，并且查询张量的大小符合预期。
    如果有文档 ID，调整查询张量的形状，并调用 IndexPart 的 rank 方法进行评分。
    对评分结果进行排序，并返回排序后的文档 ID 和对应的评分。

    只是为了重排序（re-rank）！
    """
    def rank(self, Q, pids=None):
        pids = self.retrieve(Q, verbose=False)[0] if pids is None else pids

        assert type(pids) in [list, tuple], type(pids)
        assert Q.size(0) == 1, (len(pids), Q.size())
        assert all(type(pid) is int for pid in pids)

        scores = []
        if len(pids) > 0:
            Q = Q.permute(0, 2, 1)
            scores = self.index.rank(Q, pids)

            scores_sorter = torch.tensor(scores).sort(descending=True)
            pids, scores = torch.tensor(pids)[scores_sorter.indices].tolist(), scores_sorter.values.tolist()

        return pids, scores