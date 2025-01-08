import os
import torch
import ujson

from math import ceil
from itertools import accumulate
from colbert.utils.utils import print_message, dotdict, flatten

from colbert.indexing.loaders import get_parts, load_doclens
from colbert.indexing.index_manager import load_index_part
from colbert.ranking.index_ranker import IndexRanker

# 用于处理大型索引的一部分并执行查询和文档的排名。
class IndexPart():
    def __init__(self, directory, dim=128, part_range=None, verbose=True):
        """
        初始化 IndexPart 对象，接收索引目录（directory）、向量维度（dim）、索引范围（part_range）和是否显示详细信息的标志（verbose）。
        加载索引部分和文档长度的元数据。
        计算文档偏移量和结束位置，并存储对应的文档范围。
        加载指定范围的索引部分的文档长度。
        创建一个张量用于存储加载的索引部分，并初始化 IndexRanker 对象用于排名。
        """
        first_part, last_part = (0, None) if part_range is None else (part_range.start, part_range.stop)

        # Load parts metadata
        all_parts, all_parts_paths, _ = get_parts(directory)
        self.parts = all_parts[first_part:last_part]
        self.parts_paths = all_parts_paths[first_part:last_part]

        # Load doclens metadata
        all_doclens = load_doclens(directory, flatten=False)

        self.doc_offset = sum([len(part_doclens) for part_doclens in all_doclens[:first_part]])
        self.doc_endpos = sum([len(part_doclens) for part_doclens in all_doclens[:last_part]])
        self.pids_range = range(self.doc_offset, self.doc_endpos)

        self.parts_doclens = all_doclens[first_part:last_part]
        self.doclens = flatten(self.parts_doclens)
        self.num_embeddings = sum(self.doclens)

        self.tensor = self._load_parts(dim, verbose)
        self.ranker = IndexRanker(self.tensor, self.doclens)

    # 创建一个张量 tensor 用于存储索引部分的嵌入。
    # 依次加载各个索引部分，并将它们存储到张量的对应位置。
    def _load_parts(self, dim, verbose):
        tensor = torch.zeros(self.num_embeddings + 512, dim, dtype=torch.float16)

        if verbose:
            print_message("tensor.size() = ", tensor.size())

        offset = 0
        for idx, filename in enumerate(self.parts_paths):
            print_message("|> Loading", filename, "...", condition=verbose)

            endpos = offset + sum(self.parts_doclens[idx])
            part = load_index_part(filename, verbose=verbose)

            tensor[offset:endpos] = part
            offset = endpos

        return tensor

    # 检查给定的文档ID（pid）是否在当前索引部分的范围内。
    def pid_in_range(self, pid):
        return pid in self.pids_range

    """
    对一个查询批次（Q）和文档ID（pids）进行评分。
    检查查询的数量是否与文档ID的数量匹配。
    确保所有文档ID在当前索引部分的范围内。
    调用 IndexRanker 对象的 rank 函数进行评分，并返回评分结果。
    """
    def rank(self, Q, pids):
        """
        Rank a single batch of Q x pids (e.g., 1k--10k pairs).
        """

        assert Q.size(0) in [1, len(pids)], (Q.size(0), len(pids))
        assert all(pid in self.pids_range for pid in pids), self.pids_range

        pids_ = [pid - self.doc_offset for pid in pids]
        scores = self.ranker.rank(Q, pids_)

        return scores

    """
    对一个大规模查询-文档对集合进行评分。
    确保所有文档ID在当前索引部分的范围内。
    调用 IndexRanker 对象的 batch_rank 函数进行批量评分，并返回评分结果。
    """
    def batch_rank(self, all_query_embeddings, query_indexes, pids, sorted_pids):
        """
        Rank a large, fairly dense set of query--passage pairs (e.g., 1M+ pairs).
        Higher overhead, much faster for large batches.
        """

        assert ((pids >= self.pids_range.start) & (pids < self.pids_range.stop)).sum() == pids.size(0)

        pids_ = pids - self.doc_offset
        scores = self.ranker.batch_rank(all_query_embeddings, query_indexes, pids_, sorted_pids)

        return scores