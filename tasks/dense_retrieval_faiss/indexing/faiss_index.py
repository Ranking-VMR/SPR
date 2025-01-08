import sys
import time
import math
import faiss
import torch
from loguru import logger
import numpy as np

from tasks.dense_retrieval_faiss.indexing.faiss_index_gpu import FaissIndexGPU

class FlatL2_Index():
    def __init__(self, dim):
        self.dim = dim

        self.index = self._create_index()
        self.offset = 0

    def _create_index(self):
        index = faiss.IndexFlatL2(self.dim)
        return index
    
    """
    def train(self, train_data):
        logger.info(f"#> Flat L2, no need to train...")
        return
    """

    def add(self, data):
        # need normalization before adding
        logger.info(f"Add data with shape {data.shape} (offset = {self.offset})..")
        self.index.add(data)
        self.offset += data.shape[0]

    def save(self, output_path):
        logger.info(f"Writing index to {output_path} ...")

        faiss.write_index(self.index, output_path)

class FlaIP_Index():
    def __init__(self, dim):
        self.dim = dim

        self.index = self._create_index()
        self.offset = 0

    def _create_index(self):
        index = faiss.IndexFlatIP(self.dim)
        return index

    def add(self, data):
        # need normalization before adding
        logger.info(f"Add data with shape {data.shape} (offset = {self.offset})..")
        self.index.add(data)
        self.offset += data.shape[0]

    def save(self, output_path):
        logger.info(f"Writing index to {output_path} ...")

        faiss.write_index(self.index, output_path)
        
class IVFFlat_Index():
    def __init__(self, dim, partitions):
        self.dim = dim
        self.partitions = partitions # partitions is num of cells in the IVF index
        
        #self.gpu = FaissIndexGPU()
        self.quantizer, self.index = self._create_index()
        self.offset = 0

    def _create_index(self):
        quantizer = faiss.IndexFlatL2(self.dim) 
        index = faiss.IndexIVFFlat(quantizer, self.dim, self.partitions)

        return quantizer, index

    def train(self, train_data):
        # need normalization before training
        logger.info(f"#> Training now (using CPUs)...")

        #if self.gpu.ngpu > 0:
        #    self.gpu.training_initialize(self.index, self.quantizer)

        s = time.time()
        self.index.train(train_data)
        print(time.time() - s)

        #if self.gpu.ngpu > 0:
        #    self.gpu.training_finalize()

    def add(self, data):
        # need normalization before adding
        logger.info(f"Add data with shape {data.shape} (offset = {self.offset})..")

        #if self.gpu.ngpu > 0 and self.offset == 0:
        #    self.gpu.adding_initialize(self.index)

        #if self.gpu.ngpu > 0:
        #    self.gpu.add(self.index, data, self.offset)
        #else:
        self.index.add(data)

        self.offset += data.shape[0]

    def save(self, output_path):
        logger.info(f"Writing index to {output_path} ...")

        self.index.nprobe = 10  # just a default
        faiss.write_index(self.index, output_path)

class IVFPQ_Index():
    def __init__(self, dim, partitions, m, nbits):
        self.dim = dim
        self.partitions = partitions # partitions is num of cells in the IVF index
        self.m = m
        self.nbits = nbits

        self.gpu = FaissIndexGPU()
        self.quantizer, self.index = self._create_index()
        self.offset = 0

    def _create_index(self):
        quantizer = faiss.IndexFlatL2(self.dim)  # faiss.IndexHNSWFlat(dim, 32)
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.partitions, self.m, self.nbits)

        return quantizer, index

    def train(self, train_data):
        # need normalization before training
        logger.info(f"#> Training now (using {self.gpu.ngpu} GPUs)...")

        if self.gpu.ngpu > 0:
            self.gpu.training_initialize(self.index, self.quantizer)

        s = time.time()
        self.index.train(train_data)
        print(time.time() - s)

        if self.gpu.ngpu > 0:
            self.gpu.training_finalize()

    def add(self, data):
        # need normalization before adding
        logger.info(f"Add data with shape {data.shape} (offset = {self.offset})..")

        if self.gpu.ngpu > 0 and self.offset == 0:
            self.gpu.adding_initialize(self.index)

        if self.gpu.ngpu > 0:
            self.gpu.add(self.index, data, self.offset)
        else:
            self.index.add(data)

        self.offset += data.shape[0]

    def save(self, output_path):
        logger.info(f"Writing index to {output_path} ...")

        self.index.nprobe = 10  # just a default
        faiss.write_index(self.index, output_path)