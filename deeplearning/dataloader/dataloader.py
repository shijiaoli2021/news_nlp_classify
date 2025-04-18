# coding=utf-8
import numpy as np
import pandas as pd
from tqdm import *
from vocab import *


class TextDataLoader:
    def __init__(
            self,
            vocab: Vocab,
            batch_size: int,
            split: float = 0.9,
            shuffle: bool = True,
            data: np = None,
            data_path: str = ""):
        if data is None:
            self.data = np.array(self.load_data(data_path))
        else:
            self.data = data
        self.vocab = vocab
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        self.mode = "train"
        self.train_size = int(len(self.data) * self.split)

    def load_data(self, path):
        return pd.read_csv(path, sep='\t')

    def data_iter(self):
        data = self.data
        if self.mode == "train":
            data = self.data[:self.train_size]
        if self.mode == "val":
            data = self.data[self.train_size:]

        if self.mode != "test_for_res" and self.shuffle:
            np.random.shuffle(data)

        for i in range(self.get_len()):
            cur_batch_len = self.batch_size if (i + 1) * self.batch_size < len(data) else len(
                data) - i * self.batch_size
            cash = data[i * self.batch_size: i * self.batch_size + cur_batch_len]
            yield self.data_preprocess(cash[:, :-1]), list(map(int, cash[:, -1]))

    def get_len(self):
        if self.mode == "train":
            length = int(self.train_size / self.batch_size)
            if self.train_size % self.batch_size != 0:
                return length + 1
            else:
                return length
        if self.mode == "val":
            length = int((len(self.data) - self.train_size) / self.batch_size)
            if (len(self.data) - self.train_size) % self.batch_size != 0:
                return length + 1
            else:
                return length
        if self.mode == "test_for_res":
            length = int(len(self.data) / self.batch_size)
            if len(self.data) % self.batch_size != 0:
                return length + 1
            else:
                return length
        return 0

    def data_preprocess(self, data):
        # text -> id,
        res = []
        for text in data:
            res.append([self.vocab.word2id(word) for word in text])
        return np.array(res)