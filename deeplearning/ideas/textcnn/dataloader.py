# coding=utf-8
import numpy as np
import pandas as pd
from tqdm import *
from deeplearning.ideas.textcnn.vocab import *

class TextDataLoader():
    def __init__(self, data_path: str, max_seq_len, vocab:Vocab, splitAndPad= True):
        self.data = np.array(self.load_data(data_path))
        self.vocab = vocab
        self.splitAndPad = splitAndPad


    def load_data(self, path):
        return pd.read_csv(path, sep= '\t')


    def data_iter(self, batch_size, shuffle=True):
        if shuffle:
            np.random.shuffle(self.data)

        batch_num = int(len(self.data) / float(batch_size))
        for i in range(batch_num):
            cur_batch_len = batch_size if (i+1) * batch_size < len(self.data) else len(self.data) - i * batch_size
            cash = self.data[i * batch_size : i * batch_size + cur_batch_len]
            yield self.data_prepocess(cash[:, 1]), cash[:, 0]

    def data_prepocess(self, data):
        # text -> id,
        res = []
        for text in data:
            res += [self.vocab.word2id(word) for word in text.split()]
        return np.array(res)