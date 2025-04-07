# coding=utf-8

from tqdm import *
import numpy as np
import pandas as pd
import threading

THREAD_NUM = 5


class Prv_Thread(threading.Thread):
    def __init__(self, func, args):
        super().__init__()
        self.func = func
        self.args = args
        self.res = None

    def run(self):
        self.res = self.func(*self.args)

    def get_res(self):
        return self.res


def data_preprocess(path: str, seq_len: int, save_path: str, save_keyword='train'):
    data = pd.read_csv(path, sep='\t')
    res = []
    labels = []
    print("data preprocessing...")
    page_size = int(len(data) / THREAD_NUM)
    threads = []
    for i in range(THREAD_NUM):
        startIdx = page_size * i
        endIdx = page_size * (i + 1) if i != THREAD_NUM - 1 else len(data)
        threads.append(Prv_Thread(func=preprocessing, args=(data, startIdx, endIdx, seq_len)))
    # 等待任务完成
    [t.start() for t in threads]
    [t.join() for t in threads]
    for t in threads:
        cash_res, cash_label = t.get_res()
        res += cash_res
        labels += cash_label
    res, labels = np.array(res), np.array(labels).reshape(-1, 1)
    save_data = np.hstack([res, labels])
    print("data preprocessing end, data:{}, saving...".format(save_data.shape))
    np.save(save_path + save_keyword + '.npz', save_data)


def preprocessing(data, startIdx, endIdx, seq_len):
    res = []
    labels = []
    for i in tqdm(range(startIdx, endIdx)):
        label, text = data['label'][i], data['text'][i]
        split_list = split_text(text, seq_len)
        if len(split_list) == 0:
            continue
        res += split_list
        labels += [label for j in range(len(split_list))]
    return res, labels


def split_text(text: str, seq_len: int):
    words_list = text.split()
    if len(words_list) < seq_len:
        return []
    words_len = len(words_list)
    res = []
    idx = 0
    while idx < words_len - seq_len:
        res.append(words_list[idx: idx + seq_len])
        idx = (idx + 1) * seq_len

    if idx < words_len:
        res.append(words_list[words_len - seq_len: words_len])
    return res

