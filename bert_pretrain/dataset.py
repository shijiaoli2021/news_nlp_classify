import numpy as np
import torch
from torch.utils.data.dataloader import Dataset
from count_vocab import Vocab
import bert_config
import random
from tqdm import *
from threading import Thread


THREAD_NUM = 5

class DataThread(Thread):
    def __init__(self, fn, fn_args):
        super(DataThread, self).__init__()
        self.fn = fn
        self.fn_args = fn_args
        self.res = None

    def run(self):
        self.res = self.fn(*self.fn_args)

    def get_res(self):
        return self.res

def padding(max_len:int, data, padding_idx=0):
    """
    padding data to max_len
    :param max_len: int, max sequence length
    :param data: list
    :param padding_idx: padding word2idx
    :return:
    """
    if len(data) == max_len:
        return data
    # 填充
    return data + [padding_idx for i in range(max_len - len(data))]

def seg_text(max_len, data, random_seg=False):
    """
    seg text to max_len
    :param max_len: int, max sequence length
    :param data: list
    :param random_seg: bool
    :return:
    """
    if len(data) > max_len:
        if random_seg:
            startIdx = np.random.randint(low=0, high=len(data-max_len))
            return data[startIdx:startIdx+max_len]
        return data[:max_len]
    return data


class BertDataset(Dataset):
    def __init__(self, text_data, vocab:Vocab):
        super(BertDataset, self).__init__()
        self.input_data = None
        self.mask_pos_list = None
        self.mask_token_list = None
        self.data = text_data
        self.vocab = vocab
        self.preprocess_data()

    def preprocess_data(self):
        self.input_data = []
        self.mask_pos_list = []
        self.mask_token_list = []
        print("start preprocess text data...")

        # build thread
        per_size = int(len(self.data) / THREAD_NUM) if len(self.data) % THREAD_NUM == 0 else int(len(self.data) / THREAD_NUM) + 1
        thread_list = []
        for i in range(THREAD_NUM):
            start_idx = i * per_size
            end_idx = min((i+1) * per_size, len(self.data))
            thread_list.append(DataThread(fn=self.preprocess_range_data, fn_args=(start_idx, end_idx)))

        # run
        [thread.start() for thread in thread_list]

        # wait
        [thread.join() for thread in thread_list]

        # acquire data
        for thread in thread_list:
            input_data_cash, mask_pos_cash, mask_token_cash = thread.get_res()
            # save
            self.input_data += input_data_cash
            self.mask_pos_list += mask_pos_cash
            self.mask_token_list += mask_token_cash
        # transfer to tensor
        self.list2tensor()
        print("preprocess text data over...")

    def preprocess_range_data(self, start_idx, end_idx):
        input_data = []
        mask_pos_list = []
        mask_token_list = []
        with tqdm(range(start_idx, end_idx)) as tq:
            for idx in tq:
                # todo 1) seg text by max_len. 2) mask text. 3)padding text.
                # acquire data
                text = self.data[idx]

                # split
                word_list = text.split()

                # word2idx
                input_idx_list = [self.vocab.word2idx(word) for word in word_list]

                input_idx_list = [self.vocab.word2idx(self.vocab.get_cls_word())] + input_idx_list

                # seg_text
                input_idx_list = seg_text(bert_config.max_len, input_idx_list)

                # mask
                mask_pos, mask_token, input_idx_list = self.mask_one_text(input_idx_list)

                # padding
                input_idx_list = padding(bert_config.max_len, input_idx_list,
                                         self.vocab.word2idx(self.vocab.get_pad_word()))
                mask_pos = padding(bert_config.max_pre, mask_pos)
                mask_token = padding(bert_config.max_pre, mask_token)

                # save
                input_data.append(input_idx_list)
                mask_pos_list.append(mask_pos)
                mask_token_list.append(mask_token)
        return input_data, mask_pos_list, mask_token_list


    def list2tensor(self):
        """
        transfer input to tensor for training
        """
        self.input_data = torch.LongTensor(self.input_data)
        self.mask_token_list = torch.LongTensor(self.mask_token_list)
        self.mask_pos_list = torch.LongTensor(self.mask_pos_list)


    """mask a text"""
    def mask_one_text(self, input_idx_list):
        """
        mask a text
        :param input_idx_list: []
        :return:
        """
        # pos idx
        candidate_pos_idx = [i for (i, idx) in enumerate(input_idx_list) if idx not in self.vocab.not_mask_idx_list]

        # shuffle for mask
        random.shuffle(candidate_pos_idx)

        # mask_num
        n_pre = min(bert_config.max_pre, max(1, int(len(input_idx_list) * bert_config.mask_ratio)))

        return self._mask_data(candidate_pos_idx[:n_pre], input_idx_list)

    """mask data"""
    def _mask_data(self, pos_idx_list, input_idx_list):
        """
        mask the text sequence
        :param input_idx_list:
        :param pos_idx_list: the valid(except "pad")(seq_num)
        :return: mask_data, mask_idx_from_text, mask_target_idx
        """
        mask_idx_from_text = []
        mask_words_idx = []
        for idx in pos_idx_list:
            mask_idx_from_text.append(idx)
            mask_words_idx.append(input_idx_list[idx])
            random_seed = np.random.random_sample()
            if random_seed < bert_config.p_mask:
                input_idx_list[idx] = self.vocab.word2idx(self.vocab.get_mask_word())
            elif random_seed < bert_config.p_replace:
                input_idx_list[idx] = np.random.randint(low=self.vocab.vocab_invalid_len, high=self.vocab.get_len())
        return mask_idx_from_text, mask_words_idx, input_idx_list


    def __getitem__(self, index):
        return self.input_data[index], self.mask_pos_list[index], self.mask_token_list[index]


    def __len__(self):
        return len(self.data)

