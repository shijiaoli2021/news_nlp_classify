import numpy as np
import torch
from torch.utils.data.dataloader import Dataset
from deeplearning.ideas.bert_pretrain.count_vocab import Vocab
import bert_config
import random
from tqdm import *


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
    return data.extend([padding_idx for i in range(max_len - len(data))])

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
        with tqdm(self.data) as tq:
            for text in tq:
                #todo 1) seg text by max_len. 2) mask text. 3)padding text.

                # split
                word_list = text.split()

                # word2idx
                input_idx_list = [self.vocab.word2idx(word) for word in word_list]
                input_idx_list = [self.vocab.word2idx(self.vocab.get_cls_word())] + input_idx_list

                #seg_text
                input_idx_list = seg_text(bert_config.max_len, input_idx_list)

                # mask
                mask_pos, mask_token, input_idx_list = self.mask_one_text(input_idx_list)

                # padding
                input_idx_list = padding(bert_config.max_len, input_idx_list, self.vocab.word2idx(self.vocab.get_pad_word()))
                mask_pos = padding(bert_config.max_pre, mask_pos)
                mask_token = padding(bert_config.max_len, mask_token)

                # save
                self.input_data += input_idx_list
                self.mask_pos_list += mask_pos
                self.mask_token_list += mask_token
        # transfer to tensor
        self.list2tensor()

    def list2tensor(self):
        """
        transfer input to tensor for training
        """
        self.input_data = torch.LongTensor(self.input_data)
        self.mask_token_list = torch.LongTensor(self.mask_token_list)
        self.mask_pos_list = torch.LongTensor()


    """mask a text"""
    def mask_one_text(self, input_idx_list):
        """
        mask a text
        :param input_idx_list: []
        :return:
        """
        # pos idx
        candidate_pos_idx = [i for (i, idx) in enumerate(input_idx_list) if input_idx_list[idx] not in self.vocab.not_mask_idx_list]

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
                input_idx_list[idx] = self.vocab.get_mask_word()
            elif random_seed < bert_config.p_replace:
                input_idx_list[idx] = np.random.randint(low=self.vocab.vocab_invalid_len, high=self.vocab.get_len())
        return mask_idx_from_text, mask_words_idx, input_idx_list


    def __getitem__(self, index):
        return self.input_data[index], self.mask_pos_list[index], self.mask_token_list[index]


    def __len__(self):
        return len(self.data)
