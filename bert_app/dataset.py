from bert_pretrain.dataset import padding, seg_text
import torch
from torch.utils.data.dataloader import Dataset
from bert_pretrain.count_vocab import Vocab
import bert_pretrain.bert_config as bert_config
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


def preprocess_range_text(data, vocab, start_idx, end_idx, thread_idx, input_index, label_index):
    input_data = []
    input_label = []
    with tqdm(data[start_idx: end_idx].itertuples(), total=end_idx-start_idx, desc=f"thread:{thread_idx}") as tq:
        for item in tq:
            # require data and label
            text, label = getattr(item, input_index), int(getattr(item, label_index))

            # split
            word_list = text.split()

            # word2idx
            input_idx_list = [vocab.word2idx(word) for word in word_list]
            input_idx_list = [vocab.word2idx(vocab.get_cls_word())] + input_idx_list

            # seg_text
            input_idx_list = seg_text(bert_config.max_len, input_idx_list)

            # padding
            input_idx_list = padding(bert_config.max_len, input_idx_list,
                                     vocab.word2idx(vocab.get_pad_word()))
            input_data.append(input_idx_list)
            input_label.append(label)
    return input_data, input_label

class BertAppDataset(Dataset):
    def __init__(self, text_data, vocab:Vocab, input_index:str="text", label_index:str="label"):
        super(BertAppDataset, self).__init__()
        self.data = text_data
        self.input_index = input_index
        self.label_index = label_index
        self.vocab = vocab
        self.input_data = None
        self.input_label = None
        self.preprocess_data()

    def preprocess_data(self):
        self.input_data = []
        self.input_label = []
        print("start preprocess text data...")
        per_size = int(len(self.data) / THREAD_NUM)  if len(self.data) % THREAD_NUM == 0 else (int(len(self.data) / THREAD_NUM) + 1)

        # multi-thread for data preprocess
        thread_list = []
        for i in range(THREAD_NUM):
            start_idx = i * per_size
            end_idx = min((i+1) * per_size, len(self.data))
            thread_list.append(DataThread(preprocess_range_text,
                                          fn_args=(self.data, self.vocab, start_idx, end_idx, i, self.input_index, self.label_index)))

        # start thread
        [thread.start() for thread in thread_list]

        # join
        [thread.join() for thread in thread_list]

        for thread in thread_list:
            input_data_cash, input_label_cash = thread.get_res()
            self.input_data += input_data_cash
            self.input_label += input_label_cash

        # transfer to tensor
        self.list2tensor()



    def list2tensor(self):
        """
        transfer input to tensor for training
        """
        self.input_data = torch.LongTensor(self.input_data)
        self.input_label = torch.LongTensor(self.input_label)


    def __getitem__(self, index):
        return self.input_data[index], self.input_label[index]


    def __len__(self):
        return len(self.data)



