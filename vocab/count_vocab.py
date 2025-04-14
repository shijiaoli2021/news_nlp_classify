import pandas as pd

class Vocab:
    def __init__(self, data_path="", mask_word="[MASK]", pad_word="[PAD]", cls_word = "[CLS]"):

        # 初始化词袋字典
        self.vocab_dict = {pad_word:0, cls_word:1, mask_word:2}
        self.vocab_invalid_len = 3
        self.not_mask_idx_list = []
        self.vocab_len = 3
        self.mask_word = mask_word
        self.pad_word = pad_word
        self.cls_word = cls_word
        if data_path is not None:
            self.load_data2vocab(data_path)



    def load_data2vocab(self, data_path):
        """
        dataframe: have index with ("text")
        :param data_path: ""
        """
        data = pd.read_csv(data_path, sep="\t")["text"]
        for text in data:
            self.load_text2vocab(text)

    def load_text2vocab(self, text):
        """
        load vocab for a text
        :param text: str
        """
        word_list = text.split()
        for word in word_list:
            if word not in self.vocab_dict.keys():
                self.vocab_dict[word] = self.vocab_len
                self.vocab_len += 1

    def word2idx(self, word):
        return self.vocab_dict[word]

    def vocab_size(self):
        return self.vocab_len

    def get_mask_word(self):
        return self.mask_word

    def get_pad_word(self):
        return self.pad_word

    def get_cls_word(self):
        return self.cls_word

    def get_len(self):
        return self.vocab_len
