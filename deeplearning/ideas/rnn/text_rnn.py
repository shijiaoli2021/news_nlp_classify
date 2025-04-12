#coding=utf-8

import torch.nn as nn
import torch.nn.functional as F

class TextRnn(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, layer_num, classify_num):
        super(TextRnn, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, num_layers=layer_num, batch_first=True, bidirectional=True)
        # self.l1 = nn.Linear(hidden_dim, classify_num)
        self.maxPool = nn.MaxPool2d(kernel_size=(1, hidden_dim))
        self.fc = nn.Linear(layer_num * 2, classify_num)

    def forward(self, input):

        # embedding: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        embed = self.embedding(input)

        # rnn: (batch_size, seq_len, embed_dim) -> out(batch_size, seq_len, hidden_size), hn(layer_num, batch_size, hidden_size)
        output, hn = self.rnn(embed)

        # layer_num, batch_size, hidden_size = hn.shape

        pooling = self.maxPool(hn).squeeze(-1)

        return self.fc(pooling.transpose(0, 1))
