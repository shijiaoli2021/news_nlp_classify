#coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
import torch

"""
text-CNN
通过对文本进行不能核大小的卷积，关注不同窗口长度下的词语对分类的影响，实现分类
input:(batch_size, seq_len, vocab_size)
output:(batch_size, seq_len, classify_num)
"""

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, ngrams, num_filters, classify_num):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.conv_list = nn.ModuleList([nn.Conv2d(1, num_filters, kernel_size=(k, embed_dim)) for k in ngrams])
        self.fc = nn.Linear(num_filters * len(ngrams), classify_num)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # embedding: (batch_size, seq_len, vocab_size) -> (batch_size, seq_len, embed_dim)
        x = self.embedding(x)

        # 增加通道维度: (batch_size, seq_len, embed_dim) -> (batch_size, 1, seq_len, embed_dim)
        x = x.unsqueeze(1)

        # conv2d: ngram  [(batch_size, num_filters, seq_len - kernel_size + 1)]
        convs = [F.relu(conv(x)).squeeze(-1) for conv in self.conv_list]

        # max pooling [(batch_size, num_filters)]
        convs = [F.max_pool1d(conv, kernel_size=conv.size(2)).squeeze(-1) for conv in convs]

        # cat
        cat = torch.cat(convs, dim=0)
        
        # fulling connect
        out = self.fc(cat)

        return out
