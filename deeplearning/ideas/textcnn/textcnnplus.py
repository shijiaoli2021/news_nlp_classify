# coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
import torch

"""
text-CNN
通过对文本进行不能核大小的卷积，关注不同窗口长度下的词语对分类的影响，实现分类
input:(batch_size, seq_len, vocab_size)
output:(batch_size, seq_len, classify_num)
"""


def attention(q, k, v, d):

    A = torch.matmul(q, k.transpose(-2, -1))/torch.sqrt(torch.tensor(d))
    A = F.softmax(A)
    return torch.matmul(A, v)

'''layer_norm'''
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps= 1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, input):
        output = self.alpha * (input - input.mean(dim=-1, keepdim=True))\
        /(input.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return output

class FeedForward(nn.Module):
    def __init__(self, input_dim, out_put_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_put_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.linear1(x)))

class Self_Attention(nn.Module):
    def __init__(self, embed_dim, head_num = 4, hidden_dim=256, dropout = 0.1):
        super(Self_Attention, self).__init__()
        assert embed_dim % head_num == 0
        self.multi_head_dim = embed_dim // head_num
        self.head_num = head_num
        self.embed_dim = embed_dim
        self.w = nn.Linear(embed_dim, 3 * embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.feedForward = FeedForward(embed_dim, embed_dim, hidden_dim, dropout)
        self.layerNorm = LayerNorm(embed_dim)


    def forward(self, x):
        batch_size,  embed_dim = x.shape
        W = self.w(x)

        q, k, v = W[:, : self.embed_dim], W[:, self.embed_dim : 2*self.embed_dim], W[:, 2*self.embed_dim:]
        q = q.view(batch_size, self.head_num, self.multi_head_dim).transpose(1, 2)
        k = k.view(batch_size, self.head_num, self.multi_head_dim).transpose(1, 2)
        v = v.view(batch_size, self.head_num, self.multi_head_dim).transpose(1, 2)
        att = self.dropout(attention(q, k, v, self.multi_head_dim)).view(batch_size, -1)
        # feedforward
        return self.layerNorm(self.feedForward(att))

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, ngrams, num_filters, head_num, classify_num, dropout=0.1):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.conv_list = nn.ModuleList([nn.Conv2d(1, num_filters, kernel_size=(k, embed_dim)) for k in ngrams])
        self.fc = nn.Linear(num_filters * len(ngrams), classify_num)
        self.softmax = nn.Softmax(dim=-1)
        self.att = Self_Attention(embed_dim, head_num, embed_dim, dropout)
        self.conv2att_linear = nn.Linear(len(ngrams) * num_filters, embed_dim)
        self.embed2class_linear = nn.Linear(embed_dim, classify_num)
        self.num_filters = num_filters
        self.classify_num = classify_num

    def forward(self, x):
        # embedding: (batch_size, seq_len, vocab_size) -> (batch_size, seq_len, embed_dim)
        x = self.embedding(x)

        # 增加通道维度: (batch_size, seq_len, embed_dim) -> (batch_size, 1, seq_len, embed_dim)
        x = x.unsqueeze(1)

        # conv2d: ngram  [(batch_size, num_filters, embed_dim)]
        convs = [F.relu(conv(x)).squeeze(-1) for conv in self.conv_list]
        convs = [F.max_pool1d(conv, kernel_size=conv.size(2)).squeeze(-1) for conv in convs]

        # cat (batch_size, len(ngram) * num_filters)
        cat = torch.cat(convs, dim=0)

        # conv2att
        att = self.conv2att_linear(cat)

        # self-attention (batch_size,  embed_dim)
        att = self.att(att)

        # embed2class (batch_size, embed_dim) -> (batch_size, classify_num)
        out = F.relu(self.embed2class_linear(att))

        return out
