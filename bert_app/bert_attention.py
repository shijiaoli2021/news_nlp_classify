import torch
import torch.nn as nn
from bert_pretrain.bert_model import Bert
from torch.nn import functional as F



def get_pad_mask(x: torch.Tensor, padding_idx=0):
    """
    get pad from x by pad idx
    :param padding_idx:
    :param x: tensor (batch_size, seq_len)
    :return: tensor (batch_size, seq_len, seq_len)
    """
    return x.eq(0).unsqueeze(-1).expand(-1, -1, x.shape[1])

def attention(q, k, v, d, attn_mask=None):
    """

    :param q: (batch_size, head_num, seq_len, d_model)
    :param k: (batch_size, head_num, seq_len, d_model)
    :param v: (batch_size, head_num, seq_len, d_model)
    :param d: d_model
    :param attn_mask: (batch_size, head_num, seq_len, seq_len)
    :return:
    """
    # (batch_size, head_num. seq_len, d_model) -> (batch_size, head_num, seq_len, seq_len)
    A = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d))
    if attn_mask is not None:
        A = A.masked_fill(attn_mask, 1e-9)
    A = nn.Softmax(dim=-1)(A)
    return torch.matmul(A, v)

class Attention(nn.Module):
    def __init__(
        self,
        embed_dim:int,
        dropout:float=0.1
    ):
        super().__init__()
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None)->torch.Tensor:
        """
        :param x: tensor(batch_size, seq_len, embed_dim)
        :param mask: tensor(batch_size, seq_len)
        :return: (batch_size, seq_len, embed_dim)
        """
        out = attention(self.wq(x), self.wk(x), self.wv(x), self.embed_dim, mask)
        return self.dropout(out)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        output = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
                 / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return output

class BertAttention(nn.Module):
    def __init__(
        self,
        bert:Bert,
        embed_dim,
        classify_num:int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.bert_model = bert
        self.embed_dim = embed_dim
        self.classify_num = classify_num
        self.dropout = nn.Dropout(dropout)
        # attention
        self.attention = nn.Sequential(Attention(embed_dim, dropout), LayerNorm(embed_dim))
        self.fc = nn.Linear(embed_dim, classify_num)

    def forward(self, x):

        pad_mask = get_pad_mask(x)

        # bert
        seq_out = self.bert_model(x, seq_out = True)

        seq_out = self.dropout(seq_out)

        # attention (batch_size, seq_len, embed_dim) -> (batch_size, embed_dim)
        seq_out = self.attention(seq_out, pad_mask).sum(axis=1)

        # cls and seq_out classify
        return self.fc(seq_out)


