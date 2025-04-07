import torch
import torch.nn as nn
import torch.nn.functional as F

'''
(batch_size, n, embeded_dim)
'''

class Position_Encoding(nn.Module):
    def __init__(self, feature_dim, max_num=10000, max_seq_len=1000):
        super(Position_Encoding, self).__init__()
        self.max_num = max_num
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        self.max_seq_len = 1000
        self.init_pos()

    def init_pos(self):
        pos = torch.arange(self.max_num, dtype=float)
        self.pos_encode = torch.zeros(self.max_seq_len, self.feature_dim)
        for i in range(0, self.feature_dim, 2):
            self.pos_encode[:, i] = torch.sin(pos/(self.max_num ** (i/self.feature_dim)))
            self.pos_encode[:, i+1] = torch.cos(pos/(self.max_num ** ((i+1)/self.feature_dim)))

    def forward(self, x):
        batch_size, len, feature_dim = x.shape
        return x + self.pos[:len, :].unsqueeze(0)

class Embedding(nn.Module):
    def __init__(self, num_embedding, embeded_dim, kernal_size):
        super(Embedding, self).__init__()
        self.num_embedding = num_embedding
        self.embeded_module = nn.Embedding(num_embedding, embeded_dim)
        self.Position_encoding = Position_Encoding(embeded_dim)
        self.kernal_size = kernal_size

    def forward(self, x):
        # (batch, len, 1) -> (batch, len, embeded_num)
        batch, len = x.shape
        if len < self.kernal_size:
            x = torch.stack([x, self.num_embedding * torch.ones(size=(batch, self.kernal_size-len))], dim = 1)
        return self.Position_encoding(self.embeded_module(x))


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
    def __init__(self, embed_dim, head_num = 4, hidden_dim=64, dropout = 0.1):
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
        batch_size, embed_dim = x.shape
        W = self.w(x)

        q, k, v = W[:, : self.embed_dim], W[:, self.embed_dim : 2*self.embed_dim], W[:, 2*self.embed_dim:]
        q = q.view(batch_size, self.head_num, self.multi_head_dim).transpose(1, 2)
        k = k.view(batch_size, self.head_num, self.multi_head_dim).transpose(1, 2)
        v = v.view(batch_size, self.head_num, self.multi_head_dim).transpose(1, 2)
        att = self.dropout(attention(q, k, v, self.multi_head_dim)).view(batch_size, -1)
        # feedforward
        return self.layerNorm(self.feedForward(att))


class NLP_Classify(nn.Module):
    def __init__(self, **kwargs):
        super(NLP_Classify, self).__init__()
        self.embed_dim = kwargs['embed_dim']
        self.num_embedding_dim = kwargs['num_embedding']
        self.dropout = kwargs['dropout']
        self.classify_num = kwargs['classify_num']
        self.embedding = nn.Linear(self.num_embedding_dim, self.embed_dim)
        self.self_attention1 = Self_Attention(self.embed_dim, head_num=kwargs['head_num'], hidden_dim=kwargs['hidden_dim'], dropout=self.dropout)
        self.self_attention2 = Self_Attention(self.embed_dim, head_num=kwargs['head_num'], hidden_dim=kwargs['hidden_dim'], dropout=self.dropout)
        self.linear1 = nn.Linear(self.embed_dim, kwargs['hidden_dim'])
        self.linear2 = nn.Linear(kwargs['hidden_dim'], self.classify_num)
        self.softmax = nn.Softmax()
        self.layerNorm = LayerNorm(self.num_embedding_dim)

    def forward(self, x):
        # (batch_size, len, 1)
        # return (batch_size, classify_num)
        # print(x.shape)
        x = self.embedding(self.layerNorm(x))

        #(batch_size, embed_dim)
        out = self.self_attention1(x)
        out = self.self_attention2(out)
        return self.softmax(self.linear2(self.linear1(out)))


