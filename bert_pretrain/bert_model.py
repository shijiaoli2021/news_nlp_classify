import torch
import torch.nn as nn

def get_pad_mask(x: torch.Tensor, padding_idx=0):
    """
    get pad from x by pad idx
    :param x: tensor (batch_size, seq_len)
    :return: tensor (batch_size, seq_len, seq_len)
    """
    return x.eq(0).unsqueeze(-1).expand(-1, -1, x.shape[1])

class Embedding(nn.Module):
    """
    we only want to learn the semantic of news text, so only use word embedding and pos embedding
    """
    def __init__(self, vocab_size, embed_dim, max_len, device):
        super(Embedding, self).__init__()
        self.word_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.device = device

    def forward(self, x):
        # input(batch_size, seq_len)

        # word embedding (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        word_emb = self.word_emb(x)

        # pos embedding
        pos = torch.arange(x.shape[1], dtype=torch.long, device=self.device)
        # (batch_size, seq_len)
        pos = pos.unsqueeze(0).expand_as(x)
        # (batch_size, seq_len, embed_dim)
        pos_emb = self.pos_emb(pos)

        return self.norm(word_emb + pos_emb)

class Feedforward(nn.Module):
    def __init__(self, embed_dim, dff, dropout=0.1):
        super(Feedforward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, dff)
        self.fc2 = nn.Linear(dff, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x




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
    A = torch.matmul(q, k.transpose(-2, -1))/torch.sqrt(torch.tensor(d))
    if attn_mask is not None:
        A = A.masked_fill(attn_mask, 1e-9)
    A = nn.Softmax(dim=-1)(A)
    return torch.matmul(A, v)

'''layer_norm'''
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps= 1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        output = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
                 / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_num = 4, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % head_num == 0
        self.multi_head_dim = embed_dim // head_num
        self.head_num = head_num
        self.embed_dim = embed_dim
        self.w = nn.Linear(embed_dim, 3 * embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask:torch.Tensor):
        """
        :param x: (batch_size, seq_len, embed_dim)
        :param attn_mask: (batch_size, seq_len, seq_len)
        :return: multi_head_attention
        """
        batch_size,  embed_dim = x.shape
        W = self.w(x)

        q, k, v = W[:, : self.embed_dim], W[:, self.embed_dim : 2*self.embed_dim], W[:, 2*self.embed_dim:]
        q = q.view(batch_size, self.head_num, self.multi_head_dim).transpose(1, 2)
        k = k.view(batch_size, self.head_num, self.multi_head_dim).transpose(1, 2)
        v = v.view(batch_size, self.head_num, self.multi_head_dim).transpose(1, 2)

        # mask
        mask = attn_mask.squeeze(1).repeat(1, self.head_num, 1, 1)
        att = self.dropout(attention(q, k, v, self.multi_head_dim, attn_mask=mask)).view(batch_size, -1)
        # feedforward
        return self.fc(att)

class Encoder(nn.Module):
    def __init__(self, embed_dim, head_num, dff=512, dropout=0.1):
        super(Encoder, self).__init__()
        self.multiHeadAttn = MultiHeadAttention(embed_dim, head_num, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = Feedforward(embed_dim, dff, dropout)

    def forward(self, x, pad_mask):

        residual = x

        # norm1 research express the norm before multi-head attention optimizer
        x = self.norm1(x)

        # multiHeadAttention
        x = self.multiHeadAttn(x, pad_mask) + residual

        # norm2
        x = self.norm2(x)

        # ffn
        x = self.ffn(x)

        return x + residual



class Bert(nn.Module):
    def __init__(self, max_vocab, max_len, num_layers, embed_dim, num_heads, d_ff, p_dropout, device):
        super(Bert, self).__init__()
        self.embedding = Embedding(
            vocab_size= max_vocab,
            embed_dim= embed_dim,
            max_len= max_len,
            device= device)
        self.enc_layers = nn.ModuleList([Encoder(
            embed_dim,
            num_heads,
            d_ff,
            p_dropout
        ) for i in range(num_layers)])

        # weight share between classifier and word embedding module
        self.word_emb_shared_weight = self.embedding.word_emb.weight
        self.word_classifier = nn.Linear(embed_dim, max_vocab)
        self.word_classifier.weight = self.word_emb_shared_weight
        self.gelu = nn.GELU()
        self.fc = nn.Linear(embed_dim, embed_dim)


    def forward(self, x, masked_pos):

        """

        :param x: (batch_size, seq_len)
        :param masked_pos: (batch_size, max_pre)
        :return: pool_out(batch_size, hidden_size),classifier(batch_size, classifier_num)
        """
        # embedding (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        output = self.embedding(x)

        # pad_mask (batch_size, seq_len, seq_len)
        pad_mask = get_pad_mask(x)

        # encoder_layers (batch_size, seq_len, embed_dim)
        for encoder in self.enc_layers:
            output = encoder(output, pad_mask)

        # pos_pre （batch_size, max_pre）-> (batch_size, max_pre, embed_dim)
        masked_pos = masked_pos.unsqueeze(-1).expand(-1, -1, output.shape[-1])
        h_masked = torch.gather(output, dim=1, index=masked_pos)
        h_masked = self.gelu(self.fc(h_masked))
        mlm = self.word_classifier(h_masked)

        return output[:, 0, :], mlm




