import torch
import torch.nn as nn

class BertLinear(nn.Module):

    def __init__(self, bert_model, embed_dim, classify_num, dropout = 0.1):
        super(BertLinear, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, classify_num)


    def bert_pr(self, x):

        pool_out, mlm = self.bert_model(x)
        return pool_out


    def forward(self, x):

        # x(batch_size, seq_len)

        # bert model (batch_size, seq_len) -> (batch_size, embed_dim)
        pool_out = self.bert_pr(x)

        #dropout
        pool_out = self.dropout(pool_out)

        # (batch_size, embed_dim) -> (batch_size, classify_num)
        return self.fc(pool_out)
