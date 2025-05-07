"""
LoRA: we just realize the LoRA for Bert:
we will preprocess the LoRA on the self-attention、linear and the last module
"""
from typing import Dict

import torch
import torch.nn as nn


"""
replace the module
"""
def replace_module(model, module_name, new_module):
    names = module_name.split(".")
    sub_module = model
    for name in names[:-1]:
        sub_module = getattr(sub_module, name)
    setattr(sub_module, names[-1], new_module)

def to_lora_adapter(model, adapter_info:dict=None):
    if adapter_info is None:
        return
    for replace_name in adapter_info.keys():
        for (name, m) in model.named_modules():
            if name.endswith(replace_name):
                new_m = adapter_info[replace_name]['adapter_class'](m ,**adapter_info[replace_name]['adapter_param'])
                replace_module(model, name, new_m)


def mark_only_lora_as_trainable(model:nn.Module, bias:str="none"):
    for (name, parameter) in model.named_parameters():
        if "lora_" not in name:
            parameter.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for (name, parameter) in model.named_parameters():
            if "bias" in name:
                parameter.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


class LoRALayer:
    def __init__(self, r:int, alpha:int, enable_lora:list=None):
        self.r = r
        self.alpha = alpha
        self.enable_lora = enable_lora


class LoRAAdapter(nn.Module, LoRALayer):
    def __init__(self, model:nn.Module, input_dim, output_dim, r, alpha, enable_lora:list=None):
        super(LoRAAdapter, self).__init__()
        LoRALayer.__init__(self, r, alpha, enable_lora)
        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lora_A = None
        self.lora_B = None
        self.lora_A_list = None
        self.lora_B_list = None
        self.seg_dim = None
        self.init_lora()

    def init_lora(self):
        if self.enable_lora is None:
            self.lora_A = nn.Parameter(torch.rand(size=(self.input_dim, self.r)))
            self.lora_B = nn.Parameter(torch.zeros(size=(self.r, self.output_dim)))
            return
        assert self.output_dim % len(self.enable_lora) == 0
        self.seg_dim = self.output_dim // len(self.enable_lora)
        self.lora_A_list = nn.ParameterList([nn.Parameter(torch.rand(size=(self.input_dim, self.r))) for enable in self.enable_lora if enable])
        self.lora_B_list = nn.ParameterList([nn.Parameter(torch.zeros(size=(self.r, self.seg_dim))) for enable in self.enable_lora if enable])

    def forward(self, x):

        origin_out = self.model(x)
        out_shape = origin_out.shape
        if self.enable_lora is None:
            lora = (self.alpha / self.r) *torch.matmul(torch.matmul(x, self.lora_A), self.lora_B)
            return origin_out + lora
        else:
            i = 0
            origin_out = origin_out.view(-1, out_shape[-1])
            x = x.view(-1, self.input_dim)
            for k in range(len(self.enable_lora)):
                if self.enable_lora[k]:
                    cash = (self.alpha / self.r) * torch.matmul(torch.matmul(x, self.lora_A_list[i]), self.lora_B_list[i])
                    origin_out[:, k * self.seg_dim : (k+1) * self.seg_dim] += cash
                    i += 1
            origin_out = origin_out.reshape(out_shape)
            return origin_out

class LoraEmbedding(nn.Module, LoRALayer):
    def __init__(self, embedding_module, num_embeddings, embed_dim, r:int=0, alpha:int=1, enable_lora:list=None):
        super(LoraEmbedding, self).__init__()
        LoRALayer.__init__(self, r, alpha, enable_lora)
        self.embedding_module = embedding_module
        if self.r > 0:
            self.lora_A = nn.Embedding(num_embeddings, r)
            self.lora_B = nn.Parameter(torch.zeros(size=(r, embed_dim)))
            self.init_lora_parameter()

    def init_lora_parameter(self):
        nn.init.normal_(self.lora_A.weight, std = 0.01)

    def forward(self, x):

        # origin out
        origin_out = self.embedding_module(x)

        lora_out = (self.alpha / self.r) * (self.lora_A(x) @ self.lora_B)

        return origin_out + lora_out



class LoraMha(nn.Module, LoRALayer):
    def __init__(self, mha:nn.Module, input_dim, output_dim, r, alpha, enable_lora:list=None):
        super(LoraMha, self).__init__()
        self.mha = mha
        # q, k, v (input_dim, 3*output_dim)
        self.mha.w = LoRAAdapter(mha.w, input_dim, 3 * output_dim, r, alpha, enable_lora=enable_lora)

        # fc lora
        self.mha.fc = LoRAAdapter(mha.fc, input_dim, output_dim, r, alpha)

    def forward(self, x, attn_mask):
        return self.mha(x, attn_mask)





'''
just for try
'''
# import bert_pretrain.bert_config as bert_config
# from bert_pretrain.bert_model import Bert
# from bert_app.bert_linear import BertLinear
# #device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model_param = {
#         "max_vocab": bert_config.max_vocab,
#         "embed_dim": bert_config.embed_dim,
#         "max_len": bert_config.max_len,
#         "device": device,
#         "num_heads": bert_config.num_heads,
#         "d_ff": bert_config.d_ff,
#         "p_dropout": bert_config.p_dropout,
#         "num_layers": bert_config.num_layers
#     }
#
# bert_linear_model_param = {
#     "embed_dim": 256,
#     "classify_num": 14
# }
#
# bert = Bert(**model_param)
#
# model = BertLinear(bert, **bert_linear_model_param)
#
# lora_adapter_info = {
#     "embedding": {
#         "adapter_class": LoraEmbedding,
#         "adapter_param": {
#             "num_embeddings": 7000,
#             "embed_dim": 256,
#             "r": 2,
#             "alpha": 4
#         }
#     },
#
#     "multiHeadAttn": {
#         "adapter_class": LoraMha,
#         "adapter_param": {
#             "input_dim": 256,
#             "output_dim": 256,
#             "r": 4,
#             "alpha": 12,
#             "enable_lora": [True, False, True]
#         }
#     },
#
#     "word_classifier": {
#         "adapter_class": LoRAAdapter,
#         "adapter_param": {
#             "input_dim": bert_config.embed_dim,
#             "output_dim": bert_config.embed_dim,
#             "r": 4,
#             "alpha": 12,
#             "enable_lora": None
#         }
#     }
# }
#
# to_lora_adapter(model, lora_adapter_info)
#
# for (name, m) in model.named_parameters():
#     print(name)
