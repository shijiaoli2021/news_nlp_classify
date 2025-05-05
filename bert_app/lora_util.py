"""
LoRA: we just realize the LoRA for Bert:
we will preprocess the LoRA on the self-attention、linear and the last module
"""

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

def lora_adapter(model, embed_dim, r, alpha, **kwargs):
    for (name, m) in model.named_modules():
        if name.endswith('word_classifier'):
            new_m = LoRALinear(m, embed_dim, embed_dim, r, alpha)
            replace_module(model, name, new_m)
        if name.endswith('multiHeadAttn'):
            new_m = LoraMha(m, embed_dim, 3*embed_dim, r, alpha, [True, False, True])
            replace_module(model, name, new_m)


def mark_only_lora_as_trainable(model:nn.Module):
    for (name, parameter) in model.named_parameters():
        if "lora_" not in name:
            parameter.requires_grad = False


class LoRALayer:
    def __init__(self, r:int, alpha:int, enable_lora:list=None):
        self.r = r
        self.alpha = alpha
        self.enable_lora = enable_lora


class LoRALinear(nn.Module, LoRALayer):
    def __init__(self, model:nn.Module, input_dim, output_dim, r, alpha, enable_lora:list=None):
        super(LoRALinear, self).__init__()
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
            lora = (self.alpha / self.r) *torch.mm(torch.mm(self.lora_A, x), self.lora_B)
            return origin_out + lora
        else:
            i = 0
            origin_out = origin_out.view(-1, out_shape[-1])
            for k in range(len(self.enable_lora)):
                if self.enable_lora[k]:
                    cash = (self.alpha / self.r) * torch.mm(torch.mm(self.lora_A_list[i], x), self.lora_B_list[i])
                    origin_out[:, k * self.seg_dim : (k+1) * self.seg_dim] += cash
                    i += 1
            origin_out = origin_out.reshape(out_shape)
            return origin_out

class LoraMha(nn.Module, LoRALayer):
    def __init__(self, mha:nn.Module, input_dim, output_dim, r, alpha, enable_lora:list=None):
        super(LoraMha, self).__init__()
        self.mha = mha
        # q, k, v (input_dim, 3*output_dim)
        self.mha.w = LoRALinear(mha.w, input_dim, 3 * output_dim, r, alpha, enable_lora=enable_lora)

        # fc lora
        self.mha.fc = LoRALinear(mha.fc, input_dim, output_dim, r, alpha)

    def forward(self, x):
        return self.mha(x)




'''
just for try
'''

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
# R = 4
# ALPHA = 16
# INPUT_DIM = 256
# OUTPUT_DIM = 256
# CLASSIFY_NUM = 14
# MERGE_DIM = 3 * INPUT_DIM
#
# for (name, m) in model.named_modules():
#     if name.endswith('word_classifier'):
#         new_m = LoRALinear(m, INPUT_DIM, CLASSIFY_NUM, R, ALPHA)
#         replace_module(model, name, new_m)
#     if name.endswith('multiHeadAttn'):
#         new_m = LoraMha(m, INPUT_DIM, OUTPUT_DIM, R, ALPHA, [True, False, True])
#         replace_module(model, name, new_m)
#
#
#
# for (name, m) in model.named_modules():
#     if isinstance(m, LoRALayer):
#         print(name + "_lora")
#     if name.endswith('word_classifier'):
#         print(isinstance(m, LoRALayer))

