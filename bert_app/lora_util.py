import torch
import torch.nn as nn
from bert_pretrain.bert_model import Bert
from bert_pretrain import bert_config
from peft import LoraConfig, get_peft_model, TaskType


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_param = {
        "max_vocab": bert_config.max_vocab,
        "embed_dim": bert_config.embed_dim,
        "max_len": bert_config.max_len,
        "device": device,
        "num_heads": bert_config.num_heads,
        "d_ff": bert_config.d_ff,
        "p_dropout": bert_config.p_dropout,
        "num_layers": bert_config.num_layers
    }

model = Bert(**model_param)

lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1
)

model = get_peft_model(model, lora_config)

