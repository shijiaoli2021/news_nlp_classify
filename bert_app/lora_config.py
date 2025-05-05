import bert_pretrain.bert_config as bert_config
from lora_util import LoraEmbedding, LoraMha, LoRAAdapter

lora_adapter_info = {
    "embedding": {
        "adapter_class": LoraEmbedding,
        "adapter_param": {
            "num_embeddings": bert_config.max_vocab,
            "embed_dim": bert_config.embed_dim,
            "r": 2,
            "alpha": 4
        }
    },

    "multiHeadAttn": {
        "adapter_class": LoraMha,
        "adapter_param": {
            "input_dim": bert_config.embed_dim,
            "output_dim": bert_config.embed_dim,
            "r": 4,
            "alpha": 12,
            "enable_lora": [True, False, True]
        }
    },

    "word_classifier": {
        "adapter_class": LoRAAdapter,
        "adapter_param": {
            "input_dim": bert_config.embed_dim,
            "output_dim": bert_config.embed_dim,
            "r": 4,
            "alpha": 12,
            "enable_lora": None
        }
    }
}