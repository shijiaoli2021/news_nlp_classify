# 最大序列长度
max_len = 128

# 字典最大大小
max_vocab = 7000

# mask时最大mask数量
max_pre = 10

# k和q的维度
d_k=d_v=64
d_model = 768
d_ff = d_model * 4

num_heads = 12
num_layers = 6

num_seg = 2

p_dropout = 0.1

mask_ratio = 0.15
p_mask = 0.8
p_replace = 0.1
p_do_nothing = 1 - p_mask - p_replace
