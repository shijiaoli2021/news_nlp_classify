from torchtext.vocab import *
from collections import Counter, OrderedDict


# text = ['a', 'b', 'c', 'd', 'e', 'f', 'd', 'a', 'c', 'e']
#
# counter = Counter(text)
# # print(counter.most_common())
# vocab = Vocab(counter)
# FastText()

# examples = ['chip', 'baby', 'Beautiful']
# vec = GloVe(name='6B', dim=50)
# ret = vec.get_vecs_by_tokens(examples, lower_case_backup=True)
# print(ret)