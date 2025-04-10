# coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

class Vocab:
    """
    将学习文本转换为词袋：（id：word），并记录每个词出现的数量，出现的文章数量可以计算文章TF-IDF
    用于为实时生成指定输入数据前的数据处理
    """
    def __init__(
            self,
            data_path,
            build_tfidf=False,
            min_df=1,
            max_df=1.0,
            token_pattern='\w+',
            stop_words=None,
            ngram_range=(1, 1)):
        self.min_df = min_df
        self.max_df = max_df
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.special_vectors = ['<UNK>', '<PAD>']
        self.loadVocab(data_path)
        self.build_tfidf = build_tfidf
        if self.build_tfidf:
            self.cal_tfidf()


    # 加载词袋
    def loadVocab(self, data_path):
        data = pd.read_csv(data_path, sep="\t")
        self.countVectorizer = CountVectorizer(min_df=self.min_df,
                                               max_df=self.max_df,
                                               token_pattern=self.token_pattern,
                                               stop_words=self.stop_words,
                                               ngram_range=self.ngram_range)
        self.sparse_text_feature = self.countVectorizer.fit_transform(raw_documents=data['text'])
        vector_len = len(self.countVectorizer.vocabulary_)
        self.special_vector2id = {'<UNK>': vector_len, '<PAD>': vector_len + 1}

    def word2id(self, word):
        if word in self.countVectorizer.vocabulary_:
            return self.countVectorizer.vocabulary_[word]
        if word in self.special_vectors:
            return self.special_vector2id[word]
        return self.special_vector2id[self.special_vectors[0]]

    def cal_tfidf(self):
        from sklearn.feature_extraction.text import TfidfTransformer
        tfidfTransformer = TfidfTransformer()
        self.sparse_word_tfidf = tfidfTransformer.fit_transform(self.sparse_text_feature)

    def word2tfidf(self, word, text_id):
        assert self.build_tfidf
        if word in self.countVectorizer.vocabulary_:
            return self.sparse_word_tfidf[(text_id, word)]

    def get_vocab_len(self):
        return len(self.countVectorizer.vocabulary_) + 2


# text = ["110,2 3 4 5 6 7 8", "9 9 10, 10"]
# y = [1, 2]
# vectorizer = CountVectorizer(token_pattern='\w+')
# x = vectorizer.fit_transform(raw_documents=text)
# from sklearn.feature_extraction.text import TfidfTransformer
# sparse_tfidf = TfidfTransformer().fit_transform(x)
# print(sparse_tfidf[(0, 8)])
# print(x)
# print(x.toarray())
