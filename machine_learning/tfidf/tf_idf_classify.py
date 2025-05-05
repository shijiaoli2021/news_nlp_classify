import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn import tree

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from utils import *


def save_test_res(res, path="./res.csv"):
    df = pd.DataFrame(res, columns=['label'])
    df.to_csv(path, index=False)

train_df = pd.read_csv('./news/train_set.csv', sep='\t')
test_df = pd.read_csv('./news/test_a.csv', sep='\t')

tfidf = TfidfVectorizer(ngram_range=(1,3), token_pattern='\w+', max_features=7000)
train_test = tfidf.fit_transform(pd.concat([train_df['text'], test_df['text']], ignore_index=True))

#
# clf = RandomForestClassifier(n_estimators=5)

#
clf = RidgeClassifier()
train_size = 190000
train_total = 200000
test_total = 50000
clf.fit(train_test[:train_size], train_df['label'].values[:train_size])
val_pred = clf.predict(train_test[train_size:train_total])
# print(val_pred)
print(f1_score(train_df['label'].values[train_size:], val_pred, average='macro'))

# test = tfidf.transform(test_df['text'])
res = clf.predict(train_test[train_total:])
save_test_res([str(item) for item in res])
# 0.87