import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn import tree

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


train_df = pd.read_csv('./news/train_set.csv', sep='\t', nrows=50000)

tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])

clf = RandomForestClassifier(n_estimators=5)
clf = tree.DecisionTreeClassifier(criterion="gini", max_features=3000)
clf.fit(train_test[:40000], train_df['label'].values[:40000])

val_pred = clf.predict(train_test[40000:])
print(f1_score(train_df['label'].values[40000:], val_pred, average='macro'))