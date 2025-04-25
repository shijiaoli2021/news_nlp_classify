# coding=utf-8
import fasttext
import numpy as np
import pandas as pd 
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

import utils.utils
from utils import *

SPLIT = 0.9
TRAIN = True
TEST = True

data = pd.read_csv('./news/train_set.csv', sep='\t')
split_size =int( data.shape[0] * SPLIT)
train_df = data[:split_size]
valid_df = data[split_size:]
test_df = pd.read_csv('./news/test_a.csv',sep='\t')

train_df['label_ft'] = '__label__' + train_df['label'].astype(str) #__label__number
valid_df['label_ft'] = '__label__' + valid_df['label'].astype(str) #__label__number

X_train = train_df['text']
y_train = train_df['label']
X_test = test_df['text']
X_valid = valid_df['text']
if TRAIN:
    train_df[['text','label_ft']].to_csv('fasttext_train_df.csv', header=None, index=False, sep='\t')
    # ģ�͹���
    model = fasttext.train_supervised('fasttext_train_df.csv', lr=0.1, epoch=30, wordNgrams=5,
                                      verbose=2, minCount=1, loss='hs')
    model.save_model('fasttext'+str(1)+'.bin')
    # ģ��Ԥ��
    clf = fasttext.load_model('fasttext'+str(1)+'.bin')
    ##val_pred = [int(model.predict(x)[0][0].split('__')[-1]) for x in X_train.iloc[valid_index]]
    ##print('Fasttext׼ȷ��Ϊ��',f1_score(list(y_train.iloc[valid_index]), val_pred, average='macro'))
    ##print(classification_report(list(y_train.iloc[valid_index]),val_pred,digits=4,target_names = ['�Ƽ�', '��Ʊ', '����', '����', 'ʱ��', '���', '����',  '�ƾ�','�Ҿ�',  '��Ϸ',  '����',  'ʱ��',  '��Ʊ','����']))
    # 预测
    td_pred = [int(clf.predict(x)[0][0].split('__')[-1]) for x in X_valid]
    print('f1:',f1_score(valid_df['label'].values, td_pred, average='macro'))
    print(classification_report(valid_df['label'].values,td_pred,digits=4,target_names = ['�Ƽ�', '��Ʊ', '����', '����', 'ʱ��', '���', '����',  '�ƾ�','>    �Ҿ�',  '��Ϸ',  '����',  'ʱ��',  '��Ʊ','����']))

if TEST:
    clf = fasttext.load_model('fasttext'+str(1)+'.bin')
    test_pr = [clf.predict(x)[0][0].split('__')[-1] for x in X_test]
    utils.utils.rank_out(test_pr, path='./fasttext_pre_res.csv')
    #test_pred = np.column_stack((test_pred, test_pred_))  # �������кϲ�