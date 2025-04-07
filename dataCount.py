import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRAIN_DATA = "./news/train_set.csv"
TEST_DATA = "./test_a.csv"


def getData(path):
    data = np.array(pd.read_csv(path, sep='\t'))
    np.random.shuffle(data)
    return data

def text2vec(data):
    data_X, data_Y = data[:, 1], data[:, 0]
    for i in range(data_X.shape[0]):
        word_list = [int(word) for word in data[i].split(' ')]
        data_X[i] = np.array(word_list)
    return data_X, data_Y

def textCount(data):
    len = data.shape[0]
    label = ['0', '50', '100', '150', '200', '250', '300']
    textCnt = [0 for i in range(len(label))]
    for i in range(len):
        cnt = data[i].shape[0]
        if cnt > 300:
            textCnt[label.index('300')] += 1
        elif cnt > 250:
            textCnt[label.index('250')] += 1
        elif cnt > 200:
            textCnt[label.index('200')] += 1
        elif cnt > 150:
            textCnt[label.index('150')] += 1
        elif cnt > 100:
            textCnt[label.index('100')] += 1
        elif cnt > 50:
            textCnt[label.index('50')] += 1
        else:
            textCnt[label.index('0')] += 1
    return label, textCnt


if __name__ == '__main__':
    data = getData(TRAIN_DATA)
    data_X, data_Y = text2vec(data)
    label, textCnt = textCount(data_X)
    for i in range(len(label)):
        plt.bar(i, textCnt[i], label=label[i])
    plt.savefig('./figures/textCnt.png')
    plt.show()
