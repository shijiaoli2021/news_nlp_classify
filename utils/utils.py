import numpy as np
import pandas as pd
import math
import csv



def getData(path, shuffle= False):
    data = np.array(pd.read_csv(path, sep='\t'))
    if shuffle:
        np.random.shuffle(data)
    return data

def text2vec(data):
    data_X, data_Y = data[:, 1], data[:, 0]
    for i in range(data_X.shape[0]):
        word_list = [int(word) for word in data[i].split(' ')]
        data_X[i] = np.array(word_list)
    return data_X, data_Y

def out(res, path="./res.csv"):
    headers = ['label']
    with open(path, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(res)


