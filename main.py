import torch
import numpy as np
import pandas as pd
import argparse
from model import NLP_Classify
from torch.utils.data import TensorDataset, dataloader
from tqdm import *
from torch.nn.utils.rnn import pad_sequence
import math

TRAIN_DATA = "./news/train_set.csv"
TEST_DATA = "./test_a.csv"

def get_model(args):
    return NLP_Classify(**vars(args))

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

def collate_fn(batch):
    x, y = batch
    return pad_sequence(x, batch_first=True), y


def buildDataSet(path, args):
    data = getData(path)
    print("load data:{}".format(data.shape))
    data_X, data_Y = text2vec(data)
    train_size = int(args.split * data.shape[0])
    train_data, train_label = data_X[:train_size], np.array(data_Y[:train_size], dtype=np.int8)
    test_data, test_label = data_X[train_size:], np.array(data_Y[train_size:], dtype=np.int8)
    # print(train_data)
    # print(train_label.shape)
    train = TensorDataset(torch.FloatTensor(train_data), torch.FloatTensor(train_label))
    test = TensorDataset(torch.FloatTensor(test_data), torch.FloatTensor(test_label))
    return dataloader.DataLoader(train, batch_size=args.batch_size, shuffle=True), dataloader.DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

def data_pre(data, num_embedding):
    res = []
    print("###data pre begin#####")
    for item in tqdm(data):
        res.append(data_pos_propocess(item, num_embedding))
    print("####data pre over#####")
    
    return np.array(res)


def data_pos_propocess(text, num_embedding):
    firstWordDict = {}
    out = np.zeros((num_embedding), dtype=np.float64)
    words = [int(word) for word in text.split(' ')]
    for i in range(len(words)):
        if words[i] not in firstWordDict.keys():
            firstWordDict[words[i]] = i
            if i % 2 == 0:
                out[words[i]] += (1 + math.sin((i) / (10000**(words[i] / num_embedding))))
            else:
                out[words[i]] += (1 + math.cos((i) / (10000**(words[i] / num_embedding))))
        else:
            if i % 2 == 0:
                out[words[i]] += (1 + math.sin((i - firstWordDict[words[i]]) / (10000**(words[i] / num_embedding))))
            else:
                out[words[i]] += (1 + math.cos((i - firstWordDict[words[i]]) / (10000**(words[i] / num_embedding))))
    return out



def tf_idf():
    pass

def cal_f1(pre, label, classify_num):
    f1_list = []
    pre, label = np.array(pre), np.array(label)
    for i in range(classify_num):
        pre_idx = np.where(pre == i)
        label_idx = np.where(label == i)
        # print("pre:{}, label:{}".format(pre, label))
        intersct = np.intersect1d(pre_idx, label_idx)
        pos_cnt = len(np.intersect1d(pre_idx, label_idx))
        if (pos_cnt == 0):
            f1_list.append(0)
            continue
        precision = float(pos_cnt / np.sum(pre == i))
        recall = float(pos_cnt / np.sum(label == i))
        f1 = float(2*precision * recall / (precision + recall))
        f1_list.append(f1)
    return np.mean(np.array(f1_list))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_embedding", type=int)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--classify_num", type=int, default=14)
    parser.add_argument("--head_num", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", type=float, default=0.8)


    args = parser.parse_args()
    device = torch.device('cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu')
    model = get_model(args)
    model = model.to(device)

    # dataset
    train_loader, test_loader = buildDataSet(TRAIN_DATA, args)

    # print("--------------test")
    # for (x, y) in test_loader:
    #     print(x.shape)
    #     print(y.shape)
    # print("--------------test")

    Loss = torch.nn.CrossEntropyLoss()

    optim = torch.optim.Adam(model.parameters(), lr = args.learning_rate)


    print("-----------------train begin-----------------")

    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for (x, y) in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            batch_size, _ = x.shape
            optim.zero_grad()
            # (batch, num_embedding)
            pre = model(x)
            target = torch.zeros(batch_size, args.classify_num)
            target[:, y.to(torch.int)] = 1
            target = target.to(device)
            loss = Loss(pre, target)
            total_loss += loss
            loss.backward()
            optim.step()

        print("train {} enpoch over,, total_loss{}, test begins".format(epoch, total_loss))

        model.eval()
        with torch.no_grad():
            preList = []
            labelList = []
            for (x, y) in test_loader:
                x, y = x.to(device), y.to(device)
                pre = model(x)
                pre_idx = [item.item() for item in torch.argmax(pre, dim=1)]
                preList += pre_idx
                labelList += [item.item() for item in y]
            f1 = cal_f1(preList, labelList, args.classify_num)
            print("{}enpoch test f1 mean:{}".format(epoch, f1))



