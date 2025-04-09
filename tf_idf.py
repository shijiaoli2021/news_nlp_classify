from utils.utils import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import torch
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
from deeplearning import trainer

TRAIN_DATA = "./news/train_set.csv"
TEST_DATA = "./test_a.csv"


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.linear2(self.linear1(x))
        return self.softmax(out)


class TextDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        return item

    def __len__(self):
        return len(self.data)


def data2tfidf(path):
    data = getData(TRAIN_DATA, shuffle=True)
    vector = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vector.fit_transform(data[:, 1]))
    return tfidf, data[:, 0]

def buildDataLoader(data, label, split= 0.8):
    text_len = data.shape[0]
    train_size = int(text_len * split)
    trainDataset = TextDataset(torch.FloatTensor(data[:train_size]), torch.FloatTensor(label[:train_size]))
    testDataset = TextDataset(torch.FloatTensor(data[train_size:]), torch.FloatTensor(label[train_size:]))
    trainDataloader = dataloader.DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)
    testDataloader = dataloader.DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)
    return trainDataloader, testDataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--classify_num", type=int, default=14)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", type=float, default=0.8)
    args = parser.parse_args()
    tfidfData, label = data2tfidf(TRAIN_DATA)
    text_len, word_num = tfidfData.shape
    device = torch.device("cuda:"+ str(args.gpu) if torch.cuda.is_available() else "cpu")
    model = Model(word_num, args.classify_num, args.hidden_dim).to(device)

    # dataset
    trainDataLoader, testDataLoader = buildDataLoader(tfidfData, label, args.split)

    # optim
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # loss
    Loss = torch.nn.CrossEntropyLoss()

    # trainer
    trainer = trainer.Trainer(model, optim, Loss, trainDataLoader, testDataLoader, device, args)