#coding=utf-8

import torch
from deeplearning.ideas.textcnn.dataloader import *
from tqdm import *
from sklearn.metrics import f1_score


def load_loss_fn(args):
    if args.loss_fn == "cross entropy":
         return torch.nn.CrossEntropyLoss()
    return None

def load_optimizer(args, model):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    return None

"""训练器"""
class Trainer:
    def __init__(self, model, dataLoader: TextDataLoader, device, args):
        self.model = model
        self.dataLoader = dataLoader
        self.epochs = args.epochs
        self.loss_fn = load_loss_fn(args)
        self.optimizer = load_optimizer(args, self.model)
        self.device = device
        self.total_loss_list = []

    """
    train for epochs
    """
    def train(self):
        self.print_fn()
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            self.validate()


    def train_one_epoch(self, epoch):
        self.dataLoader.mode = "train"
        self.model.train()
        total_loss = 0
        for (batch, label) in tqdm(self.dataLoader.data_iter(), desc=f"epoch:{epoch}"):
            batch, label = torch.FloatTensor(batch).to(self.device), torch.FloatTensor(label).to(self.device)
            out = self.model(batch)
            self.optimizer.zero_grad()
            loss = self.loss_fn(out, label)
            total_loss += loss
            loss.backward()
            self.optimizer.step()
            tqdm.set_postfix_str(f"Loss: {loss:.4f}")
        self.total_loss_list.append(total_loss)


    def validate(self):
        self.dataLoader.mode = "val"
        self.model.eval()
        print(f"validate begins, last total loss:{self.total_loss_list[-1]:.4f}")
        pre_list = []
        true_list = []
        for (batch, label) in tqdm(self.dataLoader.data_iter(), desc="validate"):
            batch, label = torch.FloatTensor(batch).to(self.device), torch.FloatTensor(label).to(self.device)
            out = self.model(batch)
            pre_list += [pre_value.item() for pre_value in torch.argmax(out, dim=-1)]
            true_list += [true_value.item() for true_value in label]
        f1 = f1_score(true_list, pre_list)
        print(f"validate over, f1_score:{f1:.4f}")


    def print_fn(self):
        pass
