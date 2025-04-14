import torch
from tf_idf import Model
from tqdm import *
from utils.cal_precsion import *

class Trainer:
    def __init__(self, model: Model, optim, loss_fn, trainLoader, testLoader, device: torch.device, args):
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.loss_fn = loss_fn
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.device = device
        self.args = args

    def train(self):

        print("-----------------train begin-----------------")

        for epoch in range(self.args.epochs):
            total_loss = 0
            self.model.train()
            for (x, y) in tqdm(self.trainLoader):
                x, y = x.to(self.device), y.to(self.device)
                batch_size, _ = x.shape
                self.optim.zero_grad()
                # (batch, num_embedding)
                pre = self.model(x)
                target = torch.zeros(batch_size, self.args.classify_num)
                target[:, y.to(torch.int)] = 1
                target = target.to(self.device)
                loss = self.loss_fn(pre, target)
                total_loss += loss
                loss.backward()
                self.optim.step()

            print("train {} epoch over,, total_loss{}".format(epoch, total_loss))

            if epoch % self.args.gap != 0:
                continue
            print("#####test begins#####")

            self.model.eval()
            with torch.no_grad():
                preList = []
                labelList = []
                for (x, y) in self.testLoader:
                    x, y = x.to(self.device), y.to(self.device)
                    pre = self.model(x)
                    pre_idx = [item.item() for item in torch.argmax(pre, dim=1)]
                    preList += pre_idx
                    labelList += [item.item() for item in y]
                f1 = cal_f1(preList, labelList, self.args.classify_num)
                print("{}enpoch test f1 mean:{}".format(epoch, f1))