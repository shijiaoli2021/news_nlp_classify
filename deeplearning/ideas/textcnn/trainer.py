#coding=utf-8

import torch
from deeplearning.ideas.textcnn.dataloader import *
from tqdm import *
import os
from sklearn.metrics import f1_score
import utils.utils as utils


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
    def __init__(self, model, dataLoader: TextDataLoader, device, model_param, model_save_path, args):
        self.model = model
        self.dataLoader = dataLoader
        self.epochs = args.epochs
        self.loss_fn = load_loss_fn(args)
        self.optimizer = load_optimizer(args, self.model)
        self.device = device
        self.total_loss_list = []
        self.save_best_num = args.save_best_num
        self.model_param = model_param
        self.model_save_path = model_save_path
        self.save_dict = {}
        self.args = args

    """
    train for epochs
    """

    def train(self):
        self.print_fn()
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            self.validate(epoch)

    def train_one_epoch(self, epoch):
        self.dataLoader.mode = "train"
        self.model.train()
        total_loss = 0
        with tqdm(self.dataLoader.data_iter(), total=self.dataLoader.get_len(), desc=f"epoch:{epoch}") as tq:
            for (batch, label) in tq:
                batch, label = torch.tensor(batch, dtype=torch.int32).to(self.device), torch.tensor(label,
                                                                                                    dtype=torch.int64).to(
                    self.device)
                out = self.model(batch)
                self.optimizer.zero_grad()
                loss = self.loss_fn(out, label)
                total_loss += loss
                loss.backward()
                self.optimizer.step()
                tq.set_postfix({"loss": f"{loss:.4f}"})
        self.total_loss_list.append(total_loss)

    def validate(self, epoch):
        self.dataLoader.mode = "val"
        self.model.eval()
        print(f"validate begins, last total loss:{self.total_loss_list[-1]:.4f}")
        pre_list = []
        true_list = []

        # val
        with tqdm(self.dataLoader.data_iter(), total=self.dataLoader.get_len(), desc=f"epoch:{epoch}") as tq:
            for (batch, label) in tq:
                batch, label = torch.tensor(batch, dtype=torch.int32).to(self.device), torch.tensor(label,
                                                                                                    dtype=torch.int64).to(
                    self.device)
                out = self.model(batch)
                pre_list += [pre_value.item() for pre_value in torch.argmax(out, dim=-1)]
                true_list += [true_value.item() for true_value in label]

        # cal f1
        f1 = f1_score(true_list, pre_list, average="macro")

        # save model preprocess
        self.save_preprocess(f1, epoch)

        print(f"validate over, f1_score:{f1:.4f}")

    def save_preprocess(self, f1_score, epoch):
        if len(self.save_dict) < self.save_best_num:
            save_name = f"{type(self.model).__name__}epoch{epoch}"
            self.save_dict[save_name] = f1_score
            torch.save({"model_state_dict": self.model.state_dict(), "model_param": self.model_param},
                       self.model_save_path + save_name + ".pth")
            return

        if f1_score < min(self.save_dict.values()):
            return

        save_name = f"{type(self.model).__name__}epoch{epoch}"
        self.save_dict[save_name] = f1_score

        # 保存较好的模型训练参数和初始化参数
        torch.save({"model_state_dict": self.model.state_dict(), "model_param": self.model_param},
                   self.model_save_path + save_name + ".pth")

        # 获取当前排在最后的模型保存名称，并在指定路径中删除
        min_f1_save_name = min(self.save_dict, key=self.save_dict.get)
        os.remove(self.model_save_path + min_f1_save_name + ".pth")
        self.save_dict.pop(min_f1_save_name)

    def test_for_res(self):
        self.dataLoader.mode = "test_for_res"
        pre_list = np.zeros((50000, self.args.classify_num))
        with tqdm(self.dataLoader.data_iter(), total=self.dataLoader.get_len()) as tq:
            for (batch, idx) in tq:
                # test
                batch = torch.tensor(batch, dtype=torch.int64).to(self.device)

                out = self.model(batch)

                pre_cash = [pre_item.item() for pre_item in torch.argmax(out, -1)]

                for i in range(len(idx)):
                    pre_list[idx[i], pre_cash[i]] += 1
        pre_for_save = np.argmax(pre_list, axis=1).astype(str).tolist()

        utils.out(pre_for_save, self.model_save_path + "res.csv")

    def print_fn(self):
        pass
