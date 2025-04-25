import torch
from torch.utils.checkpoint import checkpoint

from a_trainer import AbstractTrainer
import os
import utils.utils as utils
from tqdm import tqdm
from bert_linear import BertLinear
from bert_pretrain.bert_model import Bert

MODEL_SAVE_PATH = "./checkpoints/checkpoint1/"
RANK_OUT_PATH = "./checkpoints/checkpoint1/"


class Trainer(AbstractTrainer):
    def __init__(
            self,
            model,
            model_param,
            device,
            train_loader,
            valid_loader,
            test_loader,
            rank_loader,
            loss_fn,
            eval_fn,
            optimizer,
            args):
        super(Trainer, self).__init__(
            model = model,
            model_param = model_param,
            device = device,
            train_loader = train_loader,
            valid_loader = valid_loader,
            test_loader = test_loader,
            loss_fn = loss_fn,
            eval_fn=eval_fn,
            optimizer = optimizer,
            args = args
        )
        self.save_dict = {}
        self.save_best_num = args.save_best_num
        self.rank_loader = rank_loader

    def parse_input(self, data) ->(torch.Tensor, torch.Tensor):
        x, y = [item.to(self.device) for item in data]
        return x, y


    def parse_out(self, data) -> torch.Tensor:
        return data

    def valid_preprocess(self, epoch):
        if epoch % self.args.valid_interval == 0:
            self.valid(epoch)

    def save_model_for_valid(self, val_loss, epoch):
        self._save_preprocess(val_loss, epoch)

    def save_model_for_train(self, epoch):
        if self.steps % self.args.save_steps_interval == 0:
            print(f"the model has trained {self.steps} steps, saving...")
            save_name = f"{type(self.model).__name__}epoch{epoch}"
            torch.save({"model_state_dict": self.model.state_dict(), "model_param": self.model_param},
                       MODEL_SAVE_PATH + save_name + f"_{self.steps}" + ".pth")
            print("save model successfully...")


    def _save_preprocess(self, f1_score, epoch):
        if len(self.save_dict) < self.save_best_num:
            save_name = f"{type(self.model).__name__}epoch{epoch}"
            self.save_dict[save_name] = f1_score
            torch.save({"model_state_dict": self.model.state_dict(), "model_param": self.model_param},
                       MODEL_SAVE_PATH + save_name + ".pth")
            return

        if f1_score < min(self.save_dict.values()):
            return

        save_name = f"{type(self.model).__name__}epoch{epoch}"
        self.save_dict[save_name] = f1_score

        # 保存较好的模型训练参数和初始化参数
        torch.save({"model_state_dict": self.model.state_dict(), "model_param": self.model_param},
                   MODEL_SAVE_PATH + save_name + ".pth")

        # 获取当前排在最后的模型保存名称，并在指定路径中删除
        min_f1_save_name = min(self.save_dict, key=self.save_dict.get)
        os.remove(MODEL_SAVE_PATH + min_f1_save_name + ".pth")
        self.save_dict.pop(min_f1_save_name)

    def generate_rank_file(self):

        if self.rank_loader is None:
            return
        model = self.load_best_model()
        # run model for rank
        print("run model to generate rank files begin...")
        pre_list = []
        with tqdm(self.rank_loader) as tq:
            for batch in tq:
                # acquire input for model
                x, _ = self.parse_input(batch)

                # model -> (batch_size, classify_num)
                pre = model(x)

                # argmax(pre)
                pre_cash_idx = torch.argmax(pre, dim=1)

                pre_list += [pre_idx.item() for pre_idx in pre_cash_idx]

        # generate rank files
        utils.rank_out(pre_list, RANK_OUT_PATH)


    def after_train(self):
        if self.args.test_after_train:
            self.test()


    def load_best_model(self):

        print("loading the best model...")
        if len(self.save_dict) == 0:
            return self.model
        # acquire the best model
        max_score_model_path = max(self.save_dict, key=self.save_dict.get)

        # load best model
        model_checkpoint = torch.load(MODEL_SAVE_PATH + max_score_model_path + "pth")

        # model param
        model_param = model_checkpoint["model_param"]

        # pretrained_model
        pretrain_model = Bert(**model_param["pretrained_param"])

        # load model
        model = BertLinear(bert_model=pretrain_model, **model_param)
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"loading the best model successfully, eval score:{self.save_dict[max_score_model_path]:.4f}")

        return model

    # def before_train(self):
    #     self.valid(0)






