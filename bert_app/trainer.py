import typing

import torch
from a_trainer import AbstractTrainer
import os
import utils.utils as utils
from tqdm import tqdm
from bert_linear import BertLinear
from bert_pretrain.bert_model import Bert
import numpy as np
import lora_util
import lora_config

MODEL_SAVE_PATH = "./checkpoints/checkpoint3/"
RANK_OUT_PATH = "./checkpoints/checkpoint3/"


class Trainer(AbstractTrainer):
    def __init__(
        self,
        model: BertLinear,
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
            model=model,
            model_param=model_param,
            device=device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            eval_fn=eval_fn,
            optimizer=optimizer,
            args=args
        )
        self.model = model
        self.save_dict = {}
        self.save_best_num = args.save_best_num
        self.rank_loader = rank_loader
        self.stacking_model_name = None
        # fine_tuning_for_lora
        if self.args.fine_tuning_for_lora:
            print("build lora adapter for bert start...")
            lora_util.mark_only_lora_as_trainable(self.model.bert_model, bias='all')
            print("build lora adapter over...")

    def parse_input(self, data) -> (torch.Tensor, torch.Tensor):
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
            save_name = f"{type(self.model).__name__}" + (
                "_lora_" if self.args.fine_tuning_for_lora else "") + f"epoch{epoch}_{self.steps}"
            checkpoint = {"model_state_dict": self._model_state_dict(), "model_param": self.model_param,
                          "pretrained_model_path": self.args.pretrained_model_path}
            if self.args.fine_tuning_for_lora:
                checkpoint["fine_tuning_module"] = self.model.fc.state_dict()
            torch.save(checkpoint, MODEL_SAVE_PATH + save_name + ".pth")
            # test
            self.generate_rank_file()
            self.model.train()
            print("save model successfully...")

    def _model_state_dict(self):
        if not self.args.fine_tuning_for_lora:
            return self.model.state_dict()
        return lora_util.lora_state_dict(self.model, bias='all')

    def _save_preprocess(self, f1_score, epoch):
        if len(self.save_dict) < self.save_best_num:
            save_name = f"{type(self.model).__name__}" + (
                "_lora_" if self.args.fine_tuning_for_lora else "") + f"epoch{epoch}"
            self.save_dict[save_name] = f1_score
            checkpoint = {"model_state_dict": self._model_state_dict(), "model_param": self.model_param,
                          "pretrained_model_path": self.args.pretrained_model_path}
            if self.args.fine_tuning_for_lora:
                checkpoint["fine_tuning_module"] = self.model.fc.state_dict()
            torch.save(checkpoint, MODEL_SAVE_PATH + save_name + ".pth")
            return

        if f1_score < min(self.save_dict.values()):
            return

        save_name = f"{type(self.model).__name__}" + (
            "_lora_" if self.args.fine_tuning_for_lora else "") + f"epoch{epoch}"
        self.save_dict[save_name] = f1_score

        # 保存较好的模型训练参数和初始化参数
        checkpoint = {"model_state_dict": self._model_state_dict(), "model_param": self.model_param,
                      "pretrained_model_path": self.args.pretrained_model_path}
        if self.args.fine_tuning_for_lora:
            checkpoint["fine_tuning_module"] = self.model.fc.state_dict()
        torch.save(checkpoint, MODEL_SAVE_PATH + save_name + ".pth")

        # 获取当前排在最后的模型保存名称，并在指定路径中删除
        min_f1_save_name = min(self.save_dict, key=self.save_dict.get)
        os.remove(MODEL_SAVE_PATH + min_f1_save_name + ".pth")
        self.save_dict.pop(min_f1_save_name)

    def generate_rank_file(self):

        if self.rank_loader is None:
            return
        model = self.load_best_model()
        model.eval()
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
        utils.rank_out(pre_list, RANK_OUT_PATH + "res" + "_" + str(self.steps) + ".csv")

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
        model_checkpoint = torch.load(MODEL_SAVE_PATH + max_score_model_path + ".pth")

        # model param
        model_param = model_checkpoint["model_param"]

        if self.args.fine_tuning_for_lora:
            pretrain_model = self.model.bert_model
            lora_util.to_lora_adapter(pretrain_model, lora_config.lora_adapter_info)
        else:
            # pretrained_model
            pretrain_model = Bert(**model_param["pretrained_param"])
        # load model
        model = BertLinear(bert_model=pretrain_model, **model_param)
        model.load_state_dict(model_checkpoint["model_state_dict"])
        if self.args.fine_tuning_for_lora:
            model.fc.load_state_dict(model_checkpoint["fine_tuning_module"])

        print(f"loading the best model successfully, eval score:{self.save_dict[max_score_model_path]:.4f}")

        return model.to(self.device)

    def set_stacking_model_name(self, name):
        self.stacking_model_name = name

    def _get_pretrain_model_stacking(self, dataloader):

        self.model.eval()
        x_list = []
        label_list = []
        for data in tqdm(dataloader):
            x, y = self.parse_input(data)
            x_list.append(self.model.bert_out(x).detach().cpu().numpy())
            label_list.append(y.detach().cpu().numpy())
        return np.concatenate(x_list, axis=0), np.concatenate(label_list, axis=0)

    def stacking_model(self, generate_rank_file=True, data_path_dict:typing.Dict=None, save_compute_path:str=None):
        if data_path_dict is None:
            train_x, train_y = self._get_pretrain_model_stacking(self.train_loader)
            val_x, val_y = self._get_pretrain_model_stacking(self.valid_loader)
        else:
            train_data = np.load(data_path_dict["train"])
            val_data = np.load(data_path_dict["val"])
            train_x, train_y = train_data["train_x"], train_data["train_y"]
            val_x, val_y = val_data["val_x"], val_data["val_y"]

        from sklearn.metrics import f1_score
        if self.stacking_model_name == 'RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier

            # build randomForestClassifier
            classifier = RandomForestClassifier(n_estimators=100)
        elif self.stacking_model_name == "RidgeClassifier":
            from sklearn.linear_model import RidgeClassifier

            # build randomForestClassifier
            classifier = RidgeClassifier()

        elif self.stacking_model_name == "GBDT":
            from sklearn.ensemble import GradientBoostingClassifier
            classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

        else:
            raise NotImplementedError

        # fit
        classifier.fit(train_x, train_y)

        # predict
        pre = classifier.predict(val_x)
        print(f1_score(val_y, pre, average='macro'))

        # generate out file
        if generate_rank_file:
            if data_path_dict is None:
                test_x, _ = self._get_pretrain_model_stacking(self.rank_loader)
            else:
                test_x = np.load(data_path_dict["rank"])
            pre = classifier.predict(test_x)
            utils.rank_out(pre, RANK_OUT_PATH + "res_" + self.stacking_model_name + ".csv")
        else:
            test_x = None

        if save_compute_path is not None:
            np.savez(save_compute_path + "train.npz", train_x=train_x, train_y=train_y)
            np.savez(save_compute_path + "val.npz", val_x = val_x, val_y = val_y)
            np.savez(save_compute_path + "test.npz", data = test_x)

    # def before_train(self):
    #     print("test val ...")
    #     self.valid(0)





