import torch
from tqdm import tqdm


def print_fn(desc, mode: int = 0):
    if mode == 0:
        print(f"{desc} start...")
    else:
        print(f"{desc} end...")

class AbstractTrainer(object):
    def __init__(
            self,
            model,
            model_param,
            device,
            train_loader,
            valid_loader,
            test_loader,
            loss_fn,
            eval_fn,
            optimizer,
            args):
        self.model = model
        self.device = device
        self.model_param = model_param
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.steps = args.start_steps
        self.args = args
        self.eval_fn = eval_fn
        self.total_loss_list = []

    def train(self):
        # hook function
        self.before_train()

        self.model.train()

        print_fn("train", 0)

        for epoch in range(self.args.epochs):
            self._train_one_epoch(epoch)
            self.valid_preprocess(epoch)
        print_fn("train", 1)

        self.after_train()

    def valid_preprocess(self, epoch):
        pass

    def valid(self, epoch):
        self.model.eval()
        print_fn("val", 0)
        loss_list = []
        eval_list = []
        with tqdm(self.valid_loader) as tq:
            for batch in tq:
                # acquire data
                x, label = self.parse_input(batch)

                # get out from model
                out = self.model(x)

                # parse out
                pre = self.parse_out(out)

                # cal loss
                loss = self.loss_fn(pre, label)
                loss_list.append(loss.item())

                # cal eval
                eval_res = self.eval_fn(pre, label)
                eval_list.append(eval_res.item())

                tq.set_postfix_str(f"eval:{eval_res:.4f}")

        avr_eval = sum(eval_list) / len(eval_list)
        avr_loss = sum(loss_list) / len(loss_list)
        print_fn(f"valid eval:{avr_eval},valid loss:{avr_loss:.4f}", 1)
        # save for valid
        self.save_model_for_valid(val_loss=avr_eval, epoch=epoch)

    def test(self):
        model = self.load_best_model()
        if model is None:
            print("no model to test, pass...")
        model.eval()
        print_fn("test", 0)
        loss_list = []
        eval_list = []
        with tqdm(self.valid_loader) as tq:
            for batch in tq:
                # acquire data
                x, label = self.parse_input(batch)

                # get out from model
                out = model(x)

                # parse out
                pre = self.parse_out(out)

                # cal loss
                loss = self.loss_fn(pre, label)
                loss_list.append(loss.item())

                # cal eval
                eval_res = self.eval_fn(pre, label)
                eval_list.append(eval_res.item())

                tq.set_postfix_str(f"eval:{eval_res:.4f}")

        avr_eval = sum(eval_list) / len(eval_list)
        avr_loss = sum(loss_list) / len(loss_list)
        print_fn(f"test eval:{avr_eval:.4f} loss:{avr_loss:.4f}", 1)

    def _train_one_epoch(self, epoch):
        total_loss = 0
        with tqdm(self.train_loader, desc=f"epoch:{epoch}") as tq:
            for batch in tq:
                # zero_grad()
                self.optimizer.zero_grad()

                # acquire data
                x, label = self.parse_input(batch)

                # get pre from model
                out = self.model(x)

                # parse out
                pre = self.parse_out(out)

                # loss
                loss = self.loss_fn(pre, label)

                # update total loss
                total_loss += loss.item()

                # tqdm
                tq.set_postfix_str(f"loss:{loss:.4f}")

                # backward
                loss.backward()

                # step
                self.optimizer.step()

                # update steps
                self.steps += 1

        self.total_loss_list.append(total_loss)
        # save for train
        self.save_model_for_train(epoch)

    def parse_input(self, data)->(torch.Tensor, torch.Tensor):
        return data

    def parse_out(self, data) -> torch.Tensor:
        return data

    def save_model_for_train(self, epoch):
        pass

    def save_model_for_valid(self, val_loss, epoch):
        pass

    def before_train(self):
        pass

    def after_train(self):
        pass

    def load_best_model(self):
        return self.model




