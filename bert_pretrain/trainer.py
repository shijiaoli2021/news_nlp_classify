import torch
from tqdm import *
import os


MODEL_SAVE_PATH = ""


class BertTrainer:
    def __init__(self, model, model_param, device, train_loader, valid_loader, test_loader, loss_fn, optimizer, args):
        self.model = model
        self.model_param = model_param
        self.device = device
        self.trainLoader = train_loader
        self.validLoader = valid_loader
        self.testLoader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.args = args
        self.steps = args.start_steps
        self.loss_list = []
        self.max_vocab = model_param["max_vocab"]


    def train(self):
        for epoch in range(self.args.epochs):

            # train one epoch
            self._train_one_epoch(epoch)

            # valid
            if epoch % self.args.valid_interval == 0:
                self.valid()

        # test:
        self.test()

        # save_model



    def valid(self):

        self.model.eval()

        print("validation start...")

        valid_loss_list = []

        with tqdm(self.validLoader, desc="valid") as tq:

            for batch in tq:
                # data mask_pos(batch_size, max_pre) mask_token(batch_size, max_pre) input_data(batch_size, seq_len)
                input_data, mask_pos, mask_token = [data.to(self.device) for data in batch]

                # model  pool_out(batch_size, hidden_size), mlm(batch_size, max_pre, vocab_size)
                pool_out, mlm = self.model(input_data, mask_pos)

                valid_loss = self.loss_fn(mlm.view(-1, self.max_vocab), mask_token.view(-1))

                valid_loss_list.append(valid_loss.item())

        valid_loss_avr = float(sum(valid_loss_list) / len(valid_loss_list))

        print(f"validation over, average loss:{valid_loss_avr:.4f}")


    def test(self):
        self.model.eval()

        print("test start...")

        test_loss_list = []

        with tqdm(self.testLoader, desc="valid") as tq:
            for batch in tq:
                # data mask_pos(batch_size, max_pre) mask_token(batch_size, max_pre) input_data(batch_size, seq_len)
                input_data, mask_pos, mask_token = [data.to(self.device) for data in batch]

                # model  pool_out(batch_size, hidden_size), mlm(batch_size, max_pre, vocab_size)
                pool_out, mlm = self.model(input_data, mask_pos)

                test_loss = self.loss_fn(mlm.view(-1, self.max_vocab), mask_token.view(-1))

                test_loss_list.append(test_loss.item())

        test_loss_avr = float(sum(test_loss_list) / len(test_loss_list))

        print(f"test over, average loss:{test_loss_avr:.4f}")



    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        with tqdm(self.trainLoader, desc=f"epoch:{epoch}") as tq:
            for (idx, batch) in enumerate(tq):

                # data mask_pos(batch_size, max_pre) mask_token(batch_size, max_pre) input_data(batch_size, seq_len)
                input_data, mask_pos, mask_token = [data.to(self.device) for data in batch]

                # model  pool_out(batch_size, hidden_size), mlm(batch_size, max_pre, vocab_size)
                pool_out, mlm = self.model(input_data, mask_pos)

                # loss
                mlm_loss = self.loss_fn(mlm.view(-1, self.max_vocab), mask_token.view(-1))

                total_loss += mlm_loss.item()

                # zero_grad
                self.optimizer.zero_grad()

                # backward
                mlm_loss.backward()

                # optim step
                self.optimizer.step()

                if idx % 2 == 0:
                    tq.set_postfix_str(f"loss:{mlm_loss}")

                # update and check train steps
                self.steps += 1

                # save model
                self.save_model()

        self.loss_list.append(total_loss)

    def save_model(self):
        if self.steps % self.args.save_steps_interval == 0:
            print(f"the perfect bert has trained for {self.steps} steps, saving...")

            # 保存较好的模型训练参数和初始化参数
            model_name = "mini_bert_" + str(self.steps) +".pth"
            save_path = os.path.join(MODEL_SAVE_PATH, model_name)
            torch.save({"model_state_dict": self.model.state_dict(), "model_param": self.model_param}, save_path)
            print("saving successfully...")



