import torch
import argparse
from bert_app.bert_linear import BertLinear
from bert_pretrain import bert_model
from bert_pretrain import count_vocab
from bert_app.trainer import Trainer
from torchmetrics.classification import MulticlassF1Score
from bert_app.dataset import BertAppDataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import lora_util
import lora_config

pretrained_model_path = ""
TRAIN_PATH = ""
TEST_PATH = ""

model_param = {
    "embed_dim": 256,
    "classify_num": 14
}

def build_dataloader(data_path, vocab, args):
    data = pd.read_csv(data_path, sep="\t")
    length = len(data)
    train_size = int(args.train_split * length)
    val_size = int(args.val_split * length)
    return (
        build_one_dataloader(data[:train_size], vocab, args.batch_size),
        build_one_dataloader(data[train_size:train_size+val_size], vocab, args.batch_size, shuffle=False),
        build_one_dataloader(data[train_size+val_size:], vocab, args.batch_size, shuffle=False))


def build_rank_dataloader(data_path, text_vocab, batch_size):
    rank_data = pd.read_csv(data_path, sep="\t", index_col=True)
    rank_data.rename(columns={"Index":"label"}, inplace=True)
    return build_one_dataloader(rank_data, text_vocab, batch_size, label_index="label", shuffle=False)

def build_one_dataloader(data, text_vocab, batch_size, input_index="text", label_index="label", shuffle=True):
    dataset = BertAppDataset(data, text_vocab, input_index=input_index, label_index=label_index)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)



if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test", "rank out", "app_stacking"], default="train")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--classify_num", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--valid_interval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--model_on_path", type=str, default="")
    parser.add_argument("--start_steps", type=int, default=0)
    parser.add_argument("--save_best_num", type=int, default=3)
    parser.add_argument("--test_after_train", type=bool, default=True)
    parser.add_argument("--save_steps_interval", type=int, default=10000)
    parser.add_argument("--fine_tuning_for_lora", type=bool, default=False)
    parser.add_argument("--pretrained_model_path", type=str, default="")

    args = parser.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_on_path == "" or args.model_on_path is None:
        # pretrained model
        pretrained_model_checkpoint = torch.load(pretrained_model_path)
        pretrained_model = bert_model.Bert(**pretrained_model_checkpoint['model_param'])
        pretrained_model.load_state_dict(pretrained_model_checkpoint['model_state_dict'])

        # adapter for model
        if args.fine_tuning_for_lora:
            lora_util.to_lora_adapter(pretrained_model, lora_config.lora_adapter_info)

        # prediction model
        model = BertLinear(pretrained_model, **model_param).to(device)
        model_param["pretrained_param"] = pretrained_model_checkpoint["model_param"]

    else:
        checkpoint = torch.load(args.model_on_path)
        model_param = checkpoint["model_param"]
        if not args.fine_tuning_for_lora:
            # prediction model
            pretrained_model = bert_model.Bert(**model_param["pretrained_param"])

            # model
            model = BertLinear(pretrained_model, **model_param)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # pretrained model
            pretrained_model_checkpoint = torch.load(pretrained_model_path)
            pretrained_model = bert_model.Bert(**pretrained_model_checkpoint['model_param'])

            # model
            model = BertLinear(pretrained_model, **model_param)

            # lora adapter
            lora_util.to_lora_adapter(model, lora_config.lora_adapter_info)

            # lora_state_dict
            model.load_state_dict(checkpoint["model_state_dict"])

            # pretrain_model_state_dic
            model.load_state_dict(pretrained_model_checkpoint['model_state_dict'])

        model = model.to(device)


    # vocab
    vocab = count_vocab.Vocab()
    vocab.load_data2vocab(TRAIN_PATH)
    vocab.load_data2vocab(TEST_PATH)

    # optimizer: optimize for only trainable parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # loss_fn
    loss_fn = torch.nn.CrossEntropyLoss()

    # eval_fn
    eval_fn = MulticlassF1Score(num_classes=args.classify_num, average="macro").to(device)

    # dataloader
    train_loader, val_loader, test_loader = build_dataloader(TRAIN_PATH, vocab, args)
    rank_loader = build_rank_dataloader(TEST_PATH, vocab, args.batch_size)

    # build trainer
    trainer = Trainer(
        model,
        model_param,
        device,
        train_loader,
        val_loader,
        test_loader,
        rank_loader,
        loss_fn,
        eval_fn,
        optimizer,
        args
    )

    if args.mode == 'train':
        # train the model
        trainer.train()

        # train over, rank out
        trainer.generate_rank_file()

    elif args.mode == 'test':
        # test the model
        trainer.test()

    elif args.mode == 'rank_out':
        # rank out
        trainer.generate_rank_file()

    elif args.mode == 'app_stacking':
        trainer.set_stacking_model_name("RandomForestClassifier")
        trainer.stacking_model()




