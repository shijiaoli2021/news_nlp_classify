import torch
import argparse
import bert_model
import bert_config
import pandas as pd
from dataset import BertDataset
from count_vocab import Vocab
from torch.utils.data.dataloader import DataLoader
from trainer import BertTrainer

SEED = 42
DATA_PATH1 = "../news/train_set.csv"
DATA_PATH2 = "../news/test_a.csv"

def get_data():
    data1 = pd.read_csv(DATA_PATH1, sep="\t")
    data2 = pd.read_csv(DATA_PATH2, sep="\t")
    return pd.concat([data1['text'], data2['text']], axis=0)

def build_loader(bert_vocab: Vocab, bert_args):
    data = get_data()
    train_size = int(len(data) * bert_args.train_split)
    val_size = int(len(data) * bert_args.val_split)
    return (build_one_loader(data[:train_size], bert_vocab, bert_args),
            build_one_loader(data[train_size: train_size+val_size], bert_vocab, bert_args),
            build_one_loader(data[train_size+val_size:], bert_vocab, bert_args))

def build_one_loader(data, vocab, args, shuffle=True):
    dataset = BertDataset(data, vocab)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
    return dataloader

if __name__ == '__main__':

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--valid_interval", type=int, default=1)
    parser.add_argument("--save_steps_interval", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--model_on_path", type=str, default="")
    parser.add_argument("--start_steps", type=str, default=0)


    # random seed
    torch.cuda.manual_seed(SEED)

    # build args
    args = parser.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model param
    model_param = {
        "max_vocab": bert_config.max_vocab,
        "embed_dim": bert_config.embed_dim,
        "max_len": bert_config.max_len,
        "device": device,
        "num_heads": bert_config.num_heads,
        "d_ff": bert_config.d_ff,
        "p_dropout": bert_config.p_dropout,
        "num_layers": bert_config.num_layers
    }

    if args.model_on_path == "":
        # model
        model = bert_model.Bert(**model_param).to(device)

    else:
        # load model
        checkpoint = torch.load(args.model_on_path)
        model_param = checkpoint["model_param"]
        model = bert_model.Bert(**model_param).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])


    # vocab
    vocab = Vocab()
    vocab.load_data2vocab(DATA_PATH1)
    vocab.load_data2vocab(DATA_PATH2)
    print(f"loading word to vocab successfully, total {vocab.get_len()} word")

    #dataloader
    trainLoader, valLoader, testLoader = build_loader(vocab, args)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.word2idx(vocab.get_pad_word()))

    # trainer
    trainer = BertTrainer(
        model,
        model_param= model_param,
        device=device,
        train_loader=trainLoader,
        valid_loader=valLoader,
        test_loader=testLoader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args
    )

    trainer.train()

