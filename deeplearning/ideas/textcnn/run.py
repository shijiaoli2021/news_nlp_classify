#coding=utf-8
import numpy as np
import argparse
import torch
from textcnn import *
from vocab import *
from dataloader import *
from trainer import *


TRAIN_PATH = "../../../news/train_set.csv"
TEST_PATH = "../../../news/test_a.csv"
TRAIN_DATA_SAVE_PATH = "../../../news/"
MODEL_SAVE_PATH = "../../checkpoints/"
TEST_MODEL = ""


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--mode", type=str, default="data_preprocess", choices=["data_preprocess", "train", "test"])
    argparse.add_argument("--seq_len", type=int, default=128)
    argparse.add_argument("--batch_size", type=int, default=16)
    argparse.add_argument("--epochs", type=int, default=30)
    argparse.add_argument("--split", type=float, default=0.9)
    argparse.add_argument("--embed_dim", type=int, default=128)
    argparse.add_argument("--ngrams", nargs='+', type=int, default=[2, 3, 4])
    argparse.add_argument("--num_filters", type=int, default=16)
    argparse.add_argument("--classify_num", type=int, default=14)
    argparse.add_argument("--lr", type=float, default=1e-3)
    argparse.add_argument("--eps", type=float, default=2e-4)
    argparse.add_argument("--loss_fn", type=str, default="cross entropy")
    argparse.add_argument("--optimizer", type=str, default="adam")
    argparse.add_argument("--save_best_num", type=int, default=3)


    args = argparse.parse_args()
    if args.mode == "data_preprocess":
        from data_preprocess import *
        data_preprocess(TRAIN_PATH, args.seq_len, TRAIN_DATA_SAVE_PATH, save_keyword='train_split')
    if args.mode == "train":
        # device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load data  (text_num, seq_len)
        data = np.load(TRAIN_DATA_SAVE_PATH + 'train_split' +'.npz')

        # vocab
        text_vocab = Vocab(data_path=TRAIN_PATH)

        # model
        model_param = {
            "vocab_size": text_vocab.get_vocab_len(),
            "embed_dim": args.embed_dim,
            "ngrams": args.ngrams,
            "num_filters": args.num_filters,
            "classify_num": args.classify_num
        }
        model = TextCNN(
            vocab_size=model_param["vocab_size"],
            embed_dim=model_param["embed_dim"],
            ngrams=model_param["ngrams"],
            num_filters=model_param["num_filters"],
            classify_num=model_param["classify_num"]
            ).to(device)

        # dataloader
        dataloader = TextDataLoader(vocab=text_vocab, batch_size=args.batch_size, split=args.split, data=data)

        # trainer
        text_trainer = Trainer(model, dataloader, device, model_param, MODEL_SAVE_PATH, args)

        # train
        text_trainer.train()


    if args.mode == 'test':

        # device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model
        checkpoint = torch.load(TEST_MODEL)
        # initial param
        model_param = checkpoint['model_param']
        # init model
        model = TextCNN(
            vocab_size=model_param["vocab_size"],
            embed_dim=model_param["embed_dim"],
            ngrams=model_param["ngrams"],
            num_filters=model_param["num_filters"],
            classify_num=model_param["classify_num"]
            ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # test data
        test_data = pd.read_csv(args.test_path)


