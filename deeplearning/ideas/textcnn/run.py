#coding=utf-8
import numpy as np
import argparse

TRAIN_PATH = "../../news/train_set.csv"
TEST_PATH = "../../news/test.csv"
TRAIN_DATA_SAVE_PATH = "../../news/"

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("mode", type=str, default="data_preprocess", choices=["data_preprocess", "train", "test"])
    argparse.add_argument("seq_len", type=int, default=128)

    args = argparse.parse_args()
    if args.mode == "data_preprocess":
        from deeplearning.ideas.textcnn.data_preprocess import *
        data_preprocess(TEST_PATH, args.seq_len, TRAIN_DATA_SAVE_PATH, save_keyword='train_split')
    if args.mode == "train":
        pass
    if args.mode == 'test':
        pass

