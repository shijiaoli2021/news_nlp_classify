import numpy as np

def cal_f1(pre, label, classify_num):
    f1_list = []
    pre, label = np.array(pre), np.array(label)
    for i in range(classify_num):
        pre_idx = np.where(pre == i)
        label_idx = np.where(label == i)
        # print("pre:{}, label:{}".format(pre, label))
        pos_cnt = len(np.intersect1d(pre_idx, label_idx))
        if (pos_cnt == 0):
            f1_list.append(0)
            continue
        precision = float(pos_cnt / np.sum(pre == i))
        recall = float(pos_cnt / np.sum(label == i))
        f1 = float(2*precision * recall / (precision + recall))
        f1_list.append(f1)
    return np.mean(np.array(f1_list))