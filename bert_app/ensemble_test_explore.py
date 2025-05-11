import sklearn
import sklearn.ensemble as ensemble
from transformers.agents.evaluate_agent import classifier
import numpy as np
import utils.utils as utils
from sklearn.metrics import f1_score

RANK_OUT_PATH = "./checkpoints/checkpoint4/"

data_path_dict = {
            "train": "./checkpoints/checkpoint4/train.npz",
            "val": "./checkpoints/checkpoint4/val.npz",
            "test": "./checkpoints/checkpoint4/test.npz"
        }

mode = "xgboost"

if __name__ == '__main__':

    print("loading data...")
    train_data = np.load(data_path_dict["train"])
    val_data = np.load(data_path_dict["val"])
    train_x, train_y = train_data["train_x"], train_data["train_y"]
    val_x, val_y = val_data["val_x"], val_data["val_y"]
    test_x = np.load(data_path_dict["text"])
    print("loading data successfully")

    if mode == "xgboost":
        import xgboost as xgb
        classifier = xgb.XGBClassifier(n_estimators=300,
                                       tree_method="hist",
                                       early_stopping_rounds=30,
                                       learning_rate=0.05,
                                       gamma=0.5,
                                       reg_lambda=2,
                                       reg_alpha=0,
                                       max_depth=10,
                                       )

    elif mode == "RandomForestClassifier":
        classifier = ensemble.RandomForestClassifier(n_estimators=100)

    else:
        raise Exception(f"unsupported mode:{mode}")

    # fit
    classifier.fit(train_x, train_y, eval_set=[(val_x, val_y)])

    # predict
    pre = classifier.predict(val_x)
    print(f1_score(val_y, pre, average='macro'))

    pre = classifier.predict(test_x)

    utils.rank_out(pre, RANK_OUT_PATH + "res_" + mode + ".csv")

