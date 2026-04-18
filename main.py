import os
import argparse
import numpy as np
import torch
import json

from load_data import PID_DATA
from run import train, test
from utils import load_model, try_makedirs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Object(object):
    pass


def train_one_dataset(
        params,
        train_q, train_qa, train_pid,
        valid_q, valid_qa, valid_pid
):
    model = load_model(params)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    best_auc = 0
    best_epoch = 0

    for epoch in range(params.max_iter):
        train_loss, train_acc, train_auc = train(
            model, params, optimizer,
            train_q, train_qa, train_pid, label="Train"
        )

        valid_loss, valid_acc, valid_auc = test(
            model, params, optimizer,
            valid_q, valid_qa, valid_pid, label="Valid"
        )

        print(
            f"[Epoch {epoch + 1}] "
            f"Train AUC={train_auc:.4f} | "
            f"Valid AUC={valid_auc:.4f}"
        )

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch + 1

            try_makedirs("model")
            torch.save(model.state_dict(), "model/best_model.pt")

        if epoch - best_epoch > 30:
            print("Early stopping.")
            break

    return best_epoch


def test_one_dataset(params, test_q, test_qa, test_pid):
    model = load_model(params)
    model.load_state_dict(torch.load("model/best_model.pt", map_location=device))

    test_loss, test_acc, test_auc = test(
        model, params, None,
        test_q, test_qa, test_pid, label="Test"
    )

    print(f"\n[Test] AUC={test_auc:.4f}, ACC={test_acc:.4f}")
    return test_auc, test_acc


def run_pipeline(train_path, valid_path, test_path):
    params = Object()

    params.max_iter = 200
    params.batch_size = 24
    params.lr = 1e-4
    params.seed = 2024
    params.d_model = 256
    params.n_block = 1
    params.dropout = 0.1
    params.kq_same = 1
    params.l2 = 1e-5
    params.seqlen = 200

    np.random.seed(params.seed)
    torch.manual_seed(params.seed)

    dat = PID_DATA(n_question=10000, seqlen=params.seqlen, remap_ids=True)
    train_q, train_qa, train_pid = dat.load_data(train_path)
    valid_q, valid_qa, valid_pid = dat.load_data(valid_path)
    test_q, test_qa, test_pid = dat.load_data(test_path)

    params.n_question = len(dat.qid_map)
    params.n_pid = len(dat.pid_map)

    dat2 = PID_DATA(n_question=params.n_question, seqlen=params.seqlen, remap_ids=True)
    dat2.qid_map = dat.qid_map
    dat2.pid_map = dat.pid_map

    train_q, train_qa, train_pid = dat2.load_data(train_path)
    valid_q, valid_qa, valid_pid = dat2.load_data(valid_path)
    test_q, test_qa, test_pid = dat2.load_data(test_path)

    os.makedirs("model", exist_ok=True)
    with open("model/id_maps.json", "w", encoding="utf-8") as f:
        json.dump(
            {"qid_map": dat2.qid_map, "pid_map": dat2.pid_map},
            f, ensure_ascii=False
        )
    print("Saved id maps to model/id_maps.json")

    print("开始进行核心张量运算与模型训练...")
    best_epoch = train_one_dataset(
        params,
        train_q, train_qa, train_pid,
        valid_q, valid_qa, valid_pid
    )

    test_auc, test_acc = test_one_dataset(params, test_q, test_qa, test_pid)

    return test_auc, test_acc


if __name__ == "__main__":
    print("检测到本地直接运行，开始读取 data/my_data/ 目录下的数据...")

    data_dir = "data/my_data"
    local_train = os.path.join(data_dir, "my_data_train.csv")
    local_valid = os.path.join(data_dir, "my_data_valid.csv")
    local_test = os.path.join(data_dir, "my_data_test.csv")

    try:
        auc, acc = run_pipeline(local_train, local_valid, local_test)
        print("=" * 50)
        print("整个流程运行完毕！")
        print("=" * 50)
    except FileNotFoundError:
        print(f"报错：找不到数据文件！请检查 {data_dir} 下是否有那三个 csv 文件。")