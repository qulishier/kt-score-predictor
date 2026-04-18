import os
import torch
from akt import AKT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def try_makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_model(params):
    model = AKT(
        n_question=params.n_question,
        n_pid=params.n_pid,
        n_blocks=params.n_block,
        d_model=params.d_model,
        dropout=params.dropout,
        kq_same=params.kq_same,
        model_type='akt',
        l2=params.l2
    ).to(device)
    return model
