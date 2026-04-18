import numpy as np
import torch
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_entropy(target, pred):
    loss = target * torch.log(pred + 1e-8) + \
           (1 - target) * torch.log(1 - pred + 1e-8)
    return -loss.mean()


def train(model, params, optimizer, q_data, qa_data, pid_data, label="Train"):
    model.train()

    batch_size = params.batch_size
    n_batch = int(np.ceil(q_data.shape[0] / batch_size))

    total_loss = 0
    y_true, y_pred = [], []

    for idx in range(n_batch):
        optimizer.zero_grad()

        q = torch.LongTensor(
            q_data[idx * batch_size:(idx + 1) * batch_size]
        ).to(device)

        qa = torch.LongTensor(
            qa_data[idx * batch_size:(idx + 1) * batch_size]
        ).to(device)

        pid = torch.LongTensor(
            pid_data[idx * batch_size:(idx + 1) * batch_size]
        ).to(device)

        pred = model(q, qa, pid)

        mask = (q > 0)
        target = (qa > params.n_question).float()

        pred = pred[mask]
        target = target[mask]

        loss = binary_entropy(target, pred)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        y_true.extend(target.detach().cpu().numpy())
        y_pred.extend(pred.detach().cpu().numpy())

    acc = metrics.accuracy_score(
        np.array(y_true) > 0.5,
        np.array(y_pred) > 0.5
    )
    auc = metrics.roc_auc_score(y_true, y_pred)

    return total_loss / n_batch, acc, auc


def test(model, params, optimizer, q_data, qa_data, pid_data, label="Test"):
    model.eval()

    batch_size = params.batch_size
    n_batch = int(np.ceil(q_data.shape[0] / batch_size))

    total_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for idx in range(n_batch):
            q = torch.LongTensor(
                q_data[idx * batch_size:(idx + 1) * batch_size]
            ).to(device)

            qa = torch.LongTensor(
                qa_data[idx * batch_size:(idx + 1) * batch_size]
            ).to(device)

            pid = torch.LongTensor(
                pid_data[idx * batch_size:(idx + 1) * batch_size]
            ).to(device)

            pred = model(q, qa, pid)

            mask = (q > 0)
            target = (qa > params.n_question).float()

            pred = pred[mask]
            target = target[mask]

            loss = binary_entropy(target, pred)
            total_loss += loss.item()

            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    acc = metrics.accuracy_score(
        np.array(y_true) > 0.5,
        np.array(y_pred) > 0.5
    )
    auc = metrics.roc_auc_score(y_true, y_pred)

    return total_loss / n_batch, acc, auc
