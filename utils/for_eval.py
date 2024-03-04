import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

import torch

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

def accuracy(model, ds, device, OOD_ds=None):
    truth, pred_cluster = [], []
    model.eval()
    with torch.no_grad():
        for x, y, _ in ds:
            x = x.to(device)
            truth.append(y)
            q = model(x)
            pred_cluster.append(q.max(1)[1].cpu())
        if OOD_ds:
            for x, y, _ in OOD_ds:
                x = x.to(device)
                truth.append(torch.full(y.shape, 2))
                q = model(x)
                pred_cluster.append(q[:, 1].cpu())
    # y_pred = [0 if p < 0.5 else 1 for p in torch.cat(pred_cluster)]
    confusion_m = confusion_matrix(torch.cat(truth).numpy(), torch.cat(pred_cluster).numpy())
    _, col_idx = linear_sum_assignment(confusion_m, maximize=True)
    acc = np.trace(confusion_m[:, col_idx]) / confusion_m.sum()

    return acc

def print_cm(model, ds, device, save_dir, OOD_ds=None):
    truth, pred_cluster = [], []
    model.eval()
    with torch.no_grad():
        for x, labels, _ in ds:
            x = x.to(device)
            labels = torch.tensor([0 if y == 0 else 1 for y in labels])
            truth.append(labels)
            q = model(x)
            pred_cluster.append(q.max(1)[1].cpu())
        if OOD_ds:
            for x, y, _ in OOD_ds:
                x = x.to(device)
                truth.append(torch.full(y.shape, 2))
                q = model(x)
                pred_cluster.append(q.max(1)[1].cpu())

    confusion_m = confusion_matrix(torch.cat(truth).numpy(), torch.cat(pred_cluster).numpy())
    _, col_idx = linear_sum_assignment(confusion_m, maximize=True)
    confusion_m_reordered = confusion_m[:, col_idx]
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_m_reordered)
    disp.plot()
    plt.savefig(f"{save_dir}/cm.png")

    return confusion_m_reordered

