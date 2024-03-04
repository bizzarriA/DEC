from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_score
from matplotlib import pyplot as plt
import pandas as pd

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from tqdm import tqdm

from .nn import get_p
from .for_eval import accuracy
import numpy as np
from scipy.optimize import linear_sum_assignment
import wandb

DATA_PLOT = None


def normalize(X):
    return torch.exp(1 - X)


def set_data_plot(tr_ds, test_ds, device):
    global DATA_PLOT

    # select 100 sample per class
    tr_x, tr_y = [], []
    count = torch.zeros(15, dtype=torch.int)
    for batch in tr_ds:
        for data, label, id in zip(*batch):
            if count[label] < 100:
                tr_x.append(data[None])
                tr_y.append(label[None])
            count[label] += 1
    tr_x, tr_y = torch.cat(tr_x).to(device), torch.cat(tr_y).to(device)

    # select 100 sample per class
    test_x, test_y = [], []
    count = torch.zeros(15, dtype=torch.int)
    if test_ds:
        for batch in test_ds:
            for data, label, _ in zip(*batch):
                if count[label] < 100:
                    test_x.append(data[None])
                    test_y.append(label[None])
                count[label] += 1
        test_x, test_y = torch.cat(test_x).to(device), torch.cat(test_y).to(device)

    DATA_PLOT = {'train': (tr_x, tr_y),
                 'test': (test_x, test_y)}


def get_initial_center(model, ds, device, n_cluster):
    # fit
    print('\nbegin fit kmeans++ to get initial cluster centroids ...')

    model.eval()
    with torch.no_grad():
        feature = []
        for x, _, _ in ds:
            x = x.to(device)
            feature.append(model.encoder(x).cpu())

    kmeans = KMeans(n_cluster).fit(torch.cat(feature).numpy())
    center = Parameter(torch.tensor(kmeans.cluster_centers_,
                                    device=device,
                                    dtype=torch.float))

    return center


def get_best_initial_center(model, ds, device, max_cluster):
    print('\nbegin fit kmeans++ to get initial cluster centroids using silhouette score ...')

    model.eval()
    with torch.no_grad():
        feature = []
        i = 0
        for x, _, _ in ds:
            x = x.to(device)
            feature.append(model.encoder(x).cpu())
            if i == len(ds):
                break

    feature_tensor = torch.cat(feature).numpy()

    silhouette_scores = []

    for n_clusters in tqdm(range(2, max_cluster + 1)):
        kmeans = KMeans(n_clusters=n_clusters).fit(feature_tensor)
        labels = kmeans.labels_
        score = silhouette_score(feature_tensor, labels)
        silhouette_scores.append(score)

    best_n_clusters = np.argmax(silhouette_scores) + 2  # add 2 because we start from 2

    print(
        f"the best number of cluster is {best_n_clusters} with Silhouette Score of {silhouette_scores[best_n_clusters - 2]}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_cluster + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score per numero di cluster')
    plt.xlabel('Numero di cluster')
    plt.ylabel('Silhouette Score')
    plt.savefig('silhouette_score.png')

    kmeans = KMeans(n_clusters=best_n_clusters).fit(feature_tensor)

    center = Parameter(torch.tensor(kmeans.cluster_centers_,
                                    device=device,
                                    dtype=torch.float))

    return center


def map_clusters_to_labels(true_labels, predicted_clusters):
    true_labels = torch.argmax(true_labels, dim=1).cpu().numpy()
    predicted_clusters_max = torch.argmax(predicted_clusters, dim=1).cpu().numpy()

    unique_true_labels = [i for i in range(10)]
    unique_predicted_clusters = [i for i in range(10)]
    cost_matrix = confusion_matrix(true_labels, predicted_clusters_max)
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    mapping = {unique_predicted_clusters[j]: unique_true_labels[i] for i, j in zip(row_ind, col_ind)}

    indices_mapped = [mapping.get(k, 0) for k in unique_predicted_clusters]

    mapped_labels = predicted_clusters[:, indices_mapped]

    return mapped_labels


def pretrain(model, opt, ds, device, epochs, save_dir, is_wandb=True):
    print('begin train AutoEncoder ...', flush=True)
    loss_fn = nn.MSELoss()
    n_batch = len(ds)
    model.train()
    loss_h = History('min')

    # train AE
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}:', flush=True)
        print('-' * 10)
        loss = 0.
        for i, (x, y, _) in enumerate(ds, 1):
            opt.zero_grad()
            x = x.to(device)
            _, gen = model(x)
            batch_loss = loss_fn(x, gen)
            batch_loss.backward()
            opt.step()
            loss += batch_loss
            print(f'{i}/{n_batch}', end='\r')

        loss /= n_batch
        loss_h.add(loss)
        if loss_h.better:
            torch.save(model, f'{save_dir}/fine_tune_AE.pt')
        # if epoch % 10 == 0:
        #     plot_rec(model, x, save_dir, epoch)
        if is_wandb:
            wandb.log({"loss_AE": loss.item()})
        print(f'loss : {loss.item():.4f}  min loss : {loss_h.best.item():.4f}')
        print(f'lr: {opt.param_groups[0]["lr"]}')


def train(model, opt, ds, device, epochs, save_dir, alpha=1, beta=1, omega=1, label_name='Benign', is_wandb=True):
    print('begin train DEC ...')

    loss_fn_kld = nn.KLDivLoss(reduction='batchmean')
    n_sample, n_batch = len(ds.dataset), len(ds)
    loss_h, acc_cluster_h = History('min'), History('max')

    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}:', flush=True)
        print('-' * 10)
        model.train()
        closs = 0.
        kloss = 0.
        clusterloss = 0.
        for i, (x, labels, _) in enumerate(ds, 1):
            opt.zero_grad()
            x = x.to(device)
            labels = labels.to(device)
            y = torch.tensor([0 if y == label_name else 1 for y in labels])
            q = model(x)

            # kl divergence
            loss_kld = alpha * loss_fn_kld(q.log(), get_p(q))

            # contrastive loss
            centroids = model.cluster.center
            num_clusters = len(centroids)
            centroid_distances = torch.sum(torch.cdist(centroids, centroids, p=num_clusters)) / (
                    num_clusters * (num_clusters - 1))
            loss_cluster = beta * 1 / centroid_distances

            # classification loss
            y = nn.functional.one_hot(y, num_classes=num_clusters).float().to(device)
            loss_cls = omega * torch.nn.functional.cross_entropy(y, q)

            # total loss
            batch_loss = loss_kld + loss_cluster + loss_cls
            batch_loss.backward()
            opt.step()
            closs += loss_cls
            kloss += loss_kld
            clusterloss += loss_cluster
            print(f'{i}/{n_batch}', end='\r')

        loss = kloss + clusterloss + closs
        loss /= n_batch
        loss_h.add(loss)
        print(f'loss : {loss:.4f}  min loss : {loss_h.best:.4f} = ')
        print(
            f'class loss : {closs / n_batch:.4f} + \n'
            f'KL loss : {kloss / n_batch:.4f} + \n'
            f'cluster Loss: {clusterloss / n_batch:.4f}',
            flush=True
        )
        acc_cluster = accuracy(model, ds, device)
        acc_cluster_h.add(acc_cluster)
        print(f'acc cluster : {acc_cluster:.4f}  max acc : {acc_cluster_h.best:.4f}')
        print(f'lr: {opt.param_groups[0]["lr"]}', flush=True)
        if acc_cluster_h.better:
            torch.save(model, f'{save_dir}/DEC_{epoch}_best_acc.pt')
        if is_wandb:
            wandb.log({"loss": loss.item(),
                       "loss_kl": kloss / n_batch,
                       "loss_cont": clusterloss / n_batch,
                       "loss_class": closs / n_batch,
                       "accuracy": acc_cluster})
        if epoch % 5 == 0:
            plot(model, save_dir, 'train', epoch)
    torch.save(model, f'{save_dir}/DEC.pt')

    df = pd.DataFrame(zip(range(1, epoch + 1), loss_h.history, acc_cluster_h.history),
                      columns=['epoch', 'loss', 'acc_cluster'])
    df.to_excel(f'{save_dir}/train.xlsx', index=False)


class History:
    def __init__(self, target='min'):
        self.value = None
        self.best = float('inf') if target == 'min' else 0.
        self.n_no_better = 0
        self.better = False
        self.target = target
        self.history = []
        self._check(target)

    def add(self, value):
        if self.target == 'min' and value < self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        elif self.target == 'max' and value > self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        else:
            self.n_no_better += 1
            self.better = False

        self.value = value
        self.history.append(value)

    def _check(self, target):
        if target not in {'min', 'max'}:
            raise ValueError('target only allow "max" or "min" !')


def plot(model, save_dir, target='train', epoch=None):
    # plot latent space cluster in 2D
    assert target in {'train', 'test'}
    assert len(DATA_PLOT[target]) > 0
    print('plotting ...')

    model.eval()
    with torch.no_grad():
        feature = model.encoder(DATA_PLOT[target][0])
        pred = model.cluster(feature).max(1)[1].cpu().numpy()

    feature_2D = TSNE(2).fit_transform(feature.cpu().numpy())
    plt.scatter(feature_2D[:, 0], feature_2D[:, 1], 4, pred, cmap='Paired')
    if epoch is None:
        plt.title(f'Test data')
        plt.savefig(f'{save_dir}/test.png')
    else:
        plt.title(f'Epoch: {epoch}')
        plt.savefig(f'{save_dir}/epoch_{epoch}.png')
    plt.close()
    if epoch is None:
        plt.scatter(feature_2D[:, 0], feature_2D[:, 1], 16, DATA_PLOT[target][1].cpu().numpy(), cmap='Paired')
        plt.title(f'Test data real label')
        plt.savefig(f'{save_dir}/test_real_label.png')
        plt.close()


def plot_rec(model, original, save_dir, epoch):
    # plot reconstructed image
    with torch.no_grad():
        _, rec = model(original)

    for i in range(min(5, len(original))):
        original_img = original[i].cpu().numpy().transpose(1, 2, 0)
        reconstructed_img = rec[i].cpu().numpy().transpose(1, 2, 0)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(reconstructed_img)
        axes[1].set_title('Reconstructed Image')
        axes[1].axis('off')

        plt.savefig(f'{save_dir}/rec_epoch_{epoch}.png')
        plt.close()
