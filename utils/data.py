import torch
from torchvision import datasets
from torch.utils.data import Dataset, IterableDataset, DataLoader, random_split, ConcatDataset

import numpy as np
import pandas as pd


class PayloadDataset(Dataset):
    def __init__(self,  csv_path, batch_size=64, transform=None):
        self.csv = filter_columns(pd.read_csv(csv_path))
        self.batch_size = batch_size
        self.transform = transform
        self.name = csv_path.split('/')[-1]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        try:
            features = torch.tensor(row.drop(['label', 'unique_id']), dtype=torch.float32) / 255.
            unique_id = torch.tensor(row['unique_id'])
        except Exception:
            features = torch.tensor(row.drop('label'), dtype=torch.float32) / 255.
            unique_id = 0
        try:
            labels = torch.tensor(row['label'], dtype=torch.int)
        except TypeError:
            labels = torch.tensor(0 if row['label'] == 'normal' else 1)

        if self.transform:
            features = self.transform(features)

        return features, labels, unique_id

    def __len__(self):
        return len(self.csv)

class ChunkedDataset(IterableDataset):
    def __init__(self,  csv_path, batch_size=64, transform=None):
        self.batch_size = batch_size
        self.transform = transform
        self.reader = pd.read_csv(csv_path, chunksize=batch_size, iterator=True)
        self.classes = ['BENIGN',
                        'Bot',  # DT train
                        'DDoS',
                        'DoS GoldenEye',
                        'DoS Hulk',
                        'DoS Slowhttptest',
                        'DoS slowloris',
                        'FTP-Patator',
                        'Heartbleed',  # zero
                        'Infiltration',
                        'PortScan',  # zero
                        'SSH-Patator',
                        'Web Attack – Brute Force',  # zero
                        'Web Attack – Sql Injection',  # zero
                        'Web Attack – XSS']
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
    def __iter__(self):
        try:
            chunk = self.reader.get_chunk(self.batch_size)
            features = torch.tensor(chunk.iloc[:, :-1].values, dtype=torch.float32) / 255.
            chunk['label'] = chunk['label'].apply(lambda x: self.class_to_index.get(x, -1))
            labels = torch.tensor(chunk['label'].values, dtype=torch.float32)
            yield features, labels
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return self.batch_size

def filter_columns(df):
    selected_columns = [col for col in df.columns if col.startswith("payload_byte_") or col == "label" or col == 'unique_id']
    df = df[selected_columns]

    return df


def load_cyber(csv_path, batch_size, num_worker=4):

    ds = PayloadDataset(csv_path, batch_size=batch_size)
    ds = DataLoader(ds,
                    batch_size,
                    num_workers=num_worker,
                    pin_memory=True,
                    drop_last=True)
    return ds


def load_benchmark(batch_size, num_worker, path_train, path_test, path_OOD, path_val=None, t=None):
    tr_ds = datasets.ImageFolder(root=path_train, transform=t)
    if path_val:
        val_ds = datasets.ImageFolder(root=path_val, transform=t)
    else:
        train_size = int(0.8 * len(tr_ds))
        val_size = len(tr_ds) - train_size
        tr_ds, val_ds = random_split(tr_ds, [train_size, val_size])
    test_ds = datasets.ImageFolder(root=path_test, transform=t)
    # test_ds = ConcatDataset([test_ds, tr_ds, val_ds])
    OOD_ds = datasets.ImageFolder(root=path_OOD, transform=t)

    tr_ds = DataLoader(tr_ds,
                       batch_size,
                       shuffle=True,
                       num_workers=num_worker,
                       pin_memory=True)

    test_ds = DataLoader(test_ds,
                         batch_size,
                         shuffle=False,
                         num_workers=num_worker,
                         pin_memory=True)

    val_ds = DataLoader(val_ds,
                        batch_size,
                        shuffle=False,
                        num_workers=num_worker,
                        pin_memory=True)

    OOD_ds = DataLoader(OOD_ds,
                        batch_size,
                        shuffle=False,
                        num_workers=num_worker,
                        pin_memory=True)

    return tr_ds, val_ds, test_ds, OOD_ds
