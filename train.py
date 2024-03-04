from time import time

import torch
from torch.optim import SGD, AdamW

from utils import (set_data_plot, plot, print_cm,
                   AutoEncoder, DEC, AEConv,
                   pretrain, train, get_initial_center, get_best_initial_center,
                   accuracy)

def train_DEC(arg, tr_ds, val_ds, device, output_dir, alpha=1, beta=1, omega=1, is_wandb=True):

    # for visualize
    set_data_plot(tr_ds, val_ds, device)

    # train autoencoder
    t0 = time()
    if arg.pre_epoch > 0:
        ae = AutoEncoder(1500).to(device)
        print(f'\nAE param: {sum(p.numel() for p in ae.parameters()) / 1e6:.2f} M', flush=True)
        opt = AdamW(ae.parameters())
        pretrain(ae, opt, tr_ds, device, arg.pre_epoch, output_dir, is_wandb=is_wandb)
    t1 = time()

    # initial center
    t2 = time()
    try:
        ae = torch.load(f'{output_dir}/fine_tune_AE.pt', device)
    except FileNotFoundError:
        print(f'{output_dir}/fine_tune_AE.pt not found!\nCreate new AE from scratch, without pretrain...')
        ae = AutoEncoder(1500).to(device)

    center = get_initial_center(ae, tr_ds, device, arg.k)
    t3 = time()

    # train dec
    print('\nload the best encoder and build DEC ...', flush=True)
    dec = DEC(ae.encoder, center, alpha=1).to(device)
    print(f'DEC param: {sum(p.numel() for p in dec.parameters()) / 1e6:.2f} M', flush=True)
    opt = SGD(dec.parameters(), 0.01, 0.9, nesterov=True)
    t4 = time()
    train(dec, opt, tr_ds, device, arg.epoch, output_dir, alpha, beta, omega, is_wandb=is_wandb)
    t5 = time()

    print()
    print('*' * 50)
    print('load the best DEC ...', flush=True)
    dec = torch.load(f'{output_dir}/DEC.pt', device)
    print('Evaluate ...', flush=True)
    acc_cluster = accuracy(dec, val_ds, device)
    print(f'test acc cluster: {acc_cluster:.4f}', flush=True)
    print('*' * 50)
    plot(dec, output_dir, 'test')

    print(f'\ntrain AE time: {t1 - t0:.2f} s', flush=True)
    print(f'get inititial time: {t3 - t2:.2f} s', flush=True)
    print(f'train DEC time: {t5 - t4:.2f} s', flush=True)



### This is just an experiment, an idea, don't consider it! ###
def train_DEC_new(arg, tr_ds, val_ds, device, output_dir, alpha=1, beta=1, omega=1, is_wandb=True):

    # for visualize
    set_data_plot(tr_ds, val_ds, device)

    # train autoencoder
    t0 = time()
    if arg.pre_epoch > 0:
        ae = AutoEncoder(1500).to(device)
        print(f'\nAE param: {sum(p.numel() for p in ae.parameters()) / 1e6:.2f} M')
        opt = AdamW(ae.parameters())
        pretrain(ae, opt, tr_ds, device, arg.pre_epoch, output_dir, is_wandb=is_wandb)
    t1 = time()

    # initial center
    t2 = time()
    ae = torch.load(f'{output_dir}/fine_tune_AE.pt', device)
    center = get_best_initial_center(ae, tr_ds, device, arg.k)
    t3 = time()

    # train dec
    print('\nload the best encoder and build DEC ...')
    # ae = torch.load(f'{output_dir}/fine_tune_AE.pt', device)
    dec = DEC(ae.encoder, center, alpha=1).to(device)
    print(f'DEC param: {sum(p.numel() for p in dec.parameters()) / 1e6:.2f} M')
    opt = SGD(dec.parameters(), 0.01, 0.9, nesterov=True)
    t4 = time()
    train(dec, opt, tr_ds, device, arg.epoch, output_dir, alpha, 0., 0., is_wandb=is_wandb)
    t5 = time()

    print()
    print('*' * 50)
    print('load the best DEC ...')
    dec = torch.load(f'{output_dir}/DEC.pt', device)
    print('Evaluate ...')
    acc_cluster = accuracy(dec, val_ds, device)
    print(f'test acc cluster: {acc_cluster:.4f}')
    print('*' * 50)
    plot(dec, output_dir, 'test')

    print(f'\ntrain AE time: {t1 - t0:.2f} s')
    print(f'get inititial time: {t3 - t2:.2f} s')
    print(f'train DEC time: {t5 - t4:.2f} s')


