import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")
import argparse

import sys
import platform

import os

import torch

import wandb

from utils import load_cyber
from train import train_DEC
from eval import eval_DEC


def get_arg():
    # Function to parse command-line arguments using argparse
    arg = argparse.ArgumentParser()
    arg.add_argument('--name', required=True, type=str, help='Name of experiments')
    arg.add_argument('--bs', default=256, type=int, help='batch size')
    arg.add_argument('--pre_epoch', default=300, type=int, help='epochs for train Autoencoder')
    arg.add_argument('--epoch', type=int, default=200, help='epochs for train DEC')
    arg.add_argument('-k', type=int, default=2, help='num of clusters')
    arg.add_argument('--worker', default=4, type=int, help='num of workers')
    arg.add_argument('--dir', default='NeSyAI4Cyber/DEC-DT/', type=str)
    arg.add_argument('--no_train', action='store_true')
    arg = arg.parse_args()
    return arg


def main():
    def not_ios():
        # Helper function to check if the platform is not iOS
        # I used this flag to print output on file if I'm on server
        system = platform.system()
        return system != 'Darwin'

    arg = get_arg()

    output_dir = arg.dir + 'results/' + arg.name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(arg.dir + 'results/'):
        os.makedirs(arg.dir + 'results/', exist_ok=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not_ios():
        sys.stdout = open(f'stdout_{arg.name}.txt', 'w')
        sys.stderr = sys.stdout
    print(f"{arg.name} test for pretrain: {arg.pre_epoch} epochs and train: {arg.epoch} epochs")

    csv_path = 'dataset/CICIDS2017/payload_test.csv'
    test_ds = load_cyber(csv_path, arg.bs, num_worker=arg.worker)

    csv_path = 'dataset/CICIDS2017/zero_day.csv'
    ood_ds = load_cyber(csv_path, arg.bs, num_worker=arg.worker)

    if not arg.no_train:
        path_train = 'dataset/CICIDS2017/payload_train.csv'
        tr_ds = load_cyber(path_train, arg.bs, num_worker=arg.worker)

        path_val = 'dataset/CICIDS2017/payload_val.csv'
        val_ds = load_cyber(path_val, arg.bs, num_worker=arg.worker)

        try:
            wandb.init(
                project="DEC-DT",  # Set the project where this run will be logged
                name=arg.name,
            )
            is_wand = True
        except Exception:
            print("WANDB not available!", flush=True)
            is_wand = False

        train_DEC(arg, tr_ds, val_ds, device, output_dir, is_wandb=is_wand)

        if is_wand:
            wandb.finish()

        del tr_ds, val_ds  # Free up memory

    # evaluate test set and zero days. it produces the input file for next step (ML step)
    eval_DEC(arg.name, ood_ds, output_dir, test_ds=test_ds)

    del test_ds, ood_ds  # Free up memory

    print("UNSW - only for test:")
    # evaluate UNSW sample. it is input file for next step (ML step)
    csv_path = 'dataset/Payload_data_UNSW.csv'
    UNSW_ds = load_cyber(csv_path, arg.bs, num_worker=arg.worker)
    eval_DEC('UNSW', UNSW_ds, output_dir)
    del UNSW_ds


if __name__ == '__main__':
    main()
