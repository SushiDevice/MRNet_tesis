#!/usr/bin/env python3.6
"""Trains three CNN models to predict abnormalities, ACL tears and meniscal
tears for a given plane (axial, coronal or sagittal) of knee MRI images.

Usage:
  train_cnn_models.py <data_dir> <plane> <epochs> [options]
  train_cnn_models.py (-h | --help)

General options:
  -h --help             Show this screen.

Arguments:
  <data_dir>            Path to a directory where the data lives e.g. 'MRNet-v1.0'
  <plane>               MRI plane of choice ('axial', 'coronal', 'sagittal')
  <epochs>              Number of epochs e.g. 50

Training options:
  --lr=<lr>             Learning rate for nn.optim.Adam optimizer [default: 0.00001]
  --weight-decay=<wd>   Weight decay for nn.optim.Adam optimizer [default: 0.01]
  --device=<device>     Device to run code ('cpu' or 'cuda') - if not provided,
                        it will be set to the value returned by torch.cuda.is_available()
  --train-limit=<n>     Limit number of training exams (use all if omitted)
  --valid-limit=<n>     Limit number of validation exams (use all if omitted)
  --kfolds=<k>          Optional K-Folds for CNN training (>=2 enables CV on train set)
  --folds-file=<csv>    Optional CSV with columns 'case,fold' to share folds across planes
"""

import sys
import os
import numpy as np
import pandas as pd
from docopt import docopt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import make_data_loader
from dataset import make_dataset
from model import MRNet
from utils import create_output_dir, \
                  print_stats,       \
                  save_losses,       \
                  save_checkpoint
from sklearn.model_selection import KFold


def calculate_weights(data_dir, dataset_type, device):
    diagnoses = ['abnormal', 'acl', 'meniscus']

    labels_path = f'{data_dir}/{dataset_type}_labels.csv'
    labels_df = pd.read_csv(labels_path)

    weights = []

    for diagnosis in diagnoses:
        neg_count, pos_count = labels_df[diagnosis].value_counts().sort_index()
        weight = torch.tensor([neg_count / pos_count])
        weight = weight.to(device)
        weights.append(weight)

    return weights


def make_adam_optimizer(model, lr, weight_decay):
    return optim.Adam(model.parameters(), lr, weight_decay=weight_decay)


def make_lr_scheduler(optimizer,
                      mode='min',
                      factor=0.3,
                      patience=1,
                      verbose=False):
    # Nota: algunas versiones de PyTorch no soportan el parámetro 'verbose'
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                mode=mode,
                                                factor=factor,
                                                patience=patience)


def batch_forward_backprop(models, inputs, labels, criterions, optimizers):
    losses = []

    for i, (model, label, criterion, optimizer) in \
            enumerate(zip(models, labels[0], criterions, optimizers)):
        model.train()
        optimizer.zero_grad()

        out = model(inputs)
        loss = criterion(out, label.unsqueeze(0))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.array(losses)


def batch_forward(models, inputs, labels, criterions):
    preds = []
    losses = []

    for i, (model, label, criterion) in \
            enumerate(zip(models, labels[0], criterions)):
        model.eval()

        out = model(inputs)
        preds.append(out.item())
        loss = criterion(out, label.unsqueeze(0))
        losses.append(loss.item())

    return np.array(preds), np.array(losses)


def update_lr_schedulers(lr_schedulers, batch_valid_losses):
    for scheduler, v_loss in zip(lr_schedulers, batch_valid_losses):
        scheduler.step(v_loss)


def _train_one_run(data_dir, plane, epochs, lr, weight_decay, device, train_loader, valid_loader, out_dir):
    diagnoses = ['abnormal', 'acl', 'meniscus']

    out_dir, losses_path = create_output_dir(out_dir, plane)

    print(f'Creating models...')

    # Create a model for each diagnosis

    models = [MRNet().to(device), MRNet().to(device), MRNet().to(device)]

    # Calculate loss weights based on the prevalences in train set

    pos_weights = calculate_weights(data_dir, 'train', device)
    criterions = [nn.BCEWithLogitsLoss(pos_weight=weight) \
                  for weight in pos_weights]

    optimizers = [make_adam_optimizer(model, lr, weight_decay) \
                  for model in models]

    lr_schedulers = [make_lr_scheduler(optimizer) for optimizer in optimizers]

    min_valid_losses = [np.inf, np.inf, np.inf]

    print(f'Training a model using {plane} series...')
    print(f'Checkpoints and losses will be save to {out_dir}')

    for epoch, _ in enumerate(range(epochs), 1):
        print(f'=== Epoch {epoch}/{epochs} ===')

        batch_train_losses = np.array([0.0, 0.0, 0.0])
        batch_valid_losses = np.array([0.0, 0.0, 0.0])

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            batch_loss = batch_forward_backprop(models, inputs, labels,
                                                criterions, optimizers)
            batch_train_losses += batch_loss

        valid_preds = []
        valid_labels = []

        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            batch_preds, batch_loss = \
                batch_forward(models, inputs, labels, criterions)
            batch_valid_losses += batch_loss

            valid_labels.append(labels.detach().cpu().numpy().squeeze())
            valid_preds.append(batch_preds)

        batch_train_losses /= len(train_loader)
        batch_valid_losses /= len(valid_loader)

        print_stats(batch_train_losses, batch_valid_losses,
                    valid_labels, valid_preds)
        save_losses(batch_train_losses, batch_valid_losses, losses_path)

        update_lr_schedulers(lr_schedulers, batch_valid_losses)

        for i, (batch_v_loss, min_v_loss) in \
                enumerate(zip(batch_valid_losses, min_valid_losses)):

            if batch_v_loss < min_v_loss:
                save_checkpoint(epoch, plane, diagnoses[i], models[i],
                                optimizers[i], out_dir)

                min_valid_losses[i] = batch_v_loss


def _case_ids_from_dataset(dataset):
    case_ids = []
    for path in dataset.case_paths:
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            case_ids.append(int(base))
        except ValueError:
            # try zero-padded names
            case_ids.append(int(base.lstrip('0') or '0'))
    return case_ids


def main(data_dir, plane, epochs, lr, weight_decay, device=None, train_limit=None, valid_limit=None, kfolds=None, folds_file=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Folds-file branch: enforce same folds across planes using provided mapping
    if folds_file is not None:
        print('Loading shared folds mapping from CSV...')
        folds_df = pd.read_csv(folds_file)
        if 'case' not in folds_df.columns or 'fold' not in folds_df.columns:
            raise ValueError("--folds-file debe contener columnas 'case' y 'fold'")
        folds_df['case'] = folds_df['case'].astype(int)
        folds_df['fold'] = folds_df['fold'].astype(int)

        base_train_dataset = make_dataset(data_dir, 'train', plane, device=device, max_cases=train_limit, transform_type='train')
        # map dataset indices to case ids
        case_ids = _case_ids_from_dataset(base_train_dataset)
        case_to_idx = {cid: idx for idx, cid in enumerate(case_ids)}

        # derive folds present in mapping (sorted)
        fold_ids = sorted(folds_df['fold'].unique().tolist())
        exp = f'{datetime.now():%Y-%m-%d_%H-%M}'
        print(f'Using shared folds file with {len(fold_ids)} folds: {fold_ids}')

        for fold_idx in fold_ids:
            val_cases = folds_df.loc[folds_df['fold'] == fold_idx, 'case'].astype(int).tolist()
            # filter to cases that exist in this plane's dataset
            val_idx = [case_to_idx[c] for c in val_cases if c in case_to_idx]
            train_idx = [i for i in range(len(case_ids)) if i not in set(val_idx)]
            print(f'=== Fold {fold_idx}/{fold_ids[-1]} === (train={len(train_idx)}, valid={len(val_idx)})')

            train_loader = make_data_loader(
                data_dir, 'train', plane, device, shuffle=True, indices=train_idx, transform_type='train'
            )
            fold_valid_loader = make_data_loader(
                data_dir, 'train', plane, device, shuffle=False, indices=val_idx, transform_type='valid'
            )
            out_dir = f'{exp}/fold_{fold_idx}'
            _train_one_run(data_dir, plane, epochs, lr, weight_decay, device, train_loader, fold_valid_loader, out_dir)

        print('Completed shared-folds training.')
        return

    # K-Folds branch: perform CV on the train split only
    if kfolds is not None and kfolds >= 2:
        print('Creating datasets for K-Folds...')
        base_train_dataset = make_dataset(data_dir, 'train', plane, device=device, max_cases=train_limit, transform_type='train')
        n_samples = len(base_train_dataset)
        indices = np.arange(n_samples)

        exp = f'{datetime.now():%Y-%m-%d_%H-%M}'
        print(f'K-Folds training enabled: k={kfolds}')

        kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
            print(f'=== Fold {fold_idx}/{kfolds} ===')
            train_loader = make_data_loader(
                data_dir, 'train', plane, device, shuffle=True, indices=train_idx, transform_type='train'
            )
            fold_valid_loader = make_data_loader(
                data_dir, 'train', plane, device, shuffle=False, indices=val_idx, transform_type='valid'
            )
            out_dir = f'{exp}/fold_{fold_idx}'
            _train_one_run(data_dir, plane, epochs, lr, weight_decay, device, train_loader, fold_valid_loader, out_dir)

        print('Completed K-Folds training.')
        return

    # Default single-run training
    print('Creating data loaders...')

    exp = f'{datetime.now():%Y-%m-%d_%H-%M}'
    train_loader = make_data_loader(data_dir, 'train', plane, device, shuffle=True, max_cases=train_limit)
    valid_loader = make_data_loader(data_dir, 'valid', plane, device, max_cases=valid_limit)
    out_dir = exp
    _train_one_run(data_dir, plane, epochs, lr, weight_decay, device, train_loader, valid_loader, out_dir)


if __name__ == '__main__':
    arguments = docopt(__doc__)

    print('Parsing arguments...')

    train_limit = int(arguments['--train-limit']) if arguments['--train-limit'] is not None else None
    valid_limit = int(arguments['--valid-limit']) if arguments['--valid-limit'] is not None else None
    kfolds = int(arguments['--kfolds']) if arguments['--kfolds'] is not None else None
    folds_file = arguments['--folds-file']

    main(arguments['<data_dir>'],
         arguments['<plane>'],
         int(arguments['<epochs>']),
         float(arguments['--lr']),
         float(arguments['--weight-decay']),
         arguments['--device'],
         train_limit,
         valid_limit,
         kfolds,
         folds_file)
