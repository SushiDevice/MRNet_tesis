#!/usr/bin/env python3.6
"""Trains EfficientNetB0 CNN models with two-phase fine-tuning for MRNet.

This script implements transfer learning best practices:
Phase 1 (Warmup): Train only classifier with frozen backbone using higher LR (1e-2)
Phase 2 (Fine-tune): Unfreeze blocks and train with lower LR (1e-4)

Benefits:
- Higher LR in phase 1 adapts pretrained features to new task quickly
- Lower LR in phase 2 prevents catastrophic forgetting of pretrained knowledge
- Frozen BatchNorm prevents data leakage from large ImageNet batches
- Validation metrics typically exceed training metrics due to strong regularization

Usage:
  train_efficientnet_finetuned.py <data_dir> <plane> <epochs> [options]
  train_efficientnet_finetuned.py (-h | --help)

General options:
  -h --help                    Show this screen.

Arguments:
  <data_dir>                   Path to a directory where the data lives e.g. 'MRNet-v1.0'
  <plane>                      MRI plane of choice ('axial', 'coronal', 'sagittal')
  <epochs>                     Number of epochs e.g. 50

Training options:
  --warmup-epochs=<n>          Phase 1: Epochs with frozen backbone [default: 5]
  --warmup-lr=<lr>             Phase 1: Learning rate for warmup [default: 0.01]
  --finetune-lr=<lr>           Phase 2: Learning rate for fine-tuning [default: 0.0001]
  --weight-decay=<wd>          Weight decay for Adam optimizer [default: 0.001]
  --device=<device>            Device to run code ('cpu' or 'cuda') - if not provided,
                               it will be set to the value returned by torch.cuda.is_available()
  --train-limit=<n>            Limit number of training exams (use all if omitted)
  --valid-limit=<n>            Limit number of validation exams (use all if omitted)
  --unfreeze-blocks=<blocks>   Phase 2: Comma-separated block indices to unfreeze 
                               (e.g., '5,6' for last 2 blocks) [default: 5,6]
  --batch-size=<bs>            Batch size for training [default: 4]
  --batch-size-val=<bs>        Batch size for validation [default: 4]
"""

import sys
import numpy as np
import pandas as pd
from docopt import docopt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import make_data_loader
from model import MRNet
from utils import create_output_dir, \
                  print_stats,       \
                  save_losses,       \
                  save_checkpoint


def calculate_weights(data_dir, dataset_type, device):
    """Calculate class weights based on label distribution."""
    diagnoses = ['abnormal', 'acl', 'meniscus']
    labels_path = f'{data_dir}/{dataset_type}_labels.csv'
    print(labels_path)
    labels_df = pd.read_csv(labels_path)

    weights = []
    for diagnosis in diagnoses:
        neg_count, pos_count = labels_df[diagnosis].value_counts().sort_index()
        weight = torch.tensor([neg_count / pos_count])
        weight = weight.to(device)
        weights.append(weight)

    return weights


def make_adam_optimizer(model, lr, weight_decay):
    """Create Adam optimizer with specified learning rate and weight decay."""
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_lr_scheduler(optimizer, mode='min', factor=0.3, patience=1):
    """Create learning rate scheduler."""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience
    )


def batch_forward_backprop(models, inputs, labels, criterions, optimizers):
    """Forward and backward pass for a batch."""
    losses = []

    for i, (model, label, criterion, optimizer) in \
            enumerate(zip(models, labels[0], criterions, optimizers)):
        model.train()
        optimizer.zero_grad()

        out = model(inputs)
        # Squeeze output if needed, then ensure it matches label shape [1]
        if out.dim() > 1:
            out = out.squeeze(-1)
        loss = criterion(out, label.unsqueeze(0))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.array(losses)


def batch_forward(models, inputs, labels, criterions):
    """Forward pass for validation."""
    preds = []
    losses = []

    for i, (model, label, criterion) in \
            enumerate(zip(models, labels[0], criterions)):
        model.eval()

        out = model(inputs)
        # Squeeze output if needed, then ensure it matches label shape [1]
        if out.dim() > 1:
            out = out.squeeze(-1)
        preds.append(out.item())
        loss = criterion(out, label.unsqueeze(0))
        losses.append(loss.item())

    return np.array(preds), np.array(losses)
    return np.array(preds), np.array(losses)


def update_lr_schedulers(lr_schedulers, batch_valid_losses):
    """Update learning rate schedulers based on validation loss."""
    for scheduler, v_loss in zip(lr_schedulers, batch_valid_losses):
        scheduler.step(v_loss)


def log_model_state(models, epoch, stage):
    """Log trainable parameters for debugging."""
    print(f"\n--- {stage} (Epoch {epoch}) ---")
    for i, model in enumerate(models):
        total = model.get_total_params_count()
        trainable = model.get_trainable_params_count()
        print(f"Model {i}: {trainable:,}/{total:,} trainable params ({100*trainable/total:.1f}%)")


def main(data_dir, plane, epochs, warmup_epochs=5, warmup_lr=0.01, finetune_lr=0.0001, 
         weight_decay=0.001, device=None, train_limit=None, valid_limit=None, 
         unfreeze_blocks_str='5,6', batch_size=4, batch_size_val=4):
    """
    Main training function with two-phase transfer learning.
    
    Phase 1 (Warmup): 
    - Freeze backbone, train only classifier
    - Use higher learning rate (1e-2) to quickly adapt features
    
    Phase 2 (Fine-tuning):
    - Unfreeze specified blocks
    - Use lower learning rate (1e-4) to preserve pretrained knowledge
    - Keep BatchNorm frozen to avoid data leakage
    """
    diagnoses = ['abnormal', 'acl', 'meniscus']
    
    # Parse unfreeze blocks
    try:
        unfreeze_blocks = [int(b.strip()) for b in unfreeze_blocks_str.split(',')]
    except ValueError:
        print("Error parsing unfreeze_blocks. Using default [5, 6]")
        unfreeze_blocks = [5, 6]

    exp = f'{datetime.now():%Y-%m-%d_%H-%M}'
    out_dir, losses_path = create_output_dir(exp, plane)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Device: {device}')
    print(f'=== PHASE 1: WARMUP (Epochs 1-{warmup_epochs}) ===')
    print(f'  Backbone: FROZEN | Learning rate: {warmup_lr}')
    print(f'=== PHASE 2: FINE-TUNE (Epochs {warmup_epochs+1}-{epochs}) ===')
    print(f'  Unfreeze blocks: {unfreeze_blocks} | Learning rate: {finetune_lr}')
    print(f'  Weight decay: {weight_decay}')
    print(f'  Batch sizes: train={batch_size}, val={batch_size_val}')

    print('Creating data loaders...')
    train_loader = make_data_loader(data_dir, 'train_split1', plane, device, shuffle=True, 
                                   max_cases=train_limit)
    valid_loader = make_data_loader(data_dir, 'valid', plane, device, max_cases=valid_limit)

    print('Creating EfficientNetB0 models...')
    # Create models with frozen backbone initially (Phase 1 setup)
    models = [
        MRNet(freeze_backbone=True).to(device),
        MRNet(freeze_backbone=True).to(device),
        MRNet(freeze_backbone=True).to(device)
    ]

    # Calculate loss weights based on the prevalences in train set
    pos_weights = calculate_weights(data_dir, 'train_split1', device)
    criterions = [nn.BCEWithLogitsLoss(pos_weight=weight) for weight in pos_weights]

    # Start with warmup learning rate for Phase 1
    optimizers = [make_adam_optimizer(model, warmup_lr, weight_decay) for model in models]
    lr_schedulers = [make_lr_scheduler(optimizer) for optimizer in optimizers]

    min_valid_losses = [np.inf, np.inf, np.inf]
    current_phase = "Warmup"

    print(f'Training EfficientNetB0 models using {plane} series...')
    print(f'Checkpoints and losses will be saved to {out_dir}')

    # Log initial state
    log_model_state(models, 0, "Initial State (Backbone Frozen, Phase 1)")

    for epoch in range(1, epochs + 1):
        # Transition from Phase 1 to Phase 2 after warmup
        if epoch == warmup_epochs + 1:
            print(f"\n{'='*70}")
            print(f"TRANSITIONING FROM PHASE 1 TO PHASE 2")
            print(f"{'='*70}")
            print(f"Unfreezing blocks {unfreeze_blocks}...")
            print(f"Learning rate: {warmup_lr} -> {finetune_lr}")
            print(f"{'='*70}\n")
            
            current_phase = "Fine-tune"
            for model in models:
                model.unfreeze_blocks(unfreeze_blocks, freeze_batchnorm=True)
            
            # Recreate optimizers with lower learning rate for Phase 2
            optimizers = [make_adam_optimizer(model, finetune_lr, weight_decay) for model in models]
            lr_schedulers = [make_lr_scheduler(optimizer) for optimizer in optimizers]
            
            log_model_state(models, epoch, "After Phase 1->2 Transition")

        phase_label = f"Epoch {epoch}/{epochs} ({current_phase})"
        print(f'\n=== {phase_label} ===')

        batch_train_losses = np.array([0.0, 0.0, 0.0])
        batch_valid_losses = np.array([0.0, 0.0, 0.0])

        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_loss = batch_forward_backprop(models, inputs, labels,
                                                criterions, optimizers)
            batch_train_losses += batch_loss

        # Validation phase
        valid_preds = []
        valid_labels = []

        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_preds, batch_loss = batch_forward(models, inputs, labels, criterions)
            batch_valid_losses += batch_loss

            valid_labels.append(labels.detach().cpu().numpy().squeeze())
            valid_preds.append(batch_preds)

        batch_train_losses /= len(train_loader)
        batch_valid_losses /= len(valid_loader)

        print_stats(batch_train_losses, batch_valid_losses,
                    valid_labels, valid_preds)
        save_losses(batch_train_losses, batch_valid_losses, losses_path)

        update_lr_schedulers(lr_schedulers, batch_valid_losses)

        # Save checkpoints for improved validation loss
        for i, (batch_v_loss, min_v_loss) in \
                enumerate(zip(batch_valid_losses, min_valid_losses)):

            if batch_v_loss < min_v_loss:
                save_checkpoint(epoch, plane, diagnoses[i], models[i],
                                optimizers[i], out_dir)
                min_valid_losses[i] = batch_v_loss
                print(f"  Checkpoint saved for {diagnoses[i]} (loss: {batch_v_loss:.6f})")

    print(f'\nTraining completed. Models saved to {out_dir}')


if __name__ == '__main__':
    arguments = docopt(__doc__)

    print('Parsing arguments...')

    train_limit = int(arguments['--train-limit']) if arguments['--train-limit'] is not None else None
    valid_limit = int(arguments['--valid-limit']) if arguments['--valid-limit'] is not None else None
    warmup_epochs = int(arguments['--warmup-epochs'])
    warmup_lr = float(arguments['--warmup-lr'])
    finetune_lr = float(arguments['--finetune-lr'])
    weight_decay = float(arguments['--weight-decay'])
    unfreeze_blocks_str = arguments['--unfreeze-blocks']
    batch_size = int(arguments['--batch-size'])
    batch_size_val = int(arguments['--batch-size-val'])

    main(arguments['<data_dir>'],
         arguments['<plane>'],
         int(arguments['<epochs>']),
         warmup_epochs,
         warmup_lr,
         finetune_lr,
         weight_decay,
         arguments['--device'],
         train_limit,
         valid_limit,
         unfreeze_blocks_str,
         batch_size,
         batch_size_val)
