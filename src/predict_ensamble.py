#!/usr/bin/env python3.6
"""Calculates predictions on the validation dataset, using CNN ensemble models 
(AlexNet, ConvNext Tiny, and ResNext50) and logistic regression models.

Usage:
  predict_ensamble.py <valid_paths_csv> <output_dir> <models_dir>
  predict_ensamble.py (-h | --help)

General options:
  -h --help          Show this screen.

Arguments:
  <valid_paths_csv>  csv file listing paths to validation set, which needs to
                     be in a specific order - an example is provided as
                     valid-paths.csv in the root of the project
                     e.g. 'valid-paths.csv'
  <output_dir>       Directory where predictions are saved as a 3-column csv
                     file (with no header), where each column contains a
                     prediction for abnormality, ACL tear, and meniscal tear,
                     in that order
                     e.g. 'out_dir'
  <models_dir>       Directory where ensemble CNN and LR models are saved
                     e.g. 'models/2019-06-24_04-18'
"""

import os
import sys
import csv
from glob import glob
from PIL import Image
from tqdm import tqdm
from docopt import docopt

import torch
import numpy as np
import pandas as pd
import joblib
from torchvision import transforms

from model import MRNet, ConvNextTiny
from utils import preprocess_data


def main(valid_paths_csv, output_dir, models_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    planes = ['axial', 'coronal', 'sagittal']
    conditions = ['abnormal', 'acl', 'meniscus']
    architectures = ['alexnet', 'convnext', 'resnext50']

    input_files_df = pd.read_csv(valid_paths_csv, header=None)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = f'{output_dir}/predictions_ensemble.csv'

    if os.path.exists(output_file):
        os.rename(output_file, f'{output_file}.bak')
        print(f'!! {output_file} already exists, renamed to {output_file}.bak')

    # Load ensemble CNN models
    print(f'Loading CNN ensemble models from {models_dir}...')

    mrnets = [[], [], []]  # [abnormal, acl, meniscus]

    for condition_idx, condition in enumerate(conditions):
        # Load AlexNet models
        for plane in planes:
            checkpoint_pattern = glob(f'{models_dir}/alexnet_*{plane}*{condition}*.pt')
            if checkpoint_pattern:
                checkpoint_path = sorted(checkpoint_pattern)[-1]
                model = MRNet().to(device)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['state_dict'])
                mrnets[condition_idx].append(model)

        # Load ConvNext Tiny models
        for plane in planes:
            checkpoint_pattern = glob(f'{models_dir}/convnext_*{plane}*{condition}*.pt')
            if checkpoint_pattern:
                checkpoint_path = sorted(checkpoint_pattern)[-1]
                model = ConvNextTiny().to(device)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['state_dict'])
                mrnets[condition_idx].append(model)

        # Load ResNext50 models
        for plane in planes:
            checkpoint_pattern = glob(f'{models_dir}/resnext50_*{plane}*{condition}*.pt')
            if checkpoint_pattern:
                checkpoint_path = sorted(checkpoint_pattern)[-1]
                model = ResNext50().to(device)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['state_dict'])
                mrnets[condition_idx].append(model)

    # Load logistic regression ensemble models
    print(f'Loading logistic regression ensemble models from {models_dir}...')

    lrs = []
    for condition in conditions:
        lr_path = f'{models_dir}/lr_ensemble_{condition}.pkl'
        if os.path.exists(lr_path):
            lrs.append(joblib.load(lr_path))
        else:
            print(f'Warning: {lr_path} not found')

    # Parse input, 3 rows at a time (i.e. per case)

    npy_paths = [row.values[0] for _, row in input_files_df.iterrows()]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    print(f'Generating predictions per case...')
    print(f'Predictions will be saved as {output_file}')

    for i in tqdm(range(0, len(npy_paths), 3)):
        case_paths = [npy_paths[i], npy_paths[i+1], npy_paths[i+2]]

        data = []

        for case_path in case_paths:
            series = preprocess_data(case_path, transform)
            data.append(series.unsqueeze(0).to(device))

        # Make predictions per case

        case_preds = []

        for cond_idx, ensemble_models in enumerate(mrnets):  # For each condition
            X = []
            
            # Collect predictions from each plane of each architecture
            for arch_idx in range(0, len(ensemble_models), 3):  # 3 planes per architecture
                if arch_idx + 2 < len(ensemble_models):
                    sagittal_pred = ensemble_models[arch_idx](data[0]).detach().cpu().item()
                    coronal_pred = ensemble_models[arch_idx + 1](data[1]).detach().cpu().item()
                    axial_pred = ensemble_models[arch_idx + 2](data[2]).detach().cpu().item()
                    
                    X.extend([axial_pred, coronal_pred, sagittal_pred])

            # Combine predictions using logistic regression
            if cond_idx < len(lrs):
                X_array = np.array([X])
                pred = np.float64(lrs[cond_idx].predict_proba(X_array)[:, 1])
                case_preds.append(pred)

        # Write to output csv - append if it exists already

        with open(output_file, 'a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(case_preds)


if __name__ == '__main__':
    arguments = docopt(__doc__)

    print('Parsing arguments...')

    main(arguments['<valid_paths_csv>'],
         arguments['<output_dir>'],
         arguments['<models_dir>'])
