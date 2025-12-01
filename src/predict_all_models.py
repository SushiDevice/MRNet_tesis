#!/usr/bin/env python3.6
"""Calculates predictions for all 9 CNN models (3 tasks × 3 planes) on a given dataset.

Usage:
  predict_all_models.py <data_paths_csv> <models_dir> <output_dir>
  predict_all_models.py (-h | --help)

General options:
  -h --help          Show this screen.

Arguments:
  <data_paths_csv>   csv file listing paths to dataset samples
                     e.g. 'valid_paths.csv'
  <models_dir>       Path to directory containing model checkpoints organized as:
                     <models_dir>/cnn_<plane>_<task>_<number>.pt
                     Planes: axial, coronal, sagittal
                     Tasks: abnormal, acl, meniscus
                     e.g. 'models/mejores_nuevotrain'
  <output_dir>       Directory where predictions are saved as individual csv files
                     e.g. 'predictions_all'
"""

import os
import sys
import csv
from glob import glob
from tqdm import tqdm
from docopt import docopt

import torch
import numpy as np
import pandas as pd
from torchvision import transforms

from model import MRNet
from utils import preprocess_data


def main(data_paths_csv, models_dir, output_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define all models to load
    tasks = ['abnormal', 'acl', 'meniscus']
    planes = ['axial', 'coronal', 'sagittal']
    model_configs = [(task, plane) for task in tasks for plane in planes]

    print(f'Looking for models in {models_dir}...')

    # Find all checkpoint files
    checkpoint_files = {}
    for checkpoint_path in glob(os.path.join(models_dir, 'cnn_*.pt')):
        filename = os.path.basename(checkpoint_path)
        # Extract plane and task from filename: cnn_<plane>_<task>_<number>.pt
        parts = filename.replace('cnn_', '').replace('.pt', '').split('_')
        if len(parts) >= 3:
            plane = parts[0]
            task = parts[1]
            checkpoint_files[(task, plane)] = checkpoint_path

    # Load all models
    models = {}
    for task, plane in model_configs:
        key = (task, plane)
        if key in checkpoint_files:
            print(f'Loading {task} model for {plane} plane...')
            model = MRNet().to(device)
            checkpoint = torch.load(checkpoint_files[key], map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            models[key] = model
        else:
            print(f'Warning: Model checkpoint for {task} ({plane}) not found at {checkpoint_files.get(key, "N/A")}')

    if not models:
        print('Error: No models found!')
        return

    print(f'Loaded {len(models)} models')

    # Load data paths
    print(f'\nLoading data paths from {data_paths_csv}...')
    input_files_df = pd.read_csv(data_paths_csv, header=None)
    npy_paths = [row.values[0] for _, row in input_files_df.iterrows()]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    print(f'\nGenerating predictions for {len(npy_paths)} samples across {len(models)} models...')

    # Store all predictions
    all_predictions = {key: [] for key in models.keys()}

    with torch.no_grad():
        for npy_path in tqdm(npy_paths):
            series = preprocess_data(npy_path, transform)
            data = series.unsqueeze(0).to(device)

            # Make prediction with each model
            for key, model in models.items():
                pred = model(data).detach().cpu().item()
                all_predictions[key].append(pred)

    # Write predictions to separate CSV files for each model
    print('\nSaving predictions...')
    for (task, plane), predictions in all_predictions.items():
        output_file = f'{output_dir}/predictions_{task}_{plane}.csv'
        with open(output_file, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for pred in predictions:
                writer.writerow([pred])
        print(f'  Saved {output_file}')

    print('\nPredictions generation complete!')
    print(f'Output files: predictions_<task>_<plane>.csv in {output_dir}')


if __name__ == '__main__':
    arguments = docopt(__doc__)

    print('Parsing arguments...')

    main(arguments['<data_paths_csv>'],
         arguments['<models_dir>'],
         arguments['<output_dir>'])
