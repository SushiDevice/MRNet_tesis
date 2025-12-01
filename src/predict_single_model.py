#!/usr/bin/env python3.6
"""Calculates predictions on a given dataset using a single CNN model.

Usage:
  predict_single_model.py <data_paths_csv> <model_path> <output_dir>
  predict_single_model.py (-h | --help)

General options:
  -h --help          Show this screen.

Arguments:
  <data_paths_csv>   csv file listing paths to dataset
                     e.g. 'valid-paths.csv'
  <model_path>       Path to the trained CNN model checkpoint
                     e.g. 'models/2025-11-30_13-17/checkpoint_abnormal_axial.pt'
  <output_dir>       Directory where predictions are saved as a csv file
                     e.g. 'out_dir'
"""

import os
import sys
import csv
from PIL import Image
from tqdm import tqdm
from docopt import docopt

import torch
import numpy as np
import pandas as pd
from torchvision import transforms

from model import MRNet
from utils import preprocess_data


def main(data_paths_csv, model_path, output_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_files_df = pd.read_csv(data_paths_csv, header=None)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = f'{output_dir}/predictions_single.csv'

    if os.path.exists(output_file):
        os.rename(output_file, f'{output_file}.bak')
        print(f'!! {output_file} already exists, renamed to {output_file}.bak')

    # Load MRNet model
    print(f'Loading CNN model from {model_path}...')

    model = MRNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Parse input
    npy_paths = [row.values[0] for _, row in input_files_df.iterrows()]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    print(f'Generating predictions...')
    print(f'Predictions will be saved as {output_file}')

    predictions = []

    with torch.no_grad():
        for npy_path in tqdm(npy_paths):
            series = preprocess_data(npy_path, transform)
            data = series.unsqueeze(0).to(device)

            # Make prediction
            pred = model(data).detach().cpu().item()
            predictions.append(pred)

    # Write to output csv
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for pred in predictions:
            writer.writerow([pred])

    print(f'Predictions saved to {output_file}')


if __name__ == '__main__':
    arguments = docopt(__doc__)

    print('Parsing arguments...')

    main(arguments['<data_paths_csv>'],
         arguments['<model_path>'],
         arguments['<output_dir>'])
