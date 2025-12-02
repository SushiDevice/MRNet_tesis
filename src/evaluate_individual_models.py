#!/usr/bin/env python3.6
"""Evaluates individual CNN models for backbone comparison.

This script evaluates each of the 9 models independently to help you decide
which backbone/model should be swapped out for improvements.

Usage:
  evaluate_individual_models.py <model_path> <data_paths_csv> <labels_csv> <task> <plane>
  evaluate_individual_models.py (-h | --help)

General options:
  -h --help          Show this screen.

Arguments:
  <model_path>       Path to a single model checkpoint
                     e.g. 'models/mejores_nuevotrain/cnn_axial_abnormal_06.pt'
  <data_paths_csv>   csv file listing paths to dataset samples
                     e.g. 'out/all_valid_paths.csv'
  <labels_csv>       csv file containing labels for the dataset
                     e.g. 'MRNet-v1.0/valid_labels.csv'
  <task>             Task to evaluate (abnormal, acl, meniscus)
  <plane>            Plane of the model (axial, coronal, sagittal)
"""

import os
import sys
from docopt import docopt

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn import metrics
from model import MRNet
from utils import preprocess_data
from torchvision import transforms


def main(model_path, data_paths_csv, labels_csv, task, plane):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'\n=== Evaluating {task.upper()} - {plane.upper()} ===')
    print(f'Model: {model_path}\n')
    
    # Load model
    print('Loading model...')
    model = MRNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Load data paths
    print('Loading data paths...')
    all_paths = []
    with open(data_paths_csv, 'r') as f:
        all_paths = [line.strip() for line in f.readlines()]
    
    print(f'Total sample paths: {len(all_paths)}')
    
    # Filter paths to only include the specific plane
    plane_paths = [p for p in all_paths if f'/{plane}/' in p]
    print(f'Paths for {plane} plane only: {len(plane_paths)}')
    
    if not plane_paths:
        print(f'Error: No paths found for plane {plane}')
        return
    
    # Extract unique cases from filtered paths
    cases = []
    for path in plane_paths:
        case = os.path.splitext(os.path.basename(path))[0]
        if case not in cases:
            cases.append(case)
    
    print(f'Found {len(cases)} unique cases for {plane} plane\n')
    
    # Load labels
    print('Loading labels...')
    labels_df = pd.read_csv(labels_csv)
    
    if task not in labels_df.columns:
        print(f'Error: Task "{task}" not found in labels')
        return
    
    # Generate predictions
    print(f'Generating predictions for {plane} plane...')
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    
    all_predictions = []
    
    with torch.no_grad():
        for path in plane_paths:
            series = preprocess_data(path, transform)
            data = series.unsqueeze(0).to(device)
            pred = model(data).detach().cpu().item()
            all_predictions.append(pred)
    
    print(f'Generated {len(all_predictions)} predictions for {len(cases)} cases\n')
    
    # No need to aggregate by case - one prediction per case already
    predictions = np.array(all_predictions, dtype=np.float32)
    
    print(f'Evaluating on {len(predictions)} cases...\n')
    
    # Get labels for this task in the same order as cases
    labels_list = []
    for case in cases:
        case_row = labels_df[labels_df.case == int(case)]
        if len(case_row) > 0:
            labels_list.append(case_row[task].values[0])
        else:
            print(f'Warning: Case {case} not found in labels')
    
    labels = np.array(labels_list, dtype=np.float32)
    
    # Calculate AUC
    auc = metrics.roc_auc_score(labels, predictions)
    
    # Calculate Loss
    pos_weight = torch.tensor([1.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    preds_tensor = torch.tensor(predictions, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    loss = criterion(preds_tensor, labels_tensor).item()
    
    # Calculate threshold (optimal using Youden's J statistic)
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Get binary predictions using optimal threshold
    binary_preds = (predictions >= optimal_threshold).astype(int)
    
    # Calculate Sensitivity (True Positive Rate)
    sensitivity = metrics.recall_score(labels, binary_preds)
    
    # Calculate Specificity (True Negative Rate)
    tn, fp, fn, tp = metrics.confusion_matrix(labels, binary_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Calculate Accuracy
    accuracy = metrics.accuracy_score(labels, binary_preds)
    
    # Print results
    print('=' * 60)
    print(f'RESULTS FOR {task.upper()} - {plane.upper()}:')
    print('=' * 60)
    print(f'AUC:         {auc:.4f}')
    print(f'Loss:        {loss:.4f}')
    print(f'Sensitivity: {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Accuracy:    {accuracy:.4f}')
    print(f'Threshold:   {optimal_threshold:.4f}')
    print(f'\nConfusion Matrix:')
    print(f'  TP: {tp:<5} TN: {tn:<5}')
    print(f'  FP: {fp:<5} FN: {fn:<5}')
    print('=' * 60)


if __name__ == '__main__':
    arguments = docopt(__doc__)

    print('Parsing arguments...')

    main(arguments['<model_path>'],
         arguments['<data_paths_csv>'],
         arguments['<labels_csv>'],
         arguments['<task>'],
         arguments['<plane>'])
