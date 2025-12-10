#!/usr/bin/env python3.6
"""Trains logistic regression models for abnormalities, ACL tears and meniscal
tears, by combining predictions from CNN ensemble models (AlexNet, ConvNext Tiny, 
and ResNext50).

Usage:
  new_train_lr_ensamble.py <data_dir> <models_dir>
  new_train_lr_ensamble.py (-h | --help)

General options:
  -h --help         Show this screen.

Arguments:
  <data_dir>        Path to a directory where the data lives e.g. 'MRNet-v1.0'
  <models_dir>      Directory where CNN models are saved e.g. 'models/2019-06-24_04-18'
"""

import sys
from glob import glob
from tqdm import tqdm
from docopt import docopt

import torch
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import joblib

from model import MRNet, ConvNextTiny, ResNext50
from data_loader import make_data_loader


def main(data_dir, models_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    planes = ['axial', 'coronal', 'sagittal']
    conditions = ['abnormal', 'acl', 'meniscus']
    architectures = ['alexnet', 'convnext', 'resnext50']

    # models será una lista de longitud 3 (una por condición),
    # donde cada elemento es un dict:
    # { 'axial': [model1, model2, ...],
    #   'coronal': [...],
    #   'sagittal': [...] }
    models = []

    print(f'Loading best CNN ensemble models from {models_dir}...')

    for condition in conditions:
        # Un contenedor por plano para esta condición
        models_per_condition = {plane: [] for plane in planes}
        
        # Load AlexNet models
        for plane in planes:
            checkpoint_pattern = glob(f'{models_dir}/alexnet_*{plane}*{condition}*.pt')
            if checkpoint_pattern:
                checkpoint_path = sorted(checkpoint_pattern)[-1]
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                model = MRNet().to(device)
                model.load_state_dict(checkpoint['state_dict'])
                models_per_condition[plane].append(model)
        
        # Load ConvNext Tiny models
        for plane in planes:
            checkpoint_pattern = glob(f'{models_dir}/convnext_*{plane}*{condition}*.pt')
            if checkpoint_pattern:
                checkpoint_path = sorted(checkpoint_pattern)[-1]
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                model = ConvNextTiny().to(device)
                model.load_state_dict(checkpoint['state_dict'])
                models_per_condition[plane].append(model)
        
        # Load ResNext50 models
        for plane in planes:
            checkpoint_pattern = glob(f'{models_dir}/resnext50_*{plane}*{condition}*.pt')
            if checkpoint_pattern:
                checkpoint_path = sorted(checkpoint_pattern)[-1]
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                model = ResNext50().to(device)
                model.load_state_dict(checkpoint['state_dict'])
                models_per_condition[plane].append(model)

        models.append(models_per_condition)

    print(f'Creating data loaders...')

    axial_loader = make_data_loader(data_dir, 'train_split1', 'axial')
    coronal_loader = make_data_loader(data_dir, 'train_split1', 'coronal')
    sagittal_loader = make_data_loader(data_dir, 'train_split1', 'sagittal')

    print(f'Collecting predictions on train dataset from the ensemble models...')

    ys = []
    Xs = [[],[],[]]  # Abnormal, ACL, Meniscus

    with tqdm(total=len(axial_loader)) as pbar:
        for (axial_inputs, labels), (coronal_inputs, _), (sagittal_inputs, _) in \
                zip(axial_loader, coronal_loader, sagittal_loader):

            axial_inputs, coronal_inputs, sagittal_inputs = \
                axial_inputs.to(device), coronal_inputs.to(device), sagittal_inputs.to(device)

            ys.append(labels[0].cpu().tolist())

            # Para cada condición, recolectamos las predicciones de TODOS
            # los modelos asociados a cada plano.
            for cond_idx, condition_models in enumerate(models):
                X = []

                # Orden consistente de features: primero todos los axiales,
                # luego todos los coronales y por último los sagitales.
                for model in condition_models['axial']:
                    axial_pred = model(axial_inputs).detach().cpu().item()
                    X.append(axial_pred)

                for model in condition_models['coronal']:
                    coronal_pred = model(coronal_inputs).detach().cpu().item()
                    X.append(coronal_pred)

                for model in condition_models['sagittal']:
                    sagittal_pred = model(sagittal_inputs).detach().cpu().item()
                    X.append(sagittal_pred)

                Xs[cond_idx].append(X)

            pbar.update(1)

    ys = np.asarray(ys).transpose()

    print(f'Training logistic regression models for each condition...')

    clfs = []

    # Importante: cada condición puede tener un número distinto de modelos
    # (por ejemplo, algunas con AlexNet+ConvNeXt+ResNext50 y otras con menos).
    # Por eso no convertimos Xs en un único array 3D; en su lugar,
    # trabajamos condición por condición.
    for X, y in zip(Xs, ys):
        X = np.asarray(X)  # (n_casos, n_features_condicion)
        clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
        clfs.append(clf)

    # Imprimir score usando los mismos datos con los que se entrenó
    for i, (X_cond, y_cond, clf) in enumerate(zip(Xs, ys, clfs)):
        X_cond = np.asarray(X_cond)
        print(f'Cross validation score for {conditions[i]}: {clf.score(X_cond, y_cond):.3f}')
        clf_path = f'{models_dir}/lr_ensemble_{conditions[i]}.pkl'
        joblib.dump(clf, clf_path)

    print(f'Logistic regression ensemble models saved to {models_dir}')


if __name__ == '__main__':
    arguments = docopt(__doc__)

    print('Parsing arguments...')

    main(arguments['<data_dir>'],
         arguments['<models_dir>'])
