"""Copia de data_loader.py que usa dataset_no_aug (sin aumento de datos).

Misma firma que data_loader.make_data_loader. Para la etapa de regresion
logistica: las CNN congeladas deben ver las imagenes tal cual las veran en
inferencia (predict.py), no versiones rotadas/desplazadas/volteadas.
"""

import torch
from torch.utils.data import DataLoader

from dataset_no_aug import make_dataset


def make_data_loader(data_dir, dataset_type, plane, device=None, shuffle=False, max_cases=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = make_dataset(data_dir, dataset_type, plane, device=device, max_cases=max_cases)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)

    return data_loader
