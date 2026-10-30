"""Copia de dataset.py SIN aumento de datos.

Uso previsto: etapa de regresion logistica (new_train_lr_models.py,
new_train_lr_ensamble.py). Segun el paper MRNet (Bien et al., 2018) el aumento
de datos (flip horizontal + RandomAffine) se aplica solo al entrenar las CNN.
Las predicciones que alimentan la regresion se obtienen con las CNN congeladas
sobre imagenes sin transformar, en modo inferencia.

Acepta exactamente los mismos argumentos que dataset.make_dataset. La unica
diferencia: cualquier split (train, train_split1, valid, test_split1, ...) se
carga con la transformacion plana ToPILImage -> ToTensor.
"""

import torch
from torchvision import transforms

from dataset import MRNetDataset


def make_dataset(data_dir, dataset_type, plane, device=None, max_cases=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_dir = f'{data_dir}/{dataset_type}'
    labels_path = f'{data_dir}/{dataset_type}_labels.csv'

    # Sin aumento de datos para ningun split: solo inferencia.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    dataset = MRNetDataset(dataset_dir, labels_path, plane, transform=transform, device=device, max_cases=max_cases)

    return dataset
