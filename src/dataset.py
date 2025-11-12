import os
from glob import glob
from PIL import Image

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import timm

from utils import preprocess_data


class MRNetDataset(Dataset):
    def __init__(self, dataset_dir, labels_path, plane, transform=None, device=None, max_cases=None):
        self.case_paths = sorted(glob(f'{dataset_dir}/{plane}/**.npy'))
        if max_cases is not None:
            self.case_paths = self.case_paths[:int(max_cases)]
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform
        self.window = 7
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.case_paths)

    def __getitem__(self, idx):
        case_path = self.case_paths[idx]
        series = preprocess_data(case_path, self.transform)

        case_id = int(os.path.splitext(os.path.basename(case_path))[0])
        case_row = self.labels_df[self.labels_df.case == case_id]
        diagnoses = case_row.values[0,1:].astype(np.float32)
        labels = torch.tensor(diagnoses)

        return (series, labels)


def make_dataset(data_dir, dataset_type, plane, device=None, max_cases=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_dir = f'{data_dir}/{dataset_type}'
    labels_path = f'{data_dir}/{dataset_type}_labels.csv'

    # Build model-specific transforms to match EfficientNetV2-M expectations
    if dataset_type not in ('train', 'valid'):
        raise ValueError('Dataset needs to be train or valid.')
    tmp_model = timm.create_model('efficientnetv2_rw_m.agc_in1k', pretrained=True, num_classes=0, global_pool='avg')
    data_config = timm.data.resolve_model_data_config(tmp_model)
    base_transform = timm.data.create_transform(**data_config, is_training=(dataset_type == 'train'))
    # Ensure we can feed tensor slices by converting to PIL first
    transform = transforms.Compose([
        transforms.ToPILImage(),
        base_transform
    ])

    dataset = MRNetDataset(dataset_dir, labels_path, plane, transform=transform, device=device, max_cases=max_cases)

    return dataset
