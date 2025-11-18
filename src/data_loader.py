import torch
from torch.utils.data import DataLoader, Subset

from dataset import make_dataset


def make_data_loader(data_dir, dataset_type, plane, device=None, shuffle=False, max_cases=None, indices=None, transform_type=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = make_dataset(data_dir, dataset_type, plane, device=device, max_cases=max_cases, transform_type=transform_type)

    if indices is not None:
        dataset = Subset(dataset, indices)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)

    return data_loader
