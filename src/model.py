#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class MRNet(nn.Module):
    """
    MRNet usando DenseNet-121 preentrenada en ImageNet-1K como backbone.

    - Entrada: batch de tamaño B, donde cada elemento es una serie de slices
      con forma [num_slices, 3, H, W].
    - Salida: tensor 1D de tamaño B con un logit por estudio.
    """

    def __init__(self):
        super().__init__()

        # Backbone DenseNet-121 con pesos preentrenados en ImageNet-1K
        # Docs: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html
        weights = DenseNet121_Weights.IMAGENET1K_V1
        backbone = densenet121(weights=weights)

        # DenseNet-121 expone las features en backbone.features y la capa final
        # de clasificación en backbone.classifier (Linear)
        in_features = backbone.classifier.in_features

        # Definimos un extractor de características que replica el forward de DenseNet:
        # features -> ReLU -> AdaptiveAvgPool2d -> Flatten
        self.feature_extractor = nn.Sequential(
            backbone.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1)
        )

        self.fc = nn.Linear(in_features, 1)
        self.dropout = nn.Dropout(p=0.5)

    @property
    def features(self):
        return self.feature_extractor

    @property
    def classifier(self):
        return self.fc

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        batch: tensor con forma [B, num_slices, 3, H, W]
        return: tensor 1D con forma [B], un logit por estudio.
        """
        device = batch.device
        outputs = []

        for series in batch:
            # series: [num_slices, 3, H, W]
            slice_embeddings = []
            for image in series:
                # image: [3, H, W] -> [1, 3, H, W]
                emb = self.features(image.unsqueeze(0).to(device))  # [1, C]
                slice_embeddings.append(emb.squeeze(0))  # [C]

            # [num_slices, C]
            series_emb = torch.stack(slice_embeddings, dim=0)

            # Max pooling temporal sobre slices -> [C]
            series_emb, _ = series_emb.max(dim=0)

            # Clasificador final -> [1]
            logit = self.classifier(self.dropout(series_emb)).view(1)
            outputs.append(logit)

        return torch.cat(outputs, dim=0)
