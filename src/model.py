#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision.models import mnasnet1_3, MNASNet1_3_Weights


class MRNet(nn.Module):
    """
    MRNet usando MNASNet 1.3 preentrenado de torchvision como backbone.

    - Entrada: batch de tamaño B, donde cada elemento es una serie de slices
      con forma [num_slices, 3, H, W].
    - Salida: tensor 1D de tamaño B con un logit por estudio.
    """

    def __init__(self):
        super().__init__()

        # Backbone MNASNet 1.3 preentrenado
        # Docs: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mnasnet1_3.html
        weights = MNASNet1_3_Weights.IMAGENET1K_V1
        backbone = mnasnet1_3(weights=weights)

        # Obtener la dimensionalidad de las features antes del clasificador final.
        # En MNASNet, classifier es un nn.Sequential; buscamos la última capa Linear.
        last_linear = None
        for m in reversed(backbone.classifier):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is None:
            raise RuntimeError("No se encontró una capa Linear en backbone.classifier de MNASNet.")
        in_features = last_linear.in_features

        # Usamos MNASNet solo como extractor de características (sin su clasificador ImageNet).
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features, 1)

    @property
    def features(self):
        # Compatibilidad externa: devuelve el extractor de características
        return self.backbone

    @property
    def classifier(self):
        return self.fc

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrae un vector de características 1D a partir de un solo slice.
        x: tensor [1, 3, H, W]
        return: tensor [1, C]
        """
        feats = self.backbone(x)  # [1, C]
        feats = feats.view(feats.size(0), -1)
        return feats

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        batch: tensor con forma [B, num_slices, 3, H, W]
        return: tensor 1D con forma [B], un logit por estudio.
        """
        device = batch.device
        batch_out = torch.empty(0, device=device)

        for series in batch:
            # series: [num_slices, 3, H, W]
            slice_feats = torch.empty(0, self.fc.in_features, device=device)

            for image in series:
                # image: [3, H, W]
                img_batch = image.unsqueeze(0)  # [1, 3, H, W]
                feats = self._extract_features(img_batch)  # [1, C]
                slice_feats = torch.cat((slice_feats, feats), dim=0)  # [num_slices, C]

            # Max pooling temporal sobre los slices: [num_slices, C] -> [1, C]
            agg_feats, _ = slice_feats.max(dim=0, keepdim=True)

            # Clasificador final -> [1, 1] -> [1]
            logit = self.fc(self.dropout(agg_feats)).view(-1)
            batch_out = torch.cat((batch_out, logit), dim=0)

        return batch_out
