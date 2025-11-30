#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        # Usar EfficientNet-B3 como extractor de características
        self.efficientnet.classifier = nn.Identity()
        # Dimensión de embedding de EfficientNet-B3 es 1536
        self.fc = nn.Linear(1536, 1)
        
        self.dropout = nn.Dropout(p=0.5)

    @property
    def features(self):
        return self.efficientnet

    @property
    def classifier(self):
        return self.fc

    def forward(self, batch):
        batch_outputs = []
        for series in batch:
            # series: [S, C, H, W]
            feats = self.features(series)  # [S, 1536] (GAP + flatten ya aplicados en EfficientNet)
            # Agregación inter-slice en espacio de features (MRNet estricto)
            agg_feats = feats.max(dim=0).values  # [1536]
            out = self.classifier(self.dropout(agg_feats))  # [1]
            batch_outputs.append(out.squeeze())
        return torch.stack(batch_outputs, dim=0)
