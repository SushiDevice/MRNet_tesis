#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        backbone = convnext_tiny(weights=weights)
        self.feature_extractor = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(1)
        )
        in_features = backbone.classifier[2].in_features
        self.fc = nn.Linear(in_features, 1)
        self.dropout = nn.Dropout(p=0.5)

    @property
    def features(self):
        return self.feature_extractor

    @property
    def classifier(self):
        return self.fc

    def forward(self, batch):
        outputs = []

        for series in batch:
            slice_embeddings = []
            for image in series:
                emb = self.features(image.unsqueeze(0))
                slice_embeddings.append(emb.squeeze(0))

            series_emb = torch.stack(slice_embeddings)
            series_emb = series_emb.max(dim=0).values

            logit = self.classifier(self.dropout(series_emb))
            outputs.append(logit)

        return torch.cat(outputs, dim=0)
