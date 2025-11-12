#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision import models
import timm


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Use EfficientNetV2-M backbone to produce global pooled embeddings
        self.backbone = timm.create_model(
            'efficientnetv2_rw_m.agc_in1k',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        self.fc = nn.Linear(self.backbone.num_features, 1)

        self.dropout = nn.Dropout(p=0.5)

    @property
    def features(self):
        return self.backbone

    @property
    def classifier(self):
        return self.fc

    def forward(self, batch):
        batch_out = torch.tensor([]).to(batch.device)

        for series in batch:
            # Collect per-slice embeddings
            embeddings = torch.tensor([]).to(batch.device)
            for image in series:
                emb = self.features(image.unsqueeze(0)).squeeze(0)
                embeddings = torch.cat((embeddings, emb.unsqueeze(0)), 0)

            # Temporal max-pooling across slices
            out = embeddings.max(dim=0, keepdim=True)[0].squeeze()

            out = self.classifier(self.dropout(out))

            batch_out = torch.cat((batch_out, out), 0)

        return batch_out
