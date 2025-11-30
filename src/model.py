#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = models.efficientnet_b3(weights='IMAGENET1K_V1')
        # Remove the classification head to use as a feature extractor
        self.efficientnet.classifier = nn.Identity()
        self.fc = nn.Linear(1536, 1)

        self.dropout = nn.Dropout(p=0.5)

    @property
    def features(self):
        return self.efficientnet

    @property
    def classifier(self):
        return self.fc

    def forward(self, batch):
        batch_out = torch.tensor([]).to(batch.device)

        for series in batch:
            out = torch.tensor([]).to(batch.device)
            for image in series:
                # EfficientNet_B3 outputs (1, 1536) after adaptive avg pooling
                features = self.features(image.unsqueeze(0))
                # Flatten to 1D if needed
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)
                out = torch.cat((out, features), 0)

            # out shape is (num_images, 1536)
            # Take max across all images in the series
            out = out.max(dim=0, keepdim=True)[0].squeeze()

            out = self.classifier(self.dropout(out))

            batch_out = torch.cat((batch_out, out), 0)

        return batch_out
