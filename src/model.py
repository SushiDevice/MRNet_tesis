#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        regnet = models.regnet_y_3_2gf(weights='IMAGENET1K_V1')
        # Extract only the features (backbone), similar to alexnet.features
        self.features_backbone = nn.Sequential(*list(regnet.children())[:-1])
        self.fc = nn.Linear(1512, 1)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=None, padding=0)
        self.dropout = nn.Dropout(p=0.5)

    @property
    def features(self):
        return self.features_backbone

    @property
    def classifier(self):
        return self.fc

    def forward(self, batch):
        batch_out = torch.tensor([]).to(batch.device)

        for series in batch:
            out = torch.tensor([]).to(batch.device)
            for image in series:
                features = self.features(image.unsqueeze(0))
                # Flatten spatial dimensions if they exist
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)
                out = torch.cat((out, features), 0)

            out = out.max(dim=0, keepdim=True)[0].squeeze()

            out = self.classifier(self.dropout(out))

            batch_out = torch.cat((batch_out, out), 0)

        return batch_out
