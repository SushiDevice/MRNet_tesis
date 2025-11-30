#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1).features
        self.fc = nn.Linear(1536, 1)

        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=None, padding=0)
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
                out = torch.cat((out, self.features(image.unsqueeze(0))), 0)

            out = self.avg_pool(out).squeeze()
            out = out.max(dim=0, keepdim=True)[0].squeeze()

            out = self.classifier(self.dropout(out))

            batch_out = torch.cat((batch_out, out), 0)

        return batch_out
