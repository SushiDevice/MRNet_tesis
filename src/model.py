#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = models.alexnet(pretrained=True).features
        self.fc = nn.Linear(256, 1)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=None, padding=0)
        self.dropout = nn.Dropout(p=0.5)

    @property
    def features(self):
        return self.alexnet

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


class ConvNextTiny(nn.Module):
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


class ResNext50(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone ResNeXt50_32x4d preentrenada en ImageNet1K V1
        self.backbone = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        )

        # Usamos solo la parte convolucional (hasta antes de avgpool y fc)
        self.cnn_features = nn.Sequential(*list(self.backbone.children())[:-2])

        # La ResNeXt50_32x4d produce un vector de tamaño backbone.fc.in_features (normalmente 2048)
        self.fc = nn.Linear(self.backbone.fc.in_features, 1)

        # Pooling espacial adaptable a cualquier tamaño de entrada
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)

    @property
    def features(self):
        return self.cnn_features

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
