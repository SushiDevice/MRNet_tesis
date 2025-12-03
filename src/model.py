#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    """MRNet model using Swin Transformer backbone with ImageNet1K_V1 weights."""
    
    def __init__(self):
        super().__init__()
        # Load Swin Transformer with IMAGENET1K_V1 weights
        swin = models.swin_t(weights='IMAGENET1K_V1')
        
        # Extract features (all layers except the classification head)
        # This includes the patch embedding and all transformer blocks
        self.features = nn.Sequential(*list(swin.children())[:-1])
        
        # Swin-T output feature dimension is 768
        self.fc = nn.Linear(768, 1)
        
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, batch):
        batch_out = torch.tensor([]).to(batch.device)

        for series in batch:
            out = []
            for image in series:
                # Process each image through Swin backbone
                # Swin outputs a 1D feature vector (batch_size, 768) after all layers
                feat = self.features(image.unsqueeze(0))  # (1, 768)
                out.append(feat)

            # Stack all image features: (n_images, 768)
            out = torch.cat(out, dim=0)
            
            # Temporal aggregation: max pool over image sequence
            out = out.max(dim=0, keepdim=True)[0].squeeze(0)  # (768,)

            # Classification
            out = self.fc(self.dropout(out))

            batch_out = torch.cat((batch_out, out), 0)

        return batch_out
