#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    """MRNet model using EfficientNetB0 backbone with fine-tuning support."""
    def __init__(self, freeze_backbone=True, unfreeze_blocks=None):
        """
        Initialize EfficientNetB0-based MRNet.
        
        Args:
            freeze_backbone: If True, freeze all backbone parameters initially
            unfreeze_blocks: List of block indices to unfreeze (0-6 for EfficientNetB0)
                           If None and freeze_backbone=False, all blocks are trainable
        """
        super().__init__()
        
        # Load EfficientNetB0 with ImageNet1K V1 pretrained weights
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Store original classifier for reference
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with a new one for binary classification
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 1)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=0.5)
        
        # Apply initial freezing if requested
        if freeze_backbone:
            self.freeze_backbone(freeze_batchnorm=True)
        
        # Unfreeze specific blocks if requested
        if unfreeze_blocks is not None:
            self.unfreeze_blocks(unfreeze_blocks, freeze_batchnorm=True)
    
    def freeze_backbone(self, freeze_batchnorm=True):
        """Freeze backbone feature extraction, but keep classifier trainable."""
        # Freeze features but not the classifier
        for name, param in self.backbone.features.named_parameters():
            if freeze_batchnorm and 'bn' in name.lower():
                # Keep BatchNorm trainable
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Always keep classifier trainable
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze_blocks(self, block_indices, freeze_batchnorm=True):
        """
        Unfreeze specific blocks (0-6 for EfficientNetB0).
        BatchNorm layers are kept frozen unless specified otherwise.
        
        Args:
            block_indices: List of block indices to unfreeze
            freeze_batchnorm: If True, keep BatchNorm layers frozen
        """
        # First freeze everything except classifier
        self.freeze_backbone(freeze_batchnorm=freeze_batchnorm)
        
        # Then unfreeze specified blocks
        if hasattr(self.backbone, 'features'):
            features = self.backbone.features
            for block_idx in block_indices:
                if block_idx < len(features):
                    for param in features[block_idx].parameters():
                        param.requires_grad = True
    
    def unfreeze_last_n_blocks(self, n_blocks, freeze_batchnorm=True):
        """
        Unfreeze the last N blocks (0-6 for EfficientNetB0).
        
        Args:
            n_blocks: Number of blocks to unfreeze from the end
            freeze_batchnorm: If True, keep BatchNorm layers frozen
        """
        if hasattr(self.backbone, 'features'):
            total_blocks = len(self.backbone.features)
            block_indices = list(range(max(0, total_blocks - n_blocks), total_blocks))
            self.unfreeze_blocks(block_indices, freeze_batchnorm=freeze_batchnorm)
    
    def unfreeze_all(self, freeze_batchnorm=True):
        """
        Unfreeze all layers except BatchNorm (if freeze_batchnorm=True).
        """
        for name, param in self.backbone.named_parameters():
            if freeze_batchnorm and 'bn' in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def get_trainable_params_count(self):
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
    
    def get_total_params_count(self):
        """Return total count of parameters."""
        return sum(p.numel() for p in self.backbone.parameters())
    
    @property
    def features(self):
        """Returns the backbone for feature extraction."""
        return self.backbone
    
    @property
    def classifier(self):
        """Returns the classifier layer."""
        return self.backbone.classifier

    def forward(self, batch):
        """
        Forward pass for a batch of MRI series.
        
        Args:
            batch: Tensor of shape (batch_size, num_images, channels, height, width)
        
        Returns:
            Predictions of shape (batch_size,)
        """
        batch_out = []

        for series in batch:
            out_list = []
            
            for image in series:
                # Get features from backbone
                features = self.backbone.features(image.unsqueeze(0))
                out_list.append(features)
            
            # Concatenate features from all images in the series
            if out_list:
                out = torch.cat(out_list, 0)  # (num_images, num_features)
                
                # Aggregate features across images in the series
                out = self.avg_pool(out).squeeze(dim=-1).squeeze(dim=-1)  # (num_images, num_features)
                out = out.max(dim=0, keepdim=True)[0].squeeze()  # Max pooling across images

                # Classification
                out = self.backbone.classifier(self.dropout(out.unsqueeze(0)))
                batch_out.append(out)

        if batch_out:
            return torch.cat(batch_out, 0)
        else:
            return torch.tensor([], device=batch.device, dtype=torch.float32)
