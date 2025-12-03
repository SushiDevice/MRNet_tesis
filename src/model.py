#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    """MRNet model using EfficientNetB0 backbone with transfer learning support."""
    def __init__(self, freeze_backbone=True, unfreeze_blocks=None):
        """
        Initialize EfficientNetB0-based MRNet.
        
        Args:
            freeze_backbone: If True, freeze all backbone parameters initially
            unfreeze_blocks: List of block indices to unfreeze (0-6 for EfficientNetB0)
        """
        super().__init__()
        
        # Load EfficientNetB0 with ImageNet1K V1 pretrained weights
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)
        
        # Use only the features (not the classifier)
        self.features = backbone.features  # Returns features without classifier head
        
        # Get the number of output channels from EfficientNetB0
        # EfficientNetB0 outputs 1280 channels
        self.feature_dim = 1280
        
        # Simple linear classifier on top
        self.classifier = nn.Linear(self.feature_dim, 1)
        
        # Pooling and dropout match the original AlexNet design
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=None, padding=0)
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
        for name, param in self.features.named_parameters():
            if freeze_batchnorm and 'bn' in name.lower():
                # Keep BatchNorm trainable
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Always keep classifier trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze_blocks(self, block_indices, freeze_batchnorm=True):
        """
        Unfreeze specific blocks (0-6 for EfficientNetB0).
        
        Args:
            block_indices: List of block indices to unfreeze
            freeze_batchnorm: If True, keep BatchNorm layers frozen
        """
        # First freeze everything except classifier
        self.freeze_backbone(freeze_batchnorm=freeze_batchnorm)
        
        # Then unfreeze specified blocks
        for block_idx in block_indices:
            if block_idx < len(self.features):
                for param in self.features[block_idx].parameters():
                    param.requires_grad = True
    
    def unfreeze_last_n_blocks(self, n_blocks, freeze_batchnorm=True):
        """
        Unfreeze the last N blocks (0-6 for EfficientNetB0).
        
        Args:
            n_blocks: Number of blocks to unfreeze from the end
            freeze_batchnorm: If True, keep BatchNorm layers frozen
        """
        total_blocks = len(self.features)
        block_indices = list(range(max(0, total_blocks - n_blocks), total_blocks))
        self.unfreeze_blocks(block_indices, freeze_batchnorm=freeze_batchnorm)
    
    def unfreeze_all(self, freeze_batchnorm=True):
        """Unfreeze all layers except BatchNorm (if freeze_batchnorm=True)."""
        for name, param in self.features.named_parameters():
            if freeze_batchnorm and 'bn' in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def get_trainable_params_count(self):
        """Return count of trainable parameters."""
        backbone_trainable = sum(p.numel() for p in self.features.parameters() if p.requires_grad)
        classifier_trainable = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        return backbone_trainable + classifier_trainable
    
    def get_total_params_count(self):
        """Return total count of parameters."""
        backbone_total = sum(p.numel() for p in self.features.parameters())
        classifier_total = sum(p.numel() for p in self.classifier.parameters())
        return backbone_total + classifier_total

    def forward(self, batch):
        """
        Forward pass for a batch of MRI series (matches original AlexNet design).
        
        Args:
            batch: Tensor of shape (batch_size, num_images, channels, height, width)
        
        Returns:
            Predictions of shape (batch_size,)
        """
        batch_out = torch.tensor([], device=batch.device)

        for series in batch:
            out = torch.tensor([], device=batch.device)
            
            for image in series:
                # Extract features from backbone
                features = self.features(image.unsqueeze(0))  # [1, 1280, H, W]
                out = torch.cat((out, features), 0)
            
            # Aggregate features across all images in the series
            out = self.avg_pool(out).squeeze()  # [1280]
            out = out.max(dim=0, keepdim=True)[0].squeeze()  # Max pooling across images [1280]
            
            # Classification
            out = self.classifier(self.dropout(out))  # [1]
            
            batch_out = torch.cat((batch_out, out), 0)

        return batch_out

