# EfficientNetB0 Fine-Tuning Implementation Guide

## Overview

This implementation adds support for EfficientNetB0 backbone to MRNet with proper fine-tuning strategies optimized for transfer learning on the MRNet dataset.

## Key Changes

### 1. New Model Class: `MRNetEfficientNetB0`

Located in `src/model.py`, this class implements:

- **ImageNet1K V1 Pretrained Weights**: Loads EfficientNetB0 with the best publicly available ImageNet weights
- **Progressive Layer Unfreezing**: Supports unfreezing specific blocks while keeping BatchNorm frozen
- **Fine-tuning Methods**:
  - `freeze_backbone()`: Freeze all backbone parameters
  - `unfreeze_blocks()`: Unfreeze specific blocks (0-6 for EfficientNetB0)
  - `unfreeze_last_n_blocks()`: Unfreeze the last N blocks
  - `unfreeze_all()`: Unfreeze all layers except BatchNorm

### 2. New Training Script: `train_efficientnet_finetuned.py`

Implements a two-phase training strategy:

#### Phase 1: Warmup (frozen backbone)
- Train only the new classifier head
- Prevents corrupting pretrained weights
- Default: 5 epochs
- Higher learning rate is safe here

#### Phase 2: Fine-tuning (gradual unfreezing)
- Selectively unfreeze blocks after warmup
- Recommended: unfreeze blocks 5-6 (last blocks)
- Lower learning rates to preserve pretrained knowledge
- Continues with LR scheduler

## Fine-Tuning Best Practices Implemented

### ✓ BatchNorm Handling
- BatchNorm layers are **always kept frozen** during fine-tuning
- This is critical - unfrozen BatchNorm after warmup significantly reduces accuracy
- Implementation: `freeze_batchnorm=True` in all unfreezing methods

### ✓ Block-Level Unfreezing
- EfficientNetB0 has 7 blocks (0-6)
- Unfreezing respects block boundaries (shortcut connections within blocks)
- Partial unfreezing (e.g., blocks 5-6) is more efficient than unfreezing all
- Default: Unfreezes only last 2 blocks (5, 6)

### ✓ Learning Rate for Transfer Learning
- **NOT using RMSprop** (too high momentum for transfer learning)
- Using **Adam optimizer** with careful learning rate selection
- Default LR: `0.0001` (100x lower than typical training)
- Rationale: Preserves pretrained weights while adapting to new task
- Safeguard: Check loss doesn't exceed `log(num_classes)` ≈ 1.1 for 3 classes

### ✓ Weight Decay
- Default: `0.001` (lower than typically used for training from scratch)
- Prevents aggressive updates that could corrupt pretrained features

### ✓ Smaller Batch Size
- Default: `4` (much smaller than typical training)
- Smaller batches provide regularization
- Improves validation accuracy in transfer learning
- Trade-off: Slightly noisier gradients, but better generalization

### ✓ No EMA (Exponential Moving Average)
- EMA is beneficial when training from scratch
- Not used for transfer learning (can hurt performance)

## Usage

### Basic Usage

```bash
# Train with default fine-tuning settings
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50

# Train with custom learning rate
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50 --lr=0.00005

# Train all three planes
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50
python src/train_efficientnet_finetuned.py MRNet-v1.0 coronal 50
python src/train_efficientnet_finetuned.py MRNet-v1.0 sagittal 50
```

### Advanced Options

```bash
# Full control over fine-tuning strategy
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50 \
  --lr=0.0001 \
  --weight-decay=0.001 \
  --warmup-epochs=3 \
  --unfreeze-blocks=4,5,6 \
  --batch-size=4 \
  --batch-size-val=4 \
  --device=cuda

# Unfreezing different blocks
# Unfreeze only last block:
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50 --unfreeze-blocks=6

# Unfreeze last 3 blocks:
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50 --unfreeze-blocks=4,5,6

# Unfreeze all blocks (not recommended):
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50 --unfreeze-blocks=0,1,2,3,4,5,6
```

## Performance Monitoring

The training script logs trainable parameter counts at key points:

```
--- Initial State (Backbone Frozen) ---
Model 0: 9,109/4,049,571 trainable params (0.2%)

--- After Unfreezing (Epoch 6) ---
Model 0: 1,234,567/4,049,571 trainable params (30.5%)
```

This helps verify that unfreezing is working as expected.

## Troubleshooting

### Problem: Loss becomes very large (> 1.5) after unfreezing
**Solution**: Learning rate is too high. Try `--lr=0.00005` or lower.

### Problem: Validation accuracy drops after warmup
**Solution**: 
1. Check if BatchNorm layers are frozen (they should be)
2. Try longer warmup phase: `--warmup-epochs=10`
3. Reduce learning rate: `--lr=0.00005`

### Problem: Model overfitting on validation set
**Solution**:
1. Increase batch size: `--batch-size=8`
2. Use more weight decay: `--weight-decay=0.01`
3. Unfreeze fewer blocks: `--unfreeze-blocks=6`

### Problem: Training is too slow
**Solution**:
1. Reduce number of blocks to unfreeze: `--unfreeze-blocks=6`
2. Keep warmup phase short: `--warmup-epochs=2`
3. Use larger batch size: `--batch-size=8` (trades validation accuracy for speed)

## Architecture Details

### EfficientNetB0 Structure
- **Input**: 224×224 RGB images
- **Backbone Blocks**: 7 inverted residual blocks
- **Features**: ~1.3M parameters (much smaller than ResNet50)
- **Output Features**: 1280-dimensional

### MRNetEfficientNetB0 Head
- **Adaptive Average Pooling**: Aggregates spatial dimensions
- **Dropout**: 30% (matches EfficientNetB0 convention)
- **Linear Classifier**: 1280 → 1 (binary classification logits)

### Series Aggregation
- Max pooling across images in a series (same as original MRNet)
- Adaptive pooling handles variable input sizes

## Recommended Fine-Tuning Strategies

### Conservative (Recommended for smaller datasets):
```bash
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50 \
  --warmup-epochs=5 \
  --unfreeze-blocks=6 \
  --lr=0.0001 \
  --batch-size=4
```
- Only unfreezes last block (best for preventing overfitting)
- Longer warmup ensures classifier is well-trained

### Moderate (Balanced):
```bash
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50 \
  --warmup-epochs=5 \
  --unfreeze-blocks=5,6 \
  --lr=0.0001 \
  --batch-size=4
```
- Unfreezes last 2 blocks (good balance of adaptation and stability)

### Aggressive (Use with caution):
```bash
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50 \
  --warmup-epochs=10 \
  --unfreeze-blocks=3,4,5,6 \
  --lr=0.00005 \
  --batch-size=8
```
- Unfreezes more blocks for maximum adaptation
- Requires very low learning rate and longer warmup
- Risk of overfitting on smaller validation sets

## Comparison with Original MRNet

| Aspect | Original MRNet | EfficientNetB0 |
|--------|----------------|----------------|
| Backbone | AlexNet | EfficientNetB0 |
| Parameters | ~60M | ~4M |
| Pretrained | ImageNet | ImageNet1K V1 |
| Training Strategy | Standard | Progressive unfreezing |
| BatchNorm | Not frozen | Frozen during fine-tuning |
| Learning Rate | ~0.00001 | ~0.0001 (warmup) → 0.00005 |
| Batch Size | Not specified | 4 (transfer learning) |

## Expected Results

After implementing this fine-tuning strategy, you should see:

1. **Stable training loss** during warmup phase
2. **Validation loss improvement** after unfreezing appropriate blocks
3. **Better generalization** due to careful learning rate and batch size
4. **No sudden accuracy drops** after unfreezing (if BatchNorm is frozen)

## Next Steps for Optimization

If performance is still not satisfactory:

1. **Try different block combinations**: Experiment with `--unfreeze-blocks`
2. **Adjust learning rate schedule**: Modify scheduler patience/factor in code
3. **Use data augmentation**: Implement stronger augmentation for small MRNet dataset
4. **Ensemble multiple views**: Combine predictions from axial, coronal, sagittal
5. **Increase training epochs**: Fine-tuning may need more epochs (100+)
6. **Collect more data**: Transfer learning works best with sufficient training data
