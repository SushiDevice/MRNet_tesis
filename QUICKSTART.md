# Quick Start Guide - EfficientNetB0 Fine-Tuning

## 30-Second Setup

Your implementation is ready to use! Here's what to do:

### Option 1: Run Immediately (Recommended for Testing)

```bash
# Train axial plane with default fine-tuning
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50
```

**This will:**
- Use LR=0.0001, batch size=4, warmup=5 epochs
- Unfreeze blocks 5-6 after warmup
- Save checkpoints to `models/YYYY-MM-DD_HH-MM/`

### Option 2: Understand Before Running

```bash
# See what the model looks like
python inspect_model.py --all

# Then run with conservative settings
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50 \
  --unfreeze-blocks=6 \
  --batch-size=4 \
  --lr=0.0001
```

### Option 3: Run All Three Planes

```bash
# PowerShell script
foreach ($plane in "axial", "coronal", "sagittal") {
    python src/train_efficientnet_finetuned.py MRNet-v1.0 $plane 50 `
        --lr=0.0001 `
        --warmup-epochs=5 `
        --unfreeze-blocks=5,6 `
        --batch-size=4
}
```

Or on Linux/Mac:
```bash
for plane in axial coronal sagittal; do
  python src/train_efficientnet_finetuned.py MRNet-v1.0 $plane 50 \
    --lr=0.0001 --warmup-epochs=5 --unfreeze-blocks=5,6 --batch-size=4
done
```

## Key Parameters Explained

| Parameter | Default | What it does |
|-----------|---------|------------|
| `<plane>` | Required | axial, coronal, or sagittal |
| `<epochs>` | Required | Total training epochs |
| `--lr` | 0.0001 | Learning rate (lower = safer but slower) |
| `--warmup-epochs` | 5 | How long to keep backbone frozen |
| `--unfreeze-blocks` | 5,6 | Which blocks to unfreeze (0-6 available) |
| `--batch-size` | 4 | Training batch size (smaller = more regularization) |
| `--weight-decay` | 0.001 | L2 regularization strength |

## Three Quick Presets

### Conservative (Safe, Recommended)
```bash
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50 \
  --unfreeze-blocks=6 --batch-size=4 --lr=0.0001
```
- Single block unfreezing
- Smallest batch size
- Best for preventing overfitting

### Balanced (Default, Good)
```bash
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50 \
  --unfreeze-blocks=5,6 --batch-size=4 --lr=0.0001
```
- Two blocks unfreezing
- Good adaptation + stability
- Our recommended starting point

### Aggressive (Fast Adaptation)
```bash
python src/train_efficientnet_finetuned.py MRNet-v1.0 axial 50 \
  --unfreeze-blocks=3,4,5,6 --batch-size=8 --lr=0.00005
```
- More blocks unfreezing
- Larger batch size (faster training)
- Risk: needs more data to avoid overfitting

## What to Expect

### During Training (Watch the output)

```
=== Epoch 1/50 ===
Model 0: 9,109/4,049,571 trainable params (0.2%)
[abnormal]  train_loss: 0.8234  val_loss: 0.7891  auc: 0.642

=== Epoch 6/50 ===     <-- After warmup, blocks unfreeze here
Model 0: 1,234,567/4,049,571 trainable params (30.5%)
[abnormal]  train_loss: 0.6234  val_loss: 0.6542  auc: 0.721
```

**Good signs:**
- ✓ Loss decreases gradually
- ✓ Trainable params increase at epoch 6
- ✓ No sudden accuracy drop after unfreezing

**Warning signs:**
- ✗ Loss increases after unfreezing → Learning rate too high
- ✗ Loss > 1.5 → Something's wrong, stop and adjust
- ✗ Validation loss not improving → May need different settings

## Troubleshooting in 30 Seconds

| Problem | Solution |
|---------|----------|
| **Loss explodes** | `--lr=0.00005` |
| **Too slow** | `--batch-size=8` |
| **Overfitting** | `--weight-decay=0.01` |
| **Accuracy drops** | `--unfreeze-blocks=6` |
| **Out of memory** | `--batch-size=2` |

## File Organization

After training, you'll see:
```
models/
  2025-12-02_15-30/          <- Your training run
    cnn_axial_abnormal_01.pt <- Best checkpoint
    cnn_axial_acl_03.pt
    cnn_axial_meniscus_02.pt
    losses_axial.csv         <- Training history
```

## Using Trained Models

The checkpoint files are ready to load with your existing prediction code:
```python
from src.model import MRNetEfficientNetB0
import torch

model = MRNetEfficientNetB0()
model.load_state_dict(torch.load('models/2025-12-02_15-30/cnn_axial_abnormal_01.pt'))
# Use model for prediction
```

## Important Reminders

1. **Always start conservative** → adjust from there
2. **Smaller batches (4) are better** → use 8+ only if overfitting
3. **Lower learning rates are safer** → 0.0001 is standard, go lower if unsure
4. **Monitor the first epoch** → if loss is already bad, stop and adjust
5. **Don't change too many things at once** → change one parameter, retrain

## Next: Documentation

For more details, read:
- `EFFICIENTNET_FINETUNING_GUIDE.md` - Comprehensive guide
- `CONFIG_REFERENCE.py` - All configuration options
- `IMPLEMENTATION_SUMMARY.md` - What was implemented

## One Last Thing

The key to good transfer learning is **patience**:
- Don't rush to unfreeze all blocks
- Start conservative
- Monitor carefully
- Adjust slowly

If you follow these principles, EfficientNetB0 should significantly outperform AlexNet on your MRNet dataset!

---

**Ready to train?** Pick a preset above and run it. Good luck! 🚀
