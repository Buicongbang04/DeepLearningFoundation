# Project 03 — Transfer Learning on a Small Custom Dataset: Report

## Problem

4-class image classification on a custom dataset built from CIFAR-10:
`airplane`, `automobile`, `bird`, `cat`. 200 train + 100 val images per
class (**800 train / 400 val** total), upscaled to 224 × 224 with
ImageNet normalization. Metric: **top-1 validation accuracy**.

The point of this project is to *measure* the value of pretrained weights
when the labeled training set is small. We compare two runs that differ
only in `model.pretrained` and `model.freeze_backbone`.

## Method

Architecture: torchvision **ResNet-18** with the final `fc` layer
replaced by `Linear(512, 4)`. 11,178,564 total parameters in both runs.

Two configurations:

| Variant            | Config file                                | Backbone weights | Trainable params |
|--------------------|---------------------------------------------|------------------|-----------------:|
| Pretrained head-only | `configs/transfer_learning_custom.yaml`         | ImageNet (frozen) | 2,052           |
| From-scratch       | `configs/transfer_learning_custom_scratch.yaml`  | random init      | 11,178,564     |

Everything else identical:

- **Optimizer:** AdamW, `lr = 1e-3`, `weight_decay = 1e-4`.
- **Augmentation (train):** `Resize(256) → RandomCrop(224) → HorizontalFlip → ToTensor → Normalize(ImageNet stats)`.
- **Eval transform:** `Resize(256) → CenterCrop(224) → ToTensor → Normalize`.
- **Batch size:** 32, **3 epochs**, `seed: 42`.

## Results

### Headline comparison

| Variant              | Val acc @ ep 1 | Val acc @ ep 2 | Val acc @ ep 3 | **Best** |
|----------------------|---------------:|---------------:|---------------:|---------:|
| Pretrained head-only | 64.25%         | 78.00%         | **80.00%**     | **80.00%** |
| From-scratch         | 45.75%         | 50.00%         | **56.50%**     | **56.50%** |
| **Gap (points)**     | +18.5          | +28.0          | **+23.5**      | **+23.5** |

The pretrained head-only run **trains only 2,052 parameters** (the final
`Linear(512, 4)`) and yet wins by 23.5 points. The from-scratch run
updates **all 11.2 M parameters** and still loses badly.

### Training-loss view

| Epoch | Pretrained train loss | From-scratch train loss |
|------:|----------------------:|------------------------:|
| 1     | 1.267                 | 1.365                   |
| 2     | 0.866                 | 1.009                   |
| 3     | 0.662                 | 0.879                   |

Both runs decrease loss similarly *on the training set* — the difference
is **generalization**. The from-scratch run is mostly memorizing 800
images of random-init feature transforms; the pretrained run is
recombining genuinely useful ImageNet features.

### Wall-clock

- Pretrained head-only: ~15 seconds per epoch (only the new head is in
  the backward pass; the frozen backbone is forward-only).
- From-scratch: ~27 seconds per epoch (full backbone is in the backward
  pass).

Both runs took under 90 seconds total on CPU.

## Error analysis

At 80% validation accuracy on 400 images, ~80 examples are misclassified.
Inspection of misclassified images (run via the
`labs/lab_05_transfer_learning.ipynb` walkthrough) shows the typical
small-CNN confusions:

- **bird ↔ cat** — small animals at 32×32-equivalent resolution.
- **airplane ↔ automobile** — both are man-made objects with sky / road
  backgrounds; some images are ambiguous even to a human at 32×32.

The from-scratch run, by contrast, has a flatter error distribution and
fails on many "easy" examples — it has not yet learned the basic
*shape-vs-texture* split that an ImageNet-pretrained network gets for
free.

## What I would try next

**One concrete experiment, with a hypothesis:** unfreeze ResNet's
`layer4` (the last residual stage) in the pretrained variant and
fine-tune with `lr = 1e-4`. Expected validation accuracy is **≥ 88%** at
3 epochs — the last stage is most ImageNet-specific and benefits the
most from adaptation to the new four classes, while the rest of the
backbone stays generic.

If the gap is *already* this large, scaling the dataset (500 images per
class, 4 epochs) would only narrow it slightly. Pretraining wins on
small data; that is the rule of the chapter.

## Reproducibility

From the repo root:

```bash
pip install -r requirements.txt

# Pretrained head-only fine-tune.
python src/train.py     --config configs/transfer_learning_custom.yaml
python src/evaluate.py  --config configs/transfer_learning_custom.yaml \
                        --checkpoint experiments/transfer_learning_custom/checkpoints/best.pt

# From-scratch baseline.
python src/train.py     --config configs/transfer_learning_custom_scratch.yaml
python src/evaluate.py  --config configs/transfer_learning_custom_scratch.yaml \
                        --checkpoint experiments/transfer_learning_custom_scratch/checkpoints/best.pt
```

Expected end state:

- `experiments/transfer_learning_custom/checkpoints/best.pt` — 80.00%
  val acc, ~44 MB (the full ResNet-18 weights are saved, even though
  most are frozen).
- `experiments/transfer_learning_custom_scratch/checkpoints/best.pt` —
  56.50% val acc, same size.

ResNet-18 ImageNet weights are downloaded once to
`~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` (~44 MB).
