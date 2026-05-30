# Project 03 — Transfer Learning on a Small Custom Dataset

Fine-tune a pretrained ResNet-18 on a small custom dataset and compare it
to training the same architecture from scratch. The third end-to-end
project of the course, paired with Chapter 9.

**Companion chapter:** `docs/08_cnn_architectures_transfer_learning.md`.

## The "custom" dataset

To keep the project completely reproducible without external downloads,
the *custom* dataset is a 4-class subset of CIFAR-10 (`airplane`,
`automobile`, `bird`, `cat`) with 200 train + 100 val images per class,
upscaled to 224×224 with **ImageNet normalization**. The dataset factory
is `_cifar10_subset_loaders` in `src/datasets.py` and selected via
`data.name: cifar10_subset` in the config.

200 images per class is small enough that a from-scratch ResNet-18 fails,
large enough for a stable comparison.

## Reproduce both runs

From the repo root:

```bash
pip install -r requirements.txt

# Head-only fine-tune (pretrained ImageNet ResNet-18, backbone frozen).
python src/train.py     --config configs/transfer_learning_custom.yaml
python src/evaluate.py  --config configs/transfer_learning_custom.yaml \
                        --checkpoint experiments/transfer_learning_custom/checkpoints/best.pt

# From-scratch baseline (same architecture, no pretrained weights).
python src/train.py     --config configs/transfer_learning_custom_scratch.yaml
python src/evaluate.py  --config configs/transfer_learning_custom_scratch.yaml \
                        --checkpoint experiments/transfer_learning_custom_scratch/checkpoints/best.pt
```

The pretrained ResNet-18 checkpoint (~44 MB) will be downloaded by
`torchvision` on first use.

## Configuration highlights

Two configs differ only in the `model` block:

| Field            | `transfer_learning_custom.yaml` | `..._scratch.yaml` |
|------------------|---------------------------------|--------------------|
| `pretrained`     | `true`                          | `false`            |
| `freeze_backbone`| `true` (head-only)              | `false`            |

Everything else is identical (same data subset, optimizer, LR, epochs,
seed) so the comparison is apples-to-apples.

## What success looks like

- **Pretrained head-only fine-tune** should reach 75–85% val accuracy in
  3 epochs — the frozen ImageNet features already separate the four
  classes well.
- **From-scratch baseline** typically lands in the 45–60% range with 3
  epochs on 800 images. Above chance (25% for 4 classes), but the network
  is mostly memorizing rather than learning useful features.

The size of the **gap** between the two runs — roughly 20+ percentage
points — is the chapter's headline result.

## Files

- `configs/transfer_learning_custom.yaml` — pretrained head-only.
- `configs/transfer_learning_custom_scratch.yaml` — from-scratch baseline.
- `src/datasets.py` — `_cifar10_subset_loaders` factory.
- `src/models.py` — `_build_resnet18(num_classes, pretrained, freeze_backbone)`
  and registry entry under `model.name: resnet18`.
- `projects/project_03_transfer_learning_custom/report.md` — written
  report with actual training numbers and a side-by-side comparison.
- `labs/lab_05_transfer_learning.ipynb` — interactive walkthrough of the
  same recipe on an even smaller dataset.
