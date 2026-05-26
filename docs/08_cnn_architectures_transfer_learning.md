# Chapter 9 — CNN Architectures and Transfer Learning

> Reading a famous CNN paper is *not* about memorizing layer counts. It is about answering one question: "What problem did this architecture solve that the previous one could not?" — and then reusing the answer (a pretrained checkpoint) on your own dataset.

## Mục tiêu (Goal)

After this chapter you can:

- Place LeNet, AlexNet, VGG, Inception, and ResNet on a timeline and name the one idea each contributed.
- Explain *why* a skip connection lets a network with 50+ layers train at all.
- Fine-tune a pretrained ResNet on a small custom dataset with a new classification head.
- Compare from-scratch training and transfer learning on the same dataset and report which wins and by how much.

## Why this chapter

Chapter 8 built a CNN from scratch on MNIST / CIFAR-10. That works at the toy scale, but real vision problems — ImageNet-1k, custom datasets with a few thousand images per class — are dominated by **pretrained models**. You almost never train ImageNet from scratch; you download a checkpoint that someone else trained and adapt it.

- **Builds on:** Chapter 8 (CNN basics), Chapter 5 (normalization), Chapter 7 (optimization, schedulers).
- **Sets up:** Chapter 12 (the same "pretrain on a large corpus, fine-tune on a small task" recipe is the dominant pattern in NLP/Transformer too), Chapter 16 (loading a checkpoint at inference time).

## Key concepts

### A six-paper timeline

| Year | Architecture     | One-line idea                                                            |
|------|------------------|--------------------------------------------------------------------------|
| 1998 | **LeNet-5**      | Conv + Pool + FC actually works for digit recognition.                   |
| 2012 | **AlexNet**      | Same recipe + ReLU + dropout + GPU training wins ImageNet by a wide margin.|
| 2014 | **VGG-16/19**    | Stack many `3×3` convolutions; depth matters more than fancy filters.    |
| 2014 | **Inception/GoogLeNet** | Run multiple kernel sizes (`1×1, 3×3, 5×5`) in parallel inside one block.|
| 2015 | **ResNet**       | Add a *skip connection* (`y = F(x) + x`) so very deep networks can train.|
| 2019+ | **EfficientNet / ConvNeXt** | Scale depth + width + resolution jointly; modernize ResNet with Transformer-style design choices. |

You do not need to memorize every layer. You need the one-liners above.

### LeNet — the prototype

A `5 × 5` conv → pool → `5 × 5` conv → pool → two FC layers, all on `28×28` digits. Tiny by modern standards (~60k params). It established the *block* pattern that every later CNN follows.

### AlexNet — what made deep CNNs viable in 2012

Three ingredients on top of LeNet:

1. **ReLU** instead of tanh — gradient does not vanish on positive inputs, training is faster.
2. **Dropout** in the FC head — combats overfitting on ImageNet's 1.2M images.
3. **GPU training** — the network was split across two GPUs to fit in memory.

Plus: `11 × 11` first conv with stride 4 (aggressive downsampling) and local-response normalization (since superseded by BatchNorm).

### VGG — depth via small kernels

Use only `3×3` convolutions, stacked. Two `3×3` in a row have the same receptive field as one `5×5` but with fewer parameters and more non-linearity:

```
Params(5×5) = 25 · C²        vs.        Params(2 × 3×3) = 18 · C²
```

VGG-16 is 138M parameters and takes days to train from scratch. But its *feature extractor* is so general that it became the standard backbone for years.

### Inception — multi-scale inside one block

Instead of choosing one kernel size, use them all and concatenate the outputs:

```
        ┌── 1×1 conv ──┐
input ──┼── 3×3 conv ──┼── concat → output
        ├── 5×5 conv ──┤
        └── pool   ────┘
```

Critical trick: `1×1` convolutions *before* the `3×3` and `5×5` reduce the channel count, so the block stays cheap. This is the first time `1×1` conv as a "channel mixer" appears.

### ResNet — the skip connection

The headline problem: a 56-layer plain CNN has *worse* training error than a 20-layer one. The network has the capacity to be at least as good (just learn identity in the extra layers) but optimization fails to find it.

ResNet's fix is one line:

```
y = F(x; θ) + x
```

Instead of learning `H(x)` directly, the block learns the *residual* `F(x) = H(x) − x`. If the optimal mapping is close to identity, `F(x)` only needs to learn a small correction — which is far easier than learning the full identity from scratch.

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)        # the skip
```

Key effects:

- **Gradient highway** — the `+ x` term lets gradient flow back without being multiplied by many weight matrices, so deep stacks train cleanly.
- **Identity is free** — setting `F(x) ≈ 0` recovers identity, so adding more blocks cannot hurt training error.

ResNet-50 (50 layers, ~25M params) is the workhorse backbone of the 2010s. ResNet-18 is the small variant most beginners start with.

### EfficientNet, ConvNeXt — direction only

Two strands of follow-up work worth knowing exist:

- **EfficientNet (2019)** — scale depth, width, and input resolution *together* using a single compound coefficient. EfficientNet-B0 → B7 trades compute for accuracy on a Pareto curve.
- **ConvNeXt (2022)** — re-modernize ResNet with Transformer-era choices: larger kernels (`7×7` depthwise), LayerNorm instead of BatchNorm, GELU instead of ReLU. Closes most of the gap with Vision Transformers while staying purely convolutional.

For this course, ResNet-18/50 are enough. The other names are recognition-level only.

### Transfer learning — the practical payoff

Pretraining on ImageNet teaches the network a hierarchy of generic visual features. The early layers (edges, textures) transfer to almost any vision task; only the final classifier needs to be retrained for your task.

Three modes, from cheapest to most expensive:

1. **Feature extraction** — freeze the entire backbone, train only a new classification head. Use when your dataset is tiny (a few hundred images per class) or very similar to ImageNet.
2. **Fine-tuning the head + last block** — unfreeze the last conv block too, use a lower LR there. The middle ground.
3. **Full fine-tuning** — unfreeze everything, use a small LR (e.g. `1e-4`). Use when your dataset is larger or domain-shifted (medical images, satellite imagery).

In PyTorch:

```python
import torch.nn as nn
from torchvision import models

backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for p in backbone.parameters():
    p.requires_grad = False        # freeze

num_features = backbone.fc.in_features
backbone.fc = nn.Linear(num_features, num_classes)     # new head, requires_grad=True by default
```

The new `fc` head has `requires_grad=True` since it is freshly constructed, so the optimizer only updates the head's parameters. To partially unfreeze:

```python
for p in backbone.layer4.parameters():
    p.requires_grad = True
optimizer = torch.optim.AdamW(
    [p for p in backbone.parameters() if p.requires_grad], lr=1e-4
)
```

### Custom heads and input preprocessing

The pretrained backbone expects ImageNet-normalized RGB images. *Match the preprocessing exactly*:

```python
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])
```

Skipping the normalization is the #1 cause of "my fine-tuned ResNet gets 10% accuracy".

### From-scratch vs. transfer learning — what to expect

On a typical small custom dataset (5-20 classes, 100-1000 images per class), the gap is large:

| Setup                              | Typical val accuracy | Training time |
|------------------------------------|----------------------|---------------|
| ResNet-18 from scratch              | 55-65%               | hours         |
| ResNet-18 head-only fine-tune       | 80-88%               | minutes       |
| ResNet-18 full fine-tune (low LR)   | 85-92%               | tens of minutes |

This is why the field defaulted to pretraining.

## Common pitfalls

- Forgot ImageNet normalization → backbone sees out-of-distribution input → accuracy near random. Apply the `Normalize(mean=..., std=...)` block from above.
- Replaced `fc` but trained with `requires_grad=True` everywhere → effectively training from scratch with a small LR → slow, poor result. Freeze the backbone first.
- BatchNorm running stats drift during fine-tuning of a frozen backbone → set `model.eval()` for the backbone modules so their `running_mean`/`running_var` stop updating.
- Learning rate too high (`1e-3`) on a fine-tune → catastrophically forgets the pretrained weights → val accuracy collapses. Use `1e-4` or lower for full fine-tunes.
- Image size mismatch (`512×512` input on a ResNet trained at `224×224`) → first FC sees a different flattened shape → runtime error. Resize to `224` (or use `AdaptiveAvgPool2d` before the head, which most torchvision models already do).

## Learning outcomes

- Name the one new idea introduced by AlexNet, VGG, Inception, and ResNet (one sentence each).
- Write the residual block `y = F(x) + x` and explain why it helps gradient flow.
- Fine-tune a pretrained `resnet18` on a small custom image dataset with a new `Linear` head.
- Run the same dataset both from-scratch and via transfer learning, report the validation-accuracy gap.

## Quick check (self-test)

<details>
<summary>Q1 — In one sentence, why does a 56-layer plain CNN train *worse* than a 20-layer one?</summary>

Optimization, not capacity. The gradient passing through many layers is multiplied by many weight matrices and either vanishes or explodes, so SGD cannot find a good minimum even though the capacity is there. ResNet's `+ x` skip gives the gradient a direct path back, removing this optimization barrier.
</details>

<details>
<summary>Q2 — Why are `1 × 1` convolutions useful inside an Inception block?</summary>

They reduce the number of channels cheaply *before* the expensive `3×3` and `5×5` convolutions, keeping the block's parameter count under control. A `1×1` conv mixes channels without touching spatial extent.
</details>

<details>
<summary>Q3 — You freeze the ResNet backbone but accuracy is still terrible. What is the first thing to check?</summary>

The input preprocessing. The pretrained backbone needs ImageNet normalization (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`). Without it, the backbone sees out-of-distribution inputs and produces useless features.
</details>

<details>
<summary>Q4 — When is full fine-tuning preferable to feature extraction?</summary>

When your dataset is large enough (thousands of images per class) and/or domain-shifted from ImageNet (medical, satellite, microscopy). In those cases, the pretrained mid-layer features are imperfect for your domain and benefit from a low-LR full fine-tune.
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter 9 (convolutional networks), discussion of architectures.
- He et al., "Deep Residual Learning for Image Recognition" (ResNet, 2015).
- Szegedy et al., "Going Deeper with Convolutions" (Inception, 2014).
- Yosinski et al., "How transferable are features in deep neural networks?" (2014) — the empirical basis of transfer learning.
- PyTorch tutorial — *Transfer Learning for Computer Vision Tutorial*.

## Companion artifact

`labs/lab_05_transfer_learning.ipynb` — head-only fine-tune vs. full fine-tune on a small custom dataset; `projects/project_03_transfer_learning_custom/` — end-to-end transfer-learning project with from-scratch baseline.
