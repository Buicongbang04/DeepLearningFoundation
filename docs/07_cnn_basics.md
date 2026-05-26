# Chapter 8 — Convolutional Neural Networks (basics)

> An MLP throws away spatial structure: it treats the pixel at `(0, 0)` and the pixel at `(27, 27)` as two independent features. A convolution exploits the fact that an image is *locally smooth* and *translation-equivariant* — a cat detector that works in the top-left should also work in the bottom-right.

## Mục tiêu (Goal)

After this chapter you can:

- Compute the output shape of a `Conv2d` layer given input shape, kernel size, stride, padding, and dilation.
- Build a small CNN (`Conv → BN → ReLU → Pool` blocks + a final `Linear` classifier) for MNIST or CIFAR-10.
- Compare an MLP and a CNN on CIFAR-10 — training curve, validation accuracy, confusion matrix.
- Explain the role of *receptive field*, *parameter sharing*, and *local connectivity*.

## Why this chapter

This is the first time the course adds inductive bias to the architecture instead of brute-force capacity. Convolutions encode:

- **Local connectivity** — each output looks at a small neighborhood, not the whole image.
- **Parameter sharing** — the same filter is applied at every location.
- **Translation equivariance** — shifting the input shifts the output the same way.

This bias is correct for natural images. Without it, even modest vision tasks need huge MLPs.

- **Builds on:** Chapter 4 (training loop), Chapter 5 (activation/init/normalization), Chapter 6 (regularization).
- **Sets up:** Chapter 9 (CNN architectures, transfer learning), Chapter 12 (ViT shares the patch-embedding idea).

## Key concepts

### The convolution operation

A 2-D convolution slides a small filter `K ∈ R^{C_out × C_in × k_h × k_w}` across an input feature map `X ∈ R^{C_in × H × W}` to produce an output `Y ∈ R^{C_out × H' × W'}`:

```
Y[c_out, i, j] = Σ_{c_in, di, dj}  K[c_out, c_in, di, dj] · X[c_in, i + di, j + dj]   (+ bias)
```

In PyTorch:

```python
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                 stride=1, padding=1, bias=True)
```

What is a "filter" doing? Mathematically it is a learnable feature detector — gradient-descent updates each `K[c_out, c_in, :, :]` so it lights up on patterns the network finds useful (edges in early layers, textures in middle layers, parts in deeper layers).

### Stride, padding, kernel size — and output shape

For a `Conv2d` with input height `H`, kernel height `k`, padding `p`, stride `s`, dilation `d`:

```
H_out = floor( (H + 2p − d · (k − 1) − 1) / s + 1 )
```

(Same formula for width.)

For the common case of `d=1`, this simplifies to:

```
H_out = floor( (H + 2p − k) / s + 1 )
```

Memorize three patterns:

| Pattern                  | Formula      | Effect                          |
|--------------------------|--------------|---------------------------------|
| `k=3, p=1, s=1`          | `H_out = H`  | "Same" — preserve spatial size. |
| `k=3, p=1, s=2`          | `H_out = H/2`| Downsample by 2.                 |
| `k=1, p=0, s=1`          | `H_out = H`  | "Pointwise" — channel mixing.   |

### Pooling

Pooling reduces spatial size with no learnable parameters:

- **MaxPool2d(2)** — take the max over a 2×2 window, stride 2. Halves H and W. Default.
- **AvgPool2d(2)** — average instead of max.
- **AdaptiveAvgPool2d((1, 1))** — collapse to a `1×1` feature map (used right before a `Linear` classifier).

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
```

In modern architectures, strided convolutions often replace pooling.

### A standard CNN block

```python
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        return self.block(x)
```

Stack a few of these, then a global pool and a Linear:

```python
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),     # (3,32,32)  → (32,16,16)
            ConvBlock(32, 64),    # (32,16,16) → (64, 8, 8)
            ConvBlock(64, 128),   # (64, 8, 8) → (128,4, 4)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # → (128,1,1)
        self.fc   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
```

### Receptive field

The **receptive field** of a neuron is the patch of input pixels that influence its value. It grows with depth.

For an `L`-layer CNN with kernel size `k` and stride `s` at every layer, the receptive field at the top is roughly:

```
RF = 1 + Σ_{l=1..L}  (k_l − 1) · ∏_{l' < l}  s_{l'}
```

In a `Conv3x3 → Pool2x2 → Conv3x3 → Pool2x2 → Conv3x3` stack, the top neuron sees a `~20×20` patch of the input. To classify an object that fits in `40×40` pixels, you need a network *deep enough* (or with large enough strides) for the receptive field to cover it.

### Parameter count of a Conv2d

```
params = (C_in · k_h · k_w + 1) · C_out
```

A `Conv2d(64, 128, 3)` has `(64·3·3 + 1) · 128 = 73,856` parameters. Compare this to a fully-connected layer between two `64×32×32` feature maps, which would be `64·32·32 · 128·32·32 ≈ 2.7 billion` parameters. *That* is why CNNs are practical.

### Conv → ReLU → Pool → FC architecture

The classic CNN classifier pipeline:

1. Conv blocks extract features at increasing levels of abstraction.
2. Pool layers shrink spatial size; the channel dimension grows correspondingly.
3. Global pool flattens to a single feature vector per image.
4. A fully-connected layer produces class logits.

```
Input (B,3,32,32)
  → Conv+BN+ReLU+Pool → (B,32,16,16)
  → Conv+BN+ReLU+Pool → (B,64,8,8)
  → Conv+BN+ReLU+Pool → (B,128,4,4)
  → AdaptiveAvgPool → (B,128,1,1)
  → Flatten → (B,128)
  → Linear → (B,num_classes)
```

This is exactly what `SmallCNN` above does. For ImageNet-scale problems you stack many more blocks and use residual connections (Chapter 9).

## Common pitfalls

- Shape mismatch on the first `Linear` after the conv stack → forgot the final `flatten`. Print `x.shape` right before the linear and you will see immediately.
- Loss is `nan` early → exploding gradient on too-large filters. Add BatchNorm or reduce the learning rate.
- CIFAR-10 trains fine on MLP but CNN refuses to learn → check the input normalization (per-channel mean and std), check that you applied `ToTensor()` *before* `Normalize()`.
- Used `MaxPool2d(4)` on a `4×4` feature map → the feature map collapses to `1×1` too early → the deeper conv blocks see no spatial structure. Profile the shape at each layer.
- 99% train, 60% val on CIFAR-10 → over-fitting. Add augmentation + dropout in the classifier head + weight decay.

## Learning outcomes

- Compute the output shape of any `Conv2d` from `(H, k, p, s, d)`.
- Build a CNN for MNIST or CIFAR-10 that hits ≥ 70% validation accuracy.
- Compare an MLP and a CNN with the same parameter count on CIFAR-10 — the CNN should win.
- Read a confusion matrix and spot a class the model systematically confuses.

## Quick check (self-test)

<details>
<summary>Q1 — Input is <code>(B, 3, 32, 32)</code>, you apply <code>nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)</code>. What is the output shape?</summary>

`(B, 64, 32, 32)`. With kernel 3, padding 1, stride 1, the spatial size is preserved.
</details>

<details>
<summary>Q2 — Why do CNNs have so many fewer parameters than fully-connected nets for images?</summary>

Two reasons: **local connectivity** — each output neuron looks at a small `k × k` patch instead of the entire image — and **parameter sharing** — the same filter weights are used at every spatial location.
</details>

<details>
<summary>Q3 — What is a "receptive field"?</summary>

The patch of input pixels that contribute to a particular neuron's value. The receptive field grows with depth, and for a classifier it must be at least as big as the object you want to recognize.
</details>

<details>
<summary>Q4 — Why do we usually put <code>AdaptiveAvgPool2d((1,1))</code> right before the final <code>Linear</code> layer?</summary>

It collapses the spatial dimension to `1×1`, making the model insensitive to the input image size. You can then plug in any image resolution and the classifier head still has a fixed `(C, 1, 1)` input.
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter 9 (convolutional networks).
- LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998) — original LeNet.
- Stanford CS231n, *Convolutional Neural Networks for Visual Recognition* — Lecture notes.
- PyTorch docs — `torch.nn.Conv2d`, `torch.nn.MaxPool2d`, `torch.nn.AdaptiveAvgPool2d`.

## Companion artifact

`notebooks/chapter_07_cnn_basics.ipynb` — `Conv2d` shape playground, small CNN for MNIST, `projects/project_02_cnn_cifar10/` for the CIFAR-10 ablation.
