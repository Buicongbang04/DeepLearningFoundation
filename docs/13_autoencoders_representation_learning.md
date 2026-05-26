# Chapter 14 — Autoencoder and Representation Learning

> An autoencoder is the simplest possible self-supervised model: ask the network to copy its input through a narrow channel. The narrowness *is* the learning — to fit through a 32-dimensional bottleneck, the network must discover what is essential about the input and throw the rest away.

## Mục tiêu (Goal)

After this chapter you can:

- Define encoder, latent representation, decoder, and reconstruction loss.
- Build a small autoencoder for MNIST and visualize its 2-D latent space.
- Distinguish undercomplete, overcomplete, and denoising autoencoders.
- Use reconstruction error as an anomaly-detection score on a contaminated dataset.

## Why this chapter

This is the first chapter where the loss does not need labels. The autoencoder is *self-supervised*: its target is its own input. This puts the chapter on the bridge between *supervised* learning (everything up to Ch 13) and *generative* learning (Ch 15). It also sets up the encoder–latent–decoder pattern that VAE, U-Net, and many other architectures share.

- **Builds on:** Chapter 4 (training loop), Chapter 5 (activations), Chapter 8 (Conv2d for image autoencoders).
- **Sets up:** Chapter 15 (VAE is an autoencoder with a probabilistic latent; diffusion models share the denoising idea).

## Key concepts

### The encoder–decoder pattern

An autoencoder is two functions stitched together:

```
encoder:  x  →  z = f(x; θ_e)        # compress
decoder:  z  →  x̂ = g(z; θ_d)        # reconstruct
```

`z` is the **latent representation** (also called *bottleneck*, *code*, *embedding*). The training objective is to make `x̂` close to `x`:

```
L(x, x̂) = ‖x − x̂‖²                  # MSE for continuous inputs
       or  −Σ x · log(x̂)            # BCE for [0,1] pixel inputs
```

A minimal MNIST autoencoder:

```python
class MnistAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),                          # (B, 784)
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64),  nn.ReLU(),
            nn.Linear(64, latent_dim),             # bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
```

Training loop:

```python
for x, _ in train_loader:               # we ignore the label!
    x = x.to(device)
    x_hat, _ = model(x)
    loss = F.mse_loss(x_hat, x)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

Note that the dataloader returns labels, but the loss does not use them. This is what *self-supervised* means.

### Undercomplete vs. overcomplete

- **Undercomplete** — `dim(z) < dim(x)`. The bottleneck forces compression and is what makes the autoencoder learn anything useful.
- **Overcomplete** — `dim(z) ≥ dim(x)`. The encoder can trivially copy the input through identity, so reconstruction is perfect but the latent is meaningless. Use overcomplete autoencoders only with an *additional regularizer* (sparsity, denoising, contractive penalty).

For MNIST (`784` pixels), `latent_dim = 32` is a strong but tractable bottleneck. `latent_dim = 2` is more aggressive and lets you *plot* the latent space directly.

### Denoising autoencoder

A nice trick: feed in a corrupted version of `x` but ask the decoder to recover the *clean* `x`:

```python
x_noisy = x + 0.3 * torch.randn_like(x)
x_hat, _ = model(x_noisy)
loss = F.mse_loss(x_hat, x)            # target is the clean x!
```

This forces the network to learn what an MNIST digit *actually is* rather than just memorizing pixel intensities. Denoising autoencoders generalize better and produce a more semantic latent space.

Variants of corruption you can try:

- Gaussian noise (above).
- Salt-and-pepper noise (random pixels set to 0 or 1).
- Masking (zero out a random patch — the masked-autoencoder idea, MAE).

### Visualizing the latent space

For `latent_dim = 2`, you can scatter-plot the latent codes colored by the (held-out) class label:

```python
zs, ys = [], []
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        _, z = model(x.to(device))
        zs.append(z.cpu()); ys.append(y)
zs = torch.cat(zs); ys = torch.cat(ys)

plt.scatter(zs[:, 0], zs[:, 1], c=ys, cmap="tab10", s=4)
plt.colorbar()
```

A *good* autoencoder shows class clusters even though it never saw the labels — the bottleneck has discovered class-relevant structure from the pixels alone.

For larger `latent_dim`, project to 2-D with PCA or t-SNE before plotting:

```python
from sklearn.decomposition import PCA
zs_2d = PCA(n_components=2).fit_transform(zs.numpy())
```

### Anomaly detection via reconstruction error

The autoencoder is trained on *normal* data and learns to reconstruct it well. When fed an anomaly, the reconstruction is poor, so the MSE per example is a natural anomaly score:

```python
def anomaly_score(model, x):
    model.eval()
    with torch.no_grad():
        x_hat, _ = model(x)
        return (x - x_hat).pow(2).mean(dim=[1, 2, 3])   # one scalar per example
```

Workflow:

1. Train the autoencoder on a clean training set of *normal* examples only.
2. Compute the reconstruction MSE for every example in a validation set that contains a known mix.
3. Plot the score distribution; the anomalous class should have a long right tail.
4. Pick a threshold by validation precision/recall.

This is the recipe for the companion mini-project.

### Dimensionality reduction (and why autoencoders ≠ PCA)

A linear autoencoder with no nonlinearity and MSE loss reduces *exactly* to PCA — the optimal `latent_dim`-dimensional projection of the data. Add ReLU (or any nonlinearity) and the autoencoder can learn *curved* manifolds that PCA cannot, which is why it does much better on natural images.

Quick comparison:

| Method        | Captures      | Cost                      |
|---------------|---------------|---------------------------|
| PCA           | Linear        | One closed-form SVD       |
| Autoencoder   | Linear + nonlinear curved manifold | Stochastic training, GPU |
| t-SNE / UMAP  | Local clusters (visualization only) | Per-batch, not parametric |

Use PCA when you need a baseline; autoencoder when you need a *learned, reusable* low-dimensional representation.

### Conv autoencoder for images

For images, replace MLP layers with Conv2d / ConvTranspose2d so the encoder preserves spatial structure:

```python
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),    # 28 → 14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 14 →  7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 32 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1,  3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
```

`ConvTranspose2d` upsamples; the `output_padding` parameter restores the exact pre-conv spatial size. A clean alternative is `nn.Upsample(scale_factor=2) → Conv2d`, which avoids the checkerboard artifacts that `ConvTranspose2d` sometimes produces.

## Common pitfalls

- Reconstruction looks blurry → MSE loss averages over pixels and prefers blur to sharp-but-wrong → switch to BCE or train a VAE/GAN if you actually need sharp samples.
- Loss does not decrease → bottleneck too small for the data complexity, *or* you are using `sigmoid` on the output but `mse_loss` instead of `binary_cross_entropy`. Match the activation to the loss.
- Latent dim too high → autoencoder collapses to identity → reconstructions perfect, latent space useless. Reduce `latent_dim`.
- `ConvTranspose2d` produces a checkerboard pattern → known artifact of stride-2 transposed conv → use `Upsample → Conv2d` or fix the kernel-size/stride ratio.
- Computing anomaly threshold on the *training* set → threshold is much too generous → use a separate clean validation set.
- Comparing autoencoder reconstructions across runs without seeding → noisy comparison → always set `torch.manual_seed(...)`.

## Learning outcomes

- Build an MLP and a conv autoencoder for MNIST, both reaching low reconstruction MSE.
- Choose `latent_dim` deliberately and explain the trade-off between compression and reconstruction quality.
- Train a denoising autoencoder and show that the latent representation is more semantic than a vanilla autoencoder's.
- Use reconstruction error to flag anomalies on a contaminated MNIST/Fashion-MNIST set with a precision-recall curve.

## Quick check (self-test)

<details>
<summary>Q1 — In one sentence, why does the autoencoder need a bottleneck?</summary>

Without a bottleneck the network can learn the identity function — perfect reconstruction, zero learned representation. The bottleneck forces the encoder to throw information away, and *what* it chooses to keep is the representation.
</details>

<details>
<summary>Q2 — How does a denoising autoencoder differ from a vanilla one?</summary>

Input is a corrupted `x_noisy`, but the target is the *clean* `x`. The encoder cannot just copy the input through, so it has to learn what an "uncorrupted" example actually looks like — yielding a more semantic latent space.
</details>

<details>
<summary>Q3 — Why is reconstruction MSE a reasonable anomaly score?</summary>

The autoencoder was trained only on normal examples, so its decoder is good at producing things that look normal. An anomalous input is far from the training manifold; the decoder cannot reconstruct it well, so the MSE is large. Threshold the MSE on a clean validation set to set an anomaly cut-off.
</details>

<details>
<summary>Q4 — When does an autoencoder reduce exactly to PCA?</summary>

When the encoder and decoder are both single *linear* layers (no activation) and the loss is MSE. The optimal `θ` then aligns with the top-`k` principal components.
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter 14 (autoencoders).
- Vincent et al., "Stacked Denoising Autoencoders" (2010).
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (MAE, 2021) — the modern reincarnation of the masking-denoising idea at ViT scale.
- PyTorch docs — `torch.nn.ConvTranspose2d`, `torch.nn.Upsample`.

## Companion artifact

`projects/project_06_autoencoder_anomaly_detection/` — train an autoencoder on clean Fashion-MNIST, use reconstruction MSE as an anomaly score on a contaminated test set, report precision/recall.
