# Chapter 15 — Generative Models (intro)

> A discriminative model learns `p(y | x)`: given an image, what is its class? A generative model learns `p(x)`: what does a *plausible* image look like in the first place? VAE, GAN, and diffusion are three different ways of attacking the same problem.

## Mục tiêu (Goal)

After this chapter you can:

- Tell discriminative and generative models apart by what they model and how they are sampled.
- State VAE, GAN, and diffusion in one sentence each — by their training objective.
- Run a small VAE or GAN demo on MNIST and reason about what the samples look like.
- Explain why the toy demos in this course are not real generative-model research, and what would be needed to scale them.

## Why this chapter

This is a *concept* chapter. The full theory of any single one of VAE, GAN, or diffusion deserves its own multi-week course. The goal here is to give you the mental map — what each family is for, how they differ, and what to read next when you specialize.

- **Builds on:** Chapter 14 (autoencoder — the encoder/decoder/latent vocabulary), Chapter 5 (KL divergence intuition), Chapter 11 (cross-entropy / softmax).
- **Sets up:** any future Computer Vision, NLP, or Generative-AI course you take after this one.

## Key concepts

### Discriminative vs. generative

| Model type      | Models          | Example task                            |
|-----------------|-----------------|-----------------------------------------|
| Discriminative  | `p(y \| x)`     | Image → class label (everything Ch 4–13). |
| Generative      | `p(x)` or `p(x \| y)` | Sample a *new* image of a cat.             |

A generative model is more ambitious because it needs to capture the *whole data distribution*, not just decision boundaries. The trade-off: harder to train, much richer applications (synthesis, super-resolution, inpainting, conditional generation).

Within "generative", we focus on three families: **VAE**, **GAN**, **diffusion**. Each chooses a different way to define and optimize a generator `G(z) → x`.

### VAE — variational autoencoder

A VAE is the autoencoder (Ch 14) made probabilistic. The encoder outputs not a point `z` but a *distribution* `q(z | x) = N(μ(x), σ²(x))`. We sample `z` from that distribution, decode it, and use a two-term loss:

```
L_VAE(x) = ‖x − x̂‖²              (reconstruction)
        + KL( q(z|x)  ‖  N(0, I) ) (KL to a standard-normal prior)
```

- **Reconstruction** — same as a vanilla autoencoder.
- **KL term** — pulls each `q(z|x)` toward the prior `N(0, I)` so the latent space is *organized*: nearby `z`s decode to similar `x`s, and sampling `z ∼ N(0, I)` and decoding gives plausible new examples.

Sampling new data:

```python
z = torch.randn(num_samples, latent_dim)
x_new = decoder(z)
```

Strengths: stable training, principled probabilistic interpretation, latent space is interpretable.

Weakness: samples are *blurry* on natural images. The Gaussian likelihood and the KL pressure both favor smooth, averaged outputs over sharp ones.

```python
class TinyVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.mu     = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z).reshape(-1, 1, 28, 28), mu, logvar
```

The `reparameterize` trick (`mu + std * eps`) is what makes the sampling differentiable — without it you cannot backprop through a random sample.

### GAN — generative adversarial network

A GAN is **two networks trained against each other**:

- **Generator** `G(z) → x` maps a random noise vector `z` to a fake image.
- **Discriminator** `D(x) → [0, 1]` outputs the probability that `x` is real.

They play a min-max game:

```
min_G max_D  E_{x∼data} [ log D(x) ] + E_{z∼p(z)} [ log (1 − D(G(z))) ]
```

In practice you alternate:

1. Train `D` to tell real images from `G`'s fakes.
2. Train `G` to fool `D`.

A skeleton training step:

```python
# Discriminator step
real = next_real_batch()
fake = G(torch.randn(B, latent_dim))
loss_D = bce(D(real), ones) + bce(D(fake.detach()), zeros)
loss_D.backward(); opt_D.step()

# Generator step
fake = G(torch.randn(B, latent_dim))
loss_G = bce(D(fake), ones)             # fool the discriminator
loss_G.backward(); opt_G.step()
```

Strengths: produces sharp, photorealistic samples; the dominant approach for image synthesis from 2015 until diffusion took over around 2022.

Weaknesses:

- **Mode collapse** — the generator finds one or a few outputs the discriminator cannot reliably reject, and emits only those. Diversity collapses.
- **Training instability** — the adversarial dynamics oscillate; a small change in LR or architecture often kills training.
- **No likelihood** — you cannot ask "how likely is this image under the model?"

### Diffusion — model the denoising process

Diffusion is the current state of the art for image generation (DALL-E, Stable Diffusion, Imagen). The idea:

1. Define a **forward process** that gradually adds Gaussian noise to a real image over `T` steps: `x_0 → x_1 → … → x_T ≈ pure noise`.
2. Train a network `ε_θ(x_t, t)` to *predict the noise that was added* at each step.
3. To **sample**, start from `x_T ∼ N(0, I)` and run the network in reverse: subtract its predicted noise step by step until you arrive at `x_0`.

Training loss is a clean MSE:

```
L_diff = E_{x_0, t, ε}  ‖ ε − ε_θ( √(α̅_t) · x_0 + √(1 − α̅_t) · ε,  t ) ‖²
```

Read this as: "given a clean image `x_0`, a random time step `t`, and a random noise vector `ε`, mix them according to schedule `α̅_t` to form `x_t`, then ask the network to recover `ε` from `x_t` and `t`."

Sampling is iterative (50–1000 steps), each step running the network once and subtracting its noise prediction.

Strengths: highest sample quality, very stable training (just regression), supports conditional generation cleanly (concatenate the condition to the input).

Weakness: sampling is *slow* because each generated image is many forward passes. Distillation and ODE solvers (DDIM) bring this down to 4–25 steps in practice.

### Latent space and sampling — the common thread

All three families think in terms of *sampling*: produce a new `x` from noise.

- **VAE** — sample `z ∼ N(0, I)`, decode `x = G(z)`. One forward pass.
- **GAN** — same idea: sample `z ∼ N(0, I)`, decode `x = G(z)`. One forward pass.
- **Diffusion** — sample `x_T ∼ N(0, I)`, then run the denoiser `T` times to land at `x_0`. Many forward passes.

The latent space of a VAE is *interpretable* (continuous, organized by KL); a GAN's latent space is *not guaranteed* to be smooth but in practice usually is; diffusion does not have a latent space in the same sense — its "noise" is at the pixel level (or, for latent diffusion, at a VAE-encoded level).

### Mode collapse — what to watch for

A mode-collapsed model produces only a tiny fraction of the data's diversity. Quick checks:

- Sample 1000 images and look at them. Do they cover the variety in the training set?
- For MNIST, every digit class should appear. If only one or two appear, the generator has collapsed.
- Plot a 2-D projection (PCA) of `x_real` vs `x_fake` and visually compare coverage.

All three families can mode-collapse; GANs are notoriously prone to it.

### The honest limits of intro-level demos

A VAE or GAN on MNIST will produce digit-like samples. This does *not* generalize to "can I now generate photorealistic faces". The gap from MNIST to real image synthesis is:

- 4–6 orders of magnitude more data.
- 3–5 orders of magnitude more compute.
- Architecture innovations specific to each scale (StyleGAN's modulation, latent diffusion, U-Net at scale, etc.).
- Massive engineering on data filtering, EMA, gradient penalty, classifier-free guidance, etc.

Treat the chapter's MNIST samples as a **mechanism demo**, not as a benchmark. Real generative-model research is its own multi-year specialization.

## Common pitfalls

- VAE samples are blurry → expected, not a bug. Switch to a different family if sharpness is critical.
- GAN training oscillates wildly → reduce the LR, balance `D` and `G` updates (e.g., 1:1 or 1 `G` per 5 `D`), add spectral normalization, switch from BCE to Wasserstein loss.
- Mode collapse in GAN → try Wasserstein loss, increase generator capacity, minibatch discrimination, or just restart with a different seed.
- Diffusion samples look like noise → the noise schedule or `t` encoding is wrong → re-check the `α̅_t` curve and that `t` is fed into the network (commonly via sinusoidal embedding).
- VAE loss is negative → you are reporting the *negative* ELBO incorrectly, or mixed BCE and MSE for the reconstruction term. Audit each loss term separately, then sum.
- Comparing VAE/GAN/diffusion by "which looks best at epoch 10" → unfair: they have very different epoch dynamics. Compare after enough training, on a held-out set, and ideally with a quantitative metric (FID).

## Learning outcomes

- State VAE, GAN, and diffusion in one sentence each by their training objective.
- Identify mode collapse and blurriness from sample grids.
- Run a tiny VAE on MNIST and produce a sample sheet that looks like digits.
- Explain in your own words why these intro demos do not transfer directly to real image synthesis.

## Quick check (self-test)

<details>
<summary>Q1 — VAE, GAN, diffusion — one sentence each by training objective.</summary>

- VAE: maximize a *lower bound* on `p(x)` (reconstruction + KL-to-prior).
- GAN: a generator and a discriminator play a *min-max game* — generator fools discriminator, discriminator tells real from fake.
- Diffusion: regress the *noise that was added* to a clean image at a random timestep — sampling reverses this denoising step by step.
</details>

<details>
<summary>Q2 — Why are VAE samples often blurry?</summary>

The Gaussian reconstruction likelihood and the KL pressure both push the decoder toward the *mean* of the posterior over plausible reconstructions, which for ambiguous inputs is a blurry average. Sharper alternatives (GAN, diffusion) explicitly avoid averaging.
</details>

<details>
<summary>Q3 — What is mode collapse and which family is it most common in?</summary>

A generative model that produces only a small fraction of the data's diversity — e.g., a GAN trained on MNIST that only generates 3s. GANs are most prone because the discriminator can be "fooled" by a narrow but convincing set of outputs.
</details>

<details>
<summary>Q4 — In one sentence, why is diffusion sampling slow?</summary>

You denoise step by step from pure noise; each step is one forward pass of the network, and the default schedule has hundreds of steps. Distilled / ODE solvers (DDIM) cut this to 4–25 steps but still cost more than a one-shot VAE or GAN.
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter 20 (deep generative models).
- Kingma & Welling, "Auto-Encoding Variational Bayes" (VAE, 2013).
- Goodfellow et al., "Generative Adversarial Networks" (GAN, 2014).
- Ho et al., "Denoising Diffusion Probabilistic Models" (DDPM, 2020).
- Lilian Weng, *What are Diffusion Models?* (blog) — the cleanest written introduction.

## Companion artifact

`notebooks/chapter_14_vae_gan_diffusion_intro.ipynb` — train a tiny VAE on MNIST, generate samples, show a sketch of GAN training, and (concept-level) walk through one denoising step of diffusion.
