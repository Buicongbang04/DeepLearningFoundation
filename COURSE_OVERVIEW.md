# Course Overview

**Deep Learning Foundation for AI** is an open, hands-on curriculum that turns the *Deep Learning* textbook by Goodfellow, Bengio, and Courville — together with the standard PyTorch tutorial track — into a 16-week AI-Foundation course.

This document covers the *why* of the course — audience, prerequisites, and design philosophy. For the *what* and *how*, see:

- `README.md` — install instructions, repo layout, full chapter list.
- `SYLLABUS.md` — a 16-week schedule with deliverables.
- `ROADMAP.md` — chapter-by-chapter goals, topics, outcomes, and artifacts.
- `CONTRIBUTING.md` — eight-phase build plan and engineering standards.

## Who this is for

- Students of AI, Data Science, or Computer Science who have finished an intro-ML course.
- Self-learners who know classical ML (linear/logistic regression, KNN, decision trees, PCA) and want a structured path into modern Deep Learning.
- Practitioners who have used TensorFlow or PyTorch *as a black box* and now want to understand tensors, autograd, the training loop, loss landscapes, and optimizers.
- People preparing to specialize in Computer Vision, NLP, LLMs, Generative AI, or Edge AI — this is the prerequisite repo.
- Anyone building an open-source DL course or teaching repository.

## Prerequisites

- Comfortable with Python (functions, classes, list/dict comprehensions, `__init__`/`__call__`).
- Comfortable with NumPy (shape, broadcasting, slicing) and Matplotlib basics.
- Train/validation/test split, classification vs. regression, overfitting vs. underfitting at the level of a first ML course.
- Loss function, metric, and gradient descent at an *intuitive* level.
- Have at least seen linear regression, logistic regression, KNN, SVM, decision tree, and PCA.
- Comfortable with vectors, matrices, tensors, matrix multiplication, chain rule, gradient, basic probability, cross-entropy, and what "optimization" means.

The math review needed for backpropagation is folded back in during Chapters 2 and 3, so you do **not** need to revisit a full linear-algebra course before starting.

## Design Philosophy

- **Don't skip to the big model.** The course starts with `torch.Tensor`, a linear layer, an activation, a loss, an optimizer, and a training loop. Only once the loop is comfortable does the curriculum stack complexity.
- **From NumPy to PyTorch.** Forward and backward pass are first coded by hand in NumPy (Chapter 3) before `autograd` takes over. This is the same trick used in Karpathy's "micrograd" lecture and the PyTorch *Autograd Mechanics* tutorial — it prevents autograd from being a magic box.
- **Shape is everything.** Every chapter spells out batch, channel, sequence length, feature dimension, and the tensor transformation that turns the input shape into the output shape.
- **A training loop is a skill, not a snippet.** `model.train()`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`, validation under `torch.no_grad()`, checkpointing the best epoch — these are practiced every chapter.
- **Debugging is equal in weight to architecture.** Chapter 13 is an entire chapter on the 20-item debug checklist: bad data, bad shape, bad learning rate, wrong loss, mode mismatch, gradient explosion, missing normalization, label leak, augmentation that breaks labels, and so on.
- **Every chapter ships a deliverable.** A chapter without a notebook, assignment, lab, mini-project, or script is not done. This is enforced in the `ROADMAP.md` "Repo artifact" column.

## What you will be able to do

- Read a model definition (`class MyModel(nn.Module)`) and predict the shape of every intermediate tensor.
- Write a training loop from a blank file in 10 minutes that supports train/eval mode, validation, best-checkpoint saving, and logging.
- Diagnose a model that is not learning using the 20-item checklist in `docs/12_practical_methodology_debugging.md`.
- Fine-tune a pretrained ResNet on a custom image dataset.
- Build and train a small LSTM or toy Transformer.
- Train an autoencoder, visualize its latent space, and use reconstruction error for anomaly detection.
- Run a VAE or GAN demo on MNIST and explain the difference between the two.
- Convert a trained PyTorch model to a CLI inference script.
- Deliver an end-to-end project that another person can clone, install, and re-train.

## Non-goals

This course intentionally does **not** cover:

- Deep dives into Computer Vision (segmentation, detection, GAN tricks) — that is a separate CV course.
- LLM training/fine-tuning — covered by the LLM-track repo.
- MLOps, model serving, monitoring, A/B testing — covered by the AI-Engineering track.
- Hardware-specific deployment (TensorRT, Edge AI, mobile) — covered by the `Edge_Physical_AI` sister repo.
- Latest research (RLHF, mixture-of-experts, etc.) — those build on the foundation taught here.

## Beyond this course

After finishing this repo, you have the base to continue with:

- Computer Vision (segmentation, detection, ViT).
- Natural Language Processing and Large Language Models.
- Generative AI and Diffusion Models.
- Medical Image Analysis.
- AI on Edge / Physical AI / Robotics.
- MLOps for Deep Learning and Multimodal AI.
- Original research and product engineering with deep models.
