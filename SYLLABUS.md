# Syllabus — 16-Week Plan

A 16-week schedule mapping the chapters in `ROADMAP.md` to weekly deliverables. Each week pairs reading from `docs/` with a runnable artifact in `notebooks/`, `labs/`, `assignments/`, or `projects/`. Times are guidelines — most learners spend 5-8 hours per week (more in the CNN and Transformer blocks).

| Week | Chapters in `ROADMAP.md` | Topic                                                | Deliverable                                       |
|------|--------------------------|------------------------------------------------------|---------------------------------------------------|
| 1    | Ch 0                     | Deep Learning vs. ML, representation learning        | Concept note (300-500 words)                      |
| 2    | Ch 1                     | PyTorch tensor, device, autograd                     | Autograd assignment                               |
| 3    | Ch 2                     | Logistic regression → MLP, the XOR problem           | `chapter_02_mlp_from_scratch.ipynb`               |
| 4    | Ch 3, Ch 4               | Forward, loss, backprop; standard training loop      | Training-loop template                            |
| 5    | Ch 4 (cont.)             | Dataset / DataLoader / checkpoint                    | `projects/project_01_mlp_mnist/`                  |
| 6    | Ch 5                     | Activation, initialization, normalization            | Activation comparison notebook                    |
| 7    | Ch 6                     | Regularization, dropout, batchnorm                   | Regularization report                             |
| 8    | Ch 7                     | Optimization — SGD, Adam, scheduler                  | Optimizer-comparison notebook                     |
| 9    | Ch 8                     | CNN basics — Conv, Pool, FC                          | CNN on MNIST / CIFAR-10                           |
| 10   | Ch 9                     | CNN architectures, ResNet, transfer learning         | Custom-image classifier (transfer learning)       |
| 11   | Ch 10                    | RNN, GRU, LSTM                                       | `projects/project_04_text_classification_lstm/`   |
| 12   | Ch 11                    | Attention — Q, K, V                                  | `chapter_10_attention_toy.ipynb`                  |
| 13   | Ch 12                    | Transformer                                          | Toy Transformer / char-level LM                   |
| 14   | Ch 13, Ch 14             | Practical methodology, autoencoder                   | Debug checklist + autoencoder demo                |
| 15   | Ch 15, Ch 16             | Generative intro, inference & deployment basics      | VAE/GAN demo + `inference.py`                     |
| 16   | Ch 17                    | Final project presentation                           | Report + final repo                               |

## Weekly Loop

For each week:

1. Read the corresponding chapter note(s) in `docs/`.
2. Open and run the companion notebook(s) in `notebooks/`.
3. Tweak hyperparameters — learning rate, batch size, depth, width — and watch what changes.
4. Solve the assignment / lab / project listed in the deliverable column.
5. Write a one-paragraph reflection (what surprised you, what is still fuzzy).

## Checkpoints

- **After Week 5** — you can write a PyTorch training loop on a blank screen and overfit a small MLP on a subset of MNIST.
- **After Week 8** — you can read a training curve, decide whether the model is under- or overfitting, and pick the right next move (more data, more capacity, more regularization, different optimizer).
- **After Week 10** — you can fine-tune a pretrained ResNet on a custom dataset of your own.
- **After Week 13** — you can write a forward pass for a small Transformer block from memory.
- **After Week 16** — you have a polished end-to-end mini-project in `projects/` and a written report.

## Pace adjustments

- **Faster** (2x speed): pair two weeks per week, drop the weekly reflection, and skip optional further reading. Suitable for graduate students or experienced ML engineers picking up PyTorch.
- **Slower** (0.5x speed): take two weeks per row above, spend extra time on autograd (week 2) and on the training loop (weeks 4-5), and revisit Chapter 5 (activations, init, normalization) twice. Recommended for absolute beginners to Deep Learning.

## Recommended GPU access

- Weeks 1-8: CPU is fine (MNIST, small MLPs).
- Weeks 9-10 (CNN): a free Colab T4 GPU is recommended; otherwise reduce CIFAR-10 batch size to 32.
- Weeks 11-13 (RNN / Transformer): GPU strongly recommended.
- Weeks 14-16 (Autoencoder, VAE/GAN, final project): GPU strongly recommended; size the project to the hardware you have.
