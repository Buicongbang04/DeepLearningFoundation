# Deep Learning Foundation for AI

A learning-by-building repository for Deep Learning fundamentals with **PyTorch**. The goal is to take someone who has finished an *Intro Machine Learning* course and bring them to the point where they can read papers, write a clean training loop, debug a model that is not learning, and ship a small end-to-end project — *before* diving into Computer Vision, NLP, LLMs, or Edge AI.

Primary reference: *Deep Learning* by **Ian Goodfellow, Yoshua Bengio, and Aaron Courville**, paired with the official PyTorch tutorials.

This repo is the third in a three-course AI-Foundation track:

```
Python + Math        Machine Learning      Deep Learning      → CV / NLP / LLM / Generative / Edge AI
foundation     →     foundation       →    foundation  (you are here)
```

## Course Goals

After finishing this course, you will be able to:

- Explain *why* deep models work and where representation learning fits in the AI/ML/DL landscape.
- Move fluently between tensors, autograd, `nn.Module`, `DataLoader`, optimizer, and scheduler.
- Write a clean training loop with train/eval mode, validation, checkpointing, and logging.
- Build MLPs, CNNs, RNN/LSTM, attention, and a toy Transformer from scratch in PyTorch.
- Diagnose a model that is not learning — using a 20-item debugging checklist.
- Apply transfer learning with a pretrained ResNet and a custom head.
- Train an autoencoder, run a VAE/GAN demo, and understand the diffusion idea at a conceptual level.
- Deliver an end-to-end project that someone else can clone, install, train, and re-evaluate.

## Course Structure

The course is organized into 17 chapters (numbered 0 to 16), grouped into six blocks that match the eight-phase build plan in `CONTRIBUTING.md`:

1. **PyTorch foundation (Ch 0-4)** — what Deep Learning is, tensors, autograd, MLP from scratch, the standard training loop.
2. **Core neural-network engineering (Ch 5-7)** — activations, initialization, normalization, regularization, optimization.
3. **Vision foundation (Ch 8-9)** — convolutional networks, classic architectures, transfer learning.
4. **Sequence and Transformer (Ch 10-12)** — RNN/GRU/LSTM, attention, intro Transformer.
5. **Practical methodology, representation, generative (Ch 13-15)** — debugging deep models, autoencoders, generative-model concepts.
6. **Inference, deployment, final project (Ch 16-17)** — loading a checkpoint, batch/CLI inference, end-to-end capstone.

Full chapter list:

- Chapter 0 — Deep Learning là gì? (Intro Deep Learning)
- Chapter 1 — PyTorch Warmup: Tensor, Device, Autograd
- Chapter 2 — From Logistic Regression to Neural Network
- Chapter 3 — Forward Pass, Loss, Backpropagation
- Chapter 4 — A Standard Training Loop in PyTorch
- Chapter 5 — Activation, Initialization, Normalization
- Chapter 6 — Regularization for Deep Learning
- Chapter 7 — Optimization for Deep Learning
- Chapter 8 — Convolutional Neural Networks (basics)
- Chapter 9 — CNN Architectures and Transfer Learning
- Chapter 10 — Sequence Modeling with RNN, GRU, LSTM
- Chapter 11 — Attention Mechanism
- Chapter 12 — Transformer (intro)
- Chapter 13 — Practical Methodology: Debugging Deep Models
- Chapter 14 — Autoencoder and Representation Learning
- Chapter 15 — Generative Models (intro)
- Chapter 16 — Inference and Deployment (basics)
- Chapter 17 — Final Project: Deep Learning End-to-End

See `ROADMAP.md` for the per-chapter goals, topics, outcomes, and the repo artifact each chapter ships with.

## How to Use This Repo

Top-level documents and their distinct roles:

- `README.md` — entry point: what the repo is, how to install, how to navigate.
- `COURSE_OVERVIEW.md` — audience, prerequisites, design philosophy.
- `SYLLABUS.md` — a 16-week schedule that maps chapters to weekly deliverables.
- `ROADMAP.md` — per-chapter blueprint with goals, topics, outcomes, and repo artifacts.
- `CONTRIBUTING.md` — guide for adding chapters, labs, projects.
- `docs/RUBRIC.md` — final-project grading rubric (100 points).
- `docs/REPRODUCIBILITY.md` — three commands + expected metric per project; fresh-env install path.

A typical learning loop per chapter:

1. Read the chapter note in `docs/`.
2. Run the matching notebook in `notebooks/` and tweak the hyperparameters.
3. Solve the assignment in `assignments/` or the integration lab in `labs/`.
4. Once a block is finished, tackle the relevant mini-project in `projects/`.
5. At the end of the course, deliver your capstone using `projects/final_project_template/`.

## Requirements

Python 3.10 or newer is required. A CUDA-capable GPU is recommended from Chapter 8 onward but not required — every notebook also runs on CPU (with smaller batch sizes).

With pip:

```bash
pip install -r requirements.txt
```

With Conda:

```bash
conda env create -f environment.yml
conda activate dlcourse
```

## Quick Start

```bash
git clone <repo-url>
cd Deep_Learning_Foundation
pip install -r requirements.txt

# Train the first mini-project (MLP on MNIST) from CLI.
python src/train.py     --config configs/mlp_mnist.yaml
python src/evaluate.py  --config configs/mlp_mnist.yaml \
                        --checkpoint experiments/mlp_mnist/checkpoints/best.pt
python src/inference.py --checkpoint experiments/mlp_mnist/checkpoints/best.pt --input sample.png

# Confirm the install with the test suite.
pytest tests/
```

See `docs/REPRODUCIBILITY.md` for the three commands + expected metric for every shipped project.

## Course content map

| Chapter | Doc | Notebook / Lab | Project | Status |
|--------:|-----|----------------|---------|:------:|
| 0  | [`docs/00_intro_deep_learning.md`](docs/00_intro_deep_learning.md)                           | —                                                                       | —                                                              | ✓ |
| 1  | [`docs/01_tensor_autograd_pytorch.md`](docs/01_tensor_autograd_pytorch.md)                  | [`notebooks/chapter_01_tensor_autograd.ipynb`](notebooks/chapter_01_tensor_autograd.ipynb) + [`assignments/assignment_01_autograd_basics.ipynb`](assignments/assignment_01_autograd_basics.ipynb) | —                                                              | ✓ |
| 2–3| [`docs/02_mlp_backpropagation.md`](docs/02_mlp_backpropagation.md)                          | [`notebooks/chapter_02_mlp_from_scratch.ipynb`](notebooks/chapter_02_mlp_from_scratch.ipynb)               | —                                                              | ✓ |
| 4  | [`docs/03_training_loop.md`](docs/03_training_loop.md)                                       | —                                                                       | [`projects/project_01_mlp_mnist/`](projects/project_01_mlp_mnist/)             | ✓ |
| 5  | [`docs/04_activation_initialization_normalization.md`](docs/04_activation_initialization_normalization.md) | [`notebooks/chapter_04_activation_init_norm.ipynb`](notebooks/chapter_04_activation_init_norm.ipynb)         | —                                                              | ✓ |
| 6  | [`docs/05_regularization.md`](docs/05_regularization.md)                                     | [`notebooks/chapter_05_regularization_dropout_batchnorm.ipynb`](notebooks/chapter_05_regularization_dropout_batchnorm.ipynb)         | —                                                              | ✓ |
| 7  | [`docs/06_optimization.md`](docs/06_optimization.md)                                         | [`notebooks/chapter_06_optimization_adam_sgd_scheduler.ipynb`](notebooks/chapter_06_optimization_adam_sgd_scheduler.ipynb) | —                                                              | ✓ |
| 8  | [`docs/07_cnn_basics.md`](docs/07_cnn_basics.md)                                             | [`notebooks/chapter_07_cnn_basics.ipynb`](notebooks/chapter_07_cnn_basics.ipynb)                                                                       | [`projects/project_02_cnn_cifar10/`](projects/project_02_cnn_cifar10/)                | ✓ |
| 9  | [`docs/08_cnn_architectures_transfer_learning.md`](docs/08_cnn_architectures_transfer_learning.md)         | [`labs/lab_05_transfer_learning.ipynb`](labs/lab_05_transfer_learning.ipynb)                              | [`projects/project_03_transfer_learning_custom/`](projects/project_03_transfer_learning_custom/)                            | ✓ |
| 10 | [`docs/09_sequence_models.md`](docs/09_sequence_models.md)                                   | [`notebooks/chapter_09_rnn_gru_lstm.ipynb`](notebooks/chapter_09_rnn_gru_lstm.ipynb)                       | [`projects/project_04_text_classification_lstm/`](projects/project_04_text_classification_lstm/)                              | ✓ |
| 11 | [`docs/10_attention_mechanism.md`](docs/10_attention_mechanism.md)                          | [`notebooks/chapter_10_attention_toy.ipynb`](notebooks/chapter_10_attention_toy.ipynb)                  | —                                                              | ✓ |
| 12 | [`docs/11_transformer_intro.md`](docs/11_transformer_intro.md)                              | [`notebooks/chapter_11_transformer_intro.ipynb`](notebooks/chapter_11_transformer_intro.ipynb)          | [`projects/project_05_transformer_toy_language_model/`](projects/project_05_transformer_toy_language_model/)                                  | ✓ |
| 13 | [`docs/12_practical_methodology_debugging.md`](docs/12_practical_methodology_debugging.md)  | [`labs/lab_06_experiment_tracking.ipynb`](labs/lab_06_experiment_tracking.ipynb)                        | —                                                              | ✓ |
| 14 | [`docs/13_autoencoders_representation_learning.md`](docs/13_autoencoders_representation_learning.md)       | —                                                                       | [`projects/project_06_autoencoder_anomaly_detection/`](projects/project_06_autoencoder_anomaly_detection/)                                       | ✓ |
| 15 | [`docs/14_generative_models_intro.md`](docs/14_generative_models_intro.md)                  | [`notebooks/chapter_14_vae_gan_diffusion_intro.ipynb`](notebooks/chapter_14_vae_gan_diffusion_intro.ipynb) | —                                                              | ✓ |
| 16 | [`docs/15_inference_and_deployment_basics.md`](docs/15_inference_and_deployment_basics.md)  | [`labs/lab_07_inference_cli.ipynb`](labs/lab_07_inference_cli.ipynb)                                   | —                                                              | ✓ |
| 17 | [`docs/16_final_project.md`](docs/16_final_project.md) + [`docs/RUBRIC.md`](docs/RUBRIC.md)  | —                                                                       | [`projects/final_project_template/`](projects/final_project_template/)                              | ✓ |

## Repository Layout

```
.
├── README.md                # this file: what the repo is, how to install
├── COURSE_OVERVIEW.md       # audience, prerequisites, design philosophy
├── SYLLABUS.md              # 16-week schedule with deliverables
├── ROADMAP.md               # chapter-by-chapter blueprint
├── CONTRIBUTING.md          # guide for adding chapters, labs, projects
├── requirements.txt
├── environment.yml
├── LICENSE
├── docs/                    # written notes per chapter (Markdown)
├── notebooks/               # runnable companion notebooks
├── assignments/             # graded exercises
├── labs/                    # integration labs across chapters
├── projects/                # mini-projects and the final-project template
├── datasets/                # dataset documentation (large files not committed)
├── src/                     # reusable utilities + notebook builders
├── configs/                 # YAML configs (train/eval/inference)
├── experiments/             # checkpoints and run logs (gitignored)
├── tests/                   # pytest smoke tests
├── figures/                 # shared diagrams
└── reports/                 # written experiment reports
```

## License

This project is released for educational purposes. See `LICENSE` for details.

## Acknowledgements

- Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning* — primary theoretical reference.
- The PyTorch team, for the documentation, tutorials, and the framework itself.
- Vũ Hữu Tiệp, *Machine Learning cơ bản* — companion text for the classical-ML chapters this course builds on.
