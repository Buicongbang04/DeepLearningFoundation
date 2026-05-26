# Roadmap

Per-chapter blueprint for the entire course. This file is the single source of truth for chapter goals, topics, learning outcomes, and the repo artifacts each chapter ships with.

Numbering matches the chapter folders in `docs/` and the names of the notebooks in `notebooks/`. Each chapter pairs a Markdown note in `docs/` with at least one runnable notebook and (where relevant) an assignment, a lab, or a mini-project.

## Chapter 0 — Deep Learning là gì? (Intro Deep Learning)

**Goal:** place Deep Learning in the broader AI / ML / representation-learning picture.

Topics: AI vs. ML vs. DL · feature engineering vs. learned representations · hierarchy of concepts (pixels → edges → corners → parts → objects) · DL as multi-layer function optimization over data.

Outcomes: draw the AI → ML → Representation Learning → DL diagram; explain why representation matters; write a 300-500-word note comparing classical ML and DL.

Repo artifacts: `docs/00_intro_deep_learning.md`; `figures/ai_ml_dl_relationship.png`; `figures/hierarchy_of_concepts.png`.

---

## Chapter 1 — PyTorch Warmup: Tensor, Device, Autograd

**Goal:** the minimum PyTorch toolkit needed to do modern Deep Learning.

Topics: tensor — shape, dtype, device · CPU vs. GPU, broadcasting, tensor operations · automatic differentiation · computational graph · `requires_grad`, `.backward()`, `.grad`, `torch.no_grad`.

Outcomes: create 1-D, 2-D, and 4-D tensors and explain the shape; move tensors between CPU and GPU when CUDA is available; compute gradients with `autograd` and cross-check against a by-hand derivative.

Repo artifacts: `docs/01_tensor_autograd_pytorch.md`; `notebooks/chapter_01_tensor_autograd.ipynb`; `assignments/assignment_01_autograd_basics.ipynb`.

---

## Chapter 2 — From Logistic Regression to Neural Network

**Goal:** bridge the gap between classical ML and neural networks.

Topics: linear model · logistic regression as a single-layer NN with no hidden layer · perceptron · MLP, hidden layer, activation, non-linearity · the XOR problem and why a linear model cannot solve it.

Outcomes: explain why no linear model can solve XOR; build and train a small MLP with PyTorch on a toy 2-D dataset; draw the decision boundary.

Repo artifacts: `docs/02_mlp_backpropagation.md` (Part A — from logistic regression to MLP); `notebooks/chapter_02_mlp_from_scratch.ipynb`.

---

## Chapter 3 — Forward Pass, Loss, Backpropagation

**Goal:** see end-to-end what happens during one training step of a small MLP.

Topics: forward pass, prediction, loss function · chain rule, backpropagation, gradient flow · manual backprop on a small network with NumPy · autograd in practice for the same network.

Outcomes: describe the per-step training cycle in words; compute forward pass on a small MLP by hand; write a minimal backward pass in NumPy and check it against PyTorch autograd.

Repo artifacts: `docs/02_mlp_backpropagation.md` (Part B — backpropagation by hand); `notebooks/chapter_02_mlp_from_scratch.ipynb` (the same notebook, with the manual-backward section).

---

## Chapter 4 — A Standard Training Loop in PyTorch

**Goal:** write the canonical PyTorch training loop — train/eval mode, validation, metric, checkpoint.

Topics: `Dataset`, `DataLoader`, batch training · `nn.Module` · loss, optimizer, `model.train()` / `model.eval()`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` · validation loop · checkpoint save/load · random seed.

Outcomes: build an MLP pipeline on MNIST; save the best checkpoint by validation accuracy; load that checkpoint and run inference; split the code cleanly into `train.py`, `evaluate.py`, `models.py`.

Repo artifacts: `docs/03_training_loop.md`; `projects/project_01_mlp_mnist/`; `configs/mlp_mnist.yaml`; `src/train.py`, `src/evaluate.py`, `src/inference.py`.

---

## Chapter 5 — Activation, Initialization, Normalization

**Goal:** see why activation choice, weight initialization, and normalization decide whether a deep model trains at all.

Topics: sigmoid, tanh, ReLU, Leaky ReLU, GELU · vanishing and exploding gradients · Xavier and He initialization · BatchNorm, LayerNorm.

Outcomes: visually compare activations on a curve; train the same MLP with different activations and report which is more stable; show that He initialization speeds up ReLU networks.

Repo artifacts: `docs/04_activation_initialization_normalization.md`; `notebooks/chapter_04_activation_init_norm.ipynb`.

---

## Chapter 6 — Regularization for Deep Learning

**Goal:** keep a deep network from memorizing the training set.

Topics: weight decay, L1 / L2 penalty, dropout, early stopping · data augmentation, noise injection, label smoothing, ensemble (concept).

Outcomes: apply dropout and weight decay; augment images for CIFAR-10; spot overfitting from a train-vs-validation curve.

Repo artifacts: `docs/05_regularization.md`; `notebooks/chapter_05_regularization_dropout_batchnorm.ipynb`.

---

## Chapter 7 — Optimization for Deep Learning

**Goal:** the practical optimizer toolbox.

Topics: SGD, mini-batch SGD, momentum, Nesterov, RMSProp · Adam, AdamW · learning rate, scheduler, warmup, gradient clipping.

Outcomes: compare SGD, Momentum, Adam, and AdamW on the same task; plug a `lr_scheduler.OneCycleLR` into the training loop; use gradient clipping for an RNN.

Repo artifacts: `docs/06_optimization.md`; `notebooks/chapter_06_optimization_adam_sgd_scheduler.ipynb`.

---

## Chapter 8 — Convolutional Neural Networks (basics)

**Goal:** Convolutional networks as the foundational vision architecture.

Topics: local connectivity, parameter sharing, convolution · kernel/filter, stride, padding, feature map, pooling, receptive field · Conv → ReLU → Pool → FC block.

Outcomes: compute output shape after a convolution by hand; build a CNN for MNIST / CIFAR-10; compare MLP vs. CNN on CIFAR-10 with a confusion matrix and a training curve.

Repo artifacts: `docs/07_cnn_basics.md`; `notebooks/chapter_07_cnn_basics.ipynb`; `projects/project_02_cnn_cifar10/`.

---

## Chapter 9 — CNN Architectures and Transfer Learning

**Goal:** read the famous CNN architectures and reuse them via transfer learning.

Topics: LeNet, AlexNet, VGG, Inception (concept) · ResNet, skip connections, EfficientNet/ConvNeXt (direction) · transfer learning, freezing, fine-tuning, custom heads.

Outcomes: explain why a skip connection helps a deep network train; fine-tune a pretrained ResNet on a small custom dataset; compare from-scratch training and transfer learning on the same dataset.

Repo artifacts: `docs/08_cnn_architectures_transfer_learning.md`; `labs/lab_05_transfer_learning.ipynb`; `projects/project_03_transfer_learning_custom/`.

---

## Chapter 10 — Sequence Modeling with RNN, GRU, LSTM

**Goal:** model sequences before tackling attention and Transformer.

Topics: sequence data, time step, recurrent connection, unrolling · hidden state, many-to-one, one-to-many, many-to-many · vanishing gradient in RNN; GRU and LSTM cells; text classification.

Outcomes: build a small LSTM text classifier; use an embedding layer and gradient clipping; compare vanilla RNN against LSTM on the same task.

Repo artifacts: `docs/09_sequence_models.md`; `notebooks/chapter_09_rnn_lstm.ipynb`; `projects/project_04_text_classification_lstm/`.

---

## Chapter 11 — Attention Mechanism

**Goal:** the Query / Key / Value idea, before any Transformer details.

Topics: long-range dependency · Query, Key, Value · dot-product attention, scaled dot-product attention · attention weights, self-attention, multi-head attention (intuition).

Outcomes: implement scaled dot-product attention in PyTorch from scratch; explain what `Q`, `K`, `V` are projections of; plot an attention matrix on a toy sequence.

Repo artifacts: `docs/10_attention_mechanism.md`; `notebooks/chapter_10_attention_toy.ipynb`.

---

## Chapter 12 — Transformer (intro)

**Goal:** put the Transformer together as the central architecture of modern NLP, ViT, and LLMs.

Topics: token embedding, positional encoding · self-attention, multi-head attention, FFN, residual, LayerNorm · encoder block, decoder block, causal mask · encoder-only / decoder-only / encoder-decoder; pretraining vs. fine-tuning (concept).

Outcomes: draw a transformer block from memory; tell encoder-only / decoder-only / encoder-decoder apart and name one model in each family; build a toy Transformer classifier or a character-level language model.

Repo artifacts: `docs/11_transformer_intro.md`; `projects/project_05_transformer_toy_language_model/`.

---

## Chapter 13 — Practical Methodology: Debugging Deep Models

**Goal:** *learn to debug a model that is not learning*, instead of randomly changing the architecture.

Topics: check input/label/shape/learning rate/gradient norm · overfit a tiny batch, watch train vs. validation loss · `model.train()` / `model.eval()`, `no_grad`, data leakage, normalization, experiment tracking.

Outcomes: have a 20-item debug checklist memorized; tell apart a "data bug" / "model bug" / "optimizer bug" / "metric bug"; log gradient norm during training.

Repo artifacts: `docs/12_practical_methodology_debugging.md`; `labs/lab_06_experiment_tracking.ipynb`.

---

## Chapter 14 — Autoencoder and Representation Learning

**Goal:** unsupervised representation learning via the autoencoder.

Topics: encoder, latent representation, decoder · reconstruction loss, undercomplete and denoising autoencoders · anomaly detection, dimensionality reduction.

Outcomes: build an autoencoder that reconstructs MNIST; visualize the 2-D latent space; use reconstruction error to flag anomalies on a contaminated dataset.

Repo artifacts: `docs/13_autoencoders_representation_learning.md`; `projects/project_06_autoencoder_anomaly_detection/`.

---

## Chapter 15 — Generative Models (intro)

**Goal:** the three big families of generative models *at a conceptual level only*.

Topics: generative vs. discriminative · VAE, GAN, diffusion (concept) · sampling, latent space, mode collapse, denoising process.

Outcomes: tell VAE, GAN, and diffusion apart by their training objective in one sentence each; run a small VAE or GAN demo on MNIST; explain the limits of these intro-level demos.

Repo artifacts: `docs/14_generative_models_intro.md`; `notebooks/chapter_14_vae_gan_diffusion_intro.ipynb`.

---

## Chapter 16 — Inference and Deployment (basics)

**Goal:** training is not the end — load the checkpoint, preprocess new input, batch-predict.

Topics: load checkpoint, `model.eval()`, `torch.no_grad` · preprocess new input, batch inference, save predictions · CLI inference script, exporting CSV/JSON results; TorchScript/ONNX (direction only).

Outcomes: write `inference.py` for a new image or text; load the best checkpoint correctly (not the last one); output predictions and write a usage section in the README.

Repo artifacts: `docs/15_inference_and_deployment_basics.md`; `labs/lab_07_inference_cli.ipynb`.

---

## Chapter 17 — Final Project: Deep Learning End-to-End

**Goal:** apply everything to a complete end-to-end DL project that someone else can re-run.

Topics: problem description, dataset, preprocessing, `Dataset`/`DataLoader` · baseline, main model, training/validation, checkpoint, metric · training curve, error analysis, inference script, reproducibility instruction.

Outcomes: submit a clean repo with `README.md`, `configs/`, `src/`, `checkpoint/results/`, and a written report; present an experiment comparison and one direction for improvement.

Repo artifacts: `docs/16_final_project.md`; `projects/final_project_template/`.
