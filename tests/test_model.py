"""Forward-shape smoke tests for every registered model.

ResNet-18 is exercised with ``pretrained=False`` so the suite never hits
the network. All other models are pure PyTorch — instant.
"""

import pytest
import torch

from models import build_model


def test_mlp_forward_shape():
    cfg = {"model": {
        "name": "mlp", "in_dim": 28 * 28, "hidden_dim": 32,
        "num_classes": 10, "dropout": 0.0,
    }}
    model = build_model(cfg)
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 10)


def test_small_cnn_forward_shape():
    cfg = {"model": {
        "name": "small_cnn", "in_channels": 3, "num_classes": 10,
        "channels": [8, 16, 32], "dropout": 0.0,
    }}
    model = build_model(cfg)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)


def test_resnet18_scratch_forward_shape():
    cfg = {"model": {
        "name": "resnet18", "num_classes": 4,
        "pretrained": False, "freeze_backbone": False,
    }}
    model = build_model(cfg)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 4)


def test_resnet18_freeze_only_head_trainable():
    cfg = {"model": {
        "name": "resnet18", "num_classes": 4,
        "pretrained": False, "freeze_backbone": True,
    }}
    model = build_model(cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Only the new fc head should be trainable: 512 * 4 + 4 = 2052.
    assert trainable == 2052


def test_lstm_text_forward_shape():
    cfg = {"model": {
        "name": "lstm_text", "vocab_size": 100, "embed_dim": 16,
        "hidden_dim": 32, "num_classes": 4, "num_layers": 1,
        "dropout": 0.0, "pad_id": 0,
    }}
    model = build_model(cfg)
    x = torch.randint(0, 100, (3, 8))
    y = model(x)
    assert y.shape == (3, 4)


def test_lstm_text_forward_with_lengths():
    cfg = {"model": {
        "name": "lstm_text", "vocab_size": 100, "embed_dim": 16,
        "hidden_dim": 32, "num_classes": 4, "num_layers": 1,
        "dropout": 0.0, "pad_id": 0,
    }}
    model = build_model(cfg)
    x = torch.randint(1, 100, (3, 8))   # avoid id 0 (pad)
    lengths = torch.tensor([8, 5, 3])
    y = model(x, lengths=lengths)
    assert y.shape == (3, 4)


def test_char_transformer_forward_shape():
    cfg = {"model": {
        "name": "char_transformer", "vocab_size": 30, "d_model": 32,
        "num_heads": 4, "d_ff": 64, "num_layers": 2, "block_size": 16,
        "dropout": 0.0,
    }}
    model = build_model(cfg)
    x = torch.randint(0, 30, (2, 16))
    y = model(x)
    assert y.shape == (2, 16, 30)


def test_char_transformer_generate():
    cfg = {"model": {
        "name": "char_transformer", "vocab_size": 30, "d_model": 32,
        "num_heads": 4, "d_ff": 64, "num_layers": 2, "block_size": 16,
        "dropout": 0.0,
    }}
    model = build_model(cfg)
    prompt = torch.tensor([[1, 2, 3]])
    out = model.generate(prompt, max_new_tokens=5, temperature=1.0)
    assert out.shape == (1, 3 + 5)


def test_conv_autoencoder_forward_shape():
    cfg = {"model": {
        "name": "conv_autoencoder", "in_channels": 1, "latent_dim": 16,
    }}
    model = build_model(cfg)
    x = torch.rand(2, 1, 28, 28)
    x_hat, z = model(x)
    assert x_hat.shape == (2, 1, 28, 28)
    assert z.shape == (2, 16)
    # Sigmoid output must lie in [0, 1].
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        build_model({"model": {"name": "no_such_model"}})
