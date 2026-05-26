"""Predict on a single image or a directory of images using a saved checkpoint.

Usage:
    python src/inference.py \\
        --checkpoint experiments/mlp_mnist/checkpoints/best.pt \\
        --input  sample.png        # or a directory
        --output results.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import mnist_inference_transform
from models import build_model
from train import resolve_device


IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".webp"}


def load_for_inference(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    classes = ckpt.get("classes") or [str(i) for i in range(cfg["model"]["num_classes"])]

    # Pick the right transform for the dataset the model was trained on.
    if cfg["data"]["name"] == "mnist":
        transform = mnist_inference_transform(cfg["data"]["mean"], cfg["data"]["std"])
    else:
        raise ValueError(f"no inference transform registered for dataset "
                         f"{cfg['data']['name']!r}")
    return model, transform, classes, cfg


@torch.no_grad()
def predict_one(model, transform, classes, pil_image, device) -> tuple[str, float]:
    x = transform(pil_image).unsqueeze(0).to(device)
    logits = model(x)
    prob = F.softmax(logits, dim=-1).squeeze(0).cpu()
    idx = int(prob.argmax().item())
    return classes[idx], float(prob[idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="image file or directory")
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument("--device", default="auto",
                        help="'auto', 'cpu', or 'cuda'. Default auto.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, transform, classes, _ = load_for_inference(args.checkpoint, device)
    print(f"device: {device}, num_classes: {len(classes)}")

    input_path = Path(args.input)
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(p for p in input_path.iterdir()
                       if p.suffix.lower() in IMG_SUFFIXES)
        if not paths:
            raise FileNotFoundError(f"no images in {input_path}")

    rows = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        cls, prob = predict_one(model, transform, classes, img, device)
        rows.append({"path": str(p), "class": cls, "prob": prob})
        print(f"{p.name:40s} -> {cls}  ({prob:.3f})")

    out_path = Path(args.output)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nwrote {len(rows)} predictions to {out_path}")


if __name__ == "__main__":
    main()
