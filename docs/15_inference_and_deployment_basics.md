# Chapter 16 — Inference and Deployment (basics)

> Training is the *first* half. A model that nobody can run on a new input is not finished — it is a notebook. This chapter is the bridge from "I have a checkpoint" to "anyone with my repo can predict on their own image".

## Mục tiêu (Goal)

After this chapter you can:

- Load the *best* checkpoint (by validation metric) — not the last one — into a fresh model.
- Apply the *same* preprocessing at inference as at training time, and explain why mismatch silently breaks accuracy.
- Write a CLI `inference.py` that takes a single input file (or a directory) and writes predictions to CSV/JSON.
- Outline the direction-only path to TorchScript and ONNX export, and know when to take it.

## Why this chapter

Almost every project in this course ships a `train.py` that produces a checkpoint and a metric. The next question — "what do I run on a new image?" — is left to inference code that often does not exist. This chapter makes that gap explicit and writes the missing script.

- **Builds on:** Chapter 4 (checkpoint save/load), Chapter 9 (preprocessing pipeline from torchvision), Chapter 13 (`model.eval()`, `no_grad`).
- **Sets up:** Chapter 17 (every final project needs an `inference.py` to be reproducible).

## Key concepts

### Save the *right* checkpoint

A training loop that saves only the last epoch's weights is broken — by the time you stop, the model may be in the middle of overfitting. The correct pattern is to save the model whose **validation metric** is best so far:

```python
best_val_acc = 0.0
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, optimizer)
    val_acc = evaluate(model, val_loader)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "config": config_dict,
        }, "experiments/checkpoints/best.pt")
```

What to save:

- `model.state_dict()` — the weights. *Not* the model object (which couples the checkpoint to your code's import path).
- The `epoch` and the metric, so you know what this checkpoint is when you find it later.
- The config (or a hash of it) so you can reconstruct the architecture without guessing.

What *not* to save: the optimizer state for inference-only loading (save it only if you plan to resume training).

### Load a checkpoint into a fresh model

The inverse: instantiate the model with the saved config, then load the weights:

```python
def load_for_inference(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MyModel(**ckpt["config"]["model_args"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt
```

Three guarantees this function enforces:

- `map_location=device` — loads to CPU even if the checkpoint was saved from a GPU, so the script runs on a laptop.
- Architecture is constructed from the *saved* config, not from whatever `config.yaml` happens to be on disk today.
- `.eval()` is called *immediately* — no caller can forget it.

### Preprocessing parity

If you trained on `224×224` ImageNet-normalized RGB images and you preprocess your new input differently, accuracy collapses silently — no error, just wrong predictions. The fix is to *bind* the preprocessing transform to the model:

```python
class ImageClassifier:
    def __init__(self, checkpoint_path, device="cpu"):
        self.model, ckpt = load_for_inference(checkpoint_path, device)
        self.classes = ckpt["config"]["classes"]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
        self.device = device

    @torch.no_grad()
    def predict(self, pil_image):
        x = self.transform(pil_image).unsqueeze(0).to(self.device)
        logits = self.model(x)                              # (1, num_classes)
        prob = F.softmax(logits, dim=-1).squeeze(0).cpu()
        idx = int(prob.argmax().item())
        return self.classes[idx], float(prob[idx])
```

Key points:

- The same `Resize → CenterCrop → ToTensor → Normalize` pipeline that was used at training time is *exactly* what runs at inference. Save the recipe in the config or in the class, not in a different file.
- `@torch.no_grad()` — no autograd graph, less memory, faster inference.
- `softmax` is computed at inference *only* (the training loss expects raw logits).

### Batch inference

For more than a handful of inputs, batching is 5–50× faster. Wrap a folder of files in a small dataset:

```python
class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform):
        self.paths = sorted(Path(folder).glob("*"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), str(self.paths[i])
```

Then iterate with a `DataLoader`:

```python
loader = torch.utils.data.DataLoader(
    FolderDataset(args.input_dir, classifier.transform),
    batch_size=32, num_workers=2, pin_memory=True,
)
rows = []
classifier.model.eval()
with torch.no_grad():
    for x, paths in loader:
        x = x.to(device)
        logits = classifier.model(x)
        prob, idx = F.softmax(logits, dim=-1).max(dim=-1)
        for p, c, s in zip(paths, idx.cpu().tolist(), prob.cpu().tolist()):
            rows.append({"path": p, "class": classifier.classes[c], "prob": s})
```

### Save predictions to CSV/JSON

Two formats cover almost all use cases:

```python
import json, csv

# JSON — one object per line, easy to stream.
with open("results.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

# CSV — Excel-friendly, lossy for nested data.
with open("results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "class", "prob"])
    writer.writeheader()
    writer.writerows(rows)
```

Pick **JSONL** for downstream Python pipelines; **CSV** when a non-technical user will open it.

### A canonical `inference.py`

The whole script in one piece:

```python
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="image file or directory")
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    clf = ImageClassifier(args.checkpoint, device=args.device)

    input_path = Path(args.input)
    paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*"))

    rows = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        cls, prob = clf.predict(img)
        rows.append({"path": str(p), "class": cls, "prob": prob})

    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} predictions to {args.output}")

if __name__ == "__main__":
    main()
```

CLI usage:

```bash
python src/inference.py \
    --checkpoint experiments/checkpoints/best.pt \
    --input data/new_images/ \
    --output results.jsonl
```

Three things this script gets right by construction:

- The checkpoint and the architecture stay paired through the saved config.
- The preprocessing pipeline is owned by the `ImageClassifier`, not duplicated in the CLI.
- Inputs can be a single file *or* a directory — most real users want both.

### TorchScript and ONNX — direction only

For deployment outside Python (mobile, browsers, C++ services, edge devices), you need to detach the model from the PyTorch runtime.

- **TorchScript** — compile the model with `torch.jit.script(model)` or `torch.jit.trace(model, example_input)`. The result is a self-contained `.pt` file that runs in `libtorch` (C++).
- **ONNX** — export with `torch.onnx.export(model, example_input, "model.onnx")`. The result is portable across many runtimes (ONNX Runtime, TensorRT, OpenVINO).
- **Hardware-specific** — TensorRT (NVIDIA), Core ML (Apple), TFLite (mobile), OpenVINO (Intel), Triton (server).

These are *direction-only* for this course. The takeaway: when someone asks "can your model run on iPhone?", the answer is *not* `python inference.py`. Export to TorchScript or ONNX/Core ML, then run on the target.

### A short README usage section

Every project should ship a README block that the next person can copy-paste:

```markdown
## Inference

Predict on a new image or directory using a trained checkpoint:

    python src/inference.py \
        --checkpoint experiments/checkpoints/best.pt \
        --input  path/to/image.jpg \
        --output results.jsonl

Output is JSONL with one prediction per line:

    {"path": "image.jpg", "class": "cat", "prob": 0.93}
```

If this section is missing or out of date, the project is not finished.

## Common pitfalls

- Loaded the *last* checkpoint instead of the best one → val accuracy 5+ points lower than reported in `report.md`. Always name the best checkpoint `best.pt` and the last one `last.pt`.
- Re-used training preprocessing but forgot `Normalize` → silent collapse to ~10% accuracy. Bind the transform to the model wrapper.
- Forgot `model.eval()` → dropout still on → predictions are noisy. Always call `.eval()` in the loader function.
- Forgot `@torch.no_grad()` → autograd graph eats memory → CUDA OOM on long batches. Wrap the whole inference path.
- Hard-coded the device (`cuda`) in the script → fails on a CPU-only laptop. Default to `cuda if torch.cuda.is_available() else cpu`.
- Saved the model object (`torch.save(model)`) instead of `state_dict` → checkpoint breaks the moment you rename or move the model file. Save `state_dict`, reconstruct from config.
- Mixed up logits and probabilities — applied softmax during training, again during inference. Train on raw logits; softmax only at inference.

## Learning outcomes

- Write `inference.py` for a vision project: load checkpoint, preprocess, predict, save results.
- Load the best-by-validation checkpoint and re-evaluate to confirm metrics match the training report.
- Add a "Usage" block to a project README that a stranger can run without reading the rest of the repo.
- Name one direction (TorchScript or ONNX) you would take to deploy outside Python and one runtime that consumes it.

## Quick check (self-test)

<details>
<summary>Q1 — Why is loading the *last* checkpoint a common mistake?</summary>

The last checkpoint is usually past the validation-best epoch, in the over-fitting regime, so val/test metrics are worse than what the training loop's best-so-far report claimed. Always keep a separate `best.pt` keyed by validation metric.
</details>

<details>
<summary>Q2 — Why do we save `state_dict()` and not the model object?</summary>

`state_dict()` is a plain dictionary of tensors and is robust to code changes — you can rename, refactor, or restructure your model file and still load the weights. Saving the whole object couples the checkpoint to the exact Python class path at the time of save.
</details>

<details>
<summary>Q3 — Your inference predictions look random even though val accuracy was 92%. First thing to check?</summary>

Preprocessing parity — did you call the *same* `Normalize` with the *same* mean and std at inference as at training? The second-most-likely cause is forgetting `model.eval()`.
</details>

<details>
<summary>Q4 — Name one direction to deploy a PyTorch model outside Python and one consumer of that format.</summary>

TorchScript via `torch.jit.script(model)`, consumed by `libtorch` from a C++ application; *or* ONNX via `torch.onnx.export`, consumed by ONNX Runtime, TensorRT, or Core ML.
</details>

## Further reading

- PyTorch tutorial — *Saving and Loading Models*.
- PyTorch tutorial — *Exporting a Model from PyTorch to ONNX*.
- PyTorch docs — `torch.jit.script`, `torch.jit.trace`, `torch.onnx.export`.
- ONNX Runtime — getting-started guide.

## Companion artifact

`labs/lab_07_inference_cli.ipynb` — wrap a trained MLP/CNN checkpoint into a CLI inference script, run it on a folder of new images, and emit both CSV and JSONL outputs.
