"""Export a trained checkpoint to ONNX and push to a HuggingFace Hub model repo.

Produces a fully-public, MIT-licensed model repo with:
  - the original .pt
  - an ONNX export at imgsz=640 with dynamic batch
  - a model card with usage snippet, training recipe, and metrics

Usage:
    python -m src.export_and_push \\
        --weights weights/visdrone.pt \\
        --repo-id LibreYOLO/visdrone-yolo9s \\
        --imgsz 640 \\
        --metrics-file runs/train/visdrone/metrics.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from libreyolo import LibreYOLO9


VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def _model_card(repo_id: str, imgsz: int, size: str, metrics: dict, epochs: int) -> str:
    metrics_block = json.dumps({k: v for k, v in metrics.items() if v is not None}, indent=2)
    return f"""---
library_name: libreyolo
pipeline_tag: object-detection
license: mit
tags:
  - libreyolo
  - yolov9
  - visdrone
  - aerial-imagery
  - object-detection
datasets:
  - Voxel51/VisDrone2019-DET
---

# {repo_id}

YOLOv9-{size} fine-tuned on VisDrone2019-DET aerial imagery using
[LibreYOLO](https://github.com/LibreYOLO/libreyolo). Ten classes
(pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle,
bus, motor), top-down drone perspective.

**Companion use case:** [LibreYOLO/use-cases/visdrone-finetune](https://github.com/LibreYOLO/use-cases/tree/main/visdrone-finetune).

## Training

- size: `{size}`
- imgsz: `{imgsz}`
- epochs: `{epochs}`
- dataset: VisDrone2019-DET via Voxel51's HuggingFace mirror
- compute: Apple Metal Performance Shaders (MPS, M-series GPU)

## Metrics

```json
{metrics_block}
```

## Usage — Python

```python
from huggingface_hub import hf_hub_download
from libreyolo import LibreYOLO

ckpt = hf_hub_download(repo_id="{repo_id}", filename="visdrone.pt")
model = LibreYOLO(ckpt)
result = model("aerial.jpg")
for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
    print(box, ["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"][int(cls)], float(conf))
```

## Usage — ONNX (browser, edge, cross-runtime)

```python
import onnxruntime as ort
from huggingface_hub import hf_hub_download

onnx = hf_hub_download(repo_id="{repo_id}", filename="visdrone.onnx")
session = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
# Preprocess image to (1, 3, {imgsz}, {imgsz}) float32 in [0,1] then:
out = session.run(None, {{"images": preprocessed}})
```

A live browser demo using this ONNX is at
https://libreyolo.github.io/use-cases/visdrone-finetune/demo/
(zero-install, runs locally in Chrome via WebGPU/onnxruntime-web).

## Classes (index → name)

| idx | name |
|---|---|
| 0 | pedestrian |
| 1 | people |
| 2 | bicycle |
| 3 | car |
| 4 | van |
| 5 | truck |
| 6 | tricycle |
| 7 | awning-tricycle |
| 8 | bus |
| 9 | motor |

## License

MIT (the model file). Dataset (VisDrone2019-DET) is governed by its own
[license terms](http://aiskyeye.com/) — please review for your use case.
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=Path("weights/visdrone.pt"))
    p.add_argument("--repo-id", required=True, help="HF Hub model repo (e.g. ander2221/visdrone-yolo9-preview)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--size", default="s")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--metrics-file", type=Path, default=None,
                   help="JSON file with mAP / loss metrics. Optional.")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    if not args.weights.exists():
        print(f"missing {args.weights}", file=sys.stderr)
        return 1

    print(f"Loading weights {args.weights}", flush=True)
    from .load_finetuned import load_visdrone_model
    model = load_visdrone_model(args.weights, size=args.size, device="cpu")

    out_dir = Path("export")
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_dst = out_dir / "visdrone.pt"
    shutil.copyfile(args.weights, pt_dst)

    onnx_path = out_dir / "visdrone.onnx"
    print(f"Exporting ONNX -> {onnx_path}", flush=True)
    try:
        # libreyolo BaseExporter / model.export
        model.export(format="onnx", imgsz=args.imgsz, output_path=str(onnx_path), simplify=True)
    except Exception as e:
        print(f"libreyolo .export failed ({e}); falling back to direct torch.onnx", file=sys.stderr)
        import torch
        dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
        torch.onnx.export(
            model.model, dummy, str(onnx_path),
            input_names=["images"], output_names=["output"],
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17, do_constant_folding=True,
        )

    metrics = {}
    if args.metrics_file and args.metrics_file.exists():
        metrics = json.loads(args.metrics_file.read_text())
    card = _model_card(args.repo_id, args.imgsz, args.size, metrics, args.epochs)
    (out_dir / "README.md").write_text(card)

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Initial visdrone-yolo9 weights + ONNX",
    )
    print(f"\nPushed -> https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
