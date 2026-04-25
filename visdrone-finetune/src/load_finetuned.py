"""Load a VisDrone fine-tuned LibreYOLO9 checkpoint.

The fine-tune retains the COCO-pretrained 80-channel intermediate cls layers
even though the final classification head is reshaped to 10 classes (this
is how libreyolo's `_rebuild_for_new_classes` works — preserves intermediate
weights, only swaps the final conv). To load the resulting hybrid checkpoint
we have to reproduce that rebuild path:

    1. instantiate LibreYOLO9 from the COCO 80-class weights (gets the
       full-fat intermediate architecture)
    2. rebuild for 10 classes (truncates only the final conv)
    3. load_state_dict(strict=True) the trained file

This helper centralizes that.

Usage:
    from src.load_finetuned import load_visdrone_model
    model = load_visdrone_model("weights/visdrone.pt", device="mps")
"""
from __future__ import annotations

from pathlib import Path
import torch

from libreyolo.models.yolo9.model import LibreYOLO9


def load_visdrone_model(
    weights: str | Path,
    *,
    pretrained: str | Path = "weights/LibreYOLO9t.pt",
    size: str = "t",
    nb_classes: int = 10,
    device: str = "cpu",
) -> LibreYOLO9:
    """Return a LibreYOLO9 instance with the trained VisDrone weights loaded."""
    model = LibreYOLO9(str(pretrained), size=size, device=device)
    if model.nb_classes != nb_classes:
        model._rebuild_for_new_classes(nb_classes)

    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    incompat = model.model.load_state_dict(state, strict=False)
    if incompat.unexpected_keys:
        raise RuntimeError(
            f"trained ckpt {weights} has {len(incompat.unexpected_keys)} unexpected "
            f"keys (first: {incompat.unexpected_keys[:3]})"
        )
    if incompat.missing_keys:
        raise RuntimeError(
            f"trained ckpt {weights} is missing {len(incompat.missing_keys)} keys "
            f"(first: {incompat.missing_keys[:3]})"
        )
    return model
