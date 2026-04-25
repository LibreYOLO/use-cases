"""Download VisDrone2019-DET and convert it to YOLO format.

Two sources supported:

  (1) HuggingFace Voxel51 mirror (default) — FiftyOne format, ~1.5 GB. Slow
      first time, cached thereafter.
      https://huggingface.co/datasets/Voxel51/VisDrone2019-DET

  (2) A local directory with the raw VisDrone layout — skip the download if
      you already have the official release unzipped somewhere:
          <root>/VisDrone2019-DET-train/{images,annotations}/
          <root>/VisDrone2019-DET-val/{images,annotations}/

Usage:
    python -m src.download_visdrone --out data/visdrone
    python -m src.download_visdrone --out data/visdrone --raw /path/to/VisDrone
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import (
    VISDRONE_CLASSES,
    convert_fiftyone_dataset,
    convert_raw_split,
    write_data_yaml,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/visdrone"),
                   help="Where to write the YOLO-format dataset.")
    p.add_argument("--raw", type=Path, default=None,
                   help="Optional: path to already-unzipped VisDrone2019-DET root.")
    p.add_argument("--hf-repo", default="Voxel51/VisDrone2019-DET",
                   help="HF dataset repo to snapshot if --raw not given.")
    p.add_argument("--split-ratio", type=float, default=0.9,
                   help="Train fraction when using the Voxel51 single-split source.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.raw is not None:
        print(f"using raw VisDrone at {args.raw}")
        if not args.raw.exists():
            print(f"  {args.raw} does not exist", file=sys.stderr)
            return 1
        total = {}
        for split_name, src_suffix in [("train", "DET-train"), ("val", "DET-val")]:
            src = args.raw / f"VisDrone2019-{src_suffix}"
            if not src.exists():
                print(f"  missing split dir: {src}", file=sys.stderr)
                return 1
            images_dst = args.out / "images" / split_name
            images_dst.parent.mkdir(parents=True, exist_ok=True)
            if images_dst.exists():
                images_dst.unlink() if images_dst.is_symlink() else None
            if not images_dst.exists():
                images_dst.symlink_to((src / "images").resolve(), target_is_directory=True)
            labels_dst = args.out / "labels" / split_name
            summary = convert_raw_split(src / "images", src / "annotations", labels_dst)
            total[split_name] = summary
            print(f"  {split_name}: {json.dumps(summary)}")
    else:
        print(f"downloading {args.hf_repo} via HuggingFace...")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("huggingface_hub required: pip install -r requirements.txt", file=sys.stderr)
            return 1
        cache = args.out / ".hf-cache"
        cache.mkdir(exist_ok=True)
        root = Path(snapshot_download(
            repo_id=args.hf_repo, repo_type="dataset", local_dir=str(cache),
        ))
        summary = convert_fiftyone_dataset(root, args.out, split_ratio=args.split_ratio)
        print(f"  {json.dumps(summary)}")

    data_yaml = write_data_yaml(args.out)
    print(f"\ndata.yaml → {data_yaml}")
    print(f"  classes ({len(VISDRONE_CLASSES)}): {', '.join(VISDRONE_CLASSES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
