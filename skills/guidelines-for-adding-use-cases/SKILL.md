---
name: guidelines-for-adding-use-cases
description: >-
  Add a new use case to LibreYOLO/use-cases. Read this before scaffolding a
  new top-level folder. Explains what this repo is, what current folders look
  like, and what plausible future folders look like.
---

# What this repo is

`use-cases` is the **examples library** for LibreYOLO and adjacent tooling. `libreyolo` itself ships with no `examples/` folder — this repo fills that role.

Each top-level folder is **one use case**. A use case is a concrete thing somebody wants to do with the library: detect objects in a webcam, blur faces in photos, fine-tune on aerial imagery, train a license-plate detector, learn how to use the eval harness. The folder's internal layout is whatever shape that use case actually needs — not a fixed template.

# What's in the repo today

- **`chromium/`** — real-time webcam object detection running entirely in a Chromium browser. A single self-contained `index.html` at the folder root. No Python, no `src/`, no notebooks.
- **`blur-people/`** — same shape as `chromium/`: browser-only privacy filter, one `index.html` at root.
- **`blur-faces/`** — a bigger use case with three ways to use the same model:
  - `demo/index.html` — the browser version.
  - `src/` — Python scripts: `use_pretrained.py`, `train.py`, `download_widerface.py`, `blur.py`, etc.
  - `notebooks/pipeline.ipynb` — Colab walkthrough of paths 2 and 3.

`blur-faces` is **one** valid pattern, not THE pattern. Don't force unrelated use cases into its three-path layout.

# What a future use case might look like

A plausible `detect-license-plate/`:

```
detect-license-plate/
├── README.md             # what this is, two ways to use it, dataset notes
├── favicon.svg
├── inference/
│   └── inference.ipynb   # Colab: load pretrained weights from HF, run on your image/video
└── training/
    └── train.ipynb       # Colab: download dataset, fine-tune, export weights
```

**Two paths inside one use case**: an inference Colab (use the pretrained model) and a training Colab (build your own). The README tells the user which to click. No Python module, no browser demo — the Colab notebooks ARE the deliverable.

Other plausible folders:

- **`eval-harness-tour/`** — a single notebook walking through how to use the LibreYOLO eval harness on your own dataset. Just `README.md` + `notebook.ipynb`. No subfolders.
- **`onnx-export/`** — a notebook + a small script showing how to export each model family to ONNX. No demo, no training.
- **`<dataset>-finetune/`** (e.g. `visdrone-finetune`, `dota-finetune`) — a training tutorial. Either Python-module-led like `blur-faces/src/`, or notebook-led like the license-plate example. Both are fine.
- **`webcam-counting/`** — a browser demo that counts people/vehicles entering a region. Same shape as `chromium/`: one `index.html`.

The point: pick the smallest layout that does the job honestly. If a use case needs only one notebook, ship one notebook.

# What every use case folder must have

This is the actual contract. Everything else is shape-dependent.

- **`README.md`** — what this is, how to run it, what's accurate **today** (not "coming soon"). Include a Colab badge if there's a Colab. Include an HF model badge only if the HF repo actually exists.
- **`favicon.svg`** — copy any existing one and recolor.
- **Catalog registration** — add an entry in BOTH:
  - root `README.md` catalog table, AND
  - root `index.html` `<section class="grid">` cards.
  A folder missing from either is invisible to users.
- **Self-contained** — running it must not depend on sibling folders' generated artifacts.
- **MIT** — code is MIT; dataset/model licenses inherit from upstream.

# When to ask the user

- The shape isn't obvious from the use case description.
- You'd reference a published artifact (HF model, PyPI release) that doesn't exist yet — don't write README copy that depends on a future thing.
- The use case could plausibly be two folders instead of one (e.g. "fine-tune on dataset X" + "deploy that model in the browser" → two folders).
- The root catalog table or `index.html` cards don't have a column/field that fits cleanly — propose the catalog change in the same PR rather than fudging the entry.

# What to avoid

- **Forcing the `blur-faces` three-path layout** onto a use case that doesn't need it. If only one path actually works, ship one path.
- **Stubbing sections** like "Path 1: not yet available". If something doesn't work, don't list it. Add it later, in the PR that makes it work.
- **Updating the new folder's README but not the root catalog.** The folder ends up orphaned.
- **Inventing a `src/` directory** for a use case that's really just one notebook. The notebook is fine on its own.
- **Copying API calls from a fork without verifying** them against the `libreyolo` version pinned in `requirements.txt`. The API moves; running a smoke test is cheap.
