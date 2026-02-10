# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Converts hand-drawn pencil sketches of patent diagrams into clean, editable SVG line art. The system is a multi-stage Python CLI pipeline where each stage reads from the previous stage's output directory.

## Dependencies

```bash
pip install opencv-python numpy scipy scikit-image networkx
```

No pyproject.toml or package manager — flat script-based pipeline.

## Running the Pipeline

Each stage is a standalone CLI script. Stages auto-detect run directories when given paths inside `runs/`.

```bash
# Full pipeline for an input image:
python stage0_init_run.py examples/clean.png
python preprocess.py examples/clean.png --debug
python distance_transform.py runs/clean/10_preprocess/out/output_mask.png --debug
python ridge_extraction.py runs/clean/10_preprocess/out/output_mask.png runs/clean/20_distance_transform/out/dt.npy --debug
python graph_build.py runs/clean/30_ridge/out/ridge.png --mask runs/clean/10_preprocess/out/output_mask.png --debug
python graph_cleanup.py runs/clean/40_graph_raw/out/graph_raw.json --mask runs/clean/10_preprocess/out/output_mask.png --debug
python fit_primitives.py runs/clean/50_graph_clean/out/graph_clean.json --mask runs/clean/10_preprocess/out/output_mask.png --debug
python regularize_geometry.py runs/clean/60_fit/out/primitives.json --mask runs/clean/10_preprocess/out/output_mask.png --debug
python emit_svg.py runs/clean/65_regularize/out/primitives_regularized.json --mask runs/clean/10_preprocess/out/output_mask.png --debug
```

Final output: `runs/<name>/70_svg/out/output.svg`

To run a single stage in isolation, pass the appropriate input file from a previous run. All stages accept `--debug` (on by default) and `--config path/to/config.json` to override defaults.

## Pipeline Architecture

```
Input Photo → [1: Preprocess] → [2: Distance Transform] → [3: Ridge Extraction]
  → [4: Graph Build] → [5: Graph Cleanup] → [6: Primitive Fitting]
  → [6.5: Geometry Regularization] → [7: SVG Emission] → output.svg
```

| Stage | Script | Input → Output |
|-------|--------|----------------|
| 0 | `stage0_init_run.py` | Raw image → `00_input/` |
| 1 | `preprocess.py` | Photo → binary mask (`output_mask.png`) |
| 2 | `distance_transform.py` | Mask → distance field (`dt.npy`, `stroke_width.json`) |
| 3 | `ridge_extraction.py` | Mask + DT → centerlines (`ridge.png`) |
| 4 | `graph_build.py` | Ridge → topological graph (`graph_raw.json`) |
| 5 | `graph_cleanup.py` | Raw graph → cleaned graph (`graph_clean.json`) |
| 6 | `fit_primitives.py` | Clean graph → primitives (`primitives.json`) |
| 6.5 | `regularize_geometry.py` | Primitives → regularized primitives (`primitives_regularized.json`) |
| 7 | `emit_svg.py` | Regularized primitives → `output.svg`, `preview.png` |

## Key Data Formats

**Graph JSON** (stages 4-5): `{"nodes": [{"id", "position", "degree"}], "edges": [{"id", "u", "v", "polyline"}]}`

**Primitives JSON** (stages 6-7): `{"primitives": [{"id", "type": "line"|"arc"|"bezier", ...}], "buckets": {"structural": [...ids], "detail": [...ids]}}`

Edges are bucketed into **structural** (long strokes forming the main drawing) and **detail** (short strokes like hatching/cross-hatching). Stage 6.5 only regularizes structural primitives.

## Artifact System (`utils/artifacts.py`)

`StageArtifacts` manages output directories per run and stage:
- `runs/<slug>/<stage_id>_<name>/out/` — final outputs
- `runs/<slug>/<stage_id>_<name>/debug/` — numbered debug images (01_, 02_, ...)
- Key methods: `save_debug_image()`, `save_output_image()`, `save_json()`, `save_npy()`, `write_metrics()`, `path_out()`

Run directories are auto-detected: if the input path is already inside a `runs/` directory, the same run is reused.

## Stage CLI Pattern

All stages follow the same structure:

```python
DEFAULT_CONFIG = { ... }  # Stage-specific defaults

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--mask", default=None)
    parser.add_argument("--runs_root", default="runs")
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_dir = infer_run_dir(args.input_path, args.runs_root)
    artifacts = StageArtifacts(run_dir, STAGE_ID, "stage_name", debug=args.debug)
    # ... processing ...
```

When adding a new stage, follow this pattern with a unique `STAGE_ID` (multiples of 10, or N5 for intermediates).

## Test Data

Example inputs in `examples/`: `clean.png`, `detailed.png`, `lowres.jpg`. Outputs go to `runs/clean/`, `runs/detailed/`, `runs/lowres/`.
