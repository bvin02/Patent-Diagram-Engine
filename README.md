# Sketch to SVG Pipeline

Converts hand-drawn pencil sketches into clean, editable SVG line art.

### Dependencies

```bash
pip install opencv-python numpy
```

# Pipeline

## Stage 1: Pre-Processing
Input: photo
Output: clean binary mask (black ink on white for human readability, plus a foreground mask representation internally)
```bash
python preprocess.py examples/clean.png --debug
```

## Stage 2: Distance Transform
Input: binary mask
Output: DT field saved as `.npy` + a visualization PNG + `stroke_width.json`
```bash
python distance_transform.py runs/clean/10_preprocess/out/output_mask.png --debug
```

## Stage 3: Ridge Extraction
Input: binary mask + DT
Output: ridge mask PNG (centerline candidates)
```bash
python ridge_extraction.py runs/clean/10_preprocess/out/output_mask.png runs/clean/20_distance_transform/out/dt.npy --debug
```

## Stage 4: Stroke Graph Construction
Input: ridge mask
Output: `graph_raw.json` + overlay PNG (nodes, edges)
```bash
python graph_build.py runs/clean/30_ridge/out/ridge.png --mask runs/clean/10_preprocess/out/output_mask.png --debug
```

## Stage 5: Graph Stabilization
Input: `graph_raw.json` (+ optional mask for context)
Output: `graph_clean.json` + overlay PNG + cleanup metrics
```bash
python graph_cleanup.py runs/clean/40_graph_raw/out/graph_raw.json --mask runs/clean/10_preprocess/out/output_mask.png --debug
```

## Stage 6: Primitive Fitting
Input: `graph_clean.json`
Output: `primitives.json` + overlay PNG (lines, arcs, beziers)
```bash
python fit_primitives.py runs/clean/50_graph_clean/out/graph_clean.json --mask runs/clean/10_preprocess/out/output_mask.png --debug
```

## Stage 7: SVG Output
Input: `primitives.json`
Output: final editable `output.svg` + raster preview PNG
```bash
python emit_svg.py runs/clean/60_fit/out/primitives.json --mask runs/clean/10_preprocess/out/output_mask.png --debug
```