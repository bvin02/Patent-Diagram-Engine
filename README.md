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

## Stage 3: Ridge Extraction
Input: binary mask + DT
Output: ridge mask PNG (centerline candidates)

## Stage 4: Stroke Graph Construction
Input: ridge mask
Output: `graph_raw.json` + overlay PNG (nodes, edges)

## Stage 5: Graph Stabilization
Input: `graph_raw.json` (+ optional mask for context)
Output: `graph_clean.json` + overlay PNG + cleanup metrics

## Stage 6: Primitive Fitting
Input: `graph_clean.json`
Output: `primitives.json` + overlay PNG (lines, arcs, beziers)

## Stage 7: SVG Output
Input: `primitives.json`
Output: final editable `output.svg` + raster preview PNG