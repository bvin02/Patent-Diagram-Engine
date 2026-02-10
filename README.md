# Sketch to SVG Pipeline

Converts hand-drawn pencil sketches of mechanical line diagrams into clean, editable SVG line art. The pipeline transforms a photograph of a pencil drawing through 8 sequential stages — from raw pixel input to a layered, semantically-grouped SVG file optimized for editing in Illustrator, Figma, and Inkscape.

## Quick Start

```bash
pip install opencv-python numpy scipy scikit-image networkx
```

### Run the full pipeline

```bash
python run_pipeline.py examples/clean.png
```

### Run without debug output

```bash
python run_pipeline.py examples/clean.png --no-debug
```

### Resume from a specific stage

```bash
python run_pipeline.py examples/clean.png --from-stage 4
```

### Run only specific stages

```bash
python run_pipeline.py examples/clean.png --only-stages 1 2 3
```

### Skip specific stages

```bash
python run_pipeline.py examples/clean.png --skip-stages 7
```

### Run individual stages manually

```bash
python stage0_init_run.py examples/clean.png
python preprocess.py examples/clean.png --debug
python distance_transform.py runs/clean/10_preprocess/out/output_mask.png --debug
python ridge_extraction.py runs/clean/10_preprocess/out/output_mask.png runs/clean/20_distance_transform/out/dt.npy --debug
python graph_build.py runs/clean/30_ridge/out/ridge.png --mask runs/clean/10_preprocess/out/output_mask.png --debug
python graph_cleanup.py runs/clean/40_graph_raw/out/graph_raw.json --mask runs/clean/10_preprocess/out/output_mask.png --debug
python fit_primitives.py runs/clean/50_graph_clean/out/graph_clean.json --mask runs/clean/10_preprocess/out/output_mask.png --debug
python emit_svg.py runs/clean/60_fit/out/primitives.json --mask runs/clean/10_preprocess/out/output_mask.png --debug
```

---

## Pipeline Overview

```
Photo of sketch
      │
      ▼
 ┌──────────┐
 │ Stage 0   │  Initialize run directory, copy input
 │ Init Run  │
 └────┬─────┘
      │  01_input.png
      ▼
 ┌──────────┐
 │ Stage 1   │  Photo → clean binary mask
 │ Preprocess│
 └────┬─────┘
      │  output_mask.png
      ▼
 ┌──────────┐
 │ Stage 2   │  Mask → distance field + stroke width
 │ Dist. Xfm │
 └────┬─────┘
      │  dt.npy, stroke_width.json
      ▼
 ┌──────────┐
 │ Stage 3   │  DT → 1-pixel centerline skeleton
 │ Ridge     │
 └────┬─────┘
      │  ridge.png
      ▼
 ┌──────────┐
 │ Stage 4   │  Skeleton → topological graph
 │ Graph Bld │
 └────┬─────┘
      │  graph_raw.json
      ▼
 ┌──────────┐
 │ Stage 5   │  Merge, prune, simplify graph
 │ Cleanup   │
 └────┬─────┘
      │  graph_clean.json
      ▼
 ┌──────────┐
 │ Stage 6   │  Edges → lines, arcs, beziers, polylines
 │ Fit Prims │
 └────┬─────┘
      │  primitives.json
      ▼
 ┌──────────┐
 │ Stage 7   │  Primitives → editable SVG + preview
 │ Emit SVG  │
 └──────────┘
      │  output.svg, preview.png
```

---

## Output Directory Structure

Each run creates a self-contained directory under `runs/`:

```
runs/<run_name>/
├── 00_input/           # Original input copy
│   └── 01_input.png
├── 10_preprocess/      # Stage 1
│   ├── out/
│   │   └── output_mask.png
│   └── debug/          # Intermediate images (if --debug)
├── 20_distance_transform/
│   ├── out/
│   │   ├── dt.npy
│   │   └── stroke_width.json
│   └── debug/
├── 30_ridge/
│   ├── out/
│   │   ├── ridge.png
│   │   └── coverage.json
│   └── debug/
├── 40_graph_raw/
│   ├── out/
│   │   └── graph_raw.json
│   └── debug/
├── 50_graph_clean/
│   ├── out/
│   │   └── graph_clean.json
│   └── debug/
├── 60_fit/
│   ├── out/
│   │   └── primitives.json
│   └── debug/
└── 70_svg/
    ├── out/
    │   ├── output.svg
    │   ├── preview.png
    │   └── overlay_preview.png
    └── debug/
```

---

## Stage Details

### Stage 0: Init Run (`stage0_init_run.py`)

**Purpose:** Create an isolated, timestamped run directory and copy the input image for traceability.

- Slugifies the input filename to derive a safe folder name (lowercase, alphanumeric + underscores)
- Creates the directory under `runs/<slug>/`
- If the directory already exists, appends `_2`, `_3`, etc. to avoid overwrites
- Copies the input image to `00_input/01_input.<ext>`
- All subsequent stages write their outputs into subdirectories of this run directory

**Input:** Path to sketch image (any format OpenCV can read)
**Output:** `runs/<slug>/00_input/01_input.<ext>`

---

### Stage 1: Preprocessing (`preprocess.py`)

**Purpose:** Convert a photograph of a pencil sketch into a clean binary stroke mask (strokes = white, background = black).

- **Pre-denoise (bilateral filter):** An edge-preserving bilateral filter is applied to the grayscale image *before* illumination flattening to reduce paper texture and noise while preserving sharp stroke edges
- **Illumination flattening:** Estimates the background illumination using a heavy Gaussian blur (k=51), then divides the original by this estimate and rescales — this normalizes out shadows, uneven lighting, and gradients from photographing paper
- **Post-denoise (bilateral filter):** A second bilateral filter pass after flattening further reduces residual noise without blurring stroke edges
- **Adaptive thresholding:** Gaussian-weighted adaptive threshold (`ADAPTIVE_THRESH_GAUSSIAN_C`) with `THRESH_BINARY_INV` converts the denoised grayscale to binary, making dark pencil strokes white
- **Morphological cleanup:** Morphological close (elliptical kernel) fills small holes within strokes, optional morphological open removes isolated noise
- **Small component removal:** Connected components below a minimum area threshold (default 30 px) are discarded as noise
- **Overlay visualization:** Debug overlay shows detected strokes (green) on the original image with contour edges (red)

**Algorithms & techniques used:**
- Bilateral filtering (edge-preserving denoising)
- Illumination normalization via background division (Gaussian blur estimate)
- Adaptive Gaussian thresholding
- Morphological operations (close, open) with elliptical structuring elements
- Connected component analysis with area filtering

**Input:** Photograph of pencil sketch
**Output:** `10_preprocess/out/output_mask.png` (binary mask, uint8, strokes=255)

---

### Stage 2: Distance Transform (`distance_transform.py`)

**Purpose:** Compute the Euclidean distance transform of the binary mask and estimate the global stroke width.

- **Distance transform:** `cv2.distanceTransform` with L2 norm computes the distance from each foreground pixel to the nearest background pixel — values peak at stroke centerlines and fall to zero at edges
- **Stroke width estimation:** Samples interior pixels (DT ≥ 1.5) while excluding the top 1% by percentile to avoid junction blobs; takes the median of up to 5000 random samples as the robust stroke radius; stroke width = 2× median radius
- **Validation:** Checks that the input is a proper binary mask (only 0 and 255 values), rejects photographs or grayscale images with clear error messages
- **Visualization:** Percentile-clipped heatmap of the DT field, sample point overlay showing which pixels contributed to the width estimate

**Algorithms & techniques used:**
- Euclidean distance transform (L2, mask size 5)
- Robust percentile-based statistical estimation (median of filtered samples)
- Input validation with automatic thresholding for near-binary inputs

**Input:** Binary mask from Stage 1
**Output:**
- `20_distance_transform/out/dt.npy` — float32 distance transform array
- `20_distance_transform/out/stroke_width.json` — estimated stroke width with percentile statistics

---

### Stage 3: Ridge Extraction (`ridge_extraction.py`)

**Purpose:** Extract 1-pixel-wide centerline ridges from the distance transform field to locate the skeleton of each stroke.

- **Local maxima detection:** Uses `scipy.ndimage.maximum_filter` (3×3 window) to find pixels whose DT value equals the local neighborhood maximum (within epsilon tolerance) — this correctly handles plateaus where multiple adjacent pixels share the same DT value
- **Plateau thinning (optional):** Deterministic thinning that keeps plateau pixels only if they have at least one 4-neighbor with strictly lower DT, preserving connectivity
- **Dilation + thinning:** Local maxima on thin strokes are sparse discrete points; dilation with an elliptical kernel (default radius 2) bridges gaps between them, then `skimage.morphology.thin` reduces the result to a 1-pixel skeleton constrained to stay inside the foreground
- **Spur pruning:** Iteratively removes endpoint pixels (degree 1 in 8-connectivity) to trim short thinning artifacts — repeated up to `spur_prune_length` times
- **Small component removal:** Ridge components smaller than `min_ridge_component_area` are discarded
- **Coverage metrics:** Measures ridge quality by computing what fraction of foreground pixels are within 1 px and 2 px of the nearest ridge pixel using `scipy.ndimage.distance_transform_edt`

**Algorithms & techniques used:**
- Local maximum detection via maximum filter (handles plateaus with epsilon tolerance)
- Morphological dilation with elliptical structuring element (gap bridging)
- Zhang-Suen morphological thinning (`skimage.morphology.thin`)
- Iterative spur pruning (endpoint removal)
- Connected component analysis with area filtering
- Euclidean distance transform for coverage assessment

**Input:** Binary mask + DT array (`.npy`)
**Output:**
- `30_ridge/out/ridge.png` — uint8 ridge mask (centerline pixels = 255)
- `30_ridge/out/coverage.json` — coverage metrics (1 px and 2 px coverage fractions)

---

### Stage 4: Graph Build (`graph_build.py`)

**Purpose:** Convert the 1-pixel ridge skeleton into a topological graph with nodes (endpoints and junctions) and edges (polyline paths between nodes).

- **Degree map computation:** Counts the number of 8-connected ridge neighbors for each pixel using shifted arrays — degree 1 = endpoint, degree 2 = path pixel, degree ≥ 3 = junction
- **Node identification:** Connected components of endpoint/junction pixels are labeled; each component becomes a node with computed centroid, bounding box, pixel count, and type (endpoint or junction)
- **Edge tracing:** Starting from each node's boundary pixels, traces along degree-2 path pixels until reaching another node; records the full polyline (sequence of [x, y] coordinates) along the way
- **Dead-end handling:** If a trace reaches a dead end (no unvisited neighbors), a new endpoint node is created on the fly
- **Branch handling:** If a trace encounters an unmodeled branching point (≥2 valid neighbors), a new junction node is created on the fly
- **Edge length filtering:** Edges shorter than `min_edge_length` (default 3 px) are discarded as noise
- **Self-loop and multi-edge detection:** Counts self-loops and multi-edges for diagnostic purposes

**Algorithms & techniques used:**
- Pixel degree computation via shifted array accumulation (8-connectivity)
- Connected component labeling (`scipy.ndimage.label`)
- Deterministic greedy edge tracing with visited pixel tracking
- On-the-fly node creation for unmodeled topological features
- Golden-ratio hue rotation for deterministic color palettes (visualization)

**Input:** Ridge mask from Stage 3 (+ optional binary mask for overlays)
**Output:** `40_graph_raw/out/graph_raw.json` — nodes (id, type, centroid, bbox) and edges (u, v, polyline, length)

---

### Stage 5: Graph Cleanup (`graph_cleanup.py`)

**Purpose:** Stabilize and simplify the raw graph through a 5-step cleanup pipeline, reducing noise while preserving the drawing's topology.

- **Step 1 — Node merging:** Uses union-find (disjoint set) to cluster nodes within a merge radius (default 5 px); junction-junction pairs always merge; endpoint-endpoint pairs additionally check that edge directions are within an angle tolerance; endpoint-junction pairs merge if the endpoint falls within the junction's bounding box
- **Step 2 — Spur pruning:** Iteratively removes short dangling edges (default < 8 px) attached to degree-1 nodes — these are thinning artifacts, not real strokes; runs up to 3 iterations since removing one spur can expose another
- **Step 3 — Chain merging:** Dissolves degree-2 pass-through nodes by stitching their two incident edges into a single longer edge; uses direction vectors (sampled over first/last N pixels) and a collinearity angle check (default ≤ 25°) to avoid merging across actual corners; runs up to 50 iterations for thorough merging
- **Step 4 — Gap bridging (optional):** Finds pairs of degree-1 endpoint nodes within a maximum distance, checks that their edge directions align with the gap direction, and optionally verifies that the bridge path passes through foreground pixels on the mask; adds straight-line bridge edges for qualifying pairs
- **Step 5 — Polyline simplification:** Applies Ramer-Douglas-Peucker (RDP) to every edge polyline with a tolerance of 0.75 px, reducing point count while preserving shape

**Algorithms & techniques used:**
- Union-find (disjoint set) with path compression and union by rank
- Direction-aware endpoint merging (dot product angle check)
- Iterative spur pruning (degree-1 edge removal)
- Collinear chain merging through degree-2 nodes (direction sampling + angle threshold)
- Gap bridging with direction alignment and mask-based foreground verification
- Ramer-Douglas-Peucker polyline simplification
- NetworkX multigraph for adjacency operations

**Input:** `graph_raw.json` (+ optional binary mask for gap bridging, + optional ridge mask)
**Output:** `50_graph_clean/out/graph_clean.json` — cleaned graph with same schema

---

### Stage 6: Primitive Fitting (`fit_primitives.py`)

**Purpose:** Fit geometric primitives (lines, arcs, cubic Béziers, or smoothed polylines) to each graph edge, choosing the simplest representation that fits within an error tolerance.

- **Edge bucketing:** Edges ≥ 30 px are "structural" (main geometry), shorter edges are "detail" (hatching, small features); structural edges use relaxed line fitting, detail edges use strict thresholds
- **Parallel hatching detection:** Uses a KD-tree on edge midpoints to find clusters of short edges with similar directions (within 5° angle tolerance); edges with ≥ 2 parallel neighbors are flagged as hatching and forced to line fitting
- **Adaptive RDP simplification:** Chooses a larger RDP epsilon for straight polylines (ε = 2.0) and a smaller one for curves (ε = 1.25), based on a straightness score (chord / path length ratio)
- **Line fitting (PCA / total least squares):** Fits a line via SVD of centered points; computes the straightness metric (chord-length / path-length); accepts if straightness exceeds a threshold and RMS + max error are within tolerance; line wins by a "simplicity factor" — if line RMS is within 1.15× the best alternative, the simpler line is preferred
- **Arc fitting (algebraic + nonlinear refinement):** Algebraic circle fit via Kasa method (linear least squares on x² + y² = ax + by + c), refined with Huber-loss nonlinear least squares; computes arc angles with arctan2, determines CW/CCW direction from angle progression, checks sweep consistency (monotonicity of angles); accepts if radius is within bounds, error within tolerance, and sweep consistency ≥ 0.8
- **Cubic Bézier fitting:** Fixes endpoints (P0, P3) and solves for control points (P1, P2) via chord-length parameterization and regularized linear least squares (pulls control points toward the chord's 1/3 and 2/3 points); error measured via KD-tree closest-point distance between input points and sampled curve; sanity-checked by control point distance bounds
- **Polyline fallback with curve-aware smoothing:**
  - **Corner detection:** Measures turning angle at each interior vertex (arccos of consecutive direction vectors); vertices where turning angle exceeds 35° are marked as corners
  - **Segment smoothing:** Between corners, applies weighted moving-average smoothing (3-point kernel, blending weight 0.65) with drift constraint (max 2.5 px from original)
  - **Pure curve smoothing:** Polylines with no interior corners (only endpoints are corners, ≥ 4 points) are treated as smooth curves and receive aggressive smoothing:
    - Chaikin's corner-cutting subdivision (2 passes, approximately 4× point increase)
    - 5-point weighted moving-average (10 passes at weight 0.95)
    - Drift constraint via closest-point-on-segment projection (max 4.0 px)
    - Resampling to ~1.5× original point count
- **Selection logic:** For structural edges: line wins if close to best RMS (simplicity factor), otherwise best-fitting primitive (arc > cubic > polyline); for detail edges: line or polyline only; for parallel hatching: always line

**Algorithms & techniques used:**
- PCA / total least squares line fitting (SVD)
- Kasa algebraic circle fitting (linear least squares)
- Nonlinear least squares with Huber loss (`scipy.optimize.least_squares`)
- Cubic Bézier fitting with chord-length parameterization and Tikhonov regularization
- KD-tree nearest-neighbor queries (`scipy.spatial.KDTree`) for Bézier error and parallel detection
- Ramer-Douglas-Peucker polyline simplification (adaptive epsilon)
- Chaikin's corner-cutting subdivision algorithm
- Weighted moving-average smoothing with fixed-endpoint constraint
- Drift constraint via point-to-segment projection
- Arc direction determination from angle unwrapping

**Input:** `graph_clean.json` (+ optional binary mask for overlays)
**Output:** `60_fit/out/primitives.json` — per-edge chosen primitive, all candidate fits with error metrics, quality scores, and bucket labels

---

### Stage 7: SVG Emission (`emit_svg.py`)

**Purpose:** Serialize fitted primitives into an editable SVG file, organized into layers for structural and detail elements, with a raster preview for quick inspection.

- **SVG structure:** Root `<svg>` with `viewBox` matching the input image dimensions; optional white background rectangle; two layer groups (`<g>`) for structural and detail elements, each sub-grouped by primitive type (lines, arcs, cubics, polylines)
- **Element serialization:**
  - Lines → `<line x1 y1 x2 y2>`
  - Polylines → `<polyline points="x1,y1 x2,y2 ...">`; subsampled if point count exceeds safety limit (2000)
  - Cubic Béziers → `<path d="M ... C ...">`
  - Arcs → `<path d="M ... A ...">` using SVG arc commands with proper large-arc and sweep flags; large arcs (> 175°) are automatically split into multiple segments
- **Arc flag computation:** Correctly converts between the internal arc representation (center, radius, theta0, theta1, CW/CCW in math coords) and SVG's endpoint-based arc notation (large-arc-flag, sweep-flag) accounting for the y-axis flip between math and screen coordinates
- **Metadata:** Each SVG element carries `data-` attributes for edge ID, bucket, endpoint node IDs, and primitive type — enabling programmatic post-processing
- **Styling:** Configurable stroke color, width (default 1.0), linecap (round), linejoin (round)
- **Preview rendering:** Rasterizes all primitives using OpenCV (sampling Béziers and arcs to polylines) for a quick PNG preview; renders an overlay preview blending strokes (in red) over the original input or mask for alignment checking
- **Debug outputs:** Separate renders for structural-only and detail-only layers, arc debug visualization with centers and radii drawn, edge ID sanity view labeling the longest structural edges

**Algorithms & techniques used:**
- SVG arc endpoint parameterization (center → endpoint conversion)
- Arc splitting for large sweeps (> 175°) to ensure correct SVG rendering
- Cubic Bézier sampling (De Casteljau evaluation via vectorized NumPy)
- XML element tree construction with proper indentation
- Alpha-blended overlay compositing for debug visualization

**Input:** `primitives.json` from Stage 6 (+ optional mask/input for overlay)
**Output:**
- `70_svg/out/output.svg` — final editable SVG
- `70_svg/out/preview.png` — raster preview
- `70_svg/out/overlay_preview.png` — strokes overlaid on input/mask

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Image I/O, bilateral filtering, thresholding, morphology, distance transform, visualization |
| `numpy` | Array operations, linear algebra, distance computations |
| `scipy` | Nonlinear least squares (arc fitting), KD-tree (parallel detection, Bézier error), maximum filter (ridge extraction) |
| `scikit-image` | Morphological thinning (Zhang-Suen) for ridge extraction |
| `networkx` | Multigraph representation for graph cleanup operations |

```bash
pip install opencv-python numpy scipy scikit-image networkx
```

## Configuration

Every stage accepts a `--config path/to/config.json` flag. The JSON file merges with that stage's `DEFAULT_CONFIG` dict, overriding only the keys you specify. See the `DEFAULT_CONFIG` at the top of each stage file for all available parameters with their defaults and descriptions.