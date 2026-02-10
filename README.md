# Patent Diagram Generator

Converts hand-drawn pencil sketches of mechanical line diagrams into clean, patent-ready SVG drawings with numbered component labels and leader lines. The system consists of three modules:

1. **Vectorization Pipeline** -- 8 stages that transform a photograph of a pencil sketch into editable SVG line art
2. **Labelling Pipeline** -- 3 stages that identify components, route leader lines, and place numbered labels
3. **Web Application** -- A browser-based wizard that ties everything together with an embedded SVG editor

---

## User Flow

```
 +---------------------------------------------------------------------+
 |                    Patent Diagram Generator                          |
 |                                                                      |
 |  Step 1: Upload        Upload a photo of a pencil sketch             |
 |              |                                                       |
 |              v          Vectorization pipeline runs (stages 0-7)     |
 |  Step 2: Edit SVG      Edit the vector output in Method Draw         |
 |              |                                                       |
 |              v          Component identification runs (stage 8)      |
 |  Step 3: Anchors       Adjust label anchor points on the drawing     |
 |              |                                                       |
 |              v          Leader routing + label placement (9-10)      |
 |  Step 4: Review        Preview labeled SVG, download, check          |
 |                         compliance with patent office requirements    |
 +---------------------------------------------------------------------+
```

### Step-by-step

1. **Upload** -- Drag-and-drop or browse for a PNG/JPG photograph of a pencil sketch. Click "Convert to Vector" to run the vectorization pipeline.

2. **Edit SVG** -- The resulting SVG opens in an embedded [Method Draw](https://github.com/nicholidesign/Method-Draw) vector editor. Add, delete, or adjust any strokes. Click "Done Editing" when finished.

3. **Select Anchors** -- The system identifies enclosed regions in the drawing and suggests an anchor point inside each one. Drag anchors to reposition, double-click to add new ones, right-click to delete. Click "Generate Labels" to run the labelling pipeline.

4. **Review & Export** -- Preview the final labeled SVG with numbered leader lines. Download the SVG, the labels-only overlay, or both. A compliance checklist helps verify patent office requirements (black & white, legible lines, labels outside the drawing, etc.).

---

## Quick Start

### Run the web application

```bash
# Install Python dependencies
pip install opencv-python numpy scipy scikit-image networkx fastapi uvicorn python-multipart

# Install frontend dependencies
cd frontend && npm install && cd ..

# Start both servers (frontend :5173 + backend :8000)
./dev.sh

# Open http://localhost:5173
```

### Or run the pipeline from the command line

```bash
# Full pipeline (vectorization + labelling)
python run_pipeline.py examples/clean.png

# Vectorization only (stages 0-7)
python run_pipeline.py examples/clean.png --only-stages 0 1 2 3 4 5 6 7

# Resume from a specific stage
python run_pipeline.py examples/clean.png --from-stage 4

# Without debug output
python run_pipeline.py examples/clean.png --no-debug
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
python label_identify.py runs/clean/10_preprocess/out/output_mask.png --svg runs/clean/70_svg/out/output.svg --debug
python label_leaders.py runs/clean/80_label_identify/out/components.json --mask runs/clean/10_preprocess/out/output_mask.png --debug
python label_place.py runs/clean/90_label_leaders/out/leaders.json --svg runs/clean/70_svg/out/output.svg --debug
```

---

## Vectorization Pipeline (Stages 0-7)

Transforms a photograph of a pencil drawing through 8 sequential stages -- from raw pixel input to a layered, semantically-grouped SVG file optimized for editing in Illustrator, Figma, and Inkscape.

```
Photo of sketch
      |
      v
 +----------+
 | Stage 0   |  Initialize run directory, copy input
 | Init Run  |
 +----+-----+
      |  01_input.png
      v
 +----------+
 | Stage 1   |  Photo -> clean binary mask
 | Preprocess|
 +----+-----+
      |  output_mask.png
      v
 +----------+
 | Stage 2   |  Mask -> distance field + stroke width
 | Dist. Xfm |
 +----+-----+
      |  dt.npy, stroke_width.json
      v
 +----------+
 | Stage 3   |  DT -> 1-pixel centerline skeleton
 | Ridge     |
 +----+-----+
      |  ridge.png
      v
 +----------+
 | Stage 4   |  Skeleton -> topological graph
 | Graph Bld |
 +----+-----+
      |  graph_raw.json
      v
 +----------+
 | Stage 5   |  Merge, prune, simplify graph
 | Cleanup   |
 +----+-----+
      |  graph_clean.json
      v
 +----------+
 | Stage 6   |  Edges -> lines, arcs, beziers, polylines
 | Fit Prims |
 +----+-----+
      |  primitives.json
      v
 +----------+
 | Stage 7   |  Primitives -> editable SVG + preview
 | Emit SVG  |
 +----------+
      |  output.svg, preview.png
```

### Stage 0: Init Run (`stage0_init_run.py`)

**Purpose:** Create an isolated, timestamped run directory and copy the input image for traceability.

- Slugifies the input filename to derive a safe folder name (lowercase, alphanumeric + underscores)
- Creates the directory under `runs/<slug>/`
- If the directory already exists, appends `_2`, `_3`, etc. to avoid overwrites
- Copies the input image to `00_input/01_input.<ext>`
- All subsequent stages write their outputs into subdirectories of this run directory

**Input:** Path to sketch image (any format OpenCV can read)
**Output:** `runs/<slug>/00_input/01_input.<ext>`

### Stage 1: Preprocessing (`preprocess.py`)

**Purpose:** Convert a photograph of a pencil sketch into a clean binary stroke mask (strokes = white, background = black).

- **Pre-denoise (bilateral filter):** An edge-preserving bilateral filter is applied to the grayscale image *before* illumination flattening to reduce paper texture and noise while preserving sharp stroke edges
- **Illumination flattening:** Estimates the background illumination using a heavy Gaussian blur (k=51), then divides the original by this estimate and rescales -- this normalizes out shadows, uneven lighting, and gradients from photographing paper
- **Post-denoise (bilateral filter):** A second bilateral filter pass after flattening further reduces residual noise without blurring stroke edges
- **Adaptive thresholding:** Gaussian-weighted adaptive threshold (`ADAPTIVE_THRESH_GAUSSIAN_C`) with `THRESH_BINARY_INV` converts the denoised grayscale to binary, making dark pencil strokes white
- **Morphological cleanup:** Morphological close (elliptical kernel) fills small holes within strokes, optional morphological open removes isolated noise
- **Small component removal:** Connected components below a minimum area threshold (default 30 px) are discarded as noise

**Algorithms:** Bilateral filtering, illumination normalization via background division, adaptive Gaussian thresholding, morphological operations, connected component analysis

**Input:** Photograph of pencil sketch
**Output:** `10_preprocess/out/output_mask.png` (binary mask, uint8, strokes=255)

### Stage 2: Distance Transform (`distance_transform.py`)

**Purpose:** Compute the Euclidean distance transform of the binary mask and estimate the global stroke width.

- **Distance transform:** `cv2.distanceTransform` with L2 norm computes the distance from each foreground pixel to the nearest background pixel -- values peak at stroke centerlines and fall to zero at edges
- **Stroke width estimation:** Samples interior pixels (DT >= 1.5) while excluding the top 1% by percentile to avoid junction blobs; takes the median of up to 5000 random samples as the robust stroke radius; stroke width = 2x median radius
- **Validation:** Checks that the input is a proper binary mask (only 0 and 255 values), rejects photographs or grayscale images with clear error messages

**Algorithms:** Euclidean distance transform (L2), robust percentile-based statistical estimation

**Input:** Binary mask from Stage 1
**Output:** `20_distance_transform/out/dt.npy` (float32 distance field), `stroke_width.json`

### Stage 3: Ridge Extraction (`ridge_extraction.py`)

**Purpose:** Extract 1-pixel-wide centerline ridges from the distance transform field to locate the skeleton of each stroke.

- **Local maxima detection:** Uses `scipy.ndimage.maximum_filter` (3x3 window) to find pixels whose DT value equals the local neighborhood maximum -- correctly handles plateaus where multiple adjacent pixels share the same DT value
- **Dilation + thinning:** Local maxima on thin strokes are sparse; dilation with an elliptical kernel bridges gaps, then `skimage.morphology.thin` (Zhang-Suen) reduces to a 1-pixel skeleton constrained inside the foreground
- **Spur pruning:** Iteratively removes endpoint pixels (degree 1 in 8-connectivity) to trim short thinning artifacts
- **Small component removal:** Ridge components smaller than a threshold are discarded

**Algorithms:** Local maximum detection via maximum filter, morphological dilation, Zhang-Suen thinning, iterative spur pruning, connected component analysis

**Input:** Binary mask + DT array
**Output:** `30_ridge/out/ridge.png` (uint8 ridge mask, centerline pixels=255)

### Stage 4: Graph Build (`graph_build.py`)

**Purpose:** Convert the 1-pixel ridge skeleton into a topological graph with nodes (endpoints and junctions) and edges (polyline paths between nodes).

- **Degree map computation:** Counts 8-connected ridge neighbors for each pixel using shifted arrays -- degree 1 = endpoint, degree >= 3 = junction
- **Node identification:** Connected components of endpoint/junction pixels become graph nodes with centroids and types
- **Edge tracing:** Traces along degree-2 path pixels from each node boundary until reaching another node, recording the full polyline path
- **On-the-fly topology repair:** Creates new nodes for dead-ends and unexpected branching points encountered during tracing

**Algorithms:** Pixel degree computation via shifted arrays, connected component labeling, deterministic greedy edge tracing

**Input:** Ridge mask from Stage 3
**Output:** `40_graph_raw/out/graph_raw.json` -- nodes and edges with polylines

### Stage 5: Graph Cleanup (`graph_cleanup.py`)

**Purpose:** Stabilize and simplify the raw graph through a 5-step cleanup pipeline, reducing noise while preserving the drawing's topology.

1. **Node merging** -- Union-find clusters nearby nodes (junction-junction, endpoint-endpoint with direction check, endpoint-junction within bbox)
2. **Spur pruning** -- Iteratively removes short dangling edges attached to degree-1 nodes (thinning artifacts)
3. **Chain merging** -- Dissolves degree-2 pass-through nodes by stitching incident edges, with collinearity angle check to preserve real corners
4. **Gap bridging** -- Finds nearby degree-1 endpoints with aligned directions and bridges them with straight edges
5. **Polyline simplification** -- Ramer-Douglas-Peucker at 0.75 px tolerance

**Algorithms:** Union-find with path compression, direction-aware merging, iterative spur pruning, collinear chain merging, RDP simplification, NetworkX multigraph operations

**Input:** `graph_raw.json` (+ optional binary mask)
**Output:** `50_graph_clean/out/graph_clean.json`

### Stage 6: Primitive Fitting (`fit_primitives.py`)

**Purpose:** Fit geometric primitives (lines, arcs, cubic Beziers, or smoothed polylines) to each graph edge, choosing the simplest representation that fits within an error tolerance.

- **Edge bucketing:** Edges >= 30 px are "structural", shorter ones are "detail"
- **Parallel hatching detection:** KD-tree on midpoints finds clusters of short parallel edges, forced to line fitting
- **Line fitting:** PCA / total least squares via SVD with a simplicity factor -- if line RMS is within 1.15x the best alternative, the simpler line wins
- **Arc fitting:** Kasa algebraic circle fit, refined with Huber-loss nonlinear least squares; sweep consistency and radius bound checks
- **Cubic Bezier fitting:** Chord-length parameterization with regularized least squares; KD-tree error measurement
- **Polyline fallback with curve-aware smoothing:** Corner detection via turning angle (> 35 degrees); Chaikin's corner-cutting subdivision (2 passes) for smooth curves; weighted moving-average with drift constraint

**Algorithms:** PCA line fitting (SVD), Kasa circle fitting, nonlinear least squares (Huber loss), cubic Bezier fitting with Tikhonov regularization, KD-tree queries, adaptive RDP, Chaikin subdivision, weighted smoothing

**Input:** `graph_clean.json`
**Output:** `60_fit/out/primitives.json` -- per-edge primitive type, parameters, and error metrics

### Stage 7: SVG Emission (`emit_svg.py`)

**Purpose:** Serialize fitted primitives into an editable SVG file, organized into layers for structural and detail elements.

- **SVG structure:** Root `<svg>` with `viewBox` matching input dimensions; two layer groups for structural and detail elements, each sub-grouped by primitive type
- **Element serialization:** Lines -> `<line>`, polylines -> `<polyline>`, cubics -> `<path d="M...C...">`, arcs -> `<path d="M...A...">` with proper large-arc/sweep flags; large arcs (> 175 degrees) auto-split
- **Metadata:** Each element carries `data-` attributes for edge ID, bucket, node IDs, and primitive type
- **Preview rendering:** Raster PNG preview and an overlay preview for alignment checking

**Input:** `primitives.json`
**Output:** `70_svg/out/output.svg`, `preview.png`, `overlay_preview.png`

---

## Labelling Pipeline (Stages 8-10)

Adds patent-style numbered labels with leader lines to the vector drawing. Identifies enclosed regions, routes non-crossing leader lines to a margin ring, and places circled numbers.

```
output.svg (from Stage 7)
      |
      v
 +----------+
 | Stage 8   |  Flood-fill to find enclosed regions
 | Identify  |
 +----+-----+
      |  components.json (anchor points)
      v
 +----------+
 | Stage 9   |  Route leader lines to margin ring
 | Leaders   |
 +----+-----+
      |  leaders.json
      v
 +----------+
 | Stage 10  |  Place numbered labels, composite onto SVG
 | Place     |
 +----------+
      |  labelled.svg, labels_only.svg
```

### Stage 8: Component Identification (`label_identify.py`)

**Purpose:** Identify enclosed regions (components) in the binary stroke mask using flood-fill. Each region bounded by strokes gets a label anchor point.

- **Stroke dilation:** Dilates the binary mask slightly (configurable radius, default 1 px) to close thin 1-pixel gaps at junctions that would cause leaking during flood-fill
- **Flood-fill enumeration:** Starting from background pixels inside the drawing's bounding box, performs connected-component flood-fill on the inverted mask; each contiguous background region bounded by strokes is a distinct component
- **Outer background exclusion:** The largest region touching the image border is identified as the outer background and excluded from labelling
- **Anchor point selection:** For each component, computes the distance transform of the component's mask and selects the pixel farthest from any boundary stroke -- ensuring the anchor is well inside the region, not hugging an edge
- **Area filtering:** Regions smaller than `min_component_area` (default 200 px) are discarded as noise; regions larger than `max_component_area_frac` (default 50%) of the image are treated as background
- **Novel-boundary deduplication:** Tracks which boundary edge pixels have already been claimed by previous components; new components must contribute at least `novel_boundary_frac` (default 15%) novel boundary edges, otherwise they are merged/discarded to avoid duplicate labels
- **Reading-order sort:** Components are sorted top-to-bottom, left-to-right by their anchor positions for consistent numbering

**Algorithms:** Morphological dilation, connected-component flood-fill, Euclidean distance transform for anchor placement, boundary-edge novelty tracking, area filtering

**Input:** Binary mask (`output_mask.png`) + optional SVG for overlay debugging
**Output:** `80_label_identify/out/components.json` -- list of components with `id`, `anchor_x`, `anchor_y`, `area`, `bbox`, `boundary_edges`

### Stage 9: Leader Line Routing (`label_leaders.py`)

**Purpose:** Route straight leader lines from each component's anchor point to the outside margin, where a numeric label will be placed.

- **Drawing bbox computation:** Computes the tight bounding box of all strokes in the mask
- **Label ring definition:** A rectangle outside the drawing bbox with configurable margin (default 60 px), defining where label endpoints will live -- all labels are placed on this ring to keep them consistently outside the drawing
- **Natural exit direction:** For each anchor, determines the closest edge/corner of the bbox relative to the anchor position; the leader line exits in that direction
- **Ray-ring projection:** Projects each anchor outward along its exit direction until intersecting the label ring rectangle -- this gives the candidate label endpoint
- **Spacing relaxation:** If two label endpoints are closer than `min_label_spacing` (default 35 px) on the ring perimeter, a force-directed relaxation spreads them apart while keeping them on the ring
- **Crossing reduction:** Detects pairs of leader lines that cross using line-segment intersection tests; for adjacent anchors with crossing leaders, swaps their label endpoints if that reduces the total crossing count; runs up to `max_crossing_passes` (default 20) iterations
- **Canvas extension:** If label endpoints would fall outside the image bounds, optionally extends the canvas with padding

**Algorithms:** Bounding box computation, ray-rectangle intersection, force-directed spacing relaxation, greedy crossing reduction via endpoint swapping, line-segment intersection tests

**Input:** `components.json` + optional binary mask
**Output:** `90_label_leaders/out/leaders.json` -- leader lines with start/end points, label positions, assigned sides

### Stage 10: Label Placement (`label_place.py`)

**Purpose:** Assign numeric labels (1..N) to each component, render leader lines with circled numbers, and composite onto the SVG.

- **Number assignment:** Numbers 1..N assigned in reading order (top-to-bottom, left-to-right by anchor position), starting from configurable `start_number` (default 1)
- **SVG label layer construction:** Builds an SVG `<g>` group containing:
  - Leader lines -- thin black `<line>` elements from anchor to label endpoint
  - Anchor dots -- small filled `<circle>` at each anchor point
  - Circled numbers -- white-filled `<circle>` with black stroke at each label endpoint, with `<text>` centered via `text-anchor="middle"` + `dominant-baseline="central"`
  - Auto-scaling circles for two-digit numbers (>= 10)
- **SVG compositing:** Parses the Stage 7 `output.svg`, inserts the label layer as a new top-level `<g>` group, adjusts the `viewBox` if the canvas was extended
- **Standalone labels SVG:** Also emits a `labels_only.svg` containing only the label layer (leaders + numbers) without the diagram -- useful for overlaying onto separately edited SVGs
- **Raster preview:** Renders a PNG preview of the labeled diagram for quick inspection

**Algorithms:** Reading-order sorting, SVG XML tree manipulation (ElementTree), viewBox adjustment, circle + text centering

**Input:** `leaders.json` + `output.svg` + optional mask
**Output:**
- `100_label_place/out/labelled.svg` -- final SVG with diagram + numbered labels
- `100_label_place/out/labels_only.svg` -- standalone label layer SVG
- `100_label_place/out/labelled_preview.png` -- raster preview

---

## Web Application

A browser-based wizard that orchestrates the full pipeline through a 4-step UI.

### Architecture

| Layer | Tech | Port |
|---|---|---|
| Backend API | FastAPI + uvicorn | `:8000` |
| Frontend UI | React + Vite | `:5173` (dev) |
| SVG Editor | Method Draw (iframe) | served at `/editor/method-draw/` |

In production, the FastAPI backend serves the built React frontend and Method Draw as static files -- everything runs on a single `:8000` port.

### Backend API (`backend/main.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/api/vectorize` | POST | Upload image -> run stages 0-7 -> return `{run_id, svg}` |
| `/api/save-edited-svg` | POST | Save user-edited SVG back to the run directory |
| `/api/identify-components` | POST | Run stage 8 -> return detected anchor points |
| `/api/label` | POST | Accept user anchors -> run stages 9-10 -> return labeled SVG |
| `/api/runs` | GET | List all run directories |
| `/api/runs/{id}/files` | GET | List files in a run |
| `/api/runs/{id}/file/{path}` | GET | Download a specific run artifact |

### Method Draw Integration

Method Draw is embedded in Step 2 as an `<iframe>`. A postMessage bridge (`Method-Draw/src/js/postmessage-bridge.js`) enables communication:

| Direction | Message | Description |
|---|---|---|
| Parent -> Editor | `{type: "LOAD_SVG", svg: "..."}` | Load SVG into the editor canvas |
| Parent -> Editor | `{type: "REQUEST_SVG"}` | Request the current SVG content |
| Editor -> Parent | `{type: "CURRENT_SVG", svg: "..."}` | Returns the current SVG string |
| Editor -> Parent | `{type: "BRIDGE_READY"}` | Signals the editor is initialized |

### Running

```bash
# Development (hot-reload on both frontend and backend)
./dev.sh
# -> Frontend: http://localhost:5173
# -> Backend:  http://localhost:8000

# Production (single server)
cd frontend && npm run build && cd ..
cd backend && python main.py
# -> http://localhost:8000
```

---

## Output Directory Structure

Each run creates a self-contained directory under `runs/`:

```
runs/<run_name>/
+-- 00_input/                   # Original input copy
|   +-- 01_input.png
+-- 10_preprocess/              # Stage 1: Binary mask
|   +-- out/output_mask.png
|   +-- debug/
+-- 20_distance_transform/      # Stage 2: Distance field
|   +-- out/dt.npy, stroke_width.json
|   +-- debug/
+-- 30_ridge/                   # Stage 3: Centerline skeleton
|   +-- out/ridge.png, coverage.json
|   +-- debug/
+-- 40_graph_raw/               # Stage 4: Raw topological graph
|   +-- out/graph_raw.json
|   +-- debug/
+-- 50_graph_clean/             # Stage 5: Cleaned graph
|   +-- out/graph_clean.json
|   +-- debug/
+-- 60_fit/                     # Stage 6: Fitted primitives
|   +-- out/primitives.json
|   +-- debug/
+-- 70_svg/                     # Stage 7: Vector SVG output
|   +-- out/output.svg, preview.png
|   +-- debug/
+-- 80_label_identify/          # Stage 8: Component anchors
|   +-- out/components.json
|   +-- debug/
+-- 90_label_leaders/           # Stage 9: Routed leader lines
|   +-- out/leaders.json
|   +-- debug/
+-- 100_label_place/            # Stage 10: Final labeled SVG
    +-- out/labelled.svg, labels_only.svg, labelled_preview.png
    +-- debug/
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Image I/O, bilateral filtering, thresholding, morphology, distance transform, visualization |
| `numpy` | Array operations, linear algebra, distance computations |
| `scipy` | Nonlinear least squares (arc fitting), KD-tree (parallel detection, Bezier error), maximum filter (ridge extraction) |
| `scikit-image` | Morphological thinning (Zhang-Suen) for ridge extraction |
| `networkx` | Multigraph representation for graph cleanup operations |
| `fastapi` | Backend web API framework |
| `uvicorn` | ASGI server for FastAPI |
| `python-multipart` | File upload handling for FastAPI |

```bash
# Pipeline dependencies
pip install opencv-python numpy scipy scikit-image networkx

# Web application dependencies
pip install fastapi uvicorn python-multipart
cd frontend && npm install
```

## Configuration

Every pipeline stage accepts a `--config path/to/config.json` flag. The JSON file merges with that stage's `DEFAULT_CONFIG` dict, overriding only the keys you specify. See the `DEFAULT_CONFIG` at the top of each stage file for all available parameters with their defaults and descriptions.

---

## Credits

**[Method Draw](https://github.com/nicholidesign/Method-Draw)** -- The embedded SVG editor used in Step 2 of the web application is Method Draw, an open-source web-based SVG editor created by [Mark MacKay](https://github.com/nicholidesign). Method Draw is included as a Git submodule and served as a static asset. The postMessage bridge is the only modification made to the original source. Method Draw is licensed under the MIT License.
