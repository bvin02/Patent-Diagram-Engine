# Patent Draw

Convert hand-drawn patent sketches into editable SVG line art with component grouping, labeling, numbering, validation, and export.

## Overview

Patent Draw is a Python pipeline that transforms A-tier hand-drawn technical sketches into publication-ready patent drawings. The system:

- **Binarizes** input images to extract clean stroke data
- **Vectorizes** strokes using skeleton extraction and Bezier curve fitting
- **Groups** strokes into components using spatial analysis
- **Labels** components with leader lines and reference numerals
- **Validates** output against patent drawing requirements
- **Exports** to SVG and PDF formats

## Scope and Assumptions

### A-Tier Input Assumptions

This MVP assumes **A-tier inputs**:
- Clean scans or photos on white/light background
- Minimal shadows and noise
- Already roughly cropped to the drawing
- Single drawing per image (or clearly separated drawings)

**Deferred for later implementation:**
- Stage 0: PDF ingestion and image extraction
- Stage 1: Photo correction, perspective, shadow removal
- Interactive UI editor

### Non-Goals

- 3D reconstruction or CAD conversion
- Auto-exploded views
- Full auto-dimensioning

## Installation

```bash
# Clone and install
cd Patent-Diagram-Generator
pip install -e ".[dev]"

# Or install with ML extras (future)
pip install -e ".[dev,ml]"
```

### Dependencies

Core: OpenCV, NumPy, scikit-image, Shapely, NetworkX, svgwrite, cairosvg, PyYAML, Pydantic

## Quick Start

### Basic Pipeline Run

```bash
# Run on a single image
patentdraw run --inputs sketch.png --out output/ --debug

# Run on multiple images
patentdraw run --inputs view1.png view2.png view3.png --out output/ --debug

# With custom config
patentdraw run --inputs sketch.png --out output/ --config config.yaml --trace
```

### Apply Operations (Simulate Editor)

```bash
# Apply merge/split operations to existing scene
patentdraw ops apply --scene output/scene.json --ops operations.json --out output_v2/
```

### Generate Default Config

```bash
patentdraw init-config --out my_config.yaml
```

## Pipeline Stages

```
Input Image(s)
    |
    v
[Stage 2] Binarization
    - Grayscale conversion
    - Otsu/adaptive threshold
    - Morphological cleanup
    |
    v
[Stage 3] Skeleton Graph
    - Skeletonize to 1px lines
    - Build connectivity graph
    - Trace polylines between endpoints/junctions
    |
    v
[Stage 4] Vectorization
    - RDP simplification
    - Cubic Bezier curve fitting
    - SVG path emission
    |
    v
[Stage 5] Component Grouping
    - Spatial adjacency analysis
    - Union-find clustering
    - Optional ML proposals
    |
    v
[Stage 6] Label Proposals
    - Anchor point selection
    - Text position placement
    - Leader line routing
    |
    v
[Stage 7] Numbering
    - Stable ordering by centroid
    - Reference numeral assignment
    |
    v
[Stage 8] Validation + Export
    - Compliance checks
    - SVG package generation
    - PDF layout (8.5x11, margins)
    |
    v
Outputs: SVG, PDF, scene.json, validation_report.json
```

## Output Files

| File | Description |
|------|-------------|
| `final.svg` | Combined SVG with all views |
| `final.pdf` | PDF with proper page layout |
| `scene.json` | Complete scene graph (for operations) |
| `validation_report.json` | All validation check results |
| `validation_summary.txt` | Human-readable summary |
| `svg/{view_id}.svg` | Per-view SVG files |

## Debug Artifacts

When `--debug` is enabled, each stage writes artifacts to `debug/{view_id}/{stage}/`:

### Stage 2 (Binarization)
- `00_input_preview.png` - Original input
- `01_gray.png` - Grayscale conversion
- `02_binary.png` - Binary result
- `03_overlay_binary_on_input.png` - Visual verification

### Stage 3 (Skeleton)
- `01_skeleton.png` - Skeletonized image
- `02_endpoints_junctions_overlay.png` - Graph structure
- `03_polylines_overlay.png` - Traced paths

### Stage 4 (Vectorize)
- `output_strokes.svg` - Raw strokes
- `01_svg_render.png` - PNG render
- `02_stroke_bboxes_overlay.png` - Stroke IDs

### Stage 5 (Components)
- `01_components_overlay.png` - Grouped strokes
- `03_component_map.json` - Component structure

### Stage 6-7 (Labels/Numbering)
- Label overlays with leader lines
- `numbering_registry.json` - Final assignments

### Stage 8 (Export)
- Final validation and export results

## Configuration

Create a YAML config file:

```yaml
binarization:
  method: otsu          # or "adaptive"
  denoise_kernel: 3
  morph_kernel: 3

simplify:
  rdp_epsilon: 1.5

bezier:
  error_tolerance: 2.0
  max_iterations: 4

stroke:
  width: 1.5
  color: black

grouping:
  endpoint_distance_threshold: 5.0
  bbox_overlap_threshold: 0.1

label:
  text_offset: 30.0
  label_font_size: 12.0
  leader_stroke_width: 0.5

pdf:
  page_width_inches: 8.5
  page_height_inches: 11.0
  margin_inches: 1.0
  dpi: 300

numbering:
  start_number: 10
  increment: 2

ml_enabled: false
```

## Operations (Simulating Future Editor)

Operations allow deterministic modifications to the scene graph:

### Merge Components

```json
[
  {
    "op": "merge_components",
    "component_ids": ["comp_abc123", "comp_def456"]
  }
]
```

### Split Component

```json
[
  {
    "op": "split_component",
    "component_id": "comp_abc123",
    "stroke_sets": [
      ["stroke_001", "stroke_002"],
      ["stroke_003", "stroke_004"]
    ]
  }
]
```

### Move Label

```json
[
  {
    "op": "move_label",
    "label_id": "label_xyz",
    "text_pos": [150.0, 75.0]
  }
]
```

## Determinism

The pipeline is fully deterministic:

- **Stroke IDs**: Hash of polyline coordinates + view ID
- **Component IDs**: Hash of sorted stroke IDs
- **Label IDs**: Hash of component ID + positions
- **Numbering**: Sorted by (y, x) centroid position

Running the same input twice produces identical output.

## Validation Rules

| Rule | Severity | Description |
|------|----------|-------------|
| `monochrome` | ERROR | All strokes must be black/white |
| `dpi_minimum` | WARN | Input should support 300 DPI export |
| `labels_complete` | ERROR | Every component needs a numeral |
| `leader_crossings` | WARN | Leader lines cross each other |
| `text_overlap` | WARN | Label text overlaps strokes |
| `margins` | INFO | PDF margins configured |

## Adding ML Proposals

The `components/proposals_ml.py` provides a stub interface:

```python
class ProposalProvider(ABC):
    @abstractmethod
    def get_proposals(self, image, strokes):
        """Return list of Component objects from ML model."""
        pass
    
    @abstractmethod
    def is_available(self):
        """Check if model is loaded."""
        pass
```

To integrate SAM or similar:
1. Implement `ProposalProvider` subclass
2. Add model loading in `get_provider()`
3. Set `ml_enabled: true` in config

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_binarize.py -v

# Run with coverage
pytest tests/ --cov=patentdraw
```

## Tracing

Enable runtime tracing for debugging:

```bash
patentdraw run --inputs sketch.png --out output/ --trace --trace-level DEBUG

# Write to file
patentdraw run --inputs sketch.png --out output/ --trace --trace-file trace.log
```

Trace output shows hierarchical spans with timing:

```
12:01:03.114 INFO  pipeline:run_pipeline  start inputs=1
12:01:03.219 INFO    stage2_binarize:binarize  start method=otsu
12:01:03.330 INFO    stage2_binarize:binarize  end ok dt=111ms
```

## Demo Recipe

### Easy: Simple Rectangle

```bash
# Create test image with rectangle
python -c "
import cv2
import numpy as np
img = np.ones((300, 400, 3), dtype=np.uint8) * 255
cv2.rectangle(img, (50, 50), (350, 250), (0, 0, 0), 2)
cv2.imwrite('test_rect.png', img)
"

# Run pipeline
patentdraw run --inputs test_rect.png --out demo_easy/ --debug
```

### Medium: Multiple Components

```bash
patentdraw run --inputs example_good_sketch.png --out demo_medium/ --debug --trace
```

### Hard: Complex Sketch

```bash
patentdraw run --inputs example_bad_sketch.png --out demo_hard/ --debug --trace
```

### Apply Operations

```bash
# After initial run, create ops.json:
echo '[{"op": "merge_components", "component_ids": ["comp_abc", "comp_def"]}]' > merge_ops.json

# Apply and re-export
patentdraw ops apply --scene demo_medium/scene.json --ops merge_ops.json --out demo_merged/
```

## Failure Modes

| Symptom | Likely Cause | Mitigation |
|---------|--------------|------------|
| No strokes detected | Image too faint or threshold too aggressive | Adjust `binarization.adaptive_c` or use `adaptive` method |
| Too many components | Low grouping threshold | Increase `endpoint_distance_threshold` |
| Single giant component | High grouping threshold | Decrease thresholds |
| Jagged curves | RDP epsilon too high | Decrease `rdp_epsilon` |
| PDF too small | Low input resolution | Use higher DPI scans |

## Project Structure

```
src/patentdraw/
    __init__.py
    cli.py              # CLI commands
    pipeline.py         # Main orchestration
    config.py           # YAML configuration
    models.py           # Pydantic scene graph
    tracer.py           # Runtime tracing
    io/
        load_image.py   # Image loading (Stage 0 stub)
        save_artifacts.py
    preprocess/
        stage2_binarize.py
    strokes/
        skeleton_graph.py
        polyline_trace.py
        simplify.py
        bezier_fit.py
        svg_emit.py
    components/
        grouping.py
        operations.py
        proposals_ml.py
    labels/
        label_propose.py
        leader_route.py
        numbering.py
    validate/
        rules.py
        report.py
    export/
        svg_package.py
        pdf_layout.py
tests/
    conftest.py
    test_*.py
```

## License

MIT
