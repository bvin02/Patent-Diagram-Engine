"""
Stage 8: Component Identification for Labelling

Identifies enclosed regions (components) in the binary stroke mask using
flood-fill. Each region bounded by strokes gets a label anchor point —
a coordinate inside the region suitable for a leader line attachment.

Algorithm:
    1. Start from background pixels (value 0) that are INSIDE the image
       bounding box of the drawing (not the outer margin).
    2. Flood-fill each unvisited background pixel; the fill stops at
       stroke boundaries (value 255).
    3. Each connected fill region is a "component". The outer background
       (largest region touching the image border) is excluded.
    4. For each component, choose a label anchor point: the pixel farthest
       from any boundary stroke, found via the distance transform of the
       component mask.  This ensures the anchor is well inside the region.
    5. Very small regions (below min_area) are discarded as noise.
    6. Regions where ALL boundary edges are already fully "seen" by a
       previous region are merged (shared-boundary deduplication).

Usage:
    python label_identify.py runs/<run>/10_preprocess/out/output_mask.png --debug
    python label_identify.py runs/<run>/10_preprocess/out/output_mask.png --svg runs/<run>/70_svg/out/output.svg --debug

Outputs to runs/<run>/80_label_identify/out/:
    - components.json: List of components with id, anchor point, area, bbox, boundary edges
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.artifacts import StageArtifacts


# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Minimum component area in pixels to keep (discard noise regions)
    "min_component_area": 200,

    # Maximum component area as fraction of image area — regions larger
    # than this are assumed to be the outer background and excluded
    "max_component_area_frac": 0.50,

    # Minimum number of boundary edge pixels a component must have
    "min_boundary_pixels": 20,

    # Anchor point: minimum distance (px) from any stroke edge
    "anchor_min_edge_dist": 5,

    # Dilate strokes slightly before flood-fill to ensure connectivity
    # at thin junctions (prevents leaking through 1-px gaps)
    "stroke_dilate_radius": 1,

    # Fraction of novel boundary edges required to accept a new component
    # (prevents duplicate labelling of already-seen regions)
    "novel_boundary_frac": 0.15,

    # Minimum length of a boundary contour segment to count as a real edge
    "min_edge_segment_length": 8,

    # Image border margin: pixels near the image edge are considered
    # "exterior" and grouped with the outer background
    "border_margin": 5,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> dict:
    config = DEFAULT_CONFIG.copy()
    if config_path:
        p = Path(config_path)
        if p.exists():
            with open(p) as f:
                config.update(json.load(f))
    return config


def infer_run_dir(mask_path: str, runs_root: str = "runs") -> Path:
    p = Path(mask_path).resolve()
    rr = Path(runs_root).resolve()
    try:
        rel = p.relative_to(rr)
        return rr / rel.parts[0]
    except ValueError:
        return p.parent.parent.parent


# ---------------------------------------------------------------------------
# Core: Component Identification
# ---------------------------------------------------------------------------

def identify_components(
    mask: np.ndarray,
    config: dict,
    debug_callback=None,
) -> List[dict]:
    """
    Identify enclosed regions in the binary mask via flood-fill.

    Args:
        mask:  uint8 binary mask (strokes=255, background=0).
        config: Configuration dict.
        debug_callback: Optional callable(name, image) for debug output.

    Returns:
        List of component dicts, each with:
            id, anchor_x, anchor_y, area, bbox, boundary_hash
    """
    h, w = mask.shape[:2]
    image_area = h * w
    cfg = config

    # ------------------------------------------------------------------
    # 1. Prepare the boundary image: dilate strokes to seal thin gaps
    # ------------------------------------------------------------------
    dilate_r = cfg["stroke_dilate_radius"]
    if dilate_r > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_r + 1, 2 * dilate_r + 1)
        )
        boundary = cv2.dilate(mask, kernel, iterations=1)
    else:
        boundary = mask.copy()

    if debug_callback:
        debug_callback("boundary_dilated", boundary)

    # ------------------------------------------------------------------
    # 2. Label connected components of the BACKGROUND (value 0)
    #    using OpenCV connectedComponents on the inverted image.
    # ------------------------------------------------------------------
    bg_mask = (boundary == 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(bg_mask, connectivity=4)

    if debug_callback:
        # Color-code the labels for visualization
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for lbl in range(1, num_labels):
            color = _label_color(lbl)
            vis[labels == lbl] = color
        debug_callback("cc_labels_raw", vis)

    # ------------------------------------------------------------------
    # 3. Classify each label: exterior vs interior
    #    A label is exterior if it touches the image border margin.
    # ------------------------------------------------------------------
    margin = cfg["border_margin"]
    border_labels = set()
    # Top/bottom rows
    for row in range(min(margin, h)):
        border_labels.update(labels[row, :].tolist())
    for row in range(max(0, h - margin), h):
        border_labels.update(labels[row, :].tolist())
    # Left/right columns
    for col in range(min(margin, w)):
        border_labels.update(labels[:, col].tolist())
    for col in range(max(0, w - margin), w):
        border_labels.update(labels[:, col].tolist())
    border_labels.discard(0)  # label 0 is "stroke" in cc output

    # ------------------------------------------------------------------
    # 4. For each interior label, compute properties and pick anchor
    # ------------------------------------------------------------------
    min_area = cfg["min_component_area"]
    max_area = int(image_area * cfg["max_component_area_frac"])
    anchor_min_dist = cfg["anchor_min_edge_dist"]

    raw_components = []

    for lbl in range(1, num_labels):
        if lbl in border_labels:
            continue

        comp_mask = (labels == lbl).astype(np.uint8) * 255
        area = int(np.count_nonzero(comp_mask))

        if area < min_area or area > max_area:
            continue

        # Bounding box
        ys, xs = np.nonzero(comp_mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        # Distance transform inside the component to find deepest interior point
        dt = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        max_dist = dt.max()

        if max_dist < anchor_min_dist:
            # Region too thin to place a label inside comfortably
            continue

        # Pick anchor = pixel with maximum distance from boundary
        # Break ties deterministically: pick topmost, then leftmost
        candidates = np.argwhere(dt >= max_dist - 0.5)  # near-max pixels
        # argwhere returns (row, col) = (y, x)
        # Sort by y ascending, then x ascending
        idx = np.lexsort((candidates[:, 1], candidates[:, 0]))
        anchor_y, anchor_x = candidates[idx[0]]

        # ------------------------------------------------------------------
        # 5. Compute boundary signature: which stroke pixels border this region
        # ------------------------------------------------------------------
        # Dilate the component mask by 2px to find adjacent stroke pixels
        kern3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_comp = cv2.dilate(comp_mask, kern3, iterations=2)
        # Boundary stroke pixels = dilated_comp AND original mask
        boundary_px = cv2.bitwise_and(dilated_comp, mask)
        boundary_pixel_count = int(np.count_nonzero(boundary_px))

        # Encode boundary pixel locations as a frozenset of (x//4, y//4) cells
        # for efficient set comparison (quantized to 4px grid)
        by, bx = np.nonzero(boundary_px)
        boundary_cells = set(zip((bx // 4).tolist(), (by // 4).tolist()))

        raw_components.append({
            "label": lbl,
            "anchor_x": int(anchor_x),
            "anchor_y": int(anchor_y),
            "area": area,
            "bbox": bbox,
            "max_interior_dist": float(max_dist),
            "boundary_pixel_count": boundary_pixel_count,
            "boundary_cells": boundary_cells,
        })

    # ------------------------------------------------------------------
    # 6. Deduplicate: keep component only if it contributes novel boundary
    # ------------------------------------------------------------------
    # Sort by area descending — larger regions first
    raw_components.sort(key=lambda c: c["area"], reverse=True)

    seen_cells = set()
    accepted = []
    novel_frac_thresh = cfg["novel_boundary_frac"]

    for comp in raw_components:
        bc = comp["boundary_cells"]
        if not bc:
            continue

        novel = bc - seen_cells
        frac = len(novel) / len(bc) if bc else 0.0

        if frac >= novel_frac_thresh or len(accepted) == 0:
            # Accept this component
            accepted.append(comp)
            seen_cells.update(bc)

    # ------------------------------------------------------------------
    # 7. Assign sequential IDs and build output
    # ------------------------------------------------------------------
    components = []
    for idx, comp in enumerate(accepted):
        components.append({
            "id": idx,
            "anchor_x": comp["anchor_x"],
            "anchor_y": comp["anchor_y"],
            "area": comp["area"],
            "bbox": comp["bbox"],
            "max_interior_dist": round(comp["max_interior_dist"], 2),
            "boundary_pixel_count": comp["boundary_pixel_count"],
        })

    # ------------------------------------------------------------------
    # 8. Sort components in a natural reading order:
    #    top-to-bottom, left-to-right (by anchor position)
    # ------------------------------------------------------------------
    components.sort(key=lambda c: (c["anchor_y"], c["anchor_x"]))
    for idx, comp in enumerate(components):
        comp["id"] = idx

    return components


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _label_color(idx: int) -> Tuple[int, int, int]:
    """Deterministic color from index using golden-ratio hue rotation."""
    hue = int((idx * 137.508) % 180)
    hsv = np.array([[[hue, 200, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0, 0])


def draw_components_debug(
    mask: np.ndarray,
    components: List[dict],
) -> np.ndarray:
    """Draw components overlay on the mask for debugging."""
    h, w = mask.shape[:2]
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for comp in components:
        color = _label_color(comp["id"])
        ax, ay = comp["anchor_x"], comp["anchor_y"]

        # Draw anchor cross-hair
        cv2.drawMarker(
            vis, (ax, ay), color,
            markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2,
        )

        # Draw bbox
        x0, y0, x1, y1 = comp["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)

        # Label
        label = f"C{comp['id']} ({comp['area']}px)"
        cv2.putText(
            vis, label, (ax + 10, ay - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    return vis


def draw_components_on_white(
    h: int, w: int,
    mask: np.ndarray,
    components: List[dict],
) -> np.ndarray:
    """Draw clean component visualization on white background."""
    vis = np.full((h, w, 3), 255, dtype=np.uint8)

    # Draw strokes in light gray
    stroke_px = mask == 255
    vis[stroke_px] = (180, 180, 180)

    for comp in components:
        color = _label_color(comp["id"])
        ax, ay = comp["anchor_x"], comp["anchor_y"]

        # Filled circle at anchor
        cv2.circle(vis, (ax, ay), 6, color, -1)
        cv2.circle(vis, (ax, ay), 6, (0, 0, 0), 1)

        # Component number
        txt = str(comp["id"])
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            vis, txt, (ax - tw // 2, ay + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def identify(
    mask_path: str,
    svg_path: Optional[str] = None,
    debug: bool = True,
    config_path: Optional[str] = None,
    runs_root: str = "runs",
):
    """Main entry point for Stage 8."""
    print(f"Processing: {mask_path}")

    config = load_config(config_path)

    # Load mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Cannot read mask: {mask_path}")
        return None
    h, w = mask.shape
    print(f"Image size: {w}x{h}")

    # Infer run dir and init artifacts
    run_dir = infer_run_dir(mask_path, runs_root)
    print(f"Run directory: {run_dir}")
    artifacts = StageArtifacts(run_dir, stage_id=80, stage_name="label_identify", debug=debug)

    # Debug image callback
    def debug_cb(name, img):
        if debug:
            artifacts.save_debug_image(name, img)

    # Run identification
    components = identify_components(mask, config, debug_callback=debug_cb)
    print(f"Found {len(components)} components")

    # Save output
    output_data = {
        "image": {"width": w, "height": h},
        "mask_path": str(mask_path),
        "config": config,
        "components": components,
    }
    if svg_path:
        output_data["svg_path"] = str(svg_path)

    artifacts.save_json("components", output_data)

    # Debug visualizations
    if debug:
        # Anchors on mask
        dbg_mask = draw_components_debug(mask, components)
        artifacts.save_debug_image("anchors_on_mask", dbg_mask)

        # Clean white-background view
        dbg_clean = draw_components_on_white(h, w, mask, components)
        artifacts.save_debug_image("anchors_clean", dbg_clean)

    # Metrics
    artifacts.write_metrics({
        "num_components": len(components),
        "total_component_area": sum(c["area"] for c in components),
        "min_area": min((c["area"] for c in components), default=0),
        "max_area": max((c["area"] for c in components), default=0),
        "image_area": w * h,
    })

    # Print summary
    for c in components:
        print(f"  Component {c['id']:3d}  anchor=({c['anchor_x']:4d},{c['anchor_y']:4d})  "
              f"area={c['area']:6d}  dist={c['max_interior_dist']:.1f}")

    return components


def main():
    parser = argparse.ArgumentParser(
        description="Stage 8: Identify enclosed components for labelling"
    )
    parser.add_argument("mask_path", help="Path to binary mask (output_mask.png)")
    parser.add_argument("--svg", default=None,
                        help="Path to output.svg (for reference in later stages)")
    parser.add_argument("--runs_root", default="runs",
                        help="Root directory for runs (default: runs)")
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Generate debug outputs (default: True)")
    parser.add_argument("--no_debug", action="store_true",
                        help="Disable debug outputs")
    parser.add_argument("--config", help="Path to config JSON file")

    args = parser.parse_args()

    identify(
        mask_path=args.mask_path,
        svg_path=args.svg,
        debug=not args.no_debug,
        config_path=args.config,
        runs_root=args.runs_root,
    )


if __name__ == "__main__":
    main()
