#!/usr/bin/env python3
"""
Stage 9: Leader Line Routing

Creates straight leader lines from each component's interior anchor point
to the outside margin of the diagram, where a numeric label will be placed.

Design goals:
    - Lines are straight (single segment) from anchor to label position.
    - Lines do not overlap or cross each other where possible.
    - Label endpoints are placed outside the drawing's bounding box with
      comfortable margin, so they don't get cropped.
    - Lines are spread out to avoid visual congestion; angular sectors
      around the diagram are allocated to prevent bunching.
    - If crossing is unavoidable, lines are routed to minimize total
      crossings via a sweep-and-swap heuristic.

Algorithm:
    1. Compute the bounding box of all strokes (the drawing envelope).
    2. Define a label ring: a rectangle outside the bbox with configurable
       margin.  Label endpoints live on this ring.
    3. For each anchor, determine its "natural" exit direction based on
       where it sits relative to the bbox center (closest edge/corner).
    4. Project each anchor outward along that direction until hitting the
       label ring.  This is the candidate label endpoint.
    5. Detect and resolve congestion: if two label endpoints are too close,
       spread them apart along the ring perimeter.
    6. Detect and reduce crossings: for adjacent anchors whose leader
       lines cross, swap their label endpoints if that reduces crossings.
    7. Output the routed leader lines as a JSON file.

Usage:
    python label_leaders.py runs/<run>/80_label_identify/out/components.json --debug
    python label_leaders.py runs/<run>/80_label_identify/out/components.json \\
        --mask runs/<run>/10_preprocess/out/output_mask.png --debug

Outputs to runs/<run>/90_label_leaders/out/:
    - leaders.json: Leader lines with start/end points and label positions
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.artifacts import StageArtifacts


# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Margin between drawing bbox and label ring (px)
    "label_margin": 60,

    # Extra padding from image edge to ensure labels don't get cropped (px)
    "image_edge_padding": 30,

    # Minimum distance between two label endpoints on the ring (px)
    "min_label_spacing": 35,

    # Maximum number of crossing-reduction passes
    "max_crossing_passes": 20,

    # Leader line style
    "leader_elbow": False,  # Future: True = L-shaped elbow lines

    # Whether to extend image canvas if labels would be cropped
    "extend_canvas": True,

    # Extension padding when canvas is extended (px on each side)
    "canvas_extension": 100,

    # Minimum leader line length (px)
    "min_leader_length": 30,

    # Preferred directions: fraction of ring perimeter allocated to each side
    # (top, right, bottom, left) — biases label placement
    "side_weights": [1.0, 1.0, 1.0, 1.0],
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


def infer_run_dir(path: str, runs_root: str = "runs") -> Path:
    p = Path(path).resolve()
    rr = Path(runs_root).resolve()
    try:
        rel = p.relative_to(rr)
        return rr / rel.parts[0]
    except ValueError:
        return p.parent.parent.parent


# ---------------------------------------------------------------------------
# Core: Leader Line Routing
# ---------------------------------------------------------------------------

def compute_drawing_bbox(mask: np.ndarray, margin: int = 5) -> Tuple[int, int, int, int]:
    """
    Compute tight bounding box of the drawing content.

    Returns (x0, y0, x1, y1) of the minimal rectangle enclosing all strokes.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return (0, 0, w, h)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def compute_label_ring(
    drawing_bbox: Tuple[int, int, int, int],
    img_w: int, img_h: int,
    margin: int,
    edge_padding: int,
) -> Tuple[int, int, int, int, int, int]:
    """
    Compute the label ring rectangle and canvas dimensions.

    Returns:
        (ring_x0, ring_y0, ring_x1, ring_y1, canvas_w, canvas_h)

    The ring is a rectangle around the drawing bbox offset by `margin`.
    If the ring extends beyond the image, the canvas is enlarged.
    """
    dx0, dy0, dx1, dy1 = drawing_bbox

    # Ring bounds
    rx0 = dx0 - margin
    ry0 = dy0 - margin
    rx1 = dx1 + margin
    ry1 = dy1 + margin

    # Canvas must accommodate ring + edge padding
    canvas_w = max(img_w, rx1 + edge_padding)
    canvas_h = max(img_h, ry1 + edge_padding)

    # Shift ring if it goes negative
    x_shift = 0
    y_shift = 0
    if rx0 < edge_padding:
        x_shift = edge_padding - rx0
    if ry0 < edge_padding:
        y_shift = edge_padding - ry0

    rx0 += x_shift
    ry0 += y_shift
    rx1 += x_shift
    ry1 += y_shift
    canvas_w += x_shift
    canvas_h += y_shift

    return rx0, ry0, rx1, ry1, canvas_w, canvas_h, x_shift, y_shift


def closest_ring_point(
    anchor: Tuple[int, int],
    ring: Tuple[int, int, int, int],
    center: Tuple[float, float],
) -> Tuple[int, int, str]:
    """
    Project anchor outward from center to the ring rectangle.

    Returns (x, y, side) where side is "top", "bottom", "left", "right".
    """
    ax, ay = anchor
    cx, cy = center
    rx0, ry0, rx1, ry1 = ring

    dx = ax - cx
    dy = ay - cy

    # Avoid division by zero
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return rx1, int(cy), "right"

    # Find intersection with ring rectangle
    candidates = []

    # Right edge
    if dx > 0:
        t = (rx1 - ax) / dx if dx != 0 else float("inf")
        if t > 0:
            y_hit = ay + t * dy
            if ry0 <= y_hit <= ry1:
                candidates.append((rx1, int(y_hit), "right", t))

    # Left edge
    if dx < 0:
        t = (rx0 - ax) / dx if dx != 0 else float("inf")
        if t > 0:
            y_hit = ay + t * dy
            if ry0 <= y_hit <= ry1:
                candidates.append((rx0, int(y_hit), "left", t))

    # Bottom edge
    if dy > 0:
        t = (ry1 - ay) / dy if dy != 0 else float("inf")
        if t > 0:
            x_hit = ax + t * dx
            if rx0 <= x_hit <= rx1:
                candidates.append((int(x_hit), ry1, "bottom", t))

    # Top edge
    if dy < 0:
        t = (ry0 - ay) / dy if dy != 0 else float("inf")
        if t > 0:
            x_hit = ax + t * dx
            if rx0 <= x_hit <= rx1:
                candidates.append((int(x_hit), ry0, "top", t))

    if not candidates:
        # Fallback: closest ring edge midpoint
        mid_pts = [
            ((rx0 + rx1) // 2, ry0, "top"),
            ((rx0 + rx1) // 2, ry1, "bottom"),
            (rx0, (ry0 + ry1) // 2, "left"),
            (rx1, (ry0 + ry1) // 2, "right"),
        ]
        best = min(mid_pts, key=lambda p: (p[0] - ax) ** 2 + (p[1] - ay) ** 2)
        return best[0], best[1], best[2]

    # Pick the nearest intersection (smallest t)
    candidates.sort(key=lambda c: c[3])
    return candidates[0][0], candidates[0][1], candidates[0][2]


def ring_perimeter_position(
    x: int, y: int, side: str,
    ring: Tuple[int, int, int, int],
) -> float:
    """
    Map a point on the ring to a 0-1 perimeter position (clockwise from top-left).
    Used for ordering and spacing labels.
    """
    rx0, ry0, rx1, ry1 = ring
    w = rx1 - rx0
    h = ry1 - ry0
    perimeter = 2 * (w + h)
    if perimeter == 0:
        return 0.0

    if side == "top":
        return (x - rx0) / perimeter
    elif side == "right":
        return (w + (y - ry0)) / perimeter
    elif side == "bottom":
        return (w + h + (rx1 - x)) / perimeter
    elif side == "left":
        return (2 * w + h + (ry1 - y)) / perimeter
    return 0.0


def perimeter_to_ring_point(
    t: float,
    ring: Tuple[int, int, int, int],
) -> Tuple[int, int, str]:
    """Inverse of ring_perimeter_position: convert 0-1 fraction back to (x,y,side)."""
    rx0, ry0, rx1, ry1 = ring
    w = rx1 - rx0
    h = ry1 - ry0
    perimeter = 2 * (w + h)
    if perimeter == 0:
        return rx0, ry0, "top"

    t = t % 1.0
    d = t * perimeter

    if d <= w:
        return rx0 + int(d), ry0, "top"
    d -= w
    if d <= h:
        return rx1, ry0 + int(d), "right"
    d -= h
    if d <= w:
        return rx1 - int(d), ry1, "bottom"
    d -= w
    return rx0, ry1 - int(d), "left"


def segments_cross(
    a1: Tuple[int, int], a2: Tuple[int, int],
    b1: Tuple[int, int], b2: Tuple[int, int],
) -> bool:
    """Check if line segments (a1,a2) and (b1,b2) intersect (proper crossing)."""
    def cross2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross2d(b1, b2, a1)
    d2 = cross2d(b1, b2, a2)
    d3 = cross2d(a1, a2, b1)
    d4 = cross2d(a1, a2, b2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def count_crossings(leaders: List[dict]) -> int:
    """Count total pairwise leader line crossings."""
    n = len(leaders)
    crossings = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = leaders[i]
            b = leaders[j]
            if segments_cross(
                (a["anchor_x"], a["anchor_y"]),
                (a["label_x"], a["label_y"]),
                (b["anchor_x"], b["anchor_y"]),
                (b["label_x"], b["label_y"]),
            ):
                crossings += 1
    return crossings


def resolve_spacing(
    leaders: List[dict],
    ring: Tuple[int, int, int, int],
    min_spacing: float,
) -> List[dict]:
    """
    Spread out label endpoints that are too close together on the ring.
    Uses a force-directed relaxation along the ring perimeter.
    """
    if len(leaders) <= 1:
        return leaders

    rx0, ry0, rx1, ry1 = ring
    w = rx1 - rx0
    h = ry1 - ry0
    perimeter = 2 * (w + h)
    if perimeter == 0:
        return leaders

    min_frac = min_spacing / perimeter

    # Get perimeter positions
    positions = []
    for ldr in leaders:
        t = ring_perimeter_position(
            ldr["label_x"], ldr["label_y"], ldr["side"], ring
        )
        positions.append(t)

    # Sort by perimeter position
    order = sorted(range(len(positions)), key=lambda i: positions[i])

    # Iterative relaxation
    for _pass in range(50):
        moved = False
        for k in range(len(order)):
            i = order[k]
            j = order[(k + 1) % len(order)]

            ti = positions[i]
            tj = positions[j]

            # Circular distance
            gap = (tj - ti) % 1.0
            if gap < min_frac:
                # Push apart symmetrically
                push = (min_frac - gap) / 2.0
                positions[i] = (ti - push) % 1.0
                positions[j] = (tj + push) % 1.0
                moved = True

        if not moved:
            break

    # Apply new positions
    for i, ldr in enumerate(leaders):
        x, y, side = perimeter_to_ring_point(positions[i], ring)
        ldr["label_x"] = x
        ldr["label_y"] = y
        ldr["side"] = side

    return leaders


def reduce_crossings(
    leaders: List[dict],
    max_passes: int,
) -> List[dict]:
    """
    Reduce crossings by swapping label endpoints of crossing pairs.
    Greedy swap heuristic: for each crossing pair, try swapping their
    label positions; keep the swap if it reduces total crossings.
    """
    if len(leaders) <= 2:
        return leaders

    best_crossings = count_crossings(leaders)
    if best_crossings == 0:
        return leaders

    for _pass in range(max_passes):
        improved = False
        n = len(leaders)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = leaders[i], leaders[j]

                if not segments_cross(
                    (a["anchor_x"], a["anchor_y"]),
                    (a["label_x"], a["label_y"]),
                    (b["anchor_x"], b["anchor_y"]),
                    (b["label_x"], b["label_y"]),
                ):
                    continue

                # Try swapping label endpoints
                a["label_x"], b["label_x"] = b["label_x"], a["label_x"]
                a["label_y"], b["label_y"] = b["label_y"], a["label_y"]
                a["side"], b["side"] = b["side"], a["side"]

                new_crossings = count_crossings(leaders)

                if new_crossings < best_crossings:
                    best_crossings = new_crossings
                    improved = True
                else:
                    # Revert swap
                    a["label_x"], b["label_x"] = b["label_x"], a["label_x"]
                    a["label_y"], b["label_y"] = b["label_y"], a["label_y"]
                    a["side"], b["side"] = b["side"], a["side"]

        if not improved or best_crossings == 0:
            break

    return leaders


def route_leaders(
    components: List[dict],
    mask: np.ndarray,
    config: dict,
) -> dict:
    """
    Route leader lines from component anchors to a label ring.

    Returns dict with leaders list, ring info, and canvas dimensions.
    """
    h, w = mask.shape[:2]
    cfg = config

    # 1. Drawing bounding box
    drawing_bbox = compute_drawing_bbox(mask)
    dx0, dy0, dx1, dy1 = drawing_bbox
    print(f"  Drawing bbox: ({dx0},{dy0}) - ({dx1},{dy1})")

    # 2. Label ring
    ring_x0, ring_y0, ring_x1, ring_y1, canvas_w, canvas_h, x_shift, y_shift = \
        compute_label_ring(drawing_bbox, w, h, cfg["label_margin"], cfg["image_edge_padding"])
    ring = (ring_x0, ring_y0, ring_x1, ring_y1)
    print(f"  Label ring: ({ring_x0},{ring_y0}) - ({ring_x1},{ring_y1})")
    print(f"  Canvas: {canvas_w}x{canvas_h} (shift: +{x_shift},+{y_shift})")

    # Center of drawing (shifted)
    center_x = (dx0 + dx1) / 2.0 + x_shift
    center_y = (dy0 + dy1) / 2.0 + y_shift

    # 3. Compute initial leader line for each component
    leaders = []
    for comp in components:
        # Shift anchor to new canvas coordinates
        ax = comp["anchor_x"] + x_shift
        ay = comp["anchor_y"] + y_shift

        # Project to ring
        lx, ly, side = closest_ring_point((ax, ay), ring, (center_x, center_y))

        # Ensure minimum leader length
        dist = math.sqrt((lx - ax) ** 2 + (ly - ay) ** 2)
        if dist < cfg["min_leader_length"]:
            # Extend outward
            if dist > 0:
                scale = cfg["min_leader_length"] / dist
                lx = int(ax + (lx - ax) * scale)
                ly = int(ay + (ly - ay) * scale)

        leaders.append({
            "component_id": comp["id"],
            "anchor_x": ax,
            "anchor_y": ay,
            "label_x": lx,
            "label_y": ly,
            "side": side,
            "original_anchor_x": comp["anchor_x"],
            "original_anchor_y": comp["anchor_y"],
        })

    # 4. Resolve spacing
    leaders = resolve_spacing(leaders, ring, cfg["min_label_spacing"])

    # 5. Reduce crossings
    initial_crossings = count_crossings(leaders)
    leaders = reduce_crossings(leaders, cfg["max_crossing_passes"])
    final_crossings = count_crossings(leaders)
    print(f"  Crossings: {initial_crossings} → {final_crossings}")

    return {
        "image": {"width": w, "height": h},
        "canvas": {"width": canvas_w, "height": canvas_h},
        "offset": {"x": x_shift, "y": y_shift},
        "drawing_bbox": list(drawing_bbox),
        "label_ring": [ring_x0, ring_y0, ring_x1, ring_y1],
        "leaders": leaders,
        "crossings": final_crossings,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_leaders_debug(
    mask: np.ndarray,
    leader_data: dict,
) -> np.ndarray:
    """Draw leader lines on the mask for debugging."""
    canvas_w = leader_data["canvas"]["width"]
    canvas_h = leader_data["canvas"]["height"]
    x_off = leader_data["offset"]["x"]
    y_off = leader_data["offset"]["y"]

    # Create canvas
    vis = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # Paste mask (shifted)
    h, w = mask.shape[:2]
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    vis[y_off:y_off + h, x_off:x_off + w] = mask_bgr

    # Draw label ring
    ring = leader_data["label_ring"]
    cv2.rectangle(vis, (ring[0], ring[1]), (ring[2], ring[3]), (200, 200, 200), 1)

    # Draw drawing bbox
    db = leader_data["drawing_bbox"]
    cv2.rectangle(
        vis,
        (db[0] + x_off, db[1] + y_off),
        (db[2] + x_off, db[3] + y_off),
        (220, 220, 220), 1, cv2.LINE_AA,
    )

    # Draw leaders
    for ldr in leader_data["leaders"]:
        ax, ay = ldr["anchor_x"], ldr["anchor_y"]
        lx, ly = ldr["label_x"], ldr["label_y"]
        cid = ldr["component_id"]
        color = _label_color(cid)

        # Leader line
        cv2.line(vis, (ax, ay), (lx, ly), color, 1, cv2.LINE_AA)

        # Anchor dot
        cv2.circle(vis, (ax, ay), 4, color, -1)

        # Label endpoint dot
        cv2.circle(vis, (lx, ly), 5, (0, 0, 0), -1)
        cv2.circle(vis, (lx, ly), 5, color, 1)

        # Number
        txt = str(cid)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Position text outside the ring
        text_x = lx
        text_y = ly
        side = ldr["side"]
        if side == "right":
            text_x += 8
            text_y += th // 2
        elif side == "left":
            text_x -= tw + 8
            text_y += th // 2
        elif side == "top":
            text_x -= tw // 2
            text_y -= 8
        elif side == "bottom":
            text_x -= tw // 2
            text_y += th + 8

        cv2.putText(vis, txt, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return vis


def _label_color(idx: int) -> Tuple[int, int, int]:
    hue = int((idx * 137.508) % 180)
    hsv = np.array([[[hue, 200, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0, 0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def route(
    components_path: str,
    mask_path: Optional[str] = None,
    debug: bool = True,
    config_path: Optional[str] = None,
    runs_root: str = "runs",
):
    """Main entry point for Stage 9."""
    print(f"Processing: {components_path}")

    config = load_config(config_path)

    # Load components
    with open(components_path) as f:
        comp_data = json.load(f)

    components = comp_data["components"]
    img_w = comp_data["image"]["width"]
    img_h = comp_data["image"]["height"]
    print(f"Image size: {img_w}x{img_h}")
    print(f"Components: {len(components)}")

    # Infer run dir
    run_dir = infer_run_dir(components_path, runs_root)
    print(f"Run directory: {run_dir}")

    # Load mask (needed for drawing bbox)
    if mask_path is None:
        # Try to find mask in the standard location
        candidate = run_dir / "10_preprocess" / "out" / "output_mask.png"
        if candidate.exists():
            mask_path = str(candidate)

    if mask_path:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        print("Warning: No mask provided, using synthetic mask from components")
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Init artifacts
    artifacts = StageArtifacts(run_dir, stage_id=90, stage_name="label_leaders", debug=debug)

    # Route leaders
    leader_data = route_leaders(components, mask, config)
    leader_data["config"] = config

    # Persist the original components data for next stage
    leader_data["components"] = components
    if "svg_path" in comp_data:
        leader_data["svg_path"] = comp_data["svg_path"]

    # Save output
    artifacts.save_json("leaders", leader_data)

    # Debug visualization
    if debug:
        vis = draw_leaders_debug(mask, leader_data)
        artifacts.save_debug_image("leaders_routed", vis)

    # Metrics
    artifacts.write_metrics({
        "num_leaders": len(leader_data["leaders"]),
        "crossings": leader_data["crossings"],
        "canvas_width": leader_data["canvas"]["width"],
        "canvas_height": leader_data["canvas"]["height"],
        "offset_x": leader_data["offset"]["x"],
        "offset_y": leader_data["offset"]["y"],
    })

    # Print summary
    for ldr in leader_data["leaders"]:
        dist = math.sqrt(
            (ldr["label_x"] - ldr["anchor_x"]) ** 2 +
            (ldr["label_y"] - ldr["anchor_y"]) ** 2
        )
        print(f"  C{ldr['component_id']:3d}  "
              f"({ldr['anchor_x']:4d},{ldr['anchor_y']:4d}) → "
              f"({ldr['label_x']:4d},{ldr['label_y']:4d})  "
              f"side={ldr['side']:6s}  len={dist:.0f}")

    return leader_data


def main():
    parser = argparse.ArgumentParser(
        description="Stage 9: Route leader lines for component labels"
    )
    parser.add_argument("components_path", help="Path to components.json from Stage 8")
    parser.add_argument("--mask", default=None,
                        help="Path to binary mask (output_mask.png)")
    parser.add_argument("--runs_root", default="runs",
                        help="Root directory for runs (default: runs)")
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Generate debug outputs (default: True)")
    parser.add_argument("--no_debug", action="store_true",
                        help="Disable debug outputs")
    parser.add_argument("--config", help="Path to config JSON file")

    args = parser.parse_args()

    route(
        components_path=args.components_path,
        mask_path=args.mask,
        debug=not args.no_debug,
        config_path=args.config,
        runs_root=args.runs_root,
    )


if __name__ == "__main__":
    main()
