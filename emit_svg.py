#!/usr/bin/env python3
"""
Stage 7: SVG Emission

Emits final editable SVG from Stage 6.5 regularized primitives. Pure serialization 
without re-fitting. Optimized for editability in Illustrator, Figma, Inkscape.

Usage:
    python emit_svg.py runs/<run>/65_regularize/out/primitives_regularized.json --debug
    python emit_svg.py runs/<run>/65_regularize/out/primitives_regularized.json --debug --mask runs/<run>/10_preprocess/out/output_mask.png
    python emit_svg.py runs/<run>/65_regularize/out/primitives_regularized.json --debug --config configs/svg.json

Outputs to runs/<run>/70_svg/out/:
    - output.svg: Final editable SVG
    - preview.png: Raster preview of SVG
    - overlay_preview.png: SVG preview over mask/input for alignment check
    - metrics.json: Emission summary and warnings
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from xml.etree import ElementTree as ET

import cv2
import numpy as np

# Add parent to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.artifacts import StageArtifacts


# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Stroke styling
    "stroke_color": "#000000",
    "stroke_width": 1.5,
    "stroke_linecap": "round",
    "stroke_linejoin": "round",
    "background": "white",  # None for transparent, "white" for white bg
    
    # Layer naming
    "structural_layer_name": "structure",
    "detail_layer_name": "detail",
    "include_nodes_layer": False,
    "group_by_type": True,
    
    # Polyline safety
    "polyline_max_points": 2000,
    
    # Float formatting
    "float_precision": 3,
    
    # Arc handling
    "arc_split_if_large": True,
    "arc_max_sweep_deg": 175.0,
    
    # Preview rendering
    "preview_scale": 1.0,
    "preview_stroke_scale": 1.0,
    "preview_bg_color": [255, 255, 255],
    "overlay_alpha": 0.45,
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> dict:
    """Load config from JSON and merge with defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                user_config = json.load(f)
            config.update(user_config)
    return config


def infer_run_dir(primitives_path: str, runs_root: str) -> Path:
    """Infer run directory from primitives path."""
    prim_p = Path(primitives_path).resolve()
    runs_root_p = Path(runs_root).resolve()
    try:
        rel_path = prim_p.relative_to(runs_root_p)
        run_name = rel_path.parts[0]
        return runs_root_p / run_name
    except ValueError:
        # Fallback: go up from primitives path
        return prim_p.parent.parent.parent


def find_original_input(run_dir: Path) -> Optional[Path]:
    """Try to locate original input image."""
    input_dir = run_dir / "00_input"
    if not input_dir.exists():
        return None
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = input_dir / f"01_input{ext}"
        if candidate.exists():
            return candidate
    return None


def fmt(value: float, precision: int = 3) -> str:
    """Format float with fixed precision, avoiding scientific notation."""
    if abs(value) < 1e-10:
        return "0"
    formatted = f"{value:.{precision}f}"
    # Remove trailing zeros after decimal point
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.')
    return formatted


# ---------------------------------------------------------------------------
# Arc Conversion: Primitive -> SVG Arc Command
# ---------------------------------------------------------------------------

def compute_arc_endpoint(center: Tuple[float, float], radius: float, theta: float) -> Tuple[float, float]:
    """Compute point on arc given center, radius, and angle."""
    cx, cy = center
    return (cx + radius * math.cos(theta), cy + radius * math.sin(theta))


def compute_arc_sweep(theta0: float, theta1: float, cw: bool) -> float:
    """
    Compute the sweep angle from theta0 to theta1 following cw direction.
    Returns value in (-2*pi, 2*pi) with correct sign.
    """
    # Normalize angles to [0, 2*pi)
    def normalize(a):
        while a < 0:
            a += 2 * math.pi
        while a >= 2 * math.pi:
            a -= 2 * math.pi
        return a
    
    t0 = normalize(theta0)
    t1 = normalize(theta1)
    
    if cw:
        # Clockwise: we go from t0 to t1 in negative direction
        if t1 > t0:
            sweep = -(t0 + 2 * math.pi - t1)
        else:
            sweep = -(t0 - t1)
        # Ensure negative
        if sweep > 0:
            sweep -= 2 * math.pi
    else:
        # Counter-clockwise: positive direction
        if t1 < t0:
            sweep = (2 * math.pi - t0) + t1
        else:
            sweep = t1 - t0
        # Ensure positive
        if sweep < 0:
            sweep += 2 * math.pi
    
    return sweep


def arc_to_svg_path(arc: dict, precision: int, split_threshold_deg: float = 175.0, 
                   do_split: bool = True) -> Tuple[str, int]:
    """
    Convert arc primitive to SVG path d attribute.
    
    Args:
        arc: Arc primitive dict with center, radius, theta0, theta1, cw
        precision: Float precision for formatting
        split_threshold_deg: Max sweep in degrees before splitting
        do_split: Whether to split large arcs
        
    Returns:
        Tuple of (path_d_string, number_of_splits)
    """
    center = tuple(arc["center"])
    radius = arc["radius"]
    theta0 = arc["theta0"]
    theta1 = arc["theta1"]
    cw = arc["cw"]
    
    # Compute sweep
    sweep = compute_arc_sweep(theta0, theta1, cw)
    abs_sweep = abs(sweep)
    sweep_deg = math.degrees(abs_sweep)
    
    # Determine if we need to split
    n_segments = 1
    if do_split and sweep_deg > split_threshold_deg:
        n_segments = math.ceil(sweep_deg / split_threshold_deg)
    
    # Generate arc segments
    segment_sweep = sweep / n_segments
    
    # SVG flags
    # large-arc-flag: 1 if sweep > 180 degrees per segment
    # sweep-flag: In SVG with y-down coords, sweep=1 means visually clockwise on screen
    # Our cw is computed using arctan2 in standard math coords (y-up):
    #   - cw=True means angles decrease, which is CW in y-up coords
    #   - But in y-down screen coords, this appears as CCW visually
    # So we need to INVERT the mapping:
    svg_sweep_flag = 0 if cw else 1

    
    # Build path
    parts = []
    current_theta = theta0
    
    for i in range(n_segments):
        # Start point
        if i == 0:
            start = compute_arc_endpoint(center, radius, current_theta)
            parts.append(f"M {fmt(start[0], precision)} {fmt(start[1], precision)}")
        
        # End of this segment
        next_theta = current_theta + segment_sweep
        end = compute_arc_endpoint(center, radius, next_theta)
        
        # Large arc flag for this segment
        segment_abs_sweep = abs(segment_sweep)
        large_arc = 1 if segment_abs_sweep > math.pi else 0
        
        # A rx ry x-rotation large-arc sweep x y
        parts.append(
            f"A {fmt(radius, precision)} {fmt(radius, precision)} 0 "
            f"{large_arc} {svg_sweep_flag} {fmt(end[0], precision)} {fmt(end[1], precision)}"
        )
        
        current_theta = next_theta
    
    return " ".join(parts), n_segments - 1


# ---------------------------------------------------------------------------
# SVG Building
# ---------------------------------------------------------------------------

def create_svg_element(prim: dict, edge_data: dict, config: dict) -> Optional[ET.Element]:
    """
    Create an SVG element from a primitive.
    
    Args:
        prim: The chosen primitive dict
        edge_data: Full edge data including edge_id, bucket, u, v
        config: Configuration dict
        
    Returns:
        ET.Element or None if skipped
    """
    precision = config["float_precision"]
    ptype = prim["type"]
    
    # Common attributes
    attrs = {
        "id": f"edge_{edge_data['edge_id']}_{ptype}",
        "data-edge-id": str(edge_data["edge_id"]),
        "data-bucket": edge_data["bucket"],
        "data-u": str(edge_data["u"]),
        "data-v": str(edge_data["v"]),
        "data-primitive-type": ptype,
        "stroke": config["stroke_color"],
        "stroke-width": str(config["stroke_width"]),
        "fill": "none",
        "stroke-linecap": config["stroke_linecap"],
        "stroke-linejoin": config["stroke_linejoin"],
    }
    
    if ptype == "line":
        elem = ET.Element("line")
        elem.set("x1", fmt(prim["p0"][0], precision))
        elem.set("y1", fmt(prim["p0"][1], precision))
        elem.set("x2", fmt(prim["p1"][0], precision))
        elem.set("y2", fmt(prim["p1"][1], precision))
        
    elif ptype == "polyline":
        points = prim["points"]
        # Safety: subsample if too many points
        max_pts = config["polyline_max_points"]
        if len(points) > max_pts:
            # Subsample keeping endpoints
            indices = np.linspace(0, len(points) - 1, max_pts, dtype=int)
            points = [points[i] for i in indices]
        
        pts_str = " ".join(f"{fmt(p[0], precision)},{fmt(p[1], precision)}" for p in points)
        elem = ET.Element("polyline")
        elem.set("points", pts_str)
        
    elif ptype == "cubic":
        elem = ET.Element("path")
        p0 = prim["p0"]
        p1 = prim["p1"]
        p2 = prim["p2"]
        p3 = prim["p3"]
        d = (f"M {fmt(p0[0], precision)} {fmt(p0[1], precision)} "
             f"C {fmt(p1[0], precision)} {fmt(p1[1], precision)}, "
             f"{fmt(p2[0], precision)} {fmt(p2[1], precision)}, "
             f"{fmt(p3[0], precision)} {fmt(p3[1], precision)}")
        elem.set("d", d)
        
    elif ptype == "arc":
        elem = ET.Element("path")
        d, _ = arc_to_svg_path(
            prim, precision,
            split_threshold_deg=config["arc_max_sweep_deg"],
            do_split=config["arc_split_if_large"]
        )
        elem.set("d", d)
        
    else:
        return None
    
    # Apply common attributes
    for k, v in attrs.items():
        elem.set(k, v)
    
    return elem


def build_svg_tree(primitives_data: dict, config: dict) -> Tuple[ET.Element, dict]:
    """
    Build the complete SVG element tree.
    
    Args:
        primitives_data: Loaded primitives.json data
        config: Configuration dict
        
    Returns:
        Tuple of (root SVG element, metrics dict)
    """
    width = primitives_data["image"]["width"]
    height = primitives_data["image"]["height"]
    
    # Create root SVG
    root = ET.Element("svg")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("width", str(width))
    root.set("height", str(height))
    root.set("viewBox", f"0 0 {width} {height}")
    root.set("shape-rendering", "geometricPrecision")
    
    # Optional background
    if config["background"]:
        bg = ET.SubElement(root, "rect")
        bg.set("width", "100%")
        bg.set("height", "100%")
        bg.set("fill", config["background"])
    
    # Create layer groups
    struct_layer = ET.SubElement(root, "g")
    struct_layer.set("id", f"layer_{config['structural_layer_name']}")
    
    detail_layer = ET.SubElement(root, "g")
    detail_layer.set("id", f"layer_{config['detail_layer_name']}")
    
    # Type subgroups if enabled
    type_groups = {}
    if config["group_by_type"]:
        for bucket, layer in [("structural", struct_layer), ("detail", detail_layer)]:
            type_groups[bucket] = {}
            for ptype in ["lines", "arcs", "cubics", "polylines"]:
                g = ET.SubElement(layer, "g")
                g.set("id", f"{bucket}_{ptype}")
                type_groups[bucket][ptype] = g
    
    # Metrics tracking
    metrics = {
        "total_emitted_elements": 0,
        "counts": {
            "structural": {"line": 0, "arc": 0, "cubic": 0, "polyline": 0},
            "detail": {"line": 0, "arc": 0, "cubic": 0, "polyline": 0},
        },
        "polylines_subsampled": 0,
        "arcs_split": 0,
        "warnings": [],
        "bbox": {"minx": float("inf"), "miny": float("inf"), 
                 "maxx": float("-inf"), "maxy": float("-inf")},
    }
    
    # Sort primitives by edge_id for deterministic output
    prims_sorted = sorted(primitives_data["primitives"], key=lambda x: x["edge_id"])
    
    for edge in prims_sorted:
        chosen = edge["chosen"]
        bucket = edge["bucket"]
        ptype = chosen["type"]
        
        # Check polyline subsampling
        if ptype == "polyline":
            if len(chosen["points"]) > config["polyline_max_points"]:
                metrics["polylines_subsampled"] += 1
                metrics["warnings"].append(
                    f"Edge {edge['edge_id']}: polyline subsampled from {len(chosen['points'])} to {config['polyline_max_points']} points"
                )
        
        # Check arc splitting
        if ptype == "arc":
            _, splits = arc_to_svg_path(
                chosen, config["float_precision"],
                split_threshold_deg=config["arc_max_sweep_deg"],
                do_split=config["arc_split_if_large"]
            )
            if splits > 0:
                metrics["arcs_split"] += 1
        
        # Create element
        elem = create_svg_element(chosen, edge, config)
        if elem is None:
            metrics["warnings"].append(f"Edge {edge['edge_id']}: unknown type '{ptype}'")
            continue
        
        # Add to appropriate group
        if config["group_by_type"]:
            ptype_plural = ptype + "s" if ptype != "polyline" else "polylines"
            type_groups[bucket][ptype_plural].append(elem)
        else:
            if bucket == "structural":
                struct_layer.append(elem)
            else:
                detail_layer.append(elem)
        
        # Update metrics
        metrics["total_emitted_elements"] += 1
        metrics["counts"][bucket][ptype] += 1
        
        # Update bbox
        bbox = get_primitive_bbox(chosen)
        if bbox:
            metrics["bbox"]["minx"] = min(metrics["bbox"]["minx"], bbox[0])
            metrics["bbox"]["miny"] = min(metrics["bbox"]["miny"], bbox[1])
            metrics["bbox"]["maxx"] = max(metrics["bbox"]["maxx"], bbox[2])
            metrics["bbox"]["maxy"] = max(metrics["bbox"]["maxy"], bbox[3])
    
    # Add nodes layer if requested
    if config["include_nodes_layer"] and "nodes" in primitives_data:
        nodes_layer = ET.SubElement(root, "g")
        nodes_layer.set("id", "layer_nodes")
        for node in primitives_data["nodes"]:
            cx, cy = node["centroid"]
            circle = ET.SubElement(nodes_layer, "circle")
            circle.set("cx", fmt(cx, config["float_precision"]))
            circle.set("cy", fmt(cy, config["float_precision"]))
            circle.set("r", "2")
            circle.set("fill", "#ff0000")
            circle.set("data-node-id", str(node["id"]))
    
    # Finalize bbox
    if metrics["bbox"]["minx"] == float("inf"):
        metrics["bbox"] = {"minx": 0, "miny": 0, "maxx": width, "maxy": height}
    
    return root, metrics


def get_primitive_bbox(prim: dict) -> Optional[Tuple[float, float, float, float]]:
    """Get bounding box of a primitive."""
    ptype = prim["type"]
    
    if ptype == "line":
        xs = [prim["p0"][0], prim["p1"][0]]
        ys = [prim["p0"][1], prim["p1"][1]]
        
    elif ptype == "polyline":
        pts = prim["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
    elif ptype == "cubic":
        # Approximate with control points
        xs = [prim["p0"][0], prim["p1"][0], prim["p2"][0], prim["p3"][0]]
        ys = [prim["p0"][1], prim["p1"][1], prim["p2"][1], prim["p3"][1]]
        
    elif ptype == "arc":
        # Approximate with endpoints and center +/- radius
        cx, cy = prim["center"]
        r = prim["radius"]
        p0 = compute_arc_endpoint((cx, cy), r, prim["theta0"])
        p1 = compute_arc_endpoint((cx, cy), r, prim["theta1"])
        # Conservative: include circle bbox
        xs = [p0[0], p1[0], cx - r, cx + r]
        ys = [p0[1], p1[1], cy - r, cy + r]
        
    else:
        return None
    
    return (min(xs), min(ys), max(xs), max(ys))


def write_svg_file(root: ET.Element, path: Path):
    """Write SVG element tree to file with proper declaration."""
    # Pretty print with indentation
    indent_xml(root)
    
    tree = ET.ElementTree(root)
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="unicode", xml_declaration=False)


def indent_xml(elem: ET.Element, level: int = 0):
    """Add indentation to XML element tree."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


# ---------------------------------------------------------------------------
# Preview Rendering (using cv2, not SVG rasterization)
# ---------------------------------------------------------------------------

def sample_bezier(p0, p1, p2, p3, n_pts: int = 100) -> np.ndarray:
    """Sample points along cubic Bezier curve."""
    t = np.linspace(0, 1, n_pts)
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt
    
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    points = (mt3[:, None] * p0 + 
              3 * mt2[:, None] * t[:, None] * p1 + 
              3 * mt[:, None] * t2[:, None] * p2 + 
              t3[:, None] * p3)
    return points.astype(np.int32)


def sample_arc(center, radius, theta0, theta1, cw, n_pts: int = 50) -> np.ndarray:
    """Sample points along arc."""
    sweep = compute_arc_sweep(theta0, theta1, cw)
    angles = np.linspace(theta0, theta0 + sweep, n_pts)
    
    cx, cy = center
    points = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles)
    ]).astype(np.int32)
    return points


def render_primitives(primitives: List[dict], width: int, height: int, 
                      config: dict, bg_color: Tuple[int, int, int] = (255, 255, 255),
                      stroke_color: Tuple[int, int, int] = (0, 0, 0),
                      stroke_width: int = 1,
                      filter_bucket: Optional[str] = None) -> np.ndarray:
    """
    Render primitives to an image using cv2.
    
    Args:
        primitives: List of edge dicts with 'chosen' primitive
        width, height: Image size
        config: Configuration dict
        bg_color: Background color
        stroke_color: Stroke color
        stroke_width: Line thickness
        filter_bucket: If set, only render primitives with this bucket
        
    Returns:
        Rendered image
    """
    scale = config["preview_scale"]
    sw = int(max(1, stroke_width * config["preview_stroke_scale"]))
    
    img = np.full((int(height * scale), int(width * scale), 3), bg_color, dtype=np.uint8)
    
    for edge in primitives:
        if filter_bucket and edge["bucket"] != filter_bucket:
            continue
            
        prim = edge["chosen"]
        ptype = prim["type"]
        
        if ptype == "line":
            p0 = (int(prim["p0"][0] * scale), int(prim["p0"][1] * scale))
            p1 = (int(prim["p1"][0] * scale), int(prim["p1"][1] * scale))
            cv2.line(img, p0, p1, stroke_color, sw, cv2.LINE_AA)
            
        elif ptype == "polyline":
            pts = np.array(prim["points"], dtype=np.float64) * scale
            pts = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, stroke_color, sw, cv2.LINE_AA)
            
        elif ptype == "cubic":
            pts = sample_bezier(prim["p0"], prim["p1"], prim["p2"], prim["p3"])
            pts = (pts * scale).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, stroke_color, sw, cv2.LINE_AA)
            
        elif ptype == "arc":
            pts = sample_arc(
                prim["center"], prim["radius"],
                prim["theta0"], prim["theta1"], prim["cw"]
            )
            pts = (pts * scale).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, stroke_color, sw, cv2.LINE_AA)
    
    return img


def render_overlay(preview: np.ndarray, background: np.ndarray, 
                   alpha: float = 0.45) -> np.ndarray:
    """
    Overlay preview on background image.
    
    Args:
        preview: Preview rendering (colored strokes on white/black)
        background: Background image (mask or original)
        alpha: Blend alpha for background
        
    Returns:
        Blended image
    """
    # Resize background to match preview if needed
    if background.shape[:2] != preview.shape[:2]:
        background = cv2.resize(background, (preview.shape[1], preview.shape[0]))
    
    # Ensure 3 channels
    if len(background.shape) == 2:
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    if len(preview.shape) == 2:
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    
    # Create overlay: strokes (dark pixels) over semi-transparent background
    # Detect stroke pixels (significantly darker than white background)
    stroke_mask = np.any(preview < 200, axis=2)
    
    # Blend background with alpha
    result = (background.astype(np.float32) * alpha + 
              255 * (1 - alpha)).astype(np.uint8)
    
    # Overlay strokes in color
    stroke_color = (0, 0, 255)  # Red for visibility
    result[stroke_mask] = stroke_color
    
    return result


# ---------------------------------------------------------------------------
# Debug Visualization
# ---------------------------------------------------------------------------

def render_arc_debug(primitives: List[dict], width: int, height: int,
                     config: dict, every_n: int = 5) -> np.ndarray:
    """Render arc debug visualization with centers and radii."""
    img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    
    arcs = [(edge, edge["chosen"]) for edge in primitives 
            if edge["chosen"]["type"] == "arc"]
    
    if not arcs:
        cv2.putText(img, "No arcs", (width // 2 - 50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        return img
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), 
              (128, 0, 128), (0, 128, 128)]
    
    for i, (edge, arc) in enumerate(arcs[::every_n]):
        color = colors[i % len(colors)]
        center = (int(arc["center"][0]), int(arc["center"][1]))
        radius = arc["radius"]
        
        # Draw arc
        pts = sample_arc(arc["center"], radius, arc["theta0"], arc["theta1"], arc["cw"])
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
        
        # Draw center
        cv2.circle(img, center, 4, color, -1)
        cv2.circle(img, center, int(radius), color, 1)
        
        # Label
        label = f"e{edge['edge_id']} r={int(radius)}"
        cv2.putText(img, label, (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return img


def render_id_sanity(primitives: List[dict], width: int, height: int,
                     config: dict, top_n: int = 30) -> np.ndarray:
    """Render edge IDs at midpoints for top N longest structural edges."""
    img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    
    # Filter structural and sort by length
    structural = [e for e in primitives if e["bucket"] == "structural"]
    structural.sort(key=lambda x: x["length_px"], reverse=True)
    
    for edge in structural[:top_n]:
        prim = edge["chosen"]
        ptype = prim["type"]
        
        # Find midpoint
        if ptype == "line":
            mid = ((prim["p0"][0] + prim["p1"][0]) / 2,
                   (prim["p0"][1] + prim["p1"][1]) / 2)
        elif ptype == "polyline":
            pts = prim["points"]
            mid_idx = len(pts) // 2
            mid = pts[mid_idx]
        elif ptype == "cubic":
            # Sample at t=0.5
            mid = sample_bezier(prim["p0"], prim["p1"], prim["p2"], prim["p3"], 3)[1]
        elif ptype == "arc":
            pts = sample_arc(prim["center"], prim["radius"],
                            prim["theta0"], prim["theta1"], prim["cw"], 3)
            mid = pts[1]
        else:
            continue
        
        # Draw primitive
        if ptype == "line":
            p0 = (int(prim["p0"][0]), int(prim["p0"][1]))
            p1 = (int(prim["p1"][0]), int(prim["p1"][1]))
            cv2.line(img, p0, p1, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Draw ID at midpoint
        mid_pt = (int(mid[0]), int(mid[1]))
        cv2.putText(img, str(edge["edge_id"]), mid_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    return img


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def emit_svg(primitives_path: str, mask_path: Optional[str] = None,
             input_image_path: Optional[str] = None, debug: bool = True,
             config_path: Optional[str] = None, runs_root: str = "runs"):
    """
    Main entry point for Stage 7: SVG Emission.
    
    Args:
        primitives_path: Path to primitives.json from Stage 6
        mask_path: Optional path to preprocess mask for overlay
        input_image_path: Optional path to original input image
        debug: Whether to generate debug outputs
        config_path: Optional path to config JSON
        runs_root: Root directory for runs
    """
    print(f"Processing: {primitives_path}")
    
    # Load config
    config = load_config(config_path)
    
    # Load primitives
    with open(primitives_path) as f:
        primitives_data = json.load(f)
    
    width = primitives_data["image"]["width"]
    height = primitives_data["image"]["height"]
    primitives = primitives_data["primitives"]
    
    print(f"Image size: {width}x{height}")
    print(f"Primitives: {len(primitives)}")
    
    # Infer run directory
    run_dir = infer_run_dir(primitives_path, runs_root)
    print(f"Run directory: {run_dir}")
    
    # Initialize artifacts
    artifacts = StageArtifacts(run_dir, stage_id=70, stage_name="svg", debug=debug)
    
    # Build SVG tree
    svg_root, metrics = build_svg_tree(primitives_data, config)
    
    # Write SVG
    svg_path = artifacts.out_dir / "output.svg"
    write_svg_file(svg_root, svg_path)
    print(f"SVG: {svg_path}")
    
    # Render preview
    preview = render_primitives(primitives, width, height, config,
                                 bg_color=tuple(config["preview_bg_color"]),
                                 stroke_width=int(config["stroke_width"]))
    artifacts.save_output_image("preview", preview)
    
    # Load background for overlay
    background = None
    if mask_path:
        background = cv2.imread(str(mask_path))
    elif input_image_path:
        background = cv2.imread(str(input_image_path))
    else:
        # Try to find original input
        input_found = find_original_input(run_dir)
        if input_found:
            background = cv2.imread(str(input_found))
    
    # Render overlay
    if background is not None:
        overlay = render_overlay(preview, background, config["overlay_alpha"])
        artifacts.save_output_image("overlay_preview", overlay)
    else:
        # Just save preview as overlay too
        artifacts.save_output_image("overlay_preview", preview)
    
    # Debug outputs
    if debug:
        # 01: Preview (same as output)
        artifacts.save_debug_image("preview", preview)
        
        # 02: Overlay preview
        if background is not None:
            artifacts.save_debug_image("overlay_preview", overlay)
        else:
            artifacts.save_debug_image("overlay_preview", preview)
        
        # 03: Structure only
        struct_only = render_primitives(primitives, width, height, config,
                                         filter_bucket="structural")
        artifacts.save_debug_image("layer_structure_only", struct_only)
        
        # 04: Detail only
        detail_only = render_primitives(primitives, width, height, config,
                                         filter_bucket="detail")
        artifacts.save_debug_image("layer_detail_only", detail_only)
        
        # 05: Arc debug
        arc_debug = render_arc_debug(primitives, width, height, config)
        artifacts.save_debug_image("arc_debug", arc_debug)
        
        # 06: ID sanity
        id_sanity = render_id_sanity(primitives, width, height, config)
        artifacts.save_debug_image("id_sanity", id_sanity)
    
    # Write metrics
    metrics["arc_count"] = sum(
        1 for e in primitives if e["chosen"]["type"] == "arc"
    )
    artifacts.write_metrics(metrics)
    
    # Print summary
    print(f"Emitted: {metrics['total_emitted_elements']} elements")
    print(f"  Structural: {sum(metrics['counts']['structural'].values())}")
    print(f"  Detail: {sum(metrics['counts']['detail'].values())}")
    if metrics["warnings"]:
        print(f"Warnings: {len(metrics['warnings'])}")
    
    return svg_path, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Stage 7: Emit editable SVG from primitives.json"
    )
    parser.add_argument("primitives_path", help="Path to primitives.json from Stage 6")
    parser.add_argument("--runs_root", default="runs", 
                        help="Root directory for runs (default: runs)")
    parser.add_argument("--mask", help="Path to preprocess output_mask.png for overlay")
    parser.add_argument("--input_image", help="Path to original input image for overlay")
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Generate debug outputs (default: True)")
    parser.add_argument("--no_debug", action="store_true",
                        help="Disable debug outputs")
    parser.add_argument("--config", help="Path to config JSON file")
    
    args = parser.parse_args()
    
    emit_svg(
        primitives_path=args.primitives_path,
        mask_path=args.mask,
        input_image_path=args.input_image,
        debug=not args.no_debug,
        config_path=args.config,
        runs_root=args.runs_root
    )


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Notes for Debug Artifacts
# ---------------------------------------------------------------------------
# 
# out/output.svg and debug/01_preview.png -> strokes should be identical.
# 
# debug/02_overlay_preview.png
#    - Lines/arcs should align with original strokes
#    - No global offset or scaling mismatch
#    - Red strokes should follow black strokes in background
#
# debug/03_layer_structure_only.png
#    - Should show main geometry clearly (longest edges)
#    - No detail/hatching visible
#
# debug/04_layer_detail_only.png
#    - Should mostly be hatching and small features
#    - No main structure visible
#
# If SVG is opened in Illustrator/Figma/Inkscape:
#    - Each edge will have a separate selectable element
#    - Layers will be grouped correctly
#    - IDs will match edge_id from primitives.json