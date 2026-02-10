#!/usr/bin/env python3
"""
Stage 10: Label Placement & SVG Overlay

Assigns numeric labels (1..N) to each component, renders leader lines
with circled numbers, and composites this label layer onto the existing
output.svg from Stage 7.

Algorithm:
    1. Load leader data from Stage 9 (anchors, label endpoints, sides).
    2. Assign numbers 1..N in reading order (top-to-bottom, left-to-right).
    3. Build an SVG layer group containing:
       - Leader lines (thin black lines from anchor to label endpoint)
       - Small filled dots at each anchor point
       - Circled numbers at each label endpoint
    4. Parse the Stage 7 output.svg and insert the label layer as a new
       top-level <g> group.
    5. Adjust the SVG viewBox if the canvas was extended for labels.
    6. Write the composite labelled.svg.
    7. Render a raster preview for quick inspection.

Usage:
    python label_place.py runs/<run>/90_label_leaders/out/leaders.json --debug
    python label_place.py runs/<run>/90_label_leaders/out/leaders.json \\
        --svg runs/<run>/70_svg/out/output.svg --debug

Outputs to runs/<run>/100_label_place/out/:
    - labelled.svg: Final SVG with labels overlaid
    - labelled_preview.png: Raster preview
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.artifacts import StageArtifacts


# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Label numbering start
    "start_number": 1,

    # Leader line style
    "leader_stroke_color": "#000000",
    "leader_stroke_width": 0.6,

    # Anchor dot
    "anchor_dot_radius": 2.5,
    "anchor_dot_fill": "#000000",

    # Label circle
    "label_circle_radius": 10,
    "label_circle_fill": "white",
    "label_circle_stroke": "#000000",
    "label_circle_stroke_width": 0.8,

    # Label text
    "label_font_size": 10,
    "label_font_family": "Arial, Helvetica, sans-serif",
    "label_text_fill": "#000000",
    "label_font_weight": "normal",

    # Layer name
    "label_layer_name": "labels",

    # Circle radius auto-scaling: if label number >= 10, enlarge circle
    "auto_scale_circle": True,

    # SVG float precision
    "float_precision": 2,

    # Preview rendering
    "preview_bg_color": [255, 255, 255],
    "preview_leader_color": [80, 80, 80],
    "preview_label_bg_color": [255, 255, 255],
    "preview_label_text_color": [0, 0, 0],

    # Offset text nudge for centering (depends on font)
    "text_dy_offset": "0.35em",
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


def fmt(value: float, precision: int = 2) -> str:
    if abs(value) < 1e-10:
        return "0"
    formatted = f"{value:.{precision}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


# ---------------------------------------------------------------------------
# SVG Label Layer Construction
# ---------------------------------------------------------------------------

def build_label_layer(
    leaders: List[dict],
    config: dict,
    start_number: int = 1,
) -> ET.Element:
    """
    Build the SVG <g> element containing all leader lines and labels.

    Args:
        leaders:  List of leader dicts from Stage 9.
        config:   Configuration dict.
        start_number: First label number.

    Returns:
        ET.Element <g> with id="layer_labels"
    """
    prec = config["float_precision"]
    layer = ET.Element("g")
    layer.set("id", f"layer_{config['label_layer_name']}")

    # Sub-groups for organization
    lines_g = ET.SubElement(layer, "g")
    lines_g.set("id", "label_leaders")

    anchors_g = ET.SubElement(layer, "g")
    anchors_g.set("id", "label_anchors")

    circles_g = ET.SubElement(layer, "g")
    circles_g.set("id", "label_circles")

    texts_g = ET.SubElement(layer, "g")
    texts_g.set("id", "label_texts")

    for i, ldr in enumerate(leaders):
        num = start_number + i
        ax = ldr["anchor_x"]
        ay = ldr["anchor_y"]
        lx = ldr["label_x"]
        ly = ldr["label_y"]
        cid = ldr["component_id"]

        # --- Leader line ---
        line = ET.SubElement(lines_g, "line")
        line.set("x1", fmt(ax, prec))
        line.set("y1", fmt(ay, prec))
        line.set("x2", fmt(lx, prec))
        line.set("y2", fmt(ly, prec))
        line.set("stroke", config["leader_stroke_color"])
        line.set("stroke-width", str(config["leader_stroke_width"]))
        line.set("fill", "none")
        line.set("data-component-id", str(cid))

        # --- Anchor dot ---
        dot = ET.SubElement(anchors_g, "circle")
        dot.set("cx", fmt(ax, prec))
        dot.set("cy", fmt(ay, prec))
        dot.set("r", fmt(config["anchor_dot_radius"], prec))
        dot.set("fill", config["anchor_dot_fill"])
        dot.set("data-component-id", str(cid))

        # --- Label circle ---
        radius = config["label_circle_radius"]
        if config["auto_scale_circle"] and num >= 100:
            radius = radius * 1.4
        elif config["auto_scale_circle"] and num >= 10:
            radius = radius * 1.15

        circle = ET.SubElement(circles_g, "circle")
        circle.set("cx", fmt(lx, prec))
        circle.set("cy", fmt(ly, prec))
        circle.set("r", fmt(radius, prec))
        circle.set("fill", config["label_circle_fill"])
        circle.set("stroke", config["label_circle_stroke"])
        circle.set("stroke-width", str(config["label_circle_stroke_width"]))
        circle.set("data-component-id", str(cid))
        circle.set("data-label", str(num))

        # --- Label text ---
        text = ET.SubElement(texts_g, "text")
        text.set("x", fmt(lx, prec))
        text.set("y", fmt(ly, prec))
        text.set("text-anchor", "middle")
        text.set("dominant-baseline", "central")
        text.set("dy", config["text_dy_offset"])
        text.set("font-family", config["label_font_family"])
        text.set("font-size", str(config["label_font_size"]))
        text.set("font-weight", config["label_font_weight"])
        text.set("fill", config["label_text_fill"])
        text.set("data-component-id", str(cid))
        text.text = str(num)

    return layer


# ---------------------------------------------------------------------------
# SVG Composition
# ---------------------------------------------------------------------------

def composite_svg(
    base_svg_path: str,
    label_layer: ET.Element,
    canvas_w: int,
    canvas_h: int,
    offset_x: int,
    offset_y: int,
    config: dict,
) -> ET.Element:
    """
    Load the base SVG, shift its content if needed, and append the label layer.

    Args:
        base_svg_path:  Path to the Stage 7 output.svg
        label_layer:    The <g> element with labels
        canvas_w, canvas_h: New canvas dimensions
        offset_x, offset_y: Shift to apply to existing content
        config:         Configuration dict

    Returns:
        New root SVG element with labels.
    """
    # Register SVG namespace to avoid ns0: prefix
    ET.register_namespace("", "http://www.w3.org/2000/svg")

    tree = ET.parse(base_svg_path)
    root = tree.getroot()

    # Get original dimensions
    orig_w = int(root.get("width", canvas_w))
    orig_h = int(root.get("height", canvas_h))

    # If canvas changed, update viewBox and dimensions
    if canvas_w != orig_w or canvas_h != orig_h:
        root.set("width", str(canvas_w))
        root.set("height", str(canvas_h))
        root.set("viewBox", f"0 0 {canvas_w} {canvas_h}")

        # Wrap existing content in a group with a translate transform
        # to shift it by the offset
        if offset_x > 0 or offset_y > 0:
            wrapper = ET.Element("g")
            wrapper.set("transform", f"translate({offset_x},{offset_y})")
            wrapper.set("id", "original_content")

            # Move all children to wrapper
            children = list(root)
            for child in children:
                root.remove(child)
                wrapper.append(child)
            root.append(wrapper)

            # Update background rect if present
            bg_rects = wrapper.findall(".//{http://www.w3.org/2000/svg}rect")
            if not bg_rects:
                bg_rects = wrapper.findall(".//rect")
            for rect in bg_rects:
                if rect.get("width") == "100%" and rect.get("height") == "100%":
                    # Make it cover the full new canvas, positioned at origin
                    wrapper.remove(rect)
                    root.insert(0, rect)
                    break
    else:
        # No canvas change needed — if there's an offset, still wrap
        if offset_x > 0 or offset_y > 0:
            wrapper = ET.Element("g")
            wrapper.set("transform", f"translate({offset_x},{offset_y})")
            wrapper.set("id", "original_content")
            children = list(root)
            for child in children:
                root.remove(child)
                wrapper.append(child)
            root.append(wrapper)

    # Append label layer
    root.append(label_layer)

    return root


def build_standalone_label_svg(
    label_layer: ET.Element,
    canvas_w: int,
    canvas_h: int,
    config: dict,
) -> ET.Element:
    """Build a standalone SVG containing only the label layer (for cases where base SVG is unavailable)."""
    root = ET.Element("svg")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("width", str(canvas_w))
    root.set("height", str(canvas_h))
    root.set("viewBox", f"0 0 {canvas_w} {canvas_h}")

    # White background
    bg = ET.SubElement(root, "rect")
    bg.set("width", "100%")
    bg.set("height", "100%")
    bg.set("fill", "white")

    root.append(label_layer)
    return root


def write_svg_file(root: ET.Element, path: Path):
    """Write SVG with proper indentation."""
    _indent_xml(root)
    tree = ET.ElementTree(root)
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="unicode", xml_declaration=False)


def _indent_xml(elem: ET.Element, level: int = 0):
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


# ---------------------------------------------------------------------------
# Preview Rendering
# ---------------------------------------------------------------------------

def render_labelled_preview(
    mask: Optional[np.ndarray],
    leader_data: dict,
    config: dict,
    start_number: int = 1,
) -> np.ndarray:
    """
    Render a raster preview of the labelled diagram.
    """
    canvas_w = leader_data["canvas"]["width"]
    canvas_h = leader_data["canvas"]["height"]
    x_off = leader_data["offset"]["x"]
    y_off = leader_data["offset"]["y"]
    leaders = leader_data["leaders"]

    bg = tuple(config["preview_bg_color"])
    vis = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)

    # Draw the mask (shifted)
    if mask is not None:
        h, w = mask.shape[:2]
        # Draw strokes in black
        stroke_px = mask == 255
        region = vis[y_off:y_off + h, x_off:x_off + w]
        region[stroke_px] = (0, 0, 0)

    leader_color = tuple(config["preview_leader_color"])
    label_bg = tuple(config["preview_label_bg_color"])
    text_color = tuple(config["preview_label_text_color"])

    for i, ldr in enumerate(leaders):
        num = start_number + i
        ax, ay = ldr["anchor_x"], ldr["anchor_y"]
        lx, ly = ldr["label_x"], ldr["label_y"]

        # Leader line
        cv2.line(vis, (ax, ay), (lx, ly), leader_color, 1, cv2.LINE_AA)

        # Anchor dot
        cv2.circle(vis, (ax, ay), 3, (0, 0, 0), -1, cv2.LINE_AA)

        # Label circle background
        r = int(config["label_circle_radius"])
        if config["auto_scale_circle"] and num >= 100:
            r = int(r * 1.4)
        elif config["auto_scale_circle"] and num >= 10:
            r = int(r * 1.15)

        cv2.circle(vis, (lx, ly), r, label_bg, -1, cv2.LINE_AA)
        cv2.circle(vis, (lx, ly), r, (0, 0, 0), 1, cv2.LINE_AA)

        # Label number text
        txt = str(num)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.35 if num < 10 else 0.30 if num < 100 else 0.25
        (tw, th), _ = cv2.getTextSize(txt, font, scale, 1)
        cv2.putText(
            vis, txt,
            (lx - tw // 2, ly + th // 2),
            font, scale, text_color, 1, cv2.LINE_AA,
        )

    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def place_labels(
    leaders_path: str,
    svg_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    debug: bool = True,
    config_path: Optional[str] = None,
    runs_root: str = "runs",
):
    """Main entry point for Stage 10."""
    print(f"Processing: {leaders_path}")

    config = load_config(config_path)

    # Load leader data
    with open(leaders_path) as f:
        leader_data = json.load(f)

    leaders = leader_data["leaders"]
    img_w = leader_data["image"]["width"]
    img_h = leader_data["image"]["height"]
    canvas_w = leader_data["canvas"]["width"]
    canvas_h = leader_data["canvas"]["height"]
    x_off = leader_data["offset"]["x"]
    y_off = leader_data["offset"]["y"]

    print(f"Image: {img_w}x{img_h}  Canvas: {canvas_w}x{canvas_h}")
    print(f"Leaders: {len(leaders)}")

    # Infer run dir
    run_dir = infer_run_dir(leaders_path, runs_root)
    print(f"Run directory: {run_dir}")

    # Find SVG path
    if svg_path is None:
        if "svg_path" in leader_data:
            svg_path = leader_data["svg_path"]
        else:
            candidate = run_dir / "70_svg" / "out" / "output.svg"
            if candidate.exists():
                svg_path = str(candidate)

    # Find mask path
    if mask_path is None:
        candidate = run_dir / "10_preprocess" / "out" / "output_mask.png"
        if candidate.exists():
            mask_path = str(candidate)

    mask = None
    if mask_path:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Init artifacts
    artifacts = StageArtifacts(run_dir, stage_id=100, stage_name="label_place", debug=debug)

    # Build label layer
    start_num = config["start_number"]
    label_layer = build_label_layer(leaders, config, start_number=start_num)

    # Composite with base SVG
    if svg_path and Path(svg_path).exists():
        print(f"Base SVG: {svg_path}")
        composite_root = composite_svg(
            svg_path, label_layer,
            canvas_w, canvas_h, x_off, y_off, config,
        )
        svg_out_path = artifacts.out_dir / "labelled.svg"
        write_svg_file(composite_root, svg_out_path)
        print(f"Labelled SVG: {svg_out_path}")
    else:
        print("Warning: No base SVG found, creating standalone label SVG")
        standalone = build_standalone_label_svg(label_layer, canvas_w, canvas_h, config)
        svg_out_path = artifacts.out_dir / "labelled.svg"
        write_svg_file(standalone, svg_out_path)
        print(f"Standalone label SVG: {svg_out_path}")

    # Raster preview
    preview = render_labelled_preview(mask, leader_data, config, start_num)
    artifacts.save_output_image("labelled_preview", preview)

    # Debug outputs
    if debug:
        artifacts.save_debug_image("labelled_preview", preview)

        # Labels-only on white background (no strokes)
        labels_only = render_labelled_preview(None, leader_data, config, start_num)
        artifacts.save_debug_image("labels_only", labels_only)

    # Metrics
    artifacts.write_metrics({
        "num_labels": len(leaders),
        "start_number": start_num,
        "end_number": start_num + len(leaders) - 1 if leaders else start_num,
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "base_svg_used": svg_path is not None and Path(svg_path).exists(),
    })

    # Print summary
    for i, ldr in enumerate(leaders):
        num = start_num + i
        print(f"  Label {num:3d} → Component {ldr['component_id']}  "
              f"at ({ldr['label_x']},{ldr['label_y']}) {ldr['side']}")

    return svg_out_path


def main():
    parser = argparse.ArgumentParser(
        description="Stage 10: Place numeric labels and emit labelled SVG"
    )
    parser.add_argument("leaders_path", help="Path to leaders.json from Stage 9")
    parser.add_argument("--svg", default=None,
                        help="Path to output.svg from Stage 7")
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

    place_labels(
        leaders_path=args.leaders_path,
        svg_path=args.svg,
        mask_path=args.mask,
        debug=not args.no_debug,
        config_path=args.config,
        runs_root=args.runs_root,
    )


if __name__ == "__main__":
    main()
