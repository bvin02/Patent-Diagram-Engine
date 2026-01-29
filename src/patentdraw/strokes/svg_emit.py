"""
SVG emission for Patent Draw.

Generates SVG documents from strokes with proper styling for patent drawings.
"""

import svgwrite

from patentdraw.io.save_artifacts import save_svg, render_svg_to_png
from patentdraw.models import Stroke
from patentdraw.strokes.bezier_fit import bezier_to_svg_path
from patentdraw.tracer import get_tracer, trace


@trace(label="emit_strokes_svg")
def emit_strokes_svg(strokes, width, height, stroke_width=1.5, stroke_color="black"):
    """
    Create an SVG document containing all strokes.
    
    Args:
        strokes: list of Stroke objects
        width: image width in pixels
        height: image height in pixels
        stroke_width: line width
        stroke_color: stroke color (should be "black" for patent drawings)
    
    Returns:
        svgwrite.Drawing object
    """
    tracer = get_tracer()
    
    dwg = svgwrite.Drawing(size=(f"{width}px", f"{height}px"))
    dwg.viewbox(0, 0, width, height)
    
    # Add patent drawing style
    dwg.defs.add(dwg.style("""
        .stroke { stroke-linecap: round; stroke-linejoin: round; }
    """))
    
    # Group for all strokes
    stroke_group = dwg.g(id="strokes", fill="none", stroke=stroke_color, 
                         stroke_width=stroke_width, class_="stroke")
    
    for stroke in strokes:
        if stroke.svg_path:
            path = dwg.path(d=stroke.svg_path, id=stroke.stroke_id)
            stroke_group.add(path)
    
    dwg.add(stroke_group)
    
    tracer.event(f"SVG emitted with {len(strokes)} strokes")
    
    return dwg


def emit_strokes_svg_with_ids(strokes, width, height, config, debug_writer=None):
    """
    Create SVG and optionally save debug artifacts.
    """
    tracer = get_tracer()
    
    dwg = emit_strokes_svg(
        strokes, width, height, 
        stroke_width=config.stroke.width,
        stroke_color=config.stroke.color,
    )
    
    if debug_writer:
        import os
        stage_dir = debug_writer.get_stage_dir("stage4")
        svg_path = os.path.join(stage_dir, "output_strokes.svg")
        save_svg(dwg, svg_path)
        
        # Render to PNG
        png_path = os.path.join(stage_dir, "01_svg_render.png")
        render_svg_to_png(svg_path, png_path, dpi=150)
        
        # Save metrics
        total_path_length = sum(len(s.polyline) for s in strokes)
        metrics = {
            "num_strokes": len(strokes),
            "total_path_points": total_path_length,
            "stroke_width": config.stroke.width,
            "bezier_tolerance": config.bezier.error_tolerance,
        }
        debug_writer.save_json(metrics, "stage4", "stage4_metrics.json")
    
    return dwg


def create_strokes_from_beziers(view_id, polylines, bezier_lists, config):
    """
    Create Stroke objects from polylines and fitted Bezier curves.
    
    Args:
        view_id: identifier for the current view
        polylines: list of simplified polylines
        bezier_lists: list of lists of CubicBezier objects
        config: pipeline configuration
    
    Returns:
        list of Stroke objects
    """
    from patentdraw.models import generate_stroke_id, compute_bbox
    
    tracer = get_tracer()
    strokes = []
    
    for polyline, beziers in zip(polylines, bezier_lists):
        stroke_id = generate_stroke_id(polyline, view_id)
        svg_path = bezier_to_svg_path(beziers)
        bbox = compute_bbox(polyline)
        
        stroke = Stroke(
            stroke_id=stroke_id,
            polyline=polyline,
            bezier_segments=beziers,
            svg_path=svg_path,
            bbox=bbox,
        )
        strokes.append(stroke)
    
    tracer.event(f"Created {len(strokes)} stroke objects")
    
    return strokes


def create_debug_bbox_overlay(strokes, base_img, max_strokes=50):
    """
    Create an overlay image showing stroke bounding boxes.
    """
    from patentdraw.io.save_artifacts import draw_overlay
    import cv2
    
    if len(base_img.shape) == 2:
        overlay_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
    else:
        overlay_img = base_img.copy()
    
    bboxes = [s.bbox for s in strokes[:max_strokes]]
    texts = [((s.bbox[0], s.bbox[1] - 5), s.stroke_id[:8]) for s in strokes[:max_strokes]]
    
    return draw_overlay(overlay_img, bboxes=bboxes, texts=texts)
