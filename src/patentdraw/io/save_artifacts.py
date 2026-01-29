"""
Artifact saving utilities for Patent Draw.

Handles writing debug images, JSON files, and rendering SVG to PNG.
"""

import json
import os

import cv2
import numpy as np

from patentdraw.tracer import get_tracer


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def get_debug_dir(out_dir, view_id, stage_name):
    """
    Get the debug directory path for a stage.
    
    Creates the directory if it does not exist.
    """
    debug_dir = os.path.join(out_dir, "debug", view_id, stage_name)
    ensure_dir(debug_dir)
    return debug_dir


def save_image(img, path, max_edge=None):
    """
    Save an image to disk.
    
    Optionally downscales to max_edge while preserving aspect ratio.
    Handles both RGB and BGR formats (converts RGB to BGR for OpenCV).
    """
    tracer = get_tracer()
    
    # Downscale if needed
    if max_edge and max(img.shape[:2]) > max_edge:
        scale = max_edge / max(img.shape[:2])
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    
    # Convert RGB to BGR if needed (3 channels)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img_bgr)
    tracer.event(f"Saved image: {path}")


def save_json(data, path, indent=2):
    """
    Save a dictionary or Pydantic model to JSON.
    """
    tracer = get_tracer()
    
    ensure_dir(os.path.dirname(path))
    
    # Handle Pydantic models
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    
    tracer.event(f"Saved JSON: {path}")


def save_svg(svg_content, path):
    """
    Save SVG content to file.
    """
    tracer = get_tracer()
    
    ensure_dir(os.path.dirname(path))
    
    if hasattr(svg_content, "tostring"):
        content = svg_content.tostring()
    else:
        content = str(svg_content)
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    
    tracer.event(f"Saved SVG: {path}")


def render_svg_to_png(svg_path, png_path, dpi=150):
    """
    Render an SVG file to PNG using cairosvg.
    """
    tracer = get_tracer()
    
    try:
        import cairosvg
        
        ensure_dir(os.path.dirname(png_path))
        cairosvg.svg2png(url=svg_path, write_to=png_path, dpi=dpi)
        tracer.event(f"Rendered SVG to PNG: {png_path}")
    except ImportError:
        tracer.event("cairosvg not available, skipping PNG render", level="WARN")
    except Exception as e:
        tracer.event(f"Failed to render SVG: {str(e)}", level="WARN")


def draw_overlay(base_img, polylines=None, points=None, bboxes=None, texts=None, 
                 polyline_color=(0, 255, 0), point_color=(255, 0, 0), 
                 bbox_color=(0, 0, 255), text_color=(0, 0, 0)):
    """
    Draw debug overlay on an image.
    
    All inputs are optional. Creates a copy of the base image.
    
    polylines: list of [[x,y], [x,y], ...] polylines
    points: list of ([x,y], radius) tuples
    bboxes: list of [min_x, min_y, max_x, max_y] bounding boxes
    texts: list of ([x,y], text_string) tuples
    """
    # Make a copy and convert to BGR for drawing
    if len(base_img.shape) == 2:
        # Grayscale or binary - convert to 3 channel
        overlay = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    elif base_img.shape[2] == 3:
        overlay = cv2.cvtColor(base_img.copy(), cv2.COLOR_RGB2BGR)
    else:
        overlay = base_img.copy()
    
    # Draw polylines
    if polylines:
        for polyline in polylines:
            if len(polyline) < 2:
                continue
            pts = np.array(polyline, dtype=np.int32)
            cv2.polylines(overlay, [pts], isClosed=False, color=polyline_color, thickness=1)
    
    # Draw points
    if points:
        for pt, radius in points:
            center = (int(pt[0]), int(pt[1]))
            cv2.circle(overlay, center, radius, point_color, -1)
    
    # Draw bounding boxes
    if bboxes:
        for bbox in bboxes:
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(overlay, pt1, pt2, bbox_color, 1)
    
    # Draw texts
    if texts:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        for (x, y), text in texts:
            cv2.putText(overlay, str(text), (int(x), int(y)), font, font_scale, text_color, thickness)
    
    # Convert back to RGB
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def create_binary_overlay(rgb_img, binary_img, color=(255, 0, 0)):
    """
    Overlay binary edges on an RGB image.
    
    binary_img should be uint8 with 255 for foreground.
    """
    # Find edges in binary
    edges = cv2.Canny(binary_img, 50, 150)
    
    overlay = rgb_img.copy()
    overlay[edges > 0] = color
    
    return overlay


def save_metrics(metrics, out_dir, view_id, stage_name, filename="metrics.json"):
    """
    Save stage metrics to the debug directory.
    """
    debug_dir = get_debug_dir(out_dir, view_id, stage_name)
    path = os.path.join(debug_dir, filename)
    save_json(metrics, path)


class DebugArtifactWriter:
    """
    Helper class to manage debug artifact writing for a single view.
    
    Handles creation of debug directories and provides convenience methods
    for saving various artifact types.
    """
    
    def __init__(self, out_dir, view_id, enabled=True, max_edge=1600):
        self.out_dir = out_dir
        self.view_id = view_id
        self.enabled = enabled
        self.max_edge = max_edge
    
    def get_stage_dir(self, stage_name):
        """Get the debug directory for a stage."""
        return get_debug_dir(self.out_dir, self.view_id, stage_name)
    
    def save_image(self, img, stage_name, filename):
        """Save an image artifact."""
        if not self.enabled:
            return
        path = os.path.join(self.get_stage_dir(stage_name), filename)
        save_image(img, path, max_edge=self.max_edge)
    
    def save_json(self, data, stage_name, filename):
        """Save a JSON artifact."""
        if not self.enabled:
            return
        path = os.path.join(self.get_stage_dir(stage_name), filename)
        save_json(data, path)
    
    def save_svg(self, svg_content, stage_name, filename):
        """Save an SVG artifact."""
        if not self.enabled:
            return
        path = os.path.join(self.get_stage_dir(stage_name), filename)
        save_svg(svg_content, path)
    
    def save_overlay(self, base_img, stage_name, filename, **kwargs):
        """Draw and save an overlay image."""
        if not self.enabled:
            return
        overlay = draw_overlay(base_img, **kwargs)
        self.save_image(overlay, stage_name, filename)
