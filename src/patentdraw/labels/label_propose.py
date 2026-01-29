"""
Label anchor and text position proposals for Patent Draw.

Proposes label positions for each component.
"""

import numpy as np
from shapely.geometry import Point, box

from patentdraw.models import Label, LabelStatus, generate_label_id
from patentdraw.tracer import get_tracer, trace


@trace(label="propose_labels")
def propose_labels(components, strokes, image_width, image_height, config):
    """
    Propose label positions for all components.
    
    For each component:
    1. Compute anchor point on component boundary
    2. Find text position outside component bbox
    
    Returns list of Label objects with status="proposed".
    """
    tracer = get_tracer()
    
    stroke_id_to_stroke = {s.stroke_id: s for s in strokes}
    labels = []
    
    for comp in components:
        # Get all points in component
        points = []
        for stroke_id in comp.stroke_ids:
            if stroke_id in stroke_id_to_stroke:
                points.extend(stroke_id_to_stroke[stroke_id].polyline)
        
        if not points:
            continue
        
        # Compute anchor point (boundary point nearest to exterior)
        anchor = compute_anchor_point(points, comp.bbox, image_width, image_height)
        
        # Compute text position
        text_pos = compute_text_position(comp.bbox, config.label.text_offset, image_width, image_height)
        
        label = Label(
            label_id=generate_label_id(comp.component_id, anchor, text_pos),
            component_id=comp.component_id,
            anchor_point=anchor,
            text_pos=text_pos,
            leader_path=[text_pos, anchor],  # Simple straight line initially
            text="",  # Will be filled in numbering stage
            status=LabelStatus.PROPOSED,
        )
        labels.append(label)
    
    tracer.event(f"Proposed {len(labels)} labels")
    
    return labels


def compute_anchor_point(points, bbox, image_width, image_height):
    """
    Compute anchor point for a component.
    
    Finds a point on the component boundary that is well-suited for a leader line.
    Prefers points toward the exterior of the image.
    """
    if not points:
        return [0.0, 0.0]
    
    points_arr = np.array(points)
    centroid = np.mean(points_arr, axis=0)
    
    # Determine which edge of the image is nearest
    edges = [
        ("top", centroid[1]),
        ("bottom", image_height - centroid[1]),
        ("left", centroid[0]),
        ("right", image_width - centroid[0]),
    ]
    nearest_edge = min(edges, key=lambda x: x[1])[0]
    
    # Find point on component boundary nearest to that edge
    if nearest_edge == "top":
        idx = np.argmin(points_arr[:, 1])
    elif nearest_edge == "bottom":
        idx = np.argmax(points_arr[:, 1])
    elif nearest_edge == "left":
        idx = np.argmin(points_arr[:, 0])
    else:  # right
        idx = np.argmax(points_arr[:, 0])
    
    return points_arr[idx].tolist()


def compute_text_position(bbox, offset, image_width, image_height):
    """
    Compute text position for a label.
    
    Prefers positions in order: top-right, top-left, bottom-right, bottom-left.
    Stays within image bounds.
    """
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    
    # Try positions in preference order
    positions = [
        ("top-right", bbox[2] + offset, bbox[1] - offset),
        ("top-left", bbox[0] - offset, bbox[1] - offset),
        ("bottom-right", bbox[2] + offset, bbox[3] + offset),
        ("bottom-left", bbox[0] - offset, bbox[3] + offset),
        ("right", bbox[2] + offset, cy),
        ("left", bbox[0] - offset, cy),
    ]
    
    for name, x, y in positions:
        # Check if position is within image bounds (with margin)
        margin = 20
        if margin <= x <= image_width - margin and margin <= y <= image_height - margin:
            return [x, y]
    
    # Fallback: offset from centroid
    return [cx + offset, cy - offset]


def check_label_overlap(labels, strokes):
    """
    Check if any label text positions overlap with strokes.
    
    Returns list of label IDs with overlaps.
    """
    tracer = get_tracer()
    
    overlapping = []
    
    for label in labels:
        # Create text bbox (approximate size)
        text_x, text_y = label.text_pos
        text_bbox = box(text_x - 20, text_y - 10, text_x + 30, text_y + 10)
        
        for stroke in strokes:
            if len(stroke.polyline) < 2:
                continue
            
            from shapely.geometry import LineString
            stroke_line = LineString(stroke.polyline)
            
            if text_bbox.intersects(stroke_line):
                overlapping.append(label.label_id)
                break
    
    if overlapping:
        tracer.event(f"Found {len(overlapping)} labels overlapping strokes", level="WARN")
    
    return overlapping
