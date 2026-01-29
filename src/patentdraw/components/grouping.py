"""
Component grouping for Patent Draw.

Groups strokes into components using spatial adjacency and connectivity.
Provides baseline algorithm that works without ML assistance.
"""

import networkx as nx
import numpy as np
from shapely.geometry import LineString, box

from patentdraw.models import Component, generate_component_id, compute_bbox_from_bboxes, compute_centroid
from patentdraw.tracer import get_tracer, trace


@trace(label="group_strokes_baseline")
def group_strokes_baseline(strokes, config, debug_writer=None):
    """
    Group strokes into components using baseline algorithm.
    
    Uses spatial and topological relationships:
    - Shared endpoints within distance threshold
    - Intersecting or touching strokes
    - Overlapping bounding boxes
    
    Returns list of Component objects.
    """
    tracer = get_tracer()
    
    if not strokes:
        return []
    
    # Build stroke ID to index mapping
    stroke_id_to_idx = {s.stroke_id: i for i, s in enumerate(strokes)}
    n = len(strokes)
    
    # Build adjacency using union-find
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Check adjacency relationships
    with tracer.span("build_adjacency", module="grouping"):
        # Precompute geometries
        geometries = []
        for stroke in strokes:
            if len(stroke.polyline) >= 2:
                geometries.append(LineString(stroke.polyline))
            else:
                geometries.append(None)
        
        endpoint_threshold = config.grouping.endpoint_distance_threshold
        
        for i in range(n):
            for j in range(i + 1, n):
                if _are_adjacent(strokes[i], strokes[j], geometries[i], geometries[j], 
                               endpoint_threshold, config.grouping.bbox_overlap_threshold):
                    union(i, j)
    
    # Collect clusters
    with tracer.span("collect_clusters", module="grouping"):
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        
        tracer.event(f"Found {len(clusters)} clusters from {n} strokes")
    
    # Create Component objects
    components = []
    for root, indices in clusters.items():
        stroke_ids = [strokes[i].stroke_id for i in indices]
        stroke_bboxes = [strokes[i].bbox for i in indices]
        stroke_points = []
        for i in indices:
            stroke_points.extend(strokes[i].polyline)
        
        component = Component(
            component_id=generate_component_id(stroke_ids),
            stroke_ids=stroke_ids,
            bbox=compute_bbox_from_bboxes(stroke_bboxes),
            centroid=compute_centroid(stroke_points),
            proposal_sources=["baseline"],
            confidence=_compute_confidence(strokes, indices),
        )
        components.append(component)
    
    tracer.event(f"Created {len(components)} components")
    
    # Save debug artifacts
    if debug_writer:
        component_map = {c.component_id: len(c.stroke_ids) for c in components}
        
        metrics = {
            "num_components_baseline": len(components),
            "avg_strokes_per_component": sum(len(c.stroke_ids) for c in components) / len(components) if components else 0,
            "max_strokes_in_component": max(len(c.stroke_ids) for c in components) if components else 0,
        }
        debug_writer.save_json(metrics, "stage5", "stage5_metrics.json")
        debug_writer.save_json(component_map, "stage5", "03_component_map.json")
    
    return components


def _are_adjacent(stroke1, stroke2, geom1, geom2, endpoint_threshold, bbox_overlap_threshold):
    """Check if two strokes are adjacent."""
    # Check shared endpoints
    if _endpoints_close(stroke1.polyline, stroke2.polyline, endpoint_threshold):
        return True
    
    # Check intersection/touching
    if geom1 is not None and geom2 is not None:
        try:
            if geom1.intersects(geom2) or geom1.touches(geom2):
                return True
        except Exception:
            pass
    
    # Check bbox overlap
    if _bboxes_overlap(stroke1.bbox, stroke2.bbox, bbox_overlap_threshold):
        return True
    
    return False


def _endpoints_close(polyline1, polyline2, threshold):
    """Check if any endpoints of two polylines are within threshold distance."""
    if not polyline1 or not polyline2:
        return False
    
    endpoints1 = [polyline1[0], polyline1[-1]]
    endpoints2 = [polyline2[0], polyline2[-1]]
    
    for p1 in endpoints1:
        for p2 in endpoints2:
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if dist <= threshold:
                return True
    
    return False


def _bboxes_overlap(bbox1, bbox2, threshold):
    """Check if two bounding boxes overlap by more than threshold ratio."""
    b1 = box(bbox1[0], bbox1[1], bbox1[2], bbox1[3])
    b2 = box(bbox2[0], bbox2[1], bbox2[2], bbox2[3])
    
    if not b1.intersects(b2):
        return False
    
    intersection_area = b1.intersection(b2).area
    min_area = min(b1.area, b2.area)
    
    if min_area == 0:
        return False
    
    overlap_ratio = intersection_area / min_area
    return overlap_ratio > threshold


def _compute_confidence(strokes, indices):
    """
    Compute confidence score for a component cluster.
    
    Higher confidence for:
    - Strokes forming enclosed contours
    - Strokes sharing junctions
    - Tight spatial cohesion
    """
    n = len(indices)
    
    if n == 1:
        return 0.5
    
    # Base confidence from cluster size
    base = min(0.8, 0.4 + 0.1 * n)
    
    # Check spatial cohesion
    bboxes = [strokes[i].bbox for i in indices]
    combined = compute_bbox_from_bboxes(bboxes)
    combined_area = (combined[2] - combined[0]) * (combined[3] - combined[1])
    
    individual_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in bboxes)
    
    if combined_area > 0:
        cohesion = individual_area / combined_area
        base += 0.2 * min(1.0, cohesion)
    
    return min(1.0, base)


def create_component_overlay(components, strokes, base_img, max_components=20):
    """Create debug overlay showing component groupings."""
    import cv2
    import colorsys
    from patentdraw.io.save_artifacts import draw_overlay
    
    if len(base_img.shape) == 2:
        overlay_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
    else:
        overlay_img = base_img.copy()
    
    stroke_id_to_stroke = {s.stroke_id: s for s in strokes}
    
    # Generate colors for components
    colors = []
    for i in range(len(components)):
        hue = i / max(len(components), 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    
    for i, comp in enumerate(components[:max_components]):
        color = colors[i % len(colors)]
        
        polylines = []
        for stroke_id in comp.stroke_ids:
            if stroke_id in stroke_id_to_stroke:
                polylines.append(stroke_id_to_stroke[stroke_id].polyline)
        
        for polyline in polylines:
            overlay_img = draw_overlay(overlay_img, polylines=[polyline], polyline_color=color)
        
        # Add component label
        cx, cy = comp.centroid
        cv2.putText(overlay_img, comp.component_id[:8], (int(cx), int(cy)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return overlay_img
