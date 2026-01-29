"""
Leader line routing for Patent Draw.

Routes leader lines from label text positions to component anchor points.
"""

import numpy as np
from shapely.geometry import LineString, Point

from patentdraw.tracer import get_tracer, trace


@trace(label="route_leaders")
def route_leaders(labels, strokes, config):
    """
    Route leader lines for all labels.
    
    For each label:
    1. Try straight line from text_pos to anchor
    2. If too many crossings, use 1-bend orthogonal route
    
    Updates labels in place with leader_path.
    
    Returns tuple of (labels, crossing_stats).
    """
    tracer = get_tracer()
    
    stroke_lines = []
    for stroke in strokes:
        if len(stroke.polyline) >= 2:
            stroke_lines.append(LineString(stroke.polyline))
    
    total_crossings = 0
    bent_routes = 0
    
    for label in labels:
        text_pos = label.text_pos
        anchor = label.anchor_point
        
        # Try straight line
        straight_path = [text_pos, anchor]
        straight_crossings = count_crossings(straight_path, stroke_lines)
        
        if straight_crossings <= config.label.max_crossings_for_bend:
            label.leader_path = straight_path
            total_crossings += straight_crossings
        else:
            # Use 1-bend orthogonal route
            bent_path = compute_bent_path(text_pos, anchor)
            bent_crossings = count_crossings(bent_path, stroke_lines)
            
            # Use whichever has fewer crossings
            if bent_crossings < straight_crossings:
                label.leader_path = bent_path
                total_crossings += bent_crossings
                bent_routes += 1
            else:
                label.leader_path = straight_path
                total_crossings += straight_crossings
    
    crossing_stats = {
        "total_crossings": total_crossings,
        "bent_routes": bent_routes,
        "avg_crossings": total_crossings / len(labels) if labels else 0,
    }
    
    tracer.event(f"Routed {len(labels)} leaders, {bent_routes} bent, {total_crossings} total crossings")
    
    return labels, crossing_stats


def count_crossings(path, stroke_lines):
    """Count how many strokes a leader path crosses."""
    if len(path) < 2:
        return 0
    
    leader_line = LineString(path)
    crossings = 0
    
    for stroke_line in stroke_lines:
        if leader_line.intersects(stroke_line):
            intersection = leader_line.intersection(stroke_line)
            # Count number of crossing points
            if intersection.geom_type == "Point":
                crossings += 1
            elif intersection.geom_type == "MultiPoint":
                crossings += len(intersection.geoms)
            elif intersection.geom_type in ("LineString", "MultiLineString"):
                # Overlapping, count as 1
                crossings += 1
    
    return crossings


def compute_bent_path(text_pos, anchor):
    """
    Compute a 1-bend orthogonal path from text_pos to anchor.
    
    The bend point is at the horizontal level of text_pos
    and vertical level of anchor (or vice versa).
    """
    # Two possible bend points
    bend1 = [anchor[0], text_pos[1]]  # horizontal first, then vertical
    bend2 = [text_pos[0], anchor[1]]  # vertical first, then horizontal
    
    # Choose based on distances (prefer shorter total path)
    dist1 = _path_length([text_pos, bend1, anchor])
    dist2 = _path_length([text_pos, bend2, anchor])
    
    if dist1 <= dist2:
        return [text_pos, bend1, anchor]
    else:
        return [text_pos, bend2, anchor]


def _path_length(path):
    """Calculate total length of a polyline path."""
    total = 0
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        total += np.sqrt(dx*dx + dy*dy)
    return total


def check_leader_crossings(labels):
    """
    Check if leader lines cross each other.
    
    Returns list of crossing label ID pairs.
    """
    tracer = get_tracer()
    
    crossings = []
    
    leader_lines = []
    for label in labels:
        if len(label.leader_path) >= 2:
            leader_lines.append((label.label_id, LineString(label.leader_path)))
    
    n = len(leader_lines)
    for i in range(n):
        for j in range(i + 1, n):
            id1, line1 = leader_lines[i]
            id2, line2 = leader_lines[j]
            
            if line1.intersects(line2):
                # Check if it's more than just touching at endpoints
                intersection = line1.intersection(line2)
                if intersection.geom_type != "Point":
                    crossings.append((id1, id2))
                elif not _is_endpoint_touch(line1, line2, intersection):
                    crossings.append((id1, id2))
    
    if crossings:
        tracer.event(f"Found {len(crossings)} leader-leader crossings", level="WARN")
    
    return crossings


def _is_endpoint_touch(line1, line2, point):
    """Check if intersection is just endpoints touching."""
    eps = 0.01
    
    endpoints = [
        Point(line1.coords[0]),
        Point(line1.coords[-1]),
        Point(line2.coords[0]),
        Point(line2.coords[-1]),
    ]
    
    for ep in endpoints:
        if point.distance(ep) < eps:
            return True
    
    return False
