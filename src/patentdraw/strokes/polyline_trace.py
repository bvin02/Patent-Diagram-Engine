"""
Polyline tracing from skeleton graph for Patent Draw.

Trace continuous paths between endpoints and junctions to form polylines.
Each polyline becomes a stroke after simplification and vectorization.
"""

import numpy as np
import networkx as nx

from patentdraw.io.save_artifacts import draw_overlay
from patentdraw.tracer import get_tracer, trace


@trace(label="trace_polylines")
def trace_polylines(graph, endpoints, junctions, debug_writer=None, skeleton_img=None):
    """
    Trace polylines from the skeleton graph.
    
    Traces paths from endpoints to endpoints/junctions, and between junctions.
    Short segments at junctions are handled by including the junction pixel
    in adjacent paths.
    
    Returns list of polylines where each polyline is [[x, y], [x, y], ...].
    Note: converts from (y, x) graph coordinates to [x, y] output.
    """
    tracer = get_tracer()
    
    polylines = []
    visited_edges = set()
    
    # Create set of special nodes for quick lookup
    special_nodes = set(endpoints) | set(junctions)
    
    with tracer.span("trace_paths", module="polyline_trace"):
        # Start tracing from each endpoint
        for start_node in endpoints:
            for neighbor in graph.neighbors(start_node):
                edge = tuple(sorted([start_node, neighbor]))
                if edge in visited_edges:
                    continue
                
                path = _trace_path(graph, start_node, neighbor, special_nodes, visited_edges)
                if path and len(path) >= 2:
                    # Convert (y, x) to [x, y]
                    polyline = [[x, y] for y, x in path]
                    polylines.append(polyline)
        
        # Trace paths between junctions (cycles and connections)
        for start_node in junctions:
            for neighbor in graph.neighbors(start_node):
                edge = tuple(sorted([start_node, neighbor]))
                if edge in visited_edges:
                    continue
                
                path = _trace_path(graph, start_node, neighbor, special_nodes, visited_edges)
                if path and len(path) >= 2:
                    polyline = [[x, y] for y, x in path]
                    polylines.append(polyline)
        
        tracer.event(f"Traced {len(polylines)} polylines")
    
    # Calculate statistics
    lengths = [len(p) for p in polylines]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    
    tracer.event(f"Polylines: count={len(polylines)}, avg_length={avg_length:.1f}")
    
    # Save debug artifacts
    if debug_writer and skeleton_img is not None:
        import cv2
        overlay_img = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2RGB)
        
        # Draw polylines with varying colors
        colors = _generate_colors(len(polylines))
        for polyline, color in zip(polylines, colors):
            overlay = draw_overlay(overlay_img, polylines=[polyline], polyline_color=color)
            overlay_img = overlay
        
        debug_writer.save_image(overlay_img, "stage3", "03_polylines_overlay.png")
        
        # Update metrics
        metrics = {
            "num_polylines": len(polylines),
            "avg_length": round(avg_length, 2),
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
        }
        # Merge with existing metrics
        existing = debug_writer.get_stage_dir("stage3")
        import os
        import json
        metrics_path = os.path.join(existing, "stage3_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                existing_metrics = json.load(f)
            existing_metrics.update(metrics)
            metrics = existing_metrics
        debug_writer.save_json(metrics, "stage3", "stage3_metrics.json")
    
    return polylines


def _trace_path(graph, start, next_node, special_nodes, visited_edges):
    """
    Trace a path from start through next_node until reaching a special node.
    
    Marks visited edges to prevent duplicate tracing.
    """
    path = [start]
    current = start
    next_n = next_node
    
    while True:
        edge = tuple(sorted([current, next_n]))
        if edge in visited_edges:
            break
        
        visited_edges.add(edge)
        path.append(next_n)
        
        # Stop if we reach a special node (endpoint or junction)
        if next_n in special_nodes:
            break
        
        # Find the next node to continue
        neighbors = list(graph.neighbors(next_n))
        # Remove the node we came from
        neighbors = [n for n in neighbors if n != current]
        
        if len(neighbors) == 0:
            # Dead end
            break
        elif len(neighbors) == 1:
            # Continue along the path
            current = next_n
            next_n = neighbors[0]
        else:
            # Should not happen for non-junction nodes
            break
    
    return path


def _generate_colors(n):
    """Generate n distinct colors for visualization."""
    import colorsys
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def merge_short_segments(polylines, angle_threshold=30.0, min_length=5):
    """
    Merge short polyline segments that have similar direction.
    
    If two polylines share an endpoint and the angle between them
    is below the threshold, merge them into one.
    
    Returns merged list of polylines.
    """
    tracer = get_tracer()
    
    if not polylines:
        return polylines
    
    # Build endpoint index
    endpoint_to_polylines = {}
    for i, polyline in enumerate(polylines):
        if len(polyline) < 2:
            continue
        
        start = tuple(polyline[0])
        end = tuple(polyline[-1])
        
        if start not in endpoint_to_polylines:
            endpoint_to_polylines[start] = []
        endpoint_to_polylines[start].append((i, "start"))
        
        if end not in endpoint_to_polylines:
            endpoint_to_polylines[end] = []
        endpoint_to_polylines[end].append((i, "end"))
    
    # Find mergeable pairs
    merged_indices = set()
    result = []
    
    for endpoint, connections in endpoint_to_polylines.items():
        if len(connections) != 2:
            continue
        
        idx1, end1 = connections[0]
        idx2, end2 = connections[1]
        
        if idx1 in merged_indices or idx2 in merged_indices:
            continue
        
        p1 = polylines[idx1]
        p2 = polylines[idx2]
        
        # Check if both are short
        if len(p1) > min_length and len(p2) > min_length:
            continue
        
        # Check angle continuity
        angle = _compute_angle_at_junction(p1, end1, p2, end2)
        if angle > angle_threshold:
            continue
        
        # Merge
        merged = _merge_polylines(p1, end1, p2, end2)
        result.append(merged)
        merged_indices.add(idx1)
        merged_indices.add(idx2)
    
    # Add non-merged polylines
    for i, polyline in enumerate(polylines):
        if i not in merged_indices:
            result.append(polyline)
    
    tracer.event(f"Merged: {len(polylines)} -> {len(result)}")
    
    return result


def _compute_angle_at_junction(p1, end1, p2, end2):
    """Compute angle between two polylines at their shared endpoint."""
    # Get direction vectors at the shared endpoint
    if end1 == "start":
        v1 = np.array(p1[0]) - np.array(p1[min(1, len(p1)-1)])
    else:
        v1 = np.array(p1[-1]) - np.array(p1[max(0, len(p1)-2)])
    
    if end2 == "start":
        v2 = np.array(p2[min(1, len(p2)-1)]) - np.array(p2[0])
    else:
        v2 = np.array(p2[max(0, len(p2)-2)]) - np.array(p2[-1])
    
    # Compute angle
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 180.0
    
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle


def _merge_polylines(p1, end1, p2, end2):
    """Merge two polylines at their shared endpoint."""
    # Orient polylines so they can be concatenated
    if end1 == "start":
        p1 = p1[::-1]
    if end2 == "end":
        p2 = p2[::-1]
    
    # Concatenate (skip duplicate point)
    return p1 + p2[1:]
