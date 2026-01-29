"""
Polyline simplification using Ramer-Douglas-Peucker algorithm.

Reduces the number of points while preserving shape within tolerance.
"""

import numpy as np

from patentdraw.tracer import get_tracer, trace


@trace(label="simplify_polylines")
def simplify_polylines(polylines, epsilon):
    """
    Simplify all polylines using RDP algorithm.
    
    Args:
        polylines: list of polylines, each [[x, y], ...]
        epsilon: maximum perpendicular distance threshold
    
    Returns:
        list of simplified polylines
    """
    tracer = get_tracer()
    
    simplified = []
    total_points_before = 0
    total_points_after = 0
    
    for polyline in polylines:
        total_points_before += len(polyline)
        
        if len(polyline) <= 2:
            simplified.append(polyline)
            total_points_after += len(polyline)
            continue
        
        simple = rdp_simplify(polyline, epsilon)
        simplified.append(simple)
        total_points_after += len(simple)
    
    reduction = 1 - (total_points_after / total_points_before) if total_points_before > 0 else 0
    tracer.event(f"Simplified: {total_points_before} -> {total_points_after} points ({reduction:.1%} reduction)")
    
    return simplified


def rdp_simplify(points, epsilon):
    """
    Ramer-Douglas-Peucker algorithm for polyline simplification.
    
    Recursively removes points that are within epsilon distance
    of the line between endpoints.
    
    Args:
        points: list of [x, y] points
        epsilon: maximum perpendicular distance threshold
    
    Returns:
        simplified list of points
    """
    if len(points) <= 2:
        return points
    
    points_arr = np.array(points)
    
    # Find point with maximum distance from line between first and last
    start = points_arr[0]
    end = points_arr[-1]
    
    distances = _perpendicular_distances(points_arr, start, end)
    max_idx = np.argmax(distances)
    max_dist = distances[max_idx]
    
    if max_dist > epsilon:
        # Recursively simplify
        left = rdp_simplify(points[:max_idx + 1], epsilon)
        right = rdp_simplify(points[max_idx:], epsilon)
        
        # Join results (avoiding duplicate point at max_idx)
        return left[:-1] + right
    else:
        # All points within tolerance, keep only endpoints
        return [points[0], points[-1]]


def _perpendicular_distances(points, start, end):
    """
    Compute perpendicular distances from each point to the line from start to end.
    """
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    
    if line_len == 0:
        # Start and end are the same point
        return np.linalg.norm(points - start, axis=1)
    
    line_unit = line_vec / line_len
    
    # Vector from start to each point
    point_vecs = points - start
    
    # Project onto line
    projections = np.dot(point_vecs, line_unit)
    
    # Clamp projections to line segment
    projections = np.clip(projections, 0, line_len)
    
    # Nearest point on line for each input point
    nearest = start + np.outer(projections, line_unit)
    
    # Distance from each point to its nearest point on line
    distances = np.linalg.norm(points - nearest, axis=1)
    
    return distances


def remove_duplicate_points(polyline, tolerance=0.5):
    """
    Remove consecutive duplicate or near-duplicate points.
    """
    if len(polyline) <= 1:
        return polyline
    
    result = [polyline[0]]
    
    for point in polyline[1:]:
        last = result[-1]
        dist = np.sqrt((point[0] - last[0])**2 + (point[1] - last[1])**2)
        if dist > tolerance:
            result.append(point)
    
    return result


def resample_polyline(polyline, target_spacing):
    """
    Resample polyline to have approximately uniform point spacing.
    
    Useful for consistent Bezier fitting.
    """
    if len(polyline) <= 1:
        return polyline
    
    points = np.array(polyline)
    
    # Compute cumulative arc length
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    
    if total_length == 0:
        return [polyline[0]]
    
    # Generate uniform sample positions
    num_samples = max(2, int(total_length / target_spacing))
    sample_positions = np.linspace(0, total_length, num_samples)
    
    # Interpolate
    result = []
    for s in sample_positions:
        idx = np.searchsorted(cumulative, s) - 1
        idx = max(0, min(idx, len(points) - 2))
        
        t = (s - cumulative[idx]) / segment_lengths[idx] if segment_lengths[idx] > 0 else 0
        point = points[idx] + t * (points[idx + 1] - points[idx])
        result.append(point.tolist())
    
    return result
