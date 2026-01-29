"""
Bezier curve fitting for Patent Draw.

Fits cubic Bezier curves to polylines using a Schneider-style algorithm.
Each polyline is approximated by a series of cubic Bezier segments.
"""

import numpy as np

from patentdraw.models import CubicBezier
from patentdraw.tracer import get_tracer, trace


@trace(label="fit_bezier_to_polylines")
def fit_bezier_to_polylines(polylines, error_tolerance, max_iterations=4):
    """
    Fit cubic Bezier curves to all polylines.
    
    Args:
        polylines: list of polylines, each [[x, y], ...]
        error_tolerance: maximum allowed fitting error
        max_iterations: max recursion depth for subdivision
    
    Returns:
        list of lists of CubicBezier objects
    """
    tracer = get_tracer()
    
    all_beziers = []
    total_segments = 0
    
    for polyline in polylines:
        if len(polyline) < 2:
            all_beziers.append([])
            continue
        
        points = np.array(polyline)
        beziers = fit_cubic_beziers(points, error_tolerance, max_iterations)
        all_beziers.append(beziers)
        total_segments += len(beziers)
    
    tracer.event(f"Fitted {total_segments} Bezier segments for {len(polylines)} polylines")
    
    return all_beziers


def fit_cubic_beziers(points, error_tolerance, max_iterations=4):
    """
    Fit cubic Bezier curves to a single polyline.
    
    Uses the Schneider algorithm:
    1. Estimate tangent directions at endpoints
    2. Fit a single cubic Bezier
    3. If error exceeds tolerance, split at max error point and recurse
    """
    if len(points) < 2:
        return []
    
    if len(points) == 2:
        # Degenerate case: straight line
        return [_line_to_bezier(points[0], points[1])]
    
    # Compute tangent at start and end
    tangent_start = _estimate_tangent(points, 0)
    tangent_end = _estimate_tangent(points, len(points) - 1)
    
    return _fit_cubic(points, tangent_start, tangent_end, error_tolerance, max_iterations)


def _fit_cubic(points, tangent_start, tangent_end, error_tolerance, iterations_left):
    """Recursive cubic Bezier fitting."""
    if len(points) == 2:
        return [_line_to_bezier(points[0], points[1])]
    
    # Parameterize points by chord length
    t_values = _chord_length_parameterize(points)
    
    # Fit bezier with given tangents
    bezier = _fit_bezier_to_pts(points, t_values, tangent_start, tangent_end)
    
    # Compute max error
    max_error, split_point = _compute_max_error(points, bezier, t_values)
    
    if max_error < error_tolerance or iterations_left <= 0:
        return [bezier]
    
    # Split and recurse
    left_points = points[:split_point + 1]
    right_points = points[split_point:]
    
    tangent_split = _estimate_tangent(points, split_point)
    
    left_beziers = _fit_cubic(
        left_points, tangent_start, tangent_split, 
        error_tolerance, iterations_left - 1
    )
    right_beziers = _fit_cubic(
        right_points, -tangent_split, tangent_end,
        error_tolerance, iterations_left - 1
    )
    
    return left_beziers + right_beziers


def _line_to_bezier(p0, p1):
    """Create a degenerate Bezier for a straight line segment."""
    p0 = np.array(p0)
    p1 = np.array(p1)
    
    # Control points at 1/3 and 2/3 along the line
    c1 = p0 + (p1 - p0) / 3
    c2 = p0 + 2 * (p1 - p0) / 3
    
    return CubicBezier(
        p0=p0.tolist(),
        p1=c1.tolist(),
        p2=c2.tolist(),
        p3=p1.tolist(),
    )


def _estimate_tangent(points, index):
    """Estimate tangent direction at a point on the polyline."""
    n = len(points)
    
    if index == 0:
        # Start tangent
        tangent = points[min(1, n-1)] - points[0]
    elif index == n - 1:
        # End tangent
        tangent = points[-1] - points[max(0, n-2)]
    else:
        # Mid tangent (average of incoming and outgoing)
        tangent = points[min(index + 1, n-1)] - points[max(index - 1, 0)]
    
    norm = np.linalg.norm(tangent)
    if norm > 0:
        tangent = tangent / norm
    
    return tangent


def _chord_length_parameterize(points):
    """Compute parameter values based on chord length."""
    n = len(points)
    t = np.zeros(n)
    
    for i in range(1, n):
        t[i] = t[i-1] + np.linalg.norm(points[i] - points[i-1])
    
    total = t[-1]
    if total > 0:
        t = t / total
    
    return t


def _fit_bezier_to_pts(points, t_values, tangent_start, tangent_end):
    """Fit a single cubic Bezier to points with given parameterization and tangents."""
    p0 = points[0]
    p3 = points[-1]
    
    # Compute alpha values using least squares
    # Based on Schneider's algorithm
    n = len(points)
    
    # Build matrices
    a = np.zeros((n, 2, 2))
    
    for i, t in enumerate(t_values):
        b1 = _bernstein(1, t) * tangent_start
        b2 = _bernstein(2, t) * tangent_end
        a[i, 0] = b1
        a[i, 1] = b2
    
    # Compute C and X matrices
    c = np.zeros((2, 2))
    x = np.zeros(2)
    
    for i, t in enumerate(t_values):
        c[0, 0] += np.dot(a[i, 0], a[i, 0])
        c[0, 1] += np.dot(a[i, 0], a[i, 1])
        c[1, 0] = c[0, 1]
        c[1, 1] += np.dot(a[i, 1], a[i, 1])
        
        # Target point minus bezier at t with alpha=0
        b0 = _bernstein(0, t)
        b1 = _bernstein(1, t)
        b2 = _bernstein(2, t)
        b3 = _bernstein(3, t)
        
        target = points[i] - (b0 * p0 + b3 * p3)
        
        x[0] += np.dot(a[i, 0], target)
        x[1] += np.dot(a[i, 1], target)
    
    # Solve for alpha
    det = c[0, 0] * c[1, 1] - c[0, 1] * c[1, 0]
    
    if abs(det) < 1e-10:
        # Fallback to simple heuristic
        dist = np.linalg.norm(p3 - p0) / 3
        alpha1 = dist
        alpha2 = dist
    else:
        alpha1 = (c[1, 1] * x[0] - c[0, 1] * x[1]) / det
        alpha2 = (c[0, 0] * x[1] - c[1, 0] * x[0]) / det
    
    # Clamp alpha to reasonable range
    seg_len = np.linalg.norm(p3 - p0)
    alpha1 = max(0.01 * seg_len, min(alpha1, seg_len))
    alpha2 = max(0.01 * seg_len, min(alpha2, seg_len))
    
    p1 = p0 + alpha1 * tangent_start
    p2 = p3 - alpha2 * tangent_end
    
    return CubicBezier(
        p0=p0.tolist(),
        p1=p1.tolist(),
        p2=p2.tolist(),
        p3=p3.tolist(),
    )


def _bernstein(i, t):
    """Compute Bernstein basis polynomial value B_i,3(t)."""
    if i == 0:
        return (1 - t) ** 3
    elif i == 1:
        return 3 * (1 - t) ** 2 * t
    elif i == 2:
        return 3 * (1 - t) * t ** 2
    else:
        return t ** 3


def _compute_max_error(points, bezier, t_values):
    """Compute maximum error and split point for a Bezier fit."""
    max_error = 0.0
    split_point = len(points) // 2
    
    for i, t in enumerate(t_values):
        fitted = evaluate_bezier(bezier, t)
        error = np.linalg.norm(points[i] - np.array(fitted))
        
        if error > max_error:
            max_error = error
            split_point = i
    
    # Avoid splitting at endpoints
    if split_point == 0:
        split_point = 1
    if split_point == len(points) - 1:
        split_point = len(points) - 2
    
    return max_error, split_point


def evaluate_bezier(bezier, t):
    """Evaluate a cubic Bezier at parameter t."""
    p0 = np.array(bezier.p0)
    p1 = np.array(bezier.p1)
    p2 = np.array(bezier.p2)
    p3 = np.array(bezier.p3)
    
    b0 = _bernstein(0, t)
    b1 = _bernstein(1, t)
    b2 = _bernstein(2, t)
    b3 = _bernstein(3, t)
    
    return (b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3).tolist()


def bezier_to_svg_path(beziers):
    """
    Convert a list of CubicBezier objects to SVG path d attribute.
    
    Assumes beziers are connected (end of one = start of next).
    """
    if not beziers:
        return ""
    
    parts = []
    
    # Move to start
    p0 = beziers[0].p0
    parts.append(f"M {p0[0]:.2f} {p0[1]:.2f}")
    
    # Cubic bezier curves
    for bez in beziers:
        parts.append(f"C {bez.p1[0]:.2f} {bez.p1[1]:.2f} {bez.p2[0]:.2f} {bez.p2[1]:.2f} {bez.p3[0]:.2f} {bez.p3[1]:.2f}")
    
    return " ".join(parts)


def compute_bezier_bbox(beziers):
    """Compute bounding box of a list of Bezier curves."""
    if not beziers:
        return [0.0, 0.0, 0.0, 0.0]
    
    all_points = []
    for bez in beziers:
        all_points.extend([bez.p0, bez.p1, bez.p2, bez.p3])
    
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    
    return [min(xs), min(ys), max(xs), max(ys)]
