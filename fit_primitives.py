"""
Stage 6: Primitive Fitting for Sketch-to-SVG Pipeline

Fits geometric primitives (lines, arcs, beziers) to graph edges from Stage 5.
Outputs primitives.json for SVG emission in Stage 7.

Usage:
    python fit_primitives.py runs/<run>/50_graph_clean/out/graph_clean.json --debug
    python fit_primitives.py runs/<run>/50_graph_clean/out/graph_clean.json --debug --mask runs/<run>/10_preprocess/out/output_mask.png
    python fit_primitives.py runs/<run>/50_graph_clean/out/graph_clean.json --debug --config configs/fit.json
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from scipy.optimize import least_squares
from scipy.spatial import KDTree

from utils.artifacts import make_run_dir, StageArtifacts


# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Edge bucketing
    "structural_min_length_px": 30,
    "rdp_epsilon_ref": 1.25,  # For curves
    "rdp_epsilon_straight": 2.0,  # For straight/segmented lines
    "rdp_straightness_threshold": 0.98,  # If straightness > this, use epsilon_straight
    
    # Parallel line detection (cross-hatching)
    "parallel_angle_tolerance_deg": 5.0,  # Max angle diff to consider parallel
    "parallel_min_neighbors": 2,  # Min neighbors to flag as hatching
    "parallel_search_radius_px": 50.0,  # Search radius for neighbors
    
    # Line fitting
    "line_min_straightness": 0.9,  # Relaxed from 0.985 for better editability
    "line_min_straightness_detail": 0.995,
    "line_simplicity_factor": 1.15,  # Line wins if rms <= factor * best_rms
    
    # Arc fitting
    "arc_min_length_px": 40,
    "arc_max_straightness": 0.995,
    "arc_min_turning": 0.15,
    "arc_min_sweep_consistency": 0.8,
    "arc_max_radius_px": 5000,
    "arc_radius_as_line_px": 2000,  # If arc radius > this, treat as line candidate
    
    # Error tolerances (defaults, overridden by stroke width if available)
    "error_tolerance_px_default": 1.0,
    "max_error_cap_px_default": 3.0,
    
    # Bezier fitting
    "bezier_regularization": 1e-2,
    "bezier_max_ctrl_dist_factor": 2.0,
    "bezier_sample_points": 60,
    
    # Misc
    "dir_sample_len_for_turning": 3,
    "error_heatmap_bins": [0.5, 1.0, 2.0, 3.0],
}


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_config(config_path: str = None) -> dict:
    """Load config from JSON and merge with defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                user_config = json.load(f)
            config.update(user_config)
    return config


def infer_run_dir(graph_path: str, runs_root: str) -> Path:
    """Infer run directory from graph path."""
    graph_p = Path(graph_path).resolve()
    runs_root_p = Path(runs_root).resolve()
    try:
        rel_path = graph_p.relative_to(runs_root_p)
        run_name = rel_path.parts[0]
        return runs_root_p / run_name
    except ValueError:
        return make_run_dir(graph_path, runs_root)


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


def load_stroke_width(run_dir: Path) -> Optional[float]:
    """Load stroke width from Stage 2 if available."""
    sw_path = run_dir / "20_distance_transform" / "out" / "stroke_width.json"
    if sw_path.exists():
        with open(sw_path) as f:
            data = json.load(f)
        return data.get("stroke_width_px")
    return None


def rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return points
    
    def _perp_dist(pt, start, end):
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-9:
            return np.linalg.norm(pt - start)
        t = max(0, min(1, np.dot(pt - start, line_vec) / (line_len ** 2)))
        proj = start + t * line_vec
        return np.linalg.norm(pt - proj)
    
    def _rdp_recursive(pts, eps, start, end, mask):
        if end <= start + 1:
            return
        max_dist = 0.0
        max_idx = start
        for i in range(start + 1, end):
            d = _perp_dist(pts[i], pts[start], pts[end])
            if d > max_dist:
                max_dist = d
                max_idx = i
        if max_dist > eps:
            mask[max_idx] = True
            _rdp_recursive(pts, eps, start, max_idx, mask)
            _rdp_recursive(pts, eps, max_idx, end, mask)
    
    mask = np.zeros(len(points), dtype=bool)
    mask[0] = True
    mask[-1] = True
    _rdp_recursive(points, epsilon, 0, len(points) - 1, mask)
    return points[mask]


def compute_path_length(points: np.ndarray) -> float:
    """Compute total length of polyline."""
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def compute_chord_length(points: np.ndarray) -> float:
    """Compute straight-line distance between endpoints."""
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[-1] - points[0]))


def compute_straightness(points: np.ndarray) -> float:
    """Compute straightness = chord_length / path_length."""
    path_len = compute_path_length(points)
    if path_len < 1e-9:
        return 1.0
    chord_len = compute_chord_length(points)
    return min(1.0, chord_len / path_len)


def compute_turning_angles(points: np.ndarray) -> Tuple[float, float]:
    """Compute total turning angle and curvature score."""
    if len(points) < 3:
        return 0.0, 0.0
    
    total_turn = 0.0
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-9 and n2 > 1e-9:
            cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            total_turn += abs(angle)
    
    path_len = compute_path_length(points)
    curvature = total_turn / max(1e-9, path_len)
    return total_turn, curvature


def remove_duplicate_points(points: np.ndarray) -> np.ndarray:
    """Remove consecutive duplicate points."""
    if len(points) < 2:
        return points
    mask = np.ones(len(points), dtype=bool)
    for i in range(1, len(points)):
        if np.allclose(points[i], points[i-1], atol=1e-6):
            mask[i] = False
    return points[mask]


def compute_edge_direction(points: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute normalized direction vector of an edge (endpoint to endpoint).
    Returns None if endpoints are too close.
    """
    if len(points) < 2:
        return None
    direction = points[-1] - points[0]
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return None
    return direction / length


def get_adaptive_rdp_epsilon(points: np.ndarray, config: dict) -> float:
    """
    Choose RDP epsilon based on straightness of the polyline.
    Straight lines use larger epsilon (fewer points), curves use smaller.
    """
    straightness = compute_straightness(points)
    threshold = config.get("rdp_straightness_threshold", 0.98)
    
    if straightness >= threshold:
        # Fairly straight - use larger epsilon for cleaner output
        return config.get("rdp_epsilon_straight", 2.0)
    else:
        # Curved - use smaller epsilon to preserve detail
        return config.get("rdp_epsilon_ref", 1.0)


def deterministic_palette(n: int) -> List[Tuple[int, int, int]]:
    """Generate deterministic color palette."""
    colors = []
    for i in range(max(n, 1)):
        hue = (i * 0.618033988749895) % 1.0
        h = int(hue * 180)
        hsv = np.array([[[h, 200, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(map(int, bgr)))
    return colors


# ---------------------------------------------------------------------------
# Line Fitting
# ---------------------------------------------------------------------------

def fit_line_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit line using PCA (total least squares). Returns (point_on_line, direction)."""
    if len(points) < 2:
        return points[0] if len(points) == 1 else np.array([0, 0]), np.array([1, 0])
    
    mean = np.mean(points, axis=0)
    centered = points - mean
    
    # SVD for PCA
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    
    # Normalize direction
    norm = np.linalg.norm(direction)
    if norm > 1e-9:
        direction = direction / norm
    else:
        direction = np.array([1.0, 0.0])
    
    return mean, direction


def point_to_line_distance(point: np.ndarray, line_point: np.ndarray, line_dir: np.ndarray) -> float:
    """Compute perpendicular distance from point to infinite line."""
    v = point - line_point
    # Project v onto line_dir, then compute perpendicular component
    proj_len = np.dot(v, line_dir)
    proj = proj_len * line_dir
    perp = v - proj
    return float(np.linalg.norm(perp))


def compute_line_errors(points: np.ndarray, line_point: np.ndarray, line_dir: np.ndarray) -> Tuple[float, float]:
    """Compute RMS and max error for line fit."""
    if len(points) == 0:
        return 0.0, 0.0
    errors = [point_to_line_distance(p, line_point, line_dir) for p in points]
    rms = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    max_err = float(np.max(errors))
    return rms, max_err


def create_line_primitive(p0: np.ndarray, p1: np.ndarray) -> dict:
    """Create line primitive object."""
    return {
        "type": "line",
        "p0": [float(p0[0]), float(p0[1])],
        "p1": [float(p1[0]), float(p1[1])]
    }


# ---------------------------------------------------------------------------
# Arc Fitting
# ---------------------------------------------------------------------------

def fit_circle_algebraic(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Algebraic circle fit (Kasa method). Returns (center, radius)."""
    if len(points) < 3:
        center = np.mean(points, axis=0) if len(points) > 0 else np.array([0, 0])
        return center, 1.0
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Set up system: (x-cx)^2 + (y-cy)^2 = r^2
    # Linearize: a*x + b*y + c = x^2 + y^2, where cx = a/2, cy = b/2, r^2 = c + cx^2 + cy^2
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x**2 + y**2
    
    # Least squares solve
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx = result[0] / 2.0
        cy = result[1] / 2.0
        r_sq = result[2] + cx**2 + cy**2
        radius = np.sqrt(max(r_sq, 1e-9))
    except:
        cx, cy = np.mean(points, axis=0)
        radius = np.mean([np.linalg.norm(p - np.array([cx, cy])) for p in points])
    
    return np.array([cx, cy]), float(radius)


def fit_circle_nonlinear(points: np.ndarray, init_center: np.ndarray, init_radius: float) -> Tuple[np.ndarray, float]:
    """Refine circle fit with nonlinear least squares (Huber loss)."""
    
    def residuals(params):
        cx, cy, r = params
        center = np.array([cx, cy])
        dists = np.linalg.norm(points - center, axis=1)
        return dists - r
    
    x0 = [init_center[0], init_center[1], init_radius]
    
    try:
        result = least_squares(residuals, x0, loss='huber', f_scale=1.0, max_nfev=100)
        cx, cy, r = result.x
        return np.array([cx, cy]), max(abs(r), 1.0)
    except:
        return init_center, init_radius


def compute_arc_angles(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Compute angles of points relative to center."""
    diffs = points - center
    return np.arctan2(diffs[:, 1], diffs[:, 0])


def determine_arc_direction(angles: np.ndarray) -> bool:
    """Determine if arc is clockwise based on angle progression. Returns True if CW."""
    if len(angles) < 2:
        return False
    
    # Unwrap angles to detect direction
    unwrapped = np.unwrap(angles)
    total_change = unwrapped[-1] - unwrapped[0]
    return total_change < 0  # CW if decreasing


def compute_sweep_consistency(points: np.ndarray, center: np.ndarray, theta0: float, theta1: float, cw: bool) -> float:
    """Compute how consistently points follow the arc sweep."""
    if len(points) < 3:
        return 1.0
    
    angles = compute_arc_angles(points, center)
    
    # Normalize angles relative to theta0
    def normalize_angle(a, ref):
        diff = a - ref
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
    
    # Check if angles are monotonic in expected direction
    normalized = [normalize_angle(a, theta0) for a in angles]
    
    monotonic_count = 0
    for i in range(1, len(normalized)):
        if cw:
            if normalized[i] <= normalized[i-1]:
                monotonic_count += 1
        else:
            if normalized[i] >= normalized[i-1]:
                monotonic_count += 1
    
    return monotonic_count / max(1, len(normalized) - 1)


def compute_arc_errors(points: np.ndarray, center: np.ndarray, radius: float) -> Tuple[float, float]:
    """Compute radial RMS and max error for arc fit."""
    if len(points) == 0:
        return 0.0, 0.0
    dists = np.linalg.norm(points - center, axis=1)
    errors = np.abs(dists - radius)
    rms = float(np.sqrt(np.mean(errors ** 2)))
    max_err = float(np.max(errors))
    return rms, max_err


def create_arc_primitive(center: np.ndarray, radius: float, theta0: float, theta1: float, cw: bool) -> dict:
    """Create arc primitive object."""
    return {
        "type": "arc",
        "center": [float(center[0]), float(center[1])],
        "radius": float(radius),
        "theta0": float(theta0),
        "theta1": float(theta1),
        "cw": bool(cw)
    }


# ---------------------------------------------------------------------------
# Bezier Fitting
# ---------------------------------------------------------------------------

def fit_cubic_bezier(points: np.ndarray, regularization: float = 1e-2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit cubic bezier with fixed endpoints. Returns (p0, p1, p2, p3)."""
    if len(points) < 2:
        p = points[0] if len(points) == 1 else np.array([0, 0])
        return p, p, p, p
    
    p0 = points[0].copy()
    p3 = points[-1].copy()
    
    if len(points) == 2:
        # Straight line
        p1 = p0 + (p3 - p0) / 3.0
        p2 = p0 + 2.0 * (p3 - p0) / 3.0
        return p0, p1, p2, p3
    
    # Chord-length parameterization
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0], np.cumsum(dists)])
    total = cumulative[-1]
    if total < 1e-9:
        p1 = p0 + (p3 - p0) / 3.0
        p2 = p0 + 2.0 * (p3 - p0) / 3.0
        return p0, p1, p2, p3
    
    t = cumulative / total
    
    # Bezier basis functions
    def B0(t): return (1 - t) ** 3
    def B1(t): return 3 * t * (1 - t) ** 2
    def B2(t): return 3 * t ** 2 * (1 - t)
    def B3(t): return t ** 3
    
    # We want: P(t) = B0(t)*p0 + B1(t)*p1 + B2(t)*p2 + B3(t)*p3 = points
    # Rearrange for p1, p2: B1(t)*p1 + B2(t)*p2 = points - B0(t)*p0 - B3(t)*p3
    
    b0 = B0(t)
    b1 = B1(t)
    b2 = B2(t)
    b3 = B3(t)
    
    # Right hand side
    rhs_x = points[:, 0] - b0 * p0[0] - b3 * p3[0]
    rhs_y = points[:, 1] - b0 * p0[1] - b3 * p3[1]
    
    # Build least squares matrix [b1, b2]
    A = np.column_stack([b1, b2])
    
    # Add regularization toward chord line
    chord = p3 - p0
    chord_dir = chord / max(np.linalg.norm(chord), 1e-9)
    default_p1 = p0 + chord / 3.0
    default_p2 = p0 + 2.0 * chord / 3.0
    
    # Regularized solve
    reg = regularization * len(points)
    ATA = A.T @ A + reg * np.eye(2)
    
    ATb_x = A.T @ rhs_x + reg * np.array([default_p1[0], default_p2[0]])
    ATb_y = A.T @ rhs_y + reg * np.array([default_p1[1], default_p2[1]])
    
    try:
        ctrl_x = np.linalg.solve(ATA, ATb_x)
        ctrl_y = np.linalg.solve(ATA, ATb_y)
    except:
        return p0, default_p1, default_p2, p3
    
    p1 = np.array([ctrl_x[0], ctrl_y[0]])
    p2 = np.array([ctrl_x[1], ctrl_y[1]])
    
    return p0, p1, p2, p3


def sample_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int = 60) -> np.ndarray:
    """Sample points along cubic bezier."""
    t = np.linspace(0, 1, n)
    t = t.reshape(-1, 1)
    points = ((1-t)**3) * p0 + 3*((1-t)**2)*t * p1 + 3*(1-t)*(t**2) * p2 + (t**3) * p3
    return points


def compute_bezier_errors(points: np.ndarray, p0: np.ndarray, p1: np.ndarray, 
                          p2: np.ndarray, p3: np.ndarray, n_samples: int = 60) -> Tuple[float, float]:
    """Compute error between points and bezier curve using KDTree."""
    if len(points) == 0:
        return 0.0, 0.0
    
    bezier_pts = sample_bezier(p0, p1, p2, p3, n_samples)
    tree = KDTree(bezier_pts)
    dists, _ = tree.query(points)
    rms = float(np.sqrt(np.mean(dists ** 2)))
    max_err = float(np.max(dists))
    return rms, max_err


def create_bezier_primitive(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> dict:
    """Create cubic bezier primitive object."""
    return {
        "type": "cubic",
        "p0": [float(p0[0]), float(p0[1])],
        "p1": [float(p1[0]), float(p1[1])],
        "p2": [float(p2[0]), float(p2[1])],
        "p3": [float(p3[0]), float(p3[1])]
    }


def create_polyline_primitive(points: np.ndarray) -> dict:
    """Create polyline fallback primitive."""
    return {
        "type": "polyline",
        "points": [[float(p[0]), float(p[1])] for p in points]
    }


# ---------------------------------------------------------------------------
# Parallel Line Detection (Cross-Hatching)
# ---------------------------------------------------------------------------

def detect_parallel_edges(edges: List[dict], config: dict) -> set:
    """
    Detect edges that are part of parallel hatching patterns.
    
    An edge is considered parallel if it has multiple nearby neighbors
    with similar direction. These edges should be fitted as lines.
    
    Args:
        edges: List of edge dicts from graph
        config: Configuration dict
        
    Returns:
        Set of edge IDs that are parallel (hatching)
    """
    angle_tol = np.radians(config.get("parallel_angle_tolerance_deg", 5.0))
    min_neighbors = config.get("parallel_min_neighbors", 2)
    search_radius = config.get("parallel_search_radius_px", 50.0)
    
    # Precompute edge info: direction and midpoint
    edge_info = []
    for edge in edges:
        polyline = np.array(edge["polyline"], dtype=np.float64)
        if len(polyline) < 2:
            continue
        
        # Direction vector (normalized)
        direction = compute_edge_direction(polyline)
        if direction is None:
            continue
        
        # Midpoint
        midpoint = polyline[len(polyline) // 2]
        
        # Length
        length = compute_path_length(polyline)
        
        edge_info.append({
            "id": edge["id"],
            "direction": direction,
            "midpoint": midpoint,
            "length": length
        })
    
    if len(edge_info) < 3:
        return set()
    
    # Build KD-tree on midpoints
    midpoints = np.array([e["midpoint"] for e in edge_info])
    tree = KDTree(midpoints)
    
    parallel_ids = set()
    
    for i, info in enumerate(edge_info):
        # Skip very long edges (likely structural, not hatching)
        if info["length"] > 150:
            continue
        
        # Find neighbors within search radius
        indices = tree.query_ball_point(info["midpoint"], search_radius)
        
        # Count neighbors with similar direction
        parallel_count = 0
        dir_i = info["direction"]
        
        for j in indices:
            if j == i:
                continue
            
            dir_j = edge_info[j]["direction"]
            
            # Check angle between directions
            # Use absolute dot product since direction can be flipped
            dot = abs(np.dot(dir_i, dir_j))
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.arccos(dot)
            
            if angle <= angle_tol:
                parallel_count += 1
        
        if parallel_count >= min_neighbors:
            parallel_ids.add(info["id"])
    
    return parallel_ids


# ---------------------------------------------------------------------------
# Edge Processing
# ---------------------------------------------------------------------------

def process_edge(edge: dict, nodes_by_id: dict, config: dict, 
                 error_tol: float, max_error_cap: float,
                 is_parallel: bool = False) -> dict:
    """Process a single edge and fit primitives.
    
    Args:
        edge: Edge dict from graph
        nodes_by_id: Node lookup
        config: Configuration
        error_tol: RMS error tolerance
        max_error_cap: Max error cap
        is_parallel: If True, force line fitting (parallel hatching detected)
    """
    
    # Extract and clean polyline
    raw_polyline = np.array(edge["polyline"], dtype=np.float64)
    points = remove_duplicate_points(raw_polyline)
    
    if len(points) < 2:
        points = raw_polyline[:2] if len(raw_polyline) >= 2 else np.array([[0, 0], [1, 1]])
    
    # Compute length and bucket
    length_px = compute_path_length(points)
    bucket = "structural" if length_px >= config["structural_min_length_px"] else "detail"
    
    # Use adaptive RDP epsilon based on straightness
    epsilon = get_adaptive_rdp_epsilon(points, config)
    simplified = rdp_simplify(points, epsilon)
    if len(simplified) < 2:
        simplified = points[:2] if len(points) >= 2 else points
    
    # Compute geometric descriptors
    straightness = compute_straightness(points)
    turning_sum, curvature = compute_turning_angles(points)
    
    # Endpoints (must match exactly)
    p0 = points[0]
    pN = points[-1]
    
    # Generate candidates
    candidates = []
    
    # --- Line candidate ---
    line_point, line_dir = fit_line_pca(points)
    line_rms, line_max = compute_line_errors(points, line_point, line_dir)
    
    line_candidate = {
        "type": "line",
        "primitive": create_line_primitive(p0, pN),
        "rms_error": line_rms,
        "max_error": line_max,
        "passed": False,
        "fail_reason": None
    }
    
    # Line acceptance
    if bucket == "structural":
        min_straight = config["line_min_straightness"]
        if straightness >= min_straight and line_rms <= error_tol and line_max <= max_error_cap:
            line_candidate["passed"] = True
        else:
            reasons = []
            if straightness < min_straight:
                reasons.append(f"straight={straightness:.3f}<{min_straight}")
            if line_rms > error_tol:
                reasons.append(f"rms={line_rms:.2f}>{error_tol}")
            if line_max > max_error_cap:
                reasons.append(f"max={line_max:.2f}>{max_error_cap}")
            line_candidate["fail_reason"] = "; ".join(reasons)
    else:
        min_straight = config["line_min_straightness_detail"]
        strict_tol = 0.8 * error_tol
        if straightness >= min_straight and line_rms <= strict_tol:
            line_candidate["passed"] = True
        else:
            line_candidate["fail_reason"] = f"detail: straight={straightness:.3f}, rms={line_rms:.2f}"
    
    candidates.append(line_candidate)
    
    # --- Arc candidate ---
    arc_candidate = None
    attempt_arc = (bucket == "structural" and 
                   length_px >= config["arc_min_length_px"] and
                   straightness <= config["arc_max_straightness"] and
                   turning_sum >= config["arc_min_turning"])
    
    if attempt_arc:
        # Fit circle
        init_center, init_radius = fit_circle_algebraic(points)
        center, radius = fit_circle_nonlinear(points, init_center, init_radius)
        
        arc_rms, arc_max = compute_arc_errors(points, center, radius)
        
        # Compute arc parameters
        angles = compute_arc_angles(points, center)
        theta0 = float(angles[0])
        theta1 = float(angles[-1])
        cw = determine_arc_direction(angles)
        
        sweep_consistency = compute_sweep_consistency(points, center, theta0, theta1, cw)
        
        arc_candidate = {
            "type": "arc",
            "primitive": create_arc_primitive(center, radius, theta0, theta1, cw),
            "rms_error": arc_rms,
            "max_error": arc_max,
            "radius": radius,
            "sweep_consistency": sweep_consistency,
            "passed": False,
            "fail_reason": None
        }
        
        # Arc acceptance
        passed = True
        reasons = []
        
        if radius > config["arc_max_radius_px"]:
            passed = False
            reasons.append(f"radius={radius:.0f}>{config['arc_max_radius_px']}")
        if arc_rms > error_tol:
            passed = False
            reasons.append(f"rms={arc_rms:.2f}>{error_tol}")
        if arc_max > max_error_cap:
            passed = False
            reasons.append(f"max={arc_max:.2f}>{max_error_cap}")
        if sweep_consistency < config["arc_min_sweep_consistency"]:
            passed = False
            reasons.append(f"sweep={sweep_consistency:.2f}<{config['arc_min_sweep_consistency']}")
        
        arc_candidate["passed"] = passed
        if not passed:
            arc_candidate["fail_reason"] = "; ".join(reasons)
        
        candidates.append(arc_candidate)
    
    # --- Bezier candidate ---
    bp0, bp1, bp2, bp3 = fit_cubic_bezier(points, config["bezier_regularization"])
    bezier_rms, bezier_max = compute_bezier_errors(points, bp0, bp1, bp2, bp3, config["bezier_sample_points"])
    
    # Check control point sanity
    chord_len = np.linalg.norm(pN - p0)
    ctrl_dist_1 = np.linalg.norm(bp1 - p0)
    ctrl_dist_2 = np.linalg.norm(bp2 - pN)
    max_ctrl_dist = config["bezier_max_ctrl_dist_factor"] * max(chord_len, 10)
    
    bezier_sane = ctrl_dist_1 <= max_ctrl_dist and ctrl_dist_2 <= max_ctrl_dist
    
    bezier_candidate = {
        "type": "cubic",
        "primitive": create_bezier_primitive(bp0, bp1, bp2, bp3),
        "rms_error": bezier_rms,
        "max_error": bezier_max,
        "passed": bezier_sane,
        "fail_reason": None if bezier_sane else "control points too far"
    }
    candidates.append(bezier_candidate)
    
    # --- Polyline fallback ---
    polyline_candidate = {
        "type": "polyline",
        "primitive": create_polyline_primitive(simplified),
        "rms_error": 0.0,
        "max_error": 0.0,
        "passed": True,
        "fail_reason": None
    }
    candidates.append(polyline_candidate)
    
    # --- Selection logic ---
    chosen = None
    
    # Force line for parallel edges (cross-hatching)
    if is_parallel and line_max <= max_error_cap * 1.5:
        # Parallel hatching detected - force line fitting with relaxed tolerance
        chosen = line_candidate
        line_candidate["passed"] = True
        line_candidate["fail_reason"] = None
    elif bucket == "detail":
        if line_candidate["passed"]:
            chosen = line_candidate
        else:
            chosen = polyline_candidate
    else:  # structural - prioritize editability
        # Collect valid candidates with their RMS errors
        valid_candidates = []
        if line_candidate["passed"]:
            valid_candidates.append(("line", line_candidate, line_rms))
        if arc_candidate is not None and arc_candidate["passed"]:
            # Demote large-radius arcs (nearly straight, better as line)
            arc_radius = arc_candidate.get("radius", 0)
            if arc_radius <= config["arc_radius_as_line_px"]:
                valid_candidates.append(("arc", arc_candidate, arc_rms))
        if bezier_candidate["passed"] and bezier_rms < 3.0:
            valid_candidates.append(("cubic", bezier_candidate, bezier_rms))
        
        if not valid_candidates:
            # Fallback to polyline (keeps sharp turns intact)
            chosen = polyline_candidate
        else:
            # Find best candidate by RMS
            best_type, best_cand, best_rms = min(valid_candidates, key=lambda x: x[2])
            
            # "Line wins by simplicity" rule: if line is close to best, pick line
            line_simplicity = config["line_simplicity_factor"]
            if line_candidate["passed"] and line_rms <= line_simplicity * best_rms and line_max <= max_error_cap:
                chosen = line_candidate
            else:
                chosen = best_cand
    
    # Compute confidence
    if chosen["type"] == "line":
        confidence = min(1.0, 0.5 + 0.5 * straightness - 0.1 * chosen["rms_error"])
    elif chosen["type"] == "arc":
        confidence = min(1.0, 0.6 + 0.4 * arc_candidate.get("sweep_consistency", 0.8) - 0.1 * chosen["rms_error"])
    elif chosen["type"] == "cubic":
        confidence = max(0.3, 0.7 - 0.1 * chosen["rms_error"])
    else:
        confidence = 0.3
    
    confidence = max(0.0, min(1.0, confidence))
    
    # Build result
    result = {
        "edge_id": edge["id"],
        "u": edge["u"],
        "v": edge["v"],
        "length_px": length_px,
        "bucket": bucket,
        "chosen": chosen["primitive"],
        "candidates": [
            {"type": c["type"], "rms_error": float(c["rms_error"]), "max_error": float(c["max_error"]), 
             "passed": bool(c["passed"]), "fail_reason": c["fail_reason"]}
            for c in candidates
        ],
        "polyline_simplified": [[float(p[0]), float(p[1])] for p in simplified],
        "quality": {
            "rms_error": chosen["rms_error"],
            "max_error": chosen["max_error"],
            "straightness": straightness,
            "curvature_score": curvature,
            "confidence": confidence
        }
    }
    
    return result


# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------

def draw_primitive(img: np.ndarray, prim: dict, color: Tuple[int, int, int], thickness: int = 1):
    """Draw a primitive on image."""
    ptype = prim["type"]
    
    if ptype == "line":
        p0 = tuple(map(int, prim["p0"]))
        p1 = tuple(map(int, prim["p1"]))
        cv2.line(img, p0, p1, color, thickness)
    
    elif ptype == "arc":
        center = tuple(map(int, prim["center"]))
        radius = int(prim["radius"])
        theta0 = prim["theta0"]
        theta1 = prim["theta1"]
        cw = prim.get("cw", False)
        
        # Compute sweep respecting direction
        # CW means decreasing angle (negative sweep), CCW means increasing (positive sweep)
        if cw:
            # Going clockwise: theta0 to theta1 in negative direction
            if theta1 > theta0:
                theta1 -= 2 * np.pi  # Wrap to go the other way
            sweep = theta1 - theta0  # Should be negative
        else:
            # Going counter-clockwise: theta0 to theta1 in positive direction
            if theta1 < theta0:
                theta1 += 2 * np.pi  # Wrap to go the other way
            sweep = theta1 - theta0  # Should be positive
        
        # Draw arc as polyline approximation
        n_pts = max(10, int(abs(sweep) * radius / 3))
        angles = np.linspace(theta0, theta0 + sweep, n_pts)
        pts = np.array([[center[0] + radius * np.cos(a), 
                         center[1] + radius * np.sin(a)] for a in angles], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, color, thickness)
    
    elif ptype == "cubic":
        p0 = np.array(prim["p0"])
        p1 = np.array(prim["p1"])
        p2 = np.array(prim["p2"])
        p3 = np.array(prim["p3"])
        pts = sample_bezier(p0, p1, p2, p3, 50).astype(np.int32)
        cv2.polylines(img, [pts], False, color, thickness)
    
    elif ptype == "polyline":
        pts = np.array(prim["points"], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, color, thickness)


def get_error_color(rms: float, bins: List[float]) -> Tuple[int, int, int]:
    """Map RMS error to color (green to red)."""
    if rms <= bins[0]:
        return (0, 255, 0)    # Green
    elif rms <= bins[1]:
        return (0, 255, 255)  # Yellow
    elif rms <= bins[2]:
        return (0, 165, 255)  # Orange
    elif rms <= bins[3]:
        return (0, 0, 255)    # Red
    else:
        return (128, 0, 128)  # Purple (very bad)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def fit_primitives(
    graph_path: str,
    mask_path: str = None,
    runs_root: str = "runs",
    debug: bool = True,
    config_path: str = None,
) -> Tuple[Path, Path, dict]:
    """Fit primitives to graph edges."""
    
    config = load_config(config_path)
    
    # Load graph
    with open(graph_path) as f:
        graph_data = json.load(f)
    
    img_info = graph_data["image"]
    h, w = img_info["height"], img_info["width"]
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    
    nodes_by_id = {n["id"]: n for n in nodes}
    
    # Infer run directory
    run_dir = infer_run_dir(graph_path, runs_root)
    
    # Load optional assets
    mask = None
    if mask_path and Path(mask_path).exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    original_path = find_original_input(run_dir)
    original_img = None
    if original_path:
        original_img = cv2.imread(str(original_path))
        if original_img is not None and original_img.shape[:2] != (h, w):
            original_img = cv2.resize(original_img, (w, h))
    
    base_img = original_img if original_img is not None else (
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask is not None else 
        np.zeros((h, w, 3), dtype=np.uint8))
    
    # Load stroke width for error tolerances
    stroke_width = load_stroke_width(run_dir)
    if stroke_width:
        error_tol = max(1.0, 0.25 * stroke_width)
        max_error_cap = max(2.0, 0.6 * stroke_width)
    else:
        error_tol = config["error_tolerance_px_default"]
        max_error_cap = config["max_error_cap_px_default"]
    
    # Create artifacts
    artifacts = StageArtifacts(run_dir, 60, "fit", debug=debug)
    
    # Detect parallel edges (cross-hatching)
    parallel_edge_ids = detect_parallel_edges(edges, config)
    if parallel_edge_ids:
        print(f"Parallel hatching detected: {len(parallel_edge_ids)} edges")
    
    # Process edges
    primitives = []
    fit_report_entries = []
    
    for edge in sorted(edges, key=lambda e: e["id"]):
        is_parallel = edge["id"] in parallel_edge_ids
        result = process_edge(edge, nodes_by_id, config, error_tol, max_error_cap,
                             is_parallel=is_parallel)
        primitives.append(result)
        
        fit_report_entries.append({
            "edge_id": result["edge_id"],
            "length_px": result["length_px"],
            "bucket": result["bucket"],
            "straightness": result["quality"]["straightness"],
            "curvature_score": result["quality"]["curvature_score"],
            "chosen_type": result["chosen"]["type"],
            "rms_error": result["quality"]["rms_error"],
            "candidates": result["candidates"]
        })
    
    # --- Debug visualizations ---
    if debug:
        # 1) Edges bucket visualization
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for prim in primitives:
            color = (0, 255, 0) if prim["bucket"] == "structural" else (100, 100, 100)
            pts = np.array(prim["polyline_simplified"], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(vis, [pts], False, color, 1)
        for node in nodes:
            cx, cy = int(node["centroid"][0]), int(node["centroid"][1])
            cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)
        artifacts.save_debug_image("edges_bucket_vis", vis)
        
        # 2) Chosen type visualization
        vis = (base_img.astype(np.float32) * 0.4).astype(np.uint8)
        type_colors = {"line": (255, 0, 0), "arc": (255, 0, 255), 
                       "cubic": (0, 165, 255), "polyline": (100, 100, 100)}
        for prim in primitives:
            color = type_colors.get(prim["chosen"]["type"], (255, 255, 255))
            draw_primitive(vis, prim["chosen"], color, 1)
        artifacts.save_debug_image("chosen_type_vis", vis)
        
        # 3) Error heatmap
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        bins = config["error_heatmap_bins"]
        for prim in primitives:
            rms = prim["quality"]["rms_error"]
            color = get_error_color(rms, bins)
            pts = np.array(prim["polyline_simplified"], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(vis, [pts], False, color, 1)
        artifacts.save_debug_image("error_heatmap_vis", vis)
        
        # 4) Failed arc candidates
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        failed_arc_count = 0
        for prim in primitives:
            arc_cand = next((c for c in prim["candidates"] if c["type"] == "arc"), None)
            if arc_cand and not arc_cand["passed"]:
                pts = np.array(prim["polyline_simplified"], dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(vis, [pts], False, (0, 0, 255), 1)
                    if failed_arc_count % 5 == 0 and arc_cand["fail_reason"]:
                        reason = arc_cand["fail_reason"][:20]
                        cv2.putText(vis, reason, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.3, (255, 255, 255), 1)
                    failed_arc_count += 1
        artifacts.save_debug_image("failed_arc_candidates", vis)
        
        # 5) Failed line candidates with high straightness
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for prim in primitives:
            line_cand = next((c for c in prim["candidates"] if c["type"] == "line"), None)
            if line_cand and not line_cand["passed"] and prim["quality"]["straightness"] > 0.95:
                pts = np.array(prim["polyline_simplified"], dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(vis, [pts], False, (0, 255, 255), 1)
        artifacts.save_debug_image("failed_line_candidates", vis)
        
        # 6) Top 20 longest edges detail (individual crops)
        sorted_by_len = sorted(primitives, key=lambda p: p["length_px"], reverse=True)
        for idx, prim in enumerate(sorted_by_len[:20]):
            pts = np.array(prim["polyline_simplified"])
            if len(pts) < 2:
                continue
            # Compute bounding box with padding
            x_min, y_min = pts.min(axis=0) - 30
            x_max, y_max = pts.max(axis=0) + 30
            x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
            x_max, y_max = min(w, int(x_max)), min(h, int(y_max))
            
            crop = base_img[y_min:y_max, x_min:x_max].copy()
            if crop.size == 0:
                continue
            
            # Draw polyline
            shifted_pts = (pts - np.array([x_min, y_min])).astype(np.int32)
            cv2.polylines(crop, [shifted_pts], False, (0, 255, 0), 1)
            
            # Draw chosen primitive (shifted)
            prim_copy = json.loads(json.dumps(prim["chosen"]))
            if prim_copy["type"] == "line":
                prim_copy["p0"] = [prim_copy["p0"][0] - x_min, prim_copy["p0"][1] - y_min]
                prim_copy["p1"] = [prim_copy["p1"][0] - x_min, prim_copy["p1"][1] - y_min]
            elif prim_copy["type"] == "cubic":
                for key in ["p0", "p1", "p2", "p3"]:
                    prim_copy[key] = [prim_copy[key][0] - x_min, prim_copy[key][1] - y_min]
            elif prim_copy["type"] == "arc":
                prim_copy["center"] = [prim_copy["center"][0] - x_min, prim_copy["center"][1] - y_min]
            elif prim_copy["type"] == "polyline":
                prim_copy["points"] = [[p[0] - x_min, p[1] - y_min] for p in prim_copy["points"]]
            
            draw_primitive(crop, prim_copy, (255, 0, 255), 2)
            
            # Add text
            info = f"#{prim['edge_id']} L={prim['length_px']:.0f} {prim['chosen']['type']} rms={prim['quality']['rms_error']:.2f}"
            cv2.putText(crop, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            artifacts.save_debug_image(f"top20_edge_{idx:02d}", crop)
        
        # 7) Arc quality gallery
        arcs = [p for p in primitives if p["chosen"]["type"] == "arc"]
        if arcs:
            vis = (base_img.astype(np.float32) * 0.3).astype(np.uint8)
            for prim in arcs[:20]:
                draw_primitive(vis, prim["chosen"], (255, 0, 255), 2)
                # Draw center
                center = prim["chosen"]["center"]
                cv2.circle(vis, (int(center[0]), int(center[1])), 3, (0, 255, 255), -1)
        else:
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(vis, "No arcs fitted", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        artifacts.save_debug_image("arc_quality_gallery", vis)
        
        # 8) Control points for cubics
        cubics = [p for p in primitives if p["chosen"]["type"] == "cubic"]
        vis = (base_img.astype(np.float32) * 0.3).astype(np.uint8)
        for prim in cubics[:50]:
            c = prim["chosen"]
            p0, p1, p2, p3 = np.array(c["p0"]), np.array(c["p1"]), np.array(c["p2"]), np.array(c["p3"])
            # Draw bezier
            draw_primitive(vis, c, (0, 165, 255), 1)
            # Draw control lines
            cv2.line(vis, tuple(p0.astype(int)), tuple(p1.astype(int)), (128, 128, 128), 1)
            cv2.line(vis, tuple(p3.astype(int)), tuple(p2.astype(int)), (128, 128, 128), 1)
            # Draw control points
            cv2.circle(vis, tuple(p1.astype(int)), 3, (0, 255, 0), -1)
            cv2.circle(vis, tuple(p2.astype(int)), 3, (0, 255, 0), -1)
        artifacts.save_debug_image("control_point_vis", vis)
        
        # 9) Final overlay on input
        vis = base_img.copy()
        for prim in primitives:
            ptype = prim["chosen"]["type"]
            if ptype == "line":
                color = (255, 0, 0)
            elif ptype == "arc":
                color = (255, 0, 255)
            elif ptype == "cubic":
                color = (0, 165, 255)
            else:
                color = (100, 100, 100)
            draw_primitive(vis, prim["chosen"], color, 1)
        
        for node in nodes:
            cx, cy = int(node["centroid"][0]), int(node["centroid"][1])
            color = (0, 255, 0) if node["type"] == "endpoint" else (0, 0, 255)
            cv2.circle(vis, (cx, cy), 2, color, -1)
        
        # Legend
        counts = {"line": 0, "arc": 0, "cubic": 0, "polyline": 0}
        for p in primitives:
            counts[p["chosen"]["type"]] += 1
        legend = f"L:{counts['line']} A:{counts['arc']} C:{counts['cubic']} P:{counts['polyline']}"
        cv2.putText(vis, legend, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        artifacts.save_debug_image("final_overlay_on_input", vis)
    
    # --- Build output ---
    output = {
        "image": img_info,
        "params": config,
        "nodes": nodes,
        "primitives": primitives
    }
    
    primitives_path = artifacts.path_out("primitives.json")
    primitives_path.write_text(json.dumps(output, indent=2, cls=NumpyEncoder))
    
    # Fit report
    artifacts.save_json("fit_report", {"edges": fit_report_entries})
    
    # Compute metrics
    structural_prims = [p for p in primitives if p["bucket"] == "structural"]
    type_counts = {"line": 0, "arc": 0, "cubic": 0, "polyline": 0}
    for p in primitives:
        type_counts[p["chosen"]["type"]] += 1
    
    structural_lines = sum(1 for p in structural_prims if p["chosen"]["type"] == "line")
    structural_line_fraction = structural_lines / max(1, len(structural_prims))
    
    arc_attempts = sum(1 for p in primitives for c in p["candidates"] if c["type"] == "arc")
    arc_successes = type_counts["arc"]
    
    line_attempts = len(primitives)  # All edges attempt line
    line_successes = type_counts["line"]
    
    structural_errors = [p["quality"]["rms_error"] for p in structural_prims]
    
    worst_edges = sorted(primitives, key=lambda p: p["quality"]["rms_error"], reverse=True)[:20]
    
    metrics = {
        "total_edges": len(primitives),
        "structural_edges": len(structural_prims),
        "detail_edges": len(primitives) - len(structural_prims),
        "type_counts": type_counts,
        "structural_line_fraction": structural_line_fraction,
        "arc_attempt_count": arc_attempts,
        "arc_success_count": arc_successes,
        "line_attempt_count": line_attempts,
        "line_success_count": line_successes,
        "rms_median_structural": float(np.median(structural_errors)) if structural_errors else 0,
        "rms_p90_structural": float(np.percentile(structural_errors, 90)) if structural_errors else 0,
        "top_20_worst_edges": [{"id": e["edge_id"], "rms": e["quality"]["rms_error"]} for e in worst_edges],
        "readiness_flags": {}
    }
    
    if structural_line_fraction < 0.4:
        metrics["readiness_flags"]["too_few_lines"] = True
    # Count structural polylines specifically (not all polylines)  
    structural_polylines = sum(1 for p in structural_prims if p["chosen"]["type"] == "polyline")
    if structural_polylines / max(1, len(structural_prims)) > 0.7:
        metrics["readiness_flags"]["too_many_polylines"] = True
    
    artifacts.write_metrics(metrics)
    
    return run_dir, primitives_path, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Primitive fitting for sketch-to-SVG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fit_primitives.py runs/clean/50_graph_clean/out/graph_clean.json --debug
    python fit_primitives.py runs/clean/50_graph_clean/out/graph_clean.json --mask runs/clean/10_preprocess/out/output_mask.png --debug
        """
    )
    
    parser.add_argument("graph_clean_path", type=str, help="Path to graph_clean.json")
    parser.add_argument("--runs_root", type=str, default="runs")
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no_debug", dest="debug", action="store_false")
    parser.add_argument("--config", type=str, default=None)
    parser.set_defaults(debug=True)
    
    args = parser.parse_args()
    
    print(f"Processing: {args.graph_clean_path}")
    
    run_dir, prim_path, metrics = fit_primitives(
        args.graph_clean_path,
        mask_path=args.mask,
        runs_root=args.runs_root,
        debug=args.debug,
        config_path=args.config,
    )
    
    print(f"Run directory: {run_dir}")
    print(f"Primitives: {prim_path}")
    print(f"Lines: {metrics['type_counts']['line']}, Arcs: {metrics['type_counts']['arc']}, "
          f"Cubics: {metrics['type_counts']['cubic']}, Polylines: {metrics['type_counts']['polyline']}")
    print(f"Structural line fraction: {metrics['structural_line_fraction']:.1%}")
    
    if metrics["readiness_flags"]:
        print(f"Readiness flags: {metrics['readiness_flags']}")
    
    return 0


if __name__ == "__main__":
    exit(main())


# ---------------------------------------------------------------------------
# Notes for Debug Artifacts
# ---------------------------------------------------------------------------
#
# 01_edges_bucket_vis.png
#    - Main strokes are structural (green)
#    - Hatching/detail is gray
#
# 02_chosen_type_vis.png
#    - Most main strokes are blue (lines)
#    - Curved sections magenta (arcs) or orange (cubics)
#    - Details mostly gray (polylines)
#
# 03_error_heatmap_vis.png
#    - High error (red/purple) should be limited to noisy/cluttered regions
#    - Main strokes should be green/yellow (low error)
#
# <06-25>_top20_edge_XX.png (multiple images)
#    - Each longest edge should be fitted correctly with low RMS error
#    - Magenta overlay should closely follow green polyline
