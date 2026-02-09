#!/usr/bin/env python3
"""
Stage 6.5: Geometry Regularization

Produces "production-grade" patent-style geometry by enforcing global consistency:
- Straight lines truly straight with consistent dominant directions
- Collinear segments merged
- Corners crisp
- Arcs and circles stabilized
- Hatch preserved (detail bucket untouched)

Usage:
    python regularize_geometry.py runs/<run>/60_fit/out/primitives.json --debug
    python regularize_geometry.py runs/<run>/60_fit/out/primitives.json --mask runs/<run>/10_preprocess/out/output_mask.png --debug
    python regularize_geometry.py runs/<run>/60_fit/out/primitives.json --debug --config configs/regularize.json

Outputs to runs/<run>/65_regularize/out/:
    - primitives_regularized.json: Regularized primitives
    - metrics.json: Summary statistics
    - regularize_report.json: Detailed operation logs
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.artifacts import StageArtifacts


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Global angle regularization
    "enable_angle_snap": True,
    "dominant_angle_bin_deg": 2.0,
    "dominant_peak_prominence": 0.08,
    "dominant_angle_merge_deg": 6.0,
    "snap_max_delta_deg": 5.0,
    "min_line_length_for_global": 60.0,
    "apply_snap_to_min_length": 30.0,
    
    # Collinearity merging
    "enable_collinear_merge": True,
    "collinear_angle_tol_deg": 3.0,
    "collinear_offset_tol_px": 1.5,
    "endpoint_snap_tol_px": 2.5,
    "merge_gap_tol_px": 4.0,
    "min_merge_length_px": 20.0,
    
    # Corner sharpening
    "enable_corner_sharpen": True,
    "corner_min_angle_change_deg": 20.0,
    "corner_max_adjust_px": 3.0,
    
    # Arc/circle stabilization
    "enable_arc_cluster": True,
    "arc_min_radius_px": 6.0,
    "arc_center_cluster_tol_px": 3.0,
    "arc_radius_cluster_tol_px": 3.0,
    "arc_angle_snap_deg": 2.0,
    "circle_completion": False,
    
    # Hatch preservation
    "hatch_bucket_name": "detail",
    "do_not_regularize_detail": True,
    
    # Precision
    "float_precision": 3,
    "safety_max_iters": 10,
}


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def normalize_angle_0_180(angle_deg: float) -> float:
    """Normalize angle to [0, 180) range for undirected lines."""
    angle_deg = angle_deg % 360.0
    if angle_deg >= 180.0:
        angle_deg -= 180.0
    if angle_deg < 0:
        angle_deg += 180.0
    return angle_deg


def angle_diff_180(a1: float, a2: float) -> float:
    """Compute minimum angle difference in [0, 180) space."""
    d = abs(normalize_angle_0_180(a1) - normalize_angle_0_180(a2))
    return min(d, 180.0 - d)


def line_angle_deg(p0: np.ndarray, p1: np.ndarray) -> float:
    """Compute angle of line from p0 to p1 in degrees [0, 180)."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return normalize_angle_0_180(angle_deg)


def line_length(p0: np.ndarray, p1: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p0))


def point_to_line_dist(pt: np.ndarray, line_pt: np.ndarray, line_dir: np.ndarray) -> float:
    """Distance from point to infinite line through line_pt with direction line_dir."""
    v = pt - line_pt
    line_dir_norm = line_dir / (np.linalg.norm(line_dir) + 1e-12)
    proj = np.dot(v, line_dir_norm) * line_dir_norm
    perp = v - proj
    return float(np.linalg.norm(perp))


def project_point_onto_line(pt: np.ndarray, line_pt: np.ndarray, line_dir: np.ndarray) -> np.ndarray:
    """Project point onto infinite line."""
    line_dir_norm = line_dir / (np.linalg.norm(line_dir) + 1e-12)
    t = np.dot(pt - line_pt, line_dir_norm)
    return line_pt + t * line_dir_norm


def line_line_intersection(p1: np.ndarray, d1: np.ndarray, 
                           p2: np.ndarray, d2: np.ndarray) -> Optional[np.ndarray]:
    """Find intersection of two lines, returns None if parallel."""
    # Line 1: p1 + t*d1
    # Line 2: p2 + s*d2
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return None  # Parallel
    
    dp = p2 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
    return p1 + t * d1


def arc_sweep_deg(theta0: float, theta1: float, cw: bool) -> float:
    """Compute arc sweep in degrees."""
    if cw:
        if theta1 > theta0:
            sweep = theta0 + 2*math.pi - theta1
        else:
            sweep = theta0 - theta1
    else:
        if theta1 < theta0:
            sweep = 2*math.pi - theta0 + theta1
        else:
            sweep = theta1 - theta0
    return abs(math.degrees(sweep))


def arc_endpoint(center: np.ndarray, radius: float, theta: float) -> np.ndarray:
    """Compute arc endpoint from center, radius, and angle."""
    return np.array([
        center[0] + radius * math.cos(theta),
        center[1] + radius * math.sin(theta)
    ])


# =============================================================================
# UNION-FIND FOR CLUSTERING
# =============================================================================

class UnionFind:
    """Deterministic union-find data structure."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True
    
    def get_clusters(self) -> Dict[int, List[int]]:
        clusters = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        return clusters


# =============================================================================
# STEP 0: LOAD AND BUILD WORKING REPRESENTATIONS
# =============================================================================

def load_primitives(path: Path) -> dict:
    """Load primitives.json."""
    with open(path, 'r') as f:
        return json.load(f)


def build_node_map(data: dict) -> Dict[int, np.ndarray]:
    """Build map from node_id to centroid."""
    node_map = {}
    for node in data.get("nodes", []):
        node_map[node["id"]] = np.array(node["centroid"], dtype=float)
    return node_map


def extract_line_geometry(chosen: dict) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Extract line geometry: p0, p1, angle_deg, length."""
    p0 = np.array(chosen["p0"], dtype=float)
    p1 = np.array(chosen["p1"], dtype=float)
    angle = line_angle_deg(p0, p1)
    length = line_length(p0, p1)
    return p0, p1, angle, length


def extract_arc_geometry(chosen: dict) -> Tuple[np.ndarray, float, float]:
    """Extract arc geometry: center, radius, sweep_deg."""
    center = np.array(chosen["center"], dtype=float)
    radius = float(chosen["radius"])
    sweep = arc_sweep_deg(chosen["theta0"], chosen["theta1"], chosen["cw"])
    return center, radius, sweep


# =============================================================================
# RENDERING UTILITIES
# =============================================================================

def get_background_image(mask_path: Optional[Path], input_path: Optional[Path],
                         img_size: Tuple[int, int]) -> np.ndarray:
    """Get background image for overlays."""
    if mask_path and mask_path.exists():
        img = cv2.imread(str(mask_path))
        if img is not None:
            return img
    if input_path and input_path.exists():
        img = cv2.imread(str(input_path))
        if img is not None:
            return img
    # White background
    return np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255


def draw_line_prim(img: np.ndarray, chosen: dict, color: Tuple[int,int,int], 
                   thickness: int = 1) -> None:
    """Draw a line primitive."""
    p0 = tuple(map(int, chosen["p0"]))
    p1 = tuple(map(int, chosen["p1"]))
    cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


def draw_arc_prim(img: np.ndarray, chosen: dict, color: Tuple[int,int,int],
                  thickness: int = 1) -> None:
    """Draw an arc primitive as polyline."""
    center = np.array(chosen["center"])
    radius = chosen["radius"]
    theta0 = chosen["theta0"]
    theta1 = chosen["theta1"]
    cw = chosen.get("cw", False)
    
    # Compute sweep
    if cw:
        if theta1 > theta0:
            theta1 -= 2 * np.pi
        sweep = theta1 - theta0
    else:
        if theta1 < theta0:
            theta1 += 2 * np.pi
        sweep = theta1 - theta0
    
    n_pts = max(10, int(abs(sweep) * radius / 3))
    angles = np.linspace(theta0, theta0 + sweep, n_pts)
    pts = np.array([[center[0] + radius * np.cos(a),
                     center[1] + radius * np.sin(a)] for a in angles], dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)


def draw_cubic_prim(img: np.ndarray, chosen: dict, color: Tuple[int,int,int],
                    thickness: int = 1) -> None:
    """Draw a cubic bezier as polyline."""
    p0 = np.array(chosen["p0"])
    p1 = np.array(chosen["p1"])
    p2 = np.array(chosen["p2"])
    p3 = np.array(chosen["p3"])
    
    t = np.linspace(0, 1, 50).reshape(-1, 1)
    pts = ((1-t)**3) * p0 + 3*((1-t)**2)*t * p1 + 3*(1-t)*(t**2) * p2 + (t**3) * p3
    pts = pts.astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)


def draw_polyline_prim(img: np.ndarray, chosen: dict, color: Tuple[int,int,int],
                       thickness: int = 1) -> None:
    """Draw a polyline primitive."""
    pts = np.array(chosen["points"], dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)


def draw_primitive(img: np.ndarray, prim: dict, color: Tuple[int,int,int],
                   thickness: int = 1) -> None:
    """Draw any primitive type."""
    chosen = prim.get("chosen", prim)  # Handle both full prim and just chosen
    ptype = chosen.get("type", "")
    
    if ptype == "line":
        draw_line_prim(img, chosen, color, thickness)
    elif ptype == "arc":
        draw_arc_prim(img, chosen, color, thickness)
    elif ptype == "cubic":
        draw_cubic_prim(img, chosen, color, thickness)
    elif ptype == "polyline":
        draw_polyline_prim(img, chosen, color, thickness)


def draw_all_primitives(img: np.ndarray, primitives: List[dict], 
                        structural_color: Tuple[int,int,int] = (0, 0, 0),
                        detail_color: Tuple[int,int,int] = (128, 128, 128),
                        thickness: int = 1) -> None:
    """Draw all primitives on image."""
    for prim in primitives:
        bucket = prim.get("bucket", "detail")
        color = structural_color if bucket == "structural" else detail_color
        draw_primitive(img, prim, color, thickness)


# =============================================================================
# STEP 1: IDENTIFY DOMINANT DIRECTIONS
# =============================================================================

def compute_dominant_angles(primitives: List[dict], cfg: dict) -> Tuple[List[float], Dict]:
    """Identify dominant line directions from structural lines."""
    min_length = cfg["min_line_length_for_global"]
    bin_deg = cfg["dominant_angle_bin_deg"]
    merge_deg = cfg["dominant_angle_merge_deg"]
    prominence_frac = cfg["dominant_peak_prominence"]
    
    angle_weights = []
    total_length = 0.0
    
    for prim in primitives:
        if prim.get("bucket") != "structural":
            continue
        chosen = prim.get("chosen", {})
        if chosen.get("type") != "line":
            continue
        p0, p1, angle, length = extract_line_geometry(chosen)
        if length >= min_length:
            angle_weights.append((angle, length))
            total_length += length
    
    if total_length < 1e-6 or len(angle_weights) == 0:
        return [], {"histogram": [], "peaks_raw": [], "peaks_merged": [],
                    "total_length": 0, "line_count": 0}
    
    n_bins = int(180.0 / bin_deg)
    hist = np.zeros(n_bins, dtype=float)
    for angle, weight in angle_weights:
        bin_idx = int(angle / bin_deg) % n_bins
        hist[bin_idx] += weight
    
    # Smooth histogram
    kernel = np.ones(3) / 3
    hist_padded = np.concatenate([hist[-3:], hist, hist[:3]])
    hist_smooth = np.convolve(hist_padded, kernel, mode='same')[3:-3]
    
    # Find peaks
    prominence_threshold = prominence_frac * total_length
    peaks_raw = []
    for i in range(n_bins):
        left = hist_smooth[(i - 1) % n_bins]
        center = hist_smooth[i]
        right = hist_smooth[(i + 1) % n_bins]
        if center > left and center > right and center >= prominence_threshold:
            peaks_raw.append(((i + 0.5) * bin_deg, center))
    
    peaks_raw.sort(key=lambda x: -x[1])
    
    # Merge nearby peaks
    peaks_merged = []
    used = set()
    for angle, weight in peaks_raw:
        if angle in used:
            continue
        cluster = [(angle, weight)]
        for a2, w2 in peaks_raw:
            if a2 != angle and angle_diff_180(angle, a2) < merge_deg:
                cluster.append((a2, w2))
                used.add(a2)
        total_w = sum(w for a, w in cluster)
        avg_angle = sum(a * w for a, w in cluster) / total_w
        peaks_merged.append((normalize_angle_0_180(avg_angle), total_w))
        used.add(angle)
    
    peaks_merged.sort(key=lambda x: -x[1])
    dominant_angles = [p[0] for p in peaks_merged]
    
    return dominant_angles, {
        "histogram": hist.tolist(), "bin_deg": bin_deg,
        "peaks_raw": peaks_raw, "peaks_merged": peaks_merged,
        "total_length": total_length, "line_count": len(angle_weights),
        "prominence_threshold": prominence_threshold
    }


def find_nearest_dominant_angle(angle: float, dominant_angles: List[float], 
                                max_delta: float) -> Optional[float]:
    """Find nearest dominant angle within max_delta."""
    if not dominant_angles:
        return None
    best_angle, best_diff = None, float('inf')
    for dom in dominant_angles:
        diff = angle_diff_180(angle, dom)
        if diff < best_diff and diff <= max_delta:
            best_diff, best_angle = diff, dom
    return best_angle


# =============================================================================
# STEP 2: SNAP STRUCTURAL LINES TO DOMINANT DIRECTIONS
# =============================================================================

def snap_lines_to_dominant(primitives: List[dict], dominant_angles: List[float], 
                           cfg: dict) -> Tuple[List[dict], List[dict]]:
    """Snap structural lines to nearest dominant angle."""
    min_length = cfg["apply_snap_to_min_length"]
    max_delta = cfg["snap_max_delta_deg"]
    max_move = cfg["corner_max_adjust_px"] * 2
    
    snapped_prims = []
    snap_log = []
    
    for prim in primitives:
        prim = copy.deepcopy(prim)
        
        if prim.get("bucket") != "structural":
            snapped_prims.append(prim)
            continue
        
        chosen = prim.get("chosen", {})
        if chosen.get("type") != "line":
            snapped_prims.append(prim)
            continue
        
        p0, p1, angle, length = extract_line_geometry(chosen)
        
        if length < min_length:
            snapped_prims.append(prim)
            continue
        
        snap_to = find_nearest_dominant_angle(angle, dominant_angles, max_delta)
        if snap_to is None:
            snapped_prims.append(prim)
            continue
        
        # Compute snapped direction
        snap_rad = math.radians(snap_to)
        snap_dir = np.array([math.cos(snap_rad), math.sin(snap_rad)])
        
        # Project endpoints onto snapped line through midpoint
        mid = (p0 + p1) / 2
        p0_new = project_point_onto_line(p0, mid, snap_dir)
        p1_new = project_point_onto_line(p1, mid, snap_dir)
        
        # Check movement limits
        move0 = line_length(p0, p0_new)
        move1 = line_length(p1, p1_new)
        
        if move0 > max_move or move1 > max_move:
            snapped_prims.append(prim)
            continue
        
        # Store before snapshot
        prim["before"] = copy.deepcopy(prim["chosen"])
        
        # Apply snap
        prim["chosen"]["p0"] = p0_new.tolist()
        prim["chosen"]["p1"] = p1_new.tolist()
        prim["regularized"] = True
        prim.setdefault("regularize_ops", []).append("angle_snap")
        prim["after"] = copy.deepcopy(prim["chosen"])
        
        snap_log.append({
            "edge_id": prim["edge_id"],
            "original_angle": angle,
            "snap_to": snap_to,
            "delta_deg": angle_diff_180(angle, snap_to),
            "move0": move0, "move1": move1
        })
        
        snapped_prims.append(prim)
    
    return snapped_prims, snap_log


# =============================================================================
# STEP 3: CORNER SHARPENING
# =============================================================================

def build_node_adjacency(primitives: List[dict]) -> Dict[int, List[dict]]:
    """Build adjacency: node_id -> list of incident structural line primitives."""
    adj = {}
    for prim in primitives:
        if prim.get("bucket") != "structural":
            continue
        chosen = prim.get("chosen", {})
        if chosen.get("type") != "line":
            continue
        for node_id in [prim["u"], prim["v"]]:
            if node_id not in adj:
                adj[node_id] = []
            adj[node_id].append(prim)
    return adj


def sharpen_corners(primitives: List[dict], node_map: Dict[int, np.ndarray],
                    cfg: dict) -> Tuple[List[dict], Dict[int, np.ndarray], List[dict]]:
    """Sharpen corners where two structural lines meet."""
    min_angle = cfg["corner_min_angle_change_deg"]
    max_adjust = cfg["corner_max_adjust_px"]
    
    adj = build_node_adjacency(primitives)
    corner_log = []
    new_node_map = copy.deepcopy(node_map)
    
    # Find nodes with exactly 2 incident structural lines
    for node_id in sorted(adj.keys()):
        incident = adj[node_id]
        if len(incident) != 2:
            continue
        
        prim1, prim2 = incident
        ch1, ch2 = prim1["chosen"], prim2["chosen"]
        
        # Get line directions
        p0_1, p1_1, _, _ = extract_line_geometry(ch1)
        p0_2, p1_2, _, _ = extract_line_geometry(ch2)
        
        # Determine which endpoint is at node
        node_pos = new_node_map.get(node_id)
        if node_pos is None:
            continue
        
        # Get directions pointing away from node
        if line_length(p0_1, node_pos) < line_length(p1_1, node_pos):
            dir1 = p1_1 - p0_1
        else:
            dir1 = p0_1 - p1_1
        
        if line_length(p0_2, node_pos) < line_length(p1_2, node_pos):
            dir2 = p1_2 - p0_2
        else:
            dir2 = p0_2 - p1_2
        
        # Compute angle between lines
        len1, len2 = np.linalg.norm(dir1), np.linalg.norm(dir2)
        if len1 < 1e-6 or len2 < 1e-6:
            continue
        
        cos_angle = np.dot(dir1, dir2) / (len1 * len2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_deg = math.degrees(math.acos(abs(cos_angle)))
        
        if angle_deg < min_angle:
            continue
        
        # Compute intersection of infinite lines
        intersection = line_line_intersection(p0_1, p1_1 - p0_1, p0_2, p1_2 - p0_2)
        if intersection is None:
            continue
        
        move_dist = line_length(node_pos, intersection)
        if move_dist > max_adjust:
            continue
        
        # Update node position
        old_pos = node_pos.copy()
        new_node_map[node_id] = intersection
        
        corner_log.append({
            "node_id": node_id,
            "old_pos": old_pos.tolist(),
            "new_pos": intersection.tolist(),
            "move_dist": move_dist,
            "angle_deg": angle_deg
        })
    
    # Update primitives to use new node positions
    updated_prims = []
    for prim in primitives:
        prim = copy.deepcopy(prim)
        chosen = prim.get("chosen", {})
        
        if chosen.get("type") == "line":
            u_id, v_id = prim["u"], prim["v"]
            if u_id in new_node_map:
                new_p = new_node_map[u_id]
                old_p = np.array(chosen["p0"])
                if line_length(old_p, new_p) < cfg["corner_max_adjust_px"] * 1.5:
                    chosen["p0"] = new_p.tolist()
            if v_id in new_node_map:
                new_p = new_node_map[v_id]
                old_p = np.array(chosen["p1"])
                if line_length(old_p, new_p) < cfg["corner_max_adjust_px"] * 1.5:
                    chosen["p1"] = new_p.tolist()
        
        updated_prims.append(prim)
    
    return updated_prims, new_node_map, corner_log


# =============================================================================
# STEP 5: ARC CLUSTERING
# =============================================================================

def cluster_arcs(primitives: List[dict], cfg: dict) -> Tuple[List[dict], List[dict]]:
    """Cluster arcs by center and radius for consistent circular features."""
    center_tol = cfg["arc_center_cluster_tol_px"]
    radius_tol = cfg["arc_radius_cluster_tol_px"]
    min_radius = cfg["arc_min_radius_px"]
    
    # Collect structural arcs
    arc_indices = []
    arc_data = []
    
    for i, prim in enumerate(primitives):
        if prim.get("bucket") != "structural":
            continue
        chosen = prim.get("chosen", {})
        if chosen.get("type") != "arc":
            continue
        center, radius, sweep = extract_arc_geometry(chosen)
        if radius < min_radius:
            continue
        arc_indices.append(i)
        arc_data.append((center, radius, sweep))
    
    if len(arc_data) < 2:
        return primitives, []
    
    # Cluster using union-find
    n = len(arc_data)
    uf = UnionFind(n)
    
    for i in range(n):
        for j in range(i + 1, n):
            c1, r1, _ = arc_data[i]
            c2, r2, _ = arc_data[j]
            center_dist = line_length(c1, c2)
            radius_diff = abs(r1 - r2)
            if center_dist <= center_tol and radius_diff <= radius_tol:
                uf.union(i, j)
    
    clusters = uf.get_clusters()
    cluster_log = []
    
    # Update primitives
    updated_prims = copy.deepcopy(primitives)
    
    for root, members in clusters.items():
        if len(members) < 2:
            continue
        
        # Compute weighted average center and radius
        total_sweep = sum(arc_data[m][2] for m in members)
        if total_sweep < 1e-6:
            continue
        
        avg_center = np.zeros(2)
        avg_radius = 0.0
        for m in members:
            c, r, s = arc_data[m]
            avg_center += c * s
            avg_radius += r * s
        avg_center /= total_sweep
        avg_radius /= total_sweep
        
        # Apply to each arc in cluster
        for m in members:
            idx = arc_indices[m]
            prim = updated_prims[idx]
            old_chosen = copy.deepcopy(prim["chosen"])
            
            prim["chosen"]["center"] = avg_center.tolist()
            prim["chosen"]["radius"] = float(avg_radius)
            prim["regularized"] = True
            prim.setdefault("regularize_ops", []).append("arc_cluster")
            prim["before"] = old_chosen
            prim["after"] = copy.deepcopy(prim["chosen"])
        
        cluster_log.append({
            "cluster_size": len(members),
            "edge_ids": [primitives[arc_indices[m]]["edge_id"] for m in members],
            "avg_center": avg_center.tolist(),
            "avg_radius": avg_radius
        })
    
    return updated_prims, cluster_log


# =============================================================================
# MAIN REGULARIZATION PIPELINE
# =============================================================================

def regularize_primitives(data: dict, cfg: dict, artifacts: Optional[StageArtifacts],
                          mask_path: Optional[Path], input_path: Optional[Path]) -> dict:
    """Run full regularization pipeline."""
    primitives = data.get("primitives", [])
    nodes = data.get("nodes", [])
    img_size = (data.get("image", {}).get("width", 1024),
                data.get("image", {}).get("height", 1024))
    
    node_map = build_node_map(data)
    report = {"config": cfg, "operations": {}}
    
    # Debug: render input overlay
    if artifacts and artifacts.debug_enabled:
        bg = get_background_image(mask_path, input_path, img_size)
        draw_all_primitives(bg, primitives, (0, 0, 0), (180, 180, 180), 1)
        artifacts.save_debug_image("input_overlay", bg)
    
    # Step 1: Compute dominant angles
    dominant_angles = []
    if cfg.get("enable_angle_snap", True):
        dominant_angles, angle_analysis = compute_dominant_angles(primitives, cfg)
        report["operations"]["dominant_angles"] = {
            "angles_deg": dominant_angles,
            "line_count": angle_analysis.get("line_count", 0),
            "total_length": angle_analysis.get("total_length", 0)
        }
        
        # Debug: angle histogram
        if artifacts and artifacts.debug_enabled:
            with open(artifacts.debug_dir / "dominant_angles.txt", "w") as f:
                f.write("Dominant angles (deg) with weights:\n")
                for a, w in angle_analysis.get("peaks_merged", []):
                    f.write(f"  {a:.1f} deg, weight={w:.1f}\n")
    
    # Step 2: Snap lines to dominant angles
    snap_log = []
    if cfg.get("enable_angle_snap", True) and dominant_angles:
        primitives, snap_log = snap_lines_to_dominant(primitives, dominant_angles, cfg)
        report["operations"]["line_snapping"] = {
            "snapped_count": len(snap_log),
            "details": snap_log[:20]  # First 20
        }
        
        if artifacts and artifacts.debug_enabled:
            bg = get_background_image(mask_path, input_path, img_size)
            draw_all_primitives(bg, primitives, (0, 0, 0), (180, 180, 180), 1)
            artifacts.save_debug_image("after_angle_snap", bg)
    
    # Step 3: Corner sharpening
    corner_log = []
    if cfg.get("enable_corner_sharpen", True):
        primitives, node_map, corner_log = sharpen_corners(primitives, node_map, cfg)
        report["operations"]["corner_sharpening"] = {
            "corners_sharpened": len(corner_log),
            "details": corner_log[:20]
        }
        
        if artifacts and artifacts.debug_enabled:
            bg = get_background_image(mask_path, input_path, img_size)
            draw_all_primitives(bg, primitives, (0, 0, 0), (180, 180, 180), 1)
            artifacts.save_debug_image("after_corners", bg)
    
    # Step 5: Arc clustering
    arc_log = []
    if cfg.get("enable_arc_cluster", True):
        primitives, arc_log = cluster_arcs(primitives, cfg)
        report["operations"]["arc_clustering"] = {
            "clusters_count": len(arc_log),
            "details": arc_log
        }
        
        if artifacts and artifacts.debug_enabled:
            bg = get_background_image(mask_path, input_path, img_size)
            draw_all_primitives(bg, primitives, (0, 0, 0), (180, 180, 180), 1)
            artifacts.save_debug_image("after_arcs", bg)
    
    # Final overlay
    if artifacts and artifacts.debug_enabled:
        bg = get_background_image(mask_path, input_path, img_size)
        draw_all_primitives(bg, primitives, (0, 0, 0), (128, 128, 128), 1)
        artifacts.save_debug_image("final_regularized", bg)
    
    # Build output
    output_data = copy.deepcopy(data)
    output_data["primitives"] = primitives
    
    # Update nodes with new positions
    for node in output_data.get("nodes", []):
        nid = node["id"]
        if nid in node_map:
            node["centroid"] = node_map[nid].tolist()
    
    output_data["regularization"] = {
        "config": cfg,
        "summary": {
            "dominant_angles_deg": dominant_angles,
            "lines_snapped": len(snap_log),
            "corners_sharpened": len(corner_log),
            "arc_clusters": len(arc_log)
        }
    }
    
    return output_data, report


def run_regularization(primitives_path: str, mask_path: Optional[str] = None,
                       input_path: Optional[str] = None, debug: bool = True,
                       config_path: Optional[str] = None) -> None:
    """Run regularization stage."""
    prim_path = Path(primitives_path)
    
    # Determine run directory
    # Expected: runs/<run>/60_fit/out/primitives.json
    run_dir = prim_path.parent.parent.parent
    
    # Load data
    data = load_primitives(prim_path)
    print(f"Processing: {prim_path}")
    print(f"Primitives: {len(data.get('primitives', []))}")
    
    # Build config
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path:
        with open(config_path, 'r') as f:
            user_cfg = json.load(f)
            cfg.update(user_cfg)
    
    # Setup artifacts
    artifacts = StageArtifacts(run_dir, stage_id=65, stage_name="regularize", debug=debug)
    
    # Resolve paths
    mask_p = Path(mask_path) if mask_path else None
    if mask_p is None:
        mask_p = run_dir / "10_preprocess" / "out" / "output_mask.png"
    
    input_p = Path(input_path) if input_path else None
    if input_p is None:
        input_dir = run_dir / "00_input"
        if input_dir.exists():
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = input_dir / f"01_input{ext}"
                if candidate.exists():
                    input_p = candidate
                    break
    
    # Run regularization
    output_data, report = regularize_primitives(data, cfg, artifacts, mask_p, input_p)
    
    # Write outputs
    out_path = artifacts.out_dir / "primitives_regularized.json"
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Output: {out_path}")
    
    # Write report
    report_path = artifacts.out_dir / "regularize_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Write metrics
    summary = output_data.get("regularization", {}).get("summary", {})
    metrics = {
        "primitives_count": len(output_data.get("primitives", [])),
        "dominant_angles_count": len(summary.get("dominant_angles_deg", [])),
        "lines_snapped": summary.get("lines_snapped", 0),
        "corners_sharpened": summary.get("corners_sharpened", 0),
        "arc_clusters": summary.get("arc_clusters", 0)
    }
    artifacts.write_metrics(metrics)
    print(f"Metrics: {metrics}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stage 6.5: Geometry Regularization"
    )
    parser.add_argument("primitives_path", help="Path to primitives.json from Stage 6")
    parser.add_argument("--mask", help="Path to mask image for overlays")
    parser.add_argument("--input_image", help="Path to input image for overlays")
    parser.add_argument("--debug", action="store_true", default=True, 
                        help="Enable debug outputs (default: True)")
    parser.add_argument("--no_debug", action="store_true", help="Disable debug outputs")
    parser.add_argument("--config", help="Path to config JSON to merge with defaults")
    
    args = parser.parse_args()
    
    debug = True if args.debug and not args.no_debug else False
    
    run_regularization(
        args.primitives_path,
        mask_path=args.mask,
        input_path=args.input_image,
        debug=debug,
        config_path=args.config
    )


if __name__ == "__main__":
    main()


# =============================================================================
# VERIFICATION INSTRUCTIONS
# =============================================================================
"""
What "production grade" should look like after this stage:
- Main outlines become straighter and aligned to dominant directions
- Corners become crisp (no almost-intersections)
- Circular tape roll arcs look concentric and smooth
- Hatch remains unchanged (detail bucket passed through)

What to inspect:
- 01_input_overlay.png: Baseline "before" view
- 03_after_angle_snap.png: Should show straightened lines aligned to dominant angles
- 05_after_corners.png: Should show crisp corners where lines meet
- 07_after_arcs.png: Arcs should share centers where appropriate
- 09_final_regularized.png: Complete regularized output

Debug files:
- dominant_angles.txt: List of detected dominant directions
- regularize_report.json: Detailed per-operation logs
- metrics.json: Summary statistics

If lines are not snapping:
- Check if min_line_length_for_global threshold is too high
- Check if snap_max_delta_deg is too restrictive

If corners are not sharpening:
- Check corner_min_angle_change_deg threshold
- Check corner_max_adjust_px movement limit

If arcs are not clustering:
- Check arc_center_cluster_tol_px and arc_radius_cluster_tol_px
"""
