"""
Stage 4: Graph Build for Sketch-to-SVG Pipeline

Converts the ridge mask from Stage 3 into a topological graph with nodes
(endpoints and junctions) and edges (paths between nodes). This produces
a faithful raw graph with minimal normalization for Stage 5 to clean up.

Output:
- graph_raw.json: nodes and edges with polyline coordinates
- metrics.json: graph statistics
"""

# python graph_build.py runs/<run>/30_ridge/out/ridge.png --debug
# python graph_build.py runs/<run>/30_ridge/out/ridge.png --mask runs/<run>/10_preprocess/out/output_mask.png --debug
# python graph_build.py runs/<run>/30_ridge/out/ridge.png --debug --config configs/graph_build.json

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import label
from collections import defaultdict

from utils.artifacts import make_run_dir, StageArtifacts


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "connectivity": 8,                      # use 8-connected neighbors
    "min_node_component_area": 1,           # endpoints can be single pixels
    "min_edge_length": 3,                   # discard tiny edges (noise)
    "max_trace_steps": 200000,              # safety limit for tracing
    "degree_endpoint": 1,                   # degree for endpoints
    "degree_junction_min": 3,               # minimum degree for junctions
    "simplify_polyline": False,             # no RDP here, leave for Stage 5
    "allow_diagonal": True,                 # with 8-connect, diagonals happen
    "debug_draw_edge_every_n": 1,           # draw every Nth edge
    "debug_label_every_n": 10,              # label every Nth edge with id
    "min_ridge_component_area_for_noise": 1, # no removal by default
}

# 8-connectivity neighbor offsets (deterministic order)
NEIGHBOR_OFFSETS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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


def infer_run_dir(ridge_path: str, runs_root: str) -> Path:
    """
    Infer run directory from ridge path.
    
    If ridge_path is under runs/<run>/..., returns that run directory.
    Otherwise creates a new run directory.
    """
    ridge_p = Path(ridge_path).resolve()
    runs_root_p = Path(runs_root).resolve()
    
    try:
        rel_path = ridge_p.relative_to(runs_root_p)
        run_name = rel_path.parts[0]
        return runs_root_p / run_name
    except ValueError:
        return make_run_dir(ridge_path, runs_root)


def find_original_input(run_dir: Path) -> Path:
    """Try to locate original input image in run directory."""
    input_dir = run_dir / "00_input"
    if not input_dir.exists():
        return None
    
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = input_dir / f"01_input{ext}"
        if candidate.exists():
            return candidate
    return None


def deterministic_color_palette(n: int, seed: int = 42) -> list:
    """Generate deterministic color palette for visualization."""
    np.random.seed(seed)
    colors = []
    for i in range(n):
        # Use golden ratio for better color distribution
        hue = (i * 0.618033988749895) % 1.0
        # Convert to BGR
        h = int(hue * 180)
        hsv = np.array([[[h, 200, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(map(int, bgr)))
    return colors


# ---------------------------------------------------------------------------
# Part A: Load and normalize ridge
# ---------------------------------------------------------------------------

def load_ridge(ridge_path: str, config: dict) -> tuple:
    """
    Load ridge mask and convert to boolean.
    
    Returns:
        Tuple of (ridge_bool, original_ridge_count).
    """
    img = cv2.imread(str(ridge_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read ridge image: {ridge_path}")
    
    ridge = img > 0
    original_count = int(ridge.sum())
    
    # Optional noise removal
    min_area = config.get("min_ridge_component_area_for_noise", 1)
    if min_area > 1:
        labeled, num = label(ridge, structure=np.ones((3, 3)))
        sizes = np.bincount(labeled.ravel())
        keep = np.zeros_like(ridge)
        for i in range(1, num + 1):
            if sizes[i] >= min_area:
                keep |= (labeled == i)
        ridge = keep
    
    return ridge, original_count


# ---------------------------------------------------------------------------
# Part B: Compute pixel degree map
# ---------------------------------------------------------------------------

def compute_degree_map(ridge: np.ndarray) -> np.ndarray:
    """
    Compute degree (neighbor count) for each ridge pixel.
    
    Uses 8-connectivity. Returns array with degree values for ridge pixels,
    0 for background.
    """
    h, w = ridge.shape
    degree = np.zeros((h, w), dtype=np.int32)
    ridge_int = ridge.astype(np.int32)
    
    # For each neighbor offset, add 1 where both pixel and neighbor are ridge
    for dy, dx in NEIGHBOR_OFFSETS_8:
        # Shift ridge and AND with original
        shifted = np.zeros_like(ridge_int)
        
        # Source and destination slices
        if dy < 0:
            src_y = slice(0, h + dy)
            dst_y = slice(-dy, h)
        elif dy > 0:
            src_y = slice(dy, h)
            dst_y = slice(0, h - dy)
        else:
            src_y = slice(0, h)
            dst_y = slice(0, h)
            
        if dx < 0:
            src_x = slice(0, w + dx)
            dst_x = slice(-dx, w)
        elif dx > 0:
            src_x = slice(dx, w)
            dst_x = slice(0, w - dx)
        else:
            src_x = slice(0, w)
            dst_x = slice(0, w)
        
        shifted[dst_y, dst_x] = ridge_int[src_y, src_x]
        degree += ridge_int * shifted
    
    return degree


def visualize_degree_map(degree: np.ndarray, ridge: np.ndarray) -> np.ndarray:
    """
    Create BGR visualization of degree map.
    
    - Degree 0 (background): black
    - Degree 1 (endpoints): green
    - Degree 2 (path): gray
    - Degree >= 3 (junctions): red
    """
    h, w = degree.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Degree 2: gray
    vis[degree == 2] = [128, 128, 128]
    
    # Degree 1: green
    vis[degree == 1] = [0, 255, 0]
    
    # Degree >= 3: red
    vis[degree >= 3] = [0, 0, 255]
    
    return vis


# ---------------------------------------------------------------------------
# Part C: Identify node components
# ---------------------------------------------------------------------------

def identify_node_components(ridge: np.ndarray, degree: np.ndarray, config: dict) -> tuple:
    """
    Identify node components (endpoints and junctions).
    
    Returns:
        Tuple of (nodes_list, node_pixel_mask, node_labels).
    """
    # Node pixels: endpoints (degree 1) or junctions (degree >= 3)
    node_pixels = ridge & ((degree == config["degree_endpoint"]) | 
                           (degree >= config["degree_junction_min"]))
    
    # Label connected components
    structure = np.ones((3, 3))  # 8-connectivity
    node_labels, num_components = label(node_pixels, structure=structure)
    
    nodes = []
    
    for comp_id in range(1, num_components + 1):
        comp_mask = node_labels == comp_id
        ys, xs = np.where(comp_mask)
        
        if len(ys) == 0:
            continue
        
        # Get degrees of pixels in this component
        comp_degrees = degree[comp_mask]
        
        # Determine node type
        if np.all(comp_degrees == 1):
            node_type = "endpoint"
            degree_hint = 1
        else:
            node_type = "junction"
            degree_hint = int(np.max(comp_degrees))
        
        # Compute centroid (x, y format)
        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))
        
        # Compute bbox
        x0, y0 = int(np.min(xs)), int(np.min(ys))
        x1, y1 = int(np.max(xs)), int(np.max(ys))
        
        node = {
            "id": len(nodes),
            "type": node_type,
            "centroid": [centroid_x, centroid_y],
            "pixel_count": len(ys),
            "pixels_bbox": [x0, y0, x1, y1],
            "degree_hint": degree_hint,
            "_pixels": list(zip(ys.tolist(), xs.tolist())),  # internal use
        }
        nodes.append(node)
    
    return nodes, node_pixels, node_labels


def visualize_nodes_on_ridge(ridge: np.ndarray, nodes: list) -> np.ndarray:
    """Create BGR visualization with nodes overlayed on ridge."""
    h, w = ridge.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Ridge in gray
    vis[ridge] = [100, 100, 100]
    
    for node in nodes:
        cx, cy = node["centroid"]
        cx, cy = int(cx), int(cy)
        
        if node["type"] == "endpoint":
            color = (0, 255, 0)  # green
            radius = 3
        else:
            color = (0, 0, 255)  # red
            radius = 4
        
        cv2.circle(vis, (cx, cy), radius, color, -1)
        
        # Draw bbox for junctions
        if node["type"] == "junction":
            x0, y0, x1, y1 = node["pixels_bbox"]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 200), 1)
    
    return vis


def visualize_nodes_labeled(node_labels: np.ndarray, nodes: list) -> np.ndarray:
    """Create visualization with distinct colors per node component."""
    h, w = node_labels.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    palette = deterministic_color_palette(max(len(nodes), 1) + 1)
    
    for node in nodes:
        color = palette[node["id"] % len(palette)]
        mask = node_labels == (node["id"] + 1)  # labels are 1-indexed
        vis[mask] = color
    
    return vis


# ---------------------------------------------------------------------------
# Part D: Trace edges
# ---------------------------------------------------------------------------

def build_node_pixel_lookup(nodes: list, h: int, w: int) -> np.ndarray:
    """
    Build array mapping pixel (y,x) to node id.
    
    Returns array with -1 for non-node pixels, node_id otherwise.
    """
    lookup = np.full((h, w), -1, dtype=np.int32)
    
    for node in nodes:
        for y, x in node.get("_pixels", []):
            lookup[y, x] = node["id"]
    
    return lookup


def get_ridge_neighbors(y: int, x: int, ridge: np.ndarray) -> list:
    """Get list of ridge neighbor coordinates in deterministic order."""
    h, w = ridge.shape
    neighbors = []
    
    for dy, dx in NEIGHBOR_OFFSETS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and ridge[ny, nx]:
            neighbors.append((ny, nx))
    
    return neighbors


def find_edge_start_pixels(nodes: list, ridge: np.ndarray, 
                            node_pixels: np.ndarray, node_lookup: np.ndarray) -> list:
    """
    Find start pixels for edge tracing.
    
    For each node, find adjacent ridge pixels that are not node pixels.
    Returns list of (node_id, start_y, start_x) tuples.
    """
    h, w = ridge.shape
    starts = []
    
    for node in nodes:
        # Collect boundary of node component
        for y, x in node.get("_pixels", []):
            for dy, dx in NEIGHBOR_OFFSETS_8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if ridge[ny, nx] and not node_pixels[ny, nx]:
                        starts.append((node["id"], ny, nx))
    
    # Remove duplicates and sort deterministically
    starts = list(set(starts))
    starts.sort(key=lambda t: (t[0], t[1], t[2]))
    
    return starts


def trace_edge(start_node_id: int, start_y: int, start_x: int,
               ridge: np.ndarray, node_pixels: np.ndarray, 
               node_lookup: np.ndarray, visited: np.ndarray,
               nodes: list, config: dict) -> tuple:
    """
    Trace an edge from a start pixel until reaching another node.
    
    Returns:
        Tuple of (edge_dict, new_nodes_list) or (None, new_nodes_list) if invalid.
    """
    max_steps = config["max_trace_steps"]
    min_length = config["min_edge_length"]
    h, w = ridge.shape
    
    polyline = []
    new_nodes = []
    
    # Start from the start pixel
    curr_y, curr_x = start_y, start_x
    prev_y, prev_x = -1, -1  # Will be set to a node pixel
    
    # Find a node pixel adjacent to start to use as prev
    for node in nodes:
        if node["id"] == start_node_id:
            for py, px in node.get("_pixels", []):
                if abs(py - start_y) <= 1 and abs(px - start_x) <= 1:
                    prev_y, prev_x = py, px
                    break
            break
    
    if prev_y < 0:
        # Fallback: use start itself
        prev_y, prev_x = start_y, start_x
    
    steps = 0
    end_node_id = None
    
    while steps < max_steps:
        # Check if current pixel is in a node
        node_id = node_lookup[curr_y, curr_x]
        if node_id >= 0 and node_id != start_node_id:
            end_node_id = node_id
            polyline.append([int(curr_x), int(curr_y)])
            break
        
        # Check for on-the-fly junction (degree >= 3 but not in node_pixels)
        neighbors = get_ridge_neighbors(curr_y, curr_x, ridge)
        valid_neighbors = [(ny, nx) for ny, nx in neighbors 
                           if (ny, nx) != (prev_y, prev_x)]
        
        # Add current to polyline
        polyline.append([int(curr_x), int(curr_y)])
        
        # Mark as visited if not a node pixel
        if not node_pixels[curr_y, curr_x]:
            visited[curr_y, curr_x] = True
        
        if len(valid_neighbors) == 0:
            # Dead end - create endpoint node on the fly
            new_node = {
                "id": len(nodes) + len(new_nodes),
                "type": "endpoint",
                "centroid": [float(curr_x), float(curr_y)],
                "pixel_count": 1,
                "pixels_bbox": [int(curr_x), int(curr_y), int(curr_x), int(curr_y)],
                "degree_hint": 1,
                "_pixels": [(curr_y, curr_x)],
                "_created": "deadend",
            }
            new_nodes.append(new_node)
            end_node_id = new_node["id"]
            break
            
        elif len(valid_neighbors) == 1:
            # Normal path - continue
            prev_y, prev_x = curr_y, curr_x
            curr_y, curr_x = valid_neighbors[0]
            
        else:  # len(valid_neighbors) >= 2
            # Multiple choices - this is an unmodeled junction
            # Create junction node on the fly
            new_node = {
                "id": len(nodes) + len(new_nodes),
                "type": "junction",
                "centroid": [float(curr_x), float(curr_y)],
                "pixel_count": 1,
                "pixels_bbox": [int(curr_x), int(curr_y), int(curr_x), int(curr_y)],
                "degree_hint": len(valid_neighbors) + 1,
                "_pixels": [(curr_y, curr_x)],
                "_created": "branch",
            }
            new_nodes.append(new_node)
            end_node_id = new_node["id"]
            break
        
        steps += 1
    
    if steps >= max_steps:
        # Safety limit reached
        return None, new_nodes
    
    if end_node_id is None:
        return None, new_nodes
    
    # Compute edge length
    length = 0.0
    for i in range(1, len(polyline)):
        dx = polyline[i][0] - polyline[i-1][0]
        dy = polyline[i][1] - polyline[i-1][1]
        length += np.sqrt(dx*dx + dy*dy)
    
    if length < min_length:
        return None, new_nodes
    
    edge = {
        "u": start_node_id,
        "v": end_node_id,
        "polyline": polyline,
        "length_px": float(length),
        "touches_junction": False,  # Will be set later
    }
    
    return edge, new_nodes


def trace_all_edges(nodes: list, ridge: np.ndarray, node_pixels: np.ndarray,
                    config: dict) -> tuple:
    """
    Trace all edges in the graph.
    
    Returns:
        Tuple of (edges_list, updated_nodes_list, stats_dict).
    """
    h, w = ridge.shape
    
    # Build lookup
    node_lookup = build_node_pixel_lookup(nodes, h, w)
    
    # Track visited path pixels
    visited = np.zeros((h, w), dtype=bool)
    
    # Find all start pixels
    starts = find_edge_start_pixels(nodes, ridge, node_pixels, node_lookup)
    
    edges = []
    all_new_nodes = []
    stats = {
        "created_nodes_deadend_count": 0,
        "created_nodes_branch_count": 0,
    }
    
    for start_node_id, start_y, start_x in starts:
        # Skip if already visited
        if visited[start_y, start_x]:
            continue
        
        # Update node lookup with new nodes
        for nn in all_new_nodes:
            for y, x in nn.get("_pixels", []):
                node_lookup[y, x] = nn["id"]
        
        edge, new_nodes = trace_edge(
            start_node_id, start_y, start_x,
            ridge, node_pixels, node_lookup, visited,
            nodes + all_new_nodes, config
        )
        
        if edge is not None:
            edge["id"] = len(edges)
            edges.append(edge)
        
        for nn in new_nodes:
            if nn.get("_created") == "deadend":
                stats["created_nodes_deadend_count"] += 1
            elif nn.get("_created") == "branch":
                stats["created_nodes_branch_count"] += 1
            # Add to lookup
            for y, x in nn.get("_pixels", []):
                node_lookup[y, x] = nn["id"]
        
        all_new_nodes.extend(new_nodes)
    
    # Merge new nodes into nodes list
    final_nodes = nodes + all_new_nodes
    
    # Update touches_junction flag
    junction_ids = {n["id"] for n in final_nodes if n["type"] == "junction"}
    for edge in edges:
        edge["touches_junction"] = (edge["u"] in junction_ids or 
                                    edge["v"] in junction_ids)
    
    return edges, final_nodes, stats


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def visualize_edges_on_black(edges: list, nodes: list, h: int, w: int,
                              draw_every_n: int = 1) -> np.ndarray:
    """Draw edges on black background with colored polylines."""
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    palette = deterministic_color_palette(max(len(edges), 1) + 1)
    
    for i, edge in enumerate(edges):
        if i % draw_every_n != 0:
            continue
        color = palette[i % len(palette)]
        pts = np.array(edge["polyline"], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=1)
    
    # Draw node centroids
    for node in nodes:
        cx, cy = int(node["centroid"][0]), int(node["centroid"][1])
        if node["type"] == "endpoint":
            cv2.circle(vis, (cx, cy), 2, (0, 255, 0), -1)
        else:
            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
    
    return vis


def visualize_edges_on_image(edges: list, nodes: list, base_img: np.ndarray,
                              draw_every_n: int = 1) -> np.ndarray:
    """Overlay edges on a base image."""
    if len(base_img.shape) == 2:
        vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        vis = base_img.copy()
    
    # Dim the background slightly
    vis = (vis.astype(np.float32) * 0.5).astype(np.uint8)
    
    palette = deterministic_color_palette(max(len(edges), 1) + 1)
    
    for i, edge in enumerate(edges):
        if i % draw_every_n != 0:
            continue
        color = palette[i % len(palette)]
        pts = np.array(edge["polyline"], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=1)
    
    # Draw node centroids
    for node in nodes:
        cx, cy = int(node["centroid"][0]), int(node["centroid"][1])
        if node["type"] == "endpoint":
            cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)
        else:
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
    
    return vis


def visualize_edge_endpoints(edges: list, h: int, w: int) -> np.ndarray:
    """Draw edge start/end points to verify direction."""
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    for edge in edges:
        pts = edge["polyline"]
        if len(pts) >= 2:
            # Start in green
            sx, sy = pts[0]
            cv2.circle(vis, (sx, sy), 2, (0, 255, 0), -1)
            
            # End in red
            ex, ey = pts[-1]
            cv2.circle(vis, (ex, ey), 2, (0, 0, 255), -1)
    
    return vis


def visualize_edges_with_labels(edges: list, nodes: list, h: int, w: int,
                                 label_every_n: int = 10) -> np.ndarray:
    """Draw edges with id labels at midpoints."""
    vis = visualize_edges_on_black(edges, nodes, h, w)
    
    for i, edge in enumerate(edges):
        if i % label_every_n != 0:
            continue
        pts = edge["polyline"]
        if len(pts) >= 2:
            mid_idx = len(pts) // 2
            mx, my = pts[mid_idx]
            cv2.putText(vis, str(edge["id"]), (mx + 2, my - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return vis


# ---------------------------------------------------------------------------
# Main graph build function
# ---------------------------------------------------------------------------

def graph_build(
    ridge_path: str,
    mask_path: str = None,
    runs_root: str = "runs",
    debug: bool = True,
    config_path: str = None,
) -> tuple:
    """
    Build raw graph from ridge mask.
    
    Returns:
        Tuple of (run_dir, graph_path, metrics).
    """
    # Load configuration
    config = load_config(config_path)
    
    # Part A: Load ridge
    ridge, original_ridge_count = load_ridge(ridge_path, config)
    h, w = ridge.shape
    ridge_pixel_count = int(ridge.sum())
    
    # Load optional mask for overlays
    mask_img = None
    if mask_path:
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Infer run directory
    run_dir = infer_run_dir(ridge_path, runs_root)
    
    # Load original input if available
    original_input_path = find_original_input(run_dir)
    original_img = None
    if original_input_path:
        original_img = cv2.imread(str(original_input_path))
    
    # Create artifacts manager
    artifacts = StageArtifacts(run_dir, 40, "graph_raw", debug=debug)
    
    # Debug 1: input ridge
    artifacts.save_debug_image("input_ridge", (ridge.astype(np.uint8) * 255))
    
    # Part B: Compute degree map
    degree = compute_degree_map(ridge)
    
    # Compute degree histogram
    degree_values = degree[ridge]
    degree_histogram = {}
    for d in range(9):
        degree_histogram[str(d)] = int((degree_values == d).sum())
    
    # Debug 2: degree map visualization
    degree_vis = visualize_degree_map(degree, ridge)
    artifacts.save_debug_image("degree_map_vis", degree_vis)
    
    # Part C: Identify node components
    nodes, node_pixels, node_labels = identify_node_components(ridge, degree, config)
    node_pixel_count = int(node_pixels.sum())
    node_count_initial = len(nodes)
    
    # Debug 3: node pixels
    artifacts.save_debug_image("node_pixels_raw", (node_pixels.astype(np.uint8) * 255))
    
    # Debug 4: nodes labeled
    nodes_labeled_vis = visualize_nodes_labeled(node_labels, nodes)
    artifacts.save_debug_image("nodes_labeled_vis", nodes_labeled_vis)
    
    # Debug 5: nodes on ridge
    nodes_on_ridge = visualize_nodes_on_ridge(ridge, nodes)
    artifacts.save_debug_image("nodes_on_ridge", nodes_on_ridge)
    
    # Part D: Trace edges
    edges, final_nodes, trace_stats = trace_all_edges(nodes, ridge, node_pixels, config)
    
    node_count_total = len(final_nodes)
    edge_count = len(edges)
    
    # Count node types
    endpoint_count = sum(1 for n in final_nodes if n["type"] == "endpoint")
    junction_count = sum(1 for n in final_nodes if n["type"] == "junction")
    
    # Edge length statistics
    edge_lengths = [e["length_px"] for e in edges]
    if edge_lengths:
        length_stats = {
            "min": float(np.min(edge_lengths)),
            "max": float(np.max(edge_lengths)),
            "mean": float(np.mean(edge_lengths)),
            "median": float(np.median(edge_lengths)),
        }
    else:
        length_stats = {"min": 0, "max": 0, "mean": 0, "median": 0}
    
    # Count self-loops and multi-edges
    num_self_loops = sum(1 for e in edges if e["u"] == e["v"])
    edge_pairs = [(e["u"], e["v"]) for e in edges]
    edge_pairs_normalized = [tuple(sorted(p)) for p in edge_pairs]
    pair_counts = defaultdict(int)
    for p in edge_pairs_normalized:
        pair_counts[p] += 1
    num_multi_edges = sum(1 for c in pair_counts.values() if c > 1)
    
    # Debug 6: edges on black
    edges_on_black = visualize_edges_on_black(
        edges, final_nodes, h, w, config["debug_draw_edge_every_n"])
    artifacts.save_debug_image("edges_on_black", edges_on_black)
    
    # Debug 7: edges on mask (if available)
    if mask_img is not None:
        edges_on_mask = visualize_edges_on_image(
            edges, final_nodes, mask_img, config["debug_draw_edge_every_n"])
        artifacts.save_debug_image("edges_on_mask", edges_on_mask)
    
    # Debug 8: edges on input (if available)
    if original_img is not None:
        # Resize if needed
        if original_img.shape[:2] != (h, w):
            original_img_resized = cv2.resize(original_img, (w, h))
        else:
            original_img_resized = original_img
        edges_on_input = visualize_edges_on_image(
            edges, final_nodes, original_img_resized, config["debug_draw_edge_every_n"])
        artifacts.save_debug_image("edges_on_input", edges_on_input)
    
    # Debug 9: edge endpoints
    edge_endpoints_vis = visualize_edge_endpoints(edges, h, w)
    artifacts.save_debug_image("edge_endpoints_vis", edge_endpoints_vis)
    
    # Debug 10: edges with labels
    edges_labeled = visualize_edges_with_labels(
        edges, final_nodes, h, w, config["debug_label_every_n"])
    artifacts.save_debug_image("edges_labeled", edges_labeled)
    
    # Prepare output graph
    # Remove internal _pixels field
    output_nodes = []
    for node in final_nodes:
        out_node = {k: v for k, v in node.items() if not k.startswith("_")}
        output_nodes.append(out_node)
    
    output_edges = []
    for edge in edges:
        out_edge = {k: v for k, v in edge.items() if not k.startswith("_")}
        output_edges.append(out_edge)
    
    graph_data = {
        "image": {"width": w, "height": h},
        "params": config,
        "nodes": output_nodes,
        "edges": output_edges,
    }
    
    # Save graph
    graph_path = artifacts.path_out("graph_raw.json")
    graph_path.write_text(json.dumps(graph_data, indent=2, cls=NumpyEncoder))
    
    # Build metrics
    metrics = {
        "ridge_path": str(ridge_path),
        "image_w": w,
        "image_h": h,
        "ridge_pixel_count": ridge_pixel_count,
        "node_pixel_count": node_pixel_count,
        "node_count_initial": node_count_initial,
        "node_count_total": node_count_total,
        "endpoint_node_count": endpoint_count,
        "junction_node_count": junction_count,
        "edge_count": edge_count,
        "degree_histogram": degree_histogram,
        "edge_length_stats": length_stats,
        "num_self_loops": num_self_loops,
        "num_multi_edges_estimate": num_multi_edges,
        "created_nodes_deadend_count": trace_stats["created_nodes_deadend_count"],
        "created_nodes_branch_count": trace_stats["created_nodes_branch_count"],
        "graph_sanity": {
            "fragmentation_score": endpoint_count / max(1, edge_count),
        },
        "config_used": config,
    }
    artifacts.write_metrics(metrics)
    
    return run_dir, graph_path, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build raw graph from ridge mask"
    )
    parser.add_argument(
        "ridge_path",
        help="Path to ridge mask image"
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="Path to preprocess output mask (for overlays)"
    )
    parser.add_argument(
        "--runs_root",
        default="runs",
        help="Root directory for runs (default: runs)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        dest="debug",
        default=True,
        help="Save debug images (default: True)"
    )
    parser.add_argument(
        "--no_debug",
        action="store_false",
        dest="debug",
        help="Disable debug image saving"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config JSON to override defaults"
    )
    
    args = parser.parse_args()
    
    # Verify inputs exist
    ridge_p = Path(args.ridge_path)
    if not ridge_p.exists():
        print(f"Error: Ridge file not found: {args.ridge_path}")
        return 1
    
    print(f"Processing ridge: {args.ridge_path}")
    
    try:
        run_dir, graph_path, metrics = graph_build(
            args.ridge_path,
            mask_path=args.mask,
            runs_root=args.runs_root,
            debug=args.debug,
            config_path=args.config,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    print(f"Run directory: {run_dir}")
    print(f"Graph saved to: {graph_path}")
    print(f"Nodes: {metrics['node_count_total']} ({metrics['endpoint_node_count']} endpoints, {metrics['junction_node_count']} junctions)")
    print(f"Edges: {metrics['edge_count']}")
    print(f"Edge length median: {metrics['edge_length_stats']['median']:.1f} px")
    print(f"Created nodes (deadend): {metrics['created_nodes_deadend_count']}")
    print(f"Created nodes (branch): {metrics['created_nodes_branch_count']}")
    
    return 0


if __name__ == "__main__":
    exit(main())


# ---------------------------------------------------------------------------
# Notes on Debug Artifacts
# ---------------------------------------------------------------------------
#
# Run
#    python preprocess.py examples/clean.png --debug
#    python distance_transform.py runs/clean/10_preprocess/out/output_mask.png --debug
#    python ridge_extraction.py runs/clean/10_preprocess/out/output_mask.png runs/clean/20_distance_transform/out/dt.npy --debug
#    python graph_build.py runs/clean/30_ridge/out/ridge.png --mask runs/clean/10_preprocess/out/output_mask.png --debug
#
# Open runs/<run>/40_graph_raw/debug/
#
# nodes_on_ridge.png
#    - Endpoints (green) should appear at true line ends
#    - Junctions (red) should appear at true intersections
#    - If junctions appear along straight lines, ridge is too thick or noisy
#
# edges_on_black.png
#    - Each real stroke should correspond to a few continuous edges
#    - Edges should not jump across gaps or connect unrelated parts
#    - Many tiny edges means ridge is still fragmented
#
# edges_on_mask.png / edges_on_input.png
#    - Edge polylines should lie centered on the stroke region
#    - No edges crossing empty background
#
# key checks in metrics.json:
#    - endpoint_node_count: should be plausible (not thousands) for clean.png
#    - edge_count: should not be extremely large
#    - edge_length_stats.median: should not be tiny (< 5 px)
#    - created_nodes_deadend_count: should be low for clean images
#
# red flags:
#    - Many edges of length 1-5 px: ridge fragmentation
#    - Many created dead-end nodes: ridge breaks still present
#    - Junction nodes huge/smeared: ridge thick or dilation too aggressive
#    - Edges crossing background: neighbor logic wrong
