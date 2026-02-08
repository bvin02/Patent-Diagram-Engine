"""
Stage 5: Graph Cleanup for Sketch-to-SVG Pipeline

Performs graph cleanup and stabilization on the raw graph from Stage 4:
- Step 1: Merge node clusters (junction consolidation + near-duplicate nodes)
- Step 2: Spur pruning (remove tiny dangling branches)
- Step 3: Merge collinear chains (edge stitching through degree-2 nodes)
- Step 4: Optional gap bridging (very conservative, off by default)
- Step 5: Polyline simplification (RDP)

Output:
- graph_clean.json: cleaned graph with same schema as graph_raw.json
- metrics.json: statistics comparing raw vs clean
- report.json: per-step operation counts

Usage:
    python graph_cleanup.py runs/<run>/40_graph_raw/out/graph_raw.json --debug
    python graph_cleanup.py runs/<run>/40_graph_raw/out/graph_raw.json --debug --mask runs/<run>/10_preprocess/out/output_mask.png
    python graph_cleanup.py runs/<run>/40_graph_raw/out/graph_raw.json --debug --config configs/graph_cleanup.json
"""

import argparse
import json
import cv2
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

from utils.artifacts import make_run_dir, StageArtifacts


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Step 1: Node merging
    "merge_radius_px": 5.0,              # max distance to merge nodes (increased for better consolidation)
    "endpoint_merge_max_angle_deg": 30,  # angle tolerance for endpoint merge
    
    # Step 2: Spur pruning  
    "spur_max_length_px": 8.0,           # max length of spur to remove
    "spur_prune_iterations": 3,          # max iterations of spur pruning
    
    # Step 3: Chain merging
    "dir_sample_len": 5,                 # pixels along polyline for direction
    "collinear_max_angle_deg": 25,       # angle threshold for collinearity (increased for more merges)
    "min_edge_len_for_merge": 3,         # min edge length to compute direction
    "chain_merge_iterations": 50,        # max iterations (increased for thorough merging)
    
    # Step 4: Gap bridging (off by default)
    "enable_gap_bridge": True,
    "bridge_max_dist_px": 20.0,
    "bridge_max_angle_deg": 45,
    "mask_min_fg_fraction": 0.6,
    
    # Step 5: Polyline simplification
    "rdp_epsilon": 0.75,                  # RDP tolerance in pixels
}


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


# ---------------------------------------------------------------------------
# Helper functions
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
    """Try to locate original input image in run directory."""
    input_dir = run_dir / "00_input"
    if not input_dir.exists():
        return None
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = input_dir / f"01_input{ext}"
        if candidate.exists():
            return candidate
    return None


def deterministic_color_palette(n: int) -> list:
    """Generate deterministic color palette for visualization."""
    colors = []
    for i in range(max(n, 1)):
        hue = (i * 0.618033988749895) % 1.0
        h = int(hue * 180)
        hsv = np.array([[[h, 200, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(map(int, bgr)))
    return colors


def compute_polyline_length(polyline: List[List[int]]) -> float:
    """Compute total length of a polyline."""
    if len(polyline) < 2:
        return 0.0
    length = 0.0
    for i in range(1, len(polyline)):
        dx = polyline[i][0] - polyline[i-1][0]
        dy = polyline[i][1] - polyline[i-1][1]
        length += np.sqrt(dx*dx + dy*dy)
    return length


def get_edge_direction_at_node(polyline: List[List[int]], node_is_u: bool, 
                                sample_len: int) -> np.ndarray:
    """
    Get normalized direction vector of edge at a node.
    Returns vector pointing away from the node along the edge.
    """
    pts = np.array(polyline)
    if len(pts) < 2:
        return np.array([1.0, 0.0])
    
    if node_is_u:
        # Node is at start of polyline (polyline[0])
        end_idx = min(sample_len, len(pts) - 1)
        direction = pts[end_idx] - pts[0]
    else:
        # Node is at end of polyline (polyline[-1])
        start_idx = max(0, len(pts) - 1 - sample_len)
        direction = pts[start_idx] - pts[-1]
    
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.array([1.0, 0.0])
    return direction / norm


def rdp_simplify(polyline: List[List[int]], epsilon: float) -> List[List[int]]:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(polyline) < 3:
        return polyline
    
    pts = np.array(polyline, dtype=np.float64)
    
    def _perpendicular_distance(point, line_start, line_end):
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-9:
            return np.linalg.norm(point - line_start)
        t = max(0, min(1, np.dot(point - line_start, line_vec) / (line_len * line_len)))
        projection = line_start + t * line_vec
        return np.linalg.norm(point - projection)
    
    def _rdp_recursive(pts, epsilon, start, end, mask):
        if end <= start + 1:
            return
        max_dist = 0.0
        max_idx = start
        for i in range(start + 1, end):
            dist = _perpendicular_distance(pts[i], pts[start], pts[end])
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        if max_dist > epsilon:
            mask[max_idx] = True
            _rdp_recursive(pts, epsilon, start, max_idx, mask)
            _rdp_recursive(pts, epsilon, max_idx, end, mask)
    
    mask = np.zeros(len(pts), dtype=bool)
    mask[0] = True
    mask[-1] = True
    _rdp_recursive(pts, epsilon, 0, len(pts) - 1, mask)
    
    result = pts[mask].astype(int).tolist()
    return [[int(p[0]), int(p[1])] for p in result]


# ---------------------------------------------------------------------------
# Union-Find for deterministic clustering
# ---------------------------------------------------------------------------

class UnionFind:
    """Disjoint set / union-find data structure."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
    
    def get_clusters(self) -> Dict[int, List[int]]:
        clusters = defaultdict(list)
        for i in range(len(self.parent)):
            clusters[self.find(i)].append(i)
        return dict(clusters)


# ---------------------------------------------------------------------------
# Step 0: Load graph and build working structures
# ---------------------------------------------------------------------------

def load_graph_raw(graph_path: str) -> dict:
    """Load graph_raw.json and validate structure."""
    with open(graph_path) as f:
        data = json.load(f)
    required = ["image", "nodes", "edges"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required key in graph: {key}")
    return data


def build_nx_multigraph(nodes: List[dict], edges: List[dict]) -> nx.MultiGraph:
    """Build NetworkX multigraph from nodes and edges."""
    G = nx.MultiGraph()
    for node in nodes:
        G.add_node(node["id"], **node)
    for edge in edges:
        G.add_edge(edge["u"], edge["v"], key=edge["id"], **edge)
    return G


# ---------------------------------------------------------------------------
# Step 1: Merge node clusters
# ---------------------------------------------------------------------------

def should_merge_endpoints(n1: dict, n2: dict, edges: List[dict], 
                           config: dict) -> bool:
    """Check if two endpoints should be merged based on direction alignment."""
    c1 = np.array(n1["centroid"])
    c2 = np.array(n2["centroid"])
    dist = np.linalg.norm(c1 - c2)
    
    if dist > config["merge_radius_px"]:
        return False
    
    e1_list = [e for e in edges if e["u"] == n1["id"] or e["v"] == n1["id"]]
    e2_list = [e for e in edges if e["u"] == n2["id"] or e["v"] == n2["id"]]
    
    if not e1_list or not e2_list:
        return True
    
    e1 = e1_list[0]
    n1_is_u = (e1["u"] == n1["id"])
    dir1 = get_edge_direction_at_node(e1["polyline"], n1_is_u, config["dir_sample_len"])
    
    e2 = e2_list[0]
    n2_is_u = (e2["u"] == n2["id"])
    dir2 = get_edge_direction_at_node(e2["polyline"], n2_is_u, config["dir_sample_len"])
    
    dot = abs(np.dot(dir1, dir2))
    angle_deg = np.degrees(np.arccos(np.clip(dot, 0, 1)))
    
    return angle_deg <= config["endpoint_merge_max_angle_deg"]


def should_merge_nodes(n1: dict, n2: dict, edges: List[dict], config: dict) -> bool:
    """Determine if two nodes should be merged."""
    c1 = np.array(n1["centroid"])
    c2 = np.array(n2["centroid"])
    dist = np.linalg.norm(c1 - c2)
    
    if dist > config["merge_radius_px"]:
        return False
    
    t1, t2 = n1["type"], n2["type"]
    
    if t1 == "junction" and t2 == "junction":
        return True
    
    if t1 == "endpoint" and t2 == "endpoint":
        return should_merge_endpoints(n1, n2, edges, config)
    
    junction = n1 if t1 == "junction" else n2
    endpoint = n2 if t1 == "junction" else n1
    
    bbox = junction["pixels_bbox"]
    ex, ey = endpoint["centroid"]
    
    if (bbox[0] - 1 <= ex <= bbox[2] + 1 and 
        bbox[1] - 1 <= ey <= bbox[3] + 1):
        return True
    
    return dist <= config["merge_radius_px"]


def merge_node_clusters(nodes: List[dict], edges: List[dict], 
                        config: dict) -> Tuple[List[dict], List[dict], dict]:
    """Merge nearby node clusters using union-find."""
    n = len(nodes)
    if n == 0:
        return [], edges, {"clusters_formed": 0}
    
    node_by_id = {node["id"]: node for node in nodes}
    node_ids = sorted(node_by_id.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    uf = UnionFind(n)
    
    for i, id1 in enumerate(node_ids):
        for j in range(i + 1, len(node_ids)):
            id2 = node_ids[j]
            if should_merge_nodes(node_by_id[id1], node_by_id[id2], edges, config):
                uf.union(i, j)
    
    clusters = uf.get_clusters()
    
    new_nodes = []
    old_to_new_id = {}
    
    for new_id, (root, members) in enumerate(sorted(clusters.items())):
        member_ids = sorted([node_ids[m] for m in members])
        member_nodes = [node_by_id[mid] for mid in member_ids]
        
        centroids = np.array([m["centroid"] for m in member_nodes])
        merged_centroid = centroids.mean(axis=0).tolist()
        
        bboxes = [m["pixels_bbox"] for m in member_nodes]
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        
        node_type = "junction" if any(m["type"] == "junction" for m in member_nodes) else "endpoint"
        
        new_node = {
            "id": new_id,
            "type": node_type,
            "centroid": merged_centroid,
            "pixel_count": sum(m["pixel_count"] for m in member_nodes),
            "pixels_bbox": [x0, y0, x1, y1],
            "degree_hint": max(m.get("degree_hint", 1) for m in member_nodes),
            "source_node_ids": member_ids,
        }
        new_nodes.append(new_node)
        
        for mid in member_ids:
            old_to_new_id[mid] = new_id
    
    updated_edges = []
    for edge in edges:
        new_u = old_to_new_id.get(edge["u"], edge["u"])
        new_v = old_to_new_id.get(edge["v"], edge["v"])
        new_edge = edge.copy()
        new_edge["u"] = new_u
        new_edge["v"] = new_v
        updated_edges.append(new_edge)
    
    stats = {
        "clusters_formed": len([c for c in clusters.values() if len(c) > 1]),
        "nodes_merged": n - len(new_nodes),
    }
    
    return new_nodes, updated_edges, stats


# ---------------------------------------------------------------------------
# Step 2: Spur pruning
# ---------------------------------------------------------------------------

def prune_spurs(nodes: List[dict], edges: List[dict], 
                config: dict) -> Tuple[List[dict], List[dict], dict]:
    """Remove short dangling edges from degree-1 nodes."""
    max_len = config["spur_max_length_px"]
    max_iterations = config["spur_prune_iterations"]
    
    removed_edge_ids = []
    
    for iteration in range(max_iterations):
        degree = defaultdict(int)
        for e in edges:
            if e["id"] not in removed_edge_ids:
                degree[e["u"]] += 1
                degree[e["v"]] += 1
        
        to_remove = []
        for e in edges:
            if e["id"] in removed_edge_ids:
                continue
            
            length = e.get("length_px", compute_polyline_length(e["polyline"]))
            if length >= max_len:
                continue
            
            u, v = e["u"], e["v"]
            if degree[u] == 1 or degree[v] == 1:
                to_remove.append(e["id"])
        
        if not to_remove:
            break
        
        to_remove.sort()
        removed_edge_ids.extend(to_remove)
    
    final_edges = [e for e in edges if e["id"] not in removed_edge_ids]
    
    referenced_nodes = set()
    for e in final_edges:
        referenced_nodes.add(e["u"])
        referenced_nodes.add(e["v"])
    
    final_nodes = [n for n in nodes if n["id"] in referenced_nodes]
    
    stats = {
        "spurs_removed": len(removed_edge_ids),
        "nodes_removed": len(nodes) - len(final_nodes),
    }
    
    return final_nodes, final_edges, stats


# ---------------------------------------------------------------------------
# Step 3: Merge collinear chains
# ---------------------------------------------------------------------------

def merge_collinear_chains(nodes: List[dict], edges: List[dict],
                           config: dict) -> Tuple[List[dict], List[dict], dict]:
    """Merge edges through degree-2 nodes when approximately collinear."""
    threshold_deg = config["collinear_max_angle_deg"]
    sample_len = config["dir_sample_len"]
    min_edge_len = config["min_edge_len_for_merge"]
    max_iterations = config["chain_merge_iterations"]
    
    current_nodes = {n["id"]: n.copy() for n in nodes}
    current_edges = {e["id"]: e.copy() for e in edges}
    
    merges_performed = 0
    next_edge_id = max(e["id"] for e in edges) + 1 if edges else 0
    
    for iteration in range(max_iterations):
        adjacency = defaultdict(list)
        for eid, edge in current_edges.items():
            u, v = edge["u"], edge["v"]
            adjacency[u].append((eid, v))
            adjacency[v].append((eid, u))
        
        merged_this_round = False
        
        for node_id in sorted(current_nodes.keys()):
            if node_id not in current_nodes:
                continue
            
            node = current_nodes[node_id]
            adj = adjacency.get(node_id, [])
            
            if len(adj) != 2:
                continue
            
            # Note: We allow merging through any degree-2 node, including those
            # labeled as junctions. If a node has exactly 2 edges, it's a pass-through
            # regardless of how it was originally classified. The angle check below
            # ensures we don't merge actual corners.
            
            (eid1, other1), (eid2, other2) = adj
            
            if eid1 not in current_edges or eid2 not in current_edges:
                continue
            
            edge1, edge2 = current_edges[eid1], current_edges[eid2]
            
            len1 = edge1.get("length_px", compute_polyline_length(edge1["polyline"]))
            len2 = edge2.get("length_px", compute_polyline_length(edge2["polyline"]))
            
            if len1 < min_edge_len or len2 < min_edge_len:
                continue
            
            n_is_u1 = (edge1["u"] == node_id)
            n_is_u2 = (edge2["u"] == node_id)
            
            dir1 = get_edge_direction_at_node(edge1["polyline"], n_is_u1, sample_len)
            dir2 = get_edge_direction_at_node(edge2["polyline"], n_is_u2, sample_len)
            
            # Collinear if angle close to 180 deg (dot near -1)
            dot = np.dot(dir1, dir2)
            angle_deg = np.degrees(np.arccos(np.clip(-dot, -1.0, 1.0)))
            
            if angle_deg > threshold_deg:
                continue
            
            poly1 = edge1["polyline"]
            poly2 = edge2["polyline"]
            
            if edge1["u"] == node_id:
                poly1 = poly1[::-1]
            if edge2["v"] == node_id:
                poly2 = poly2[::-1]
            
            merged_poly = poly1 + poly2[1:]
            merged_length = compute_polyline_length(merged_poly)
            
            new_u = other1
            new_v = other2
            
            sources1 = edge1.get("source_edge_ids", [edge1["id"]])
            sources2 = edge2.get("source_edge_ids", [edge2["id"]])
            source_ids = sorted(set(sources1 + sources2))
            
            new_edge = {
                "id": next_edge_id,
                "u": new_u,
                "v": new_v,
                "polyline": merged_poly,
                "length_px": merged_length,
                "source_edge_ids": source_ids,
            }
            
            del current_edges[eid1]
            del current_edges[eid2]
            del current_nodes[node_id]
            current_edges[next_edge_id] = new_edge
            next_edge_id += 1
            
            merges_performed += 1
            merged_this_round = True
            break
        
        if not merged_this_round:
            break
    
    final_nodes = list(current_nodes.values())
    final_edges = list(current_edges.values())
    
    stats = {"chain_merges": merges_performed}
    
    return final_nodes, final_edges, stats


# ---------------------------------------------------------------------------
# Step 4: Gap bridging (optional)
# ---------------------------------------------------------------------------

def find_gap_bridge_candidates(nodes: List[dict], edges: List[dict], 
                                mask: Optional[np.ndarray],
                                config: dict) -> List[Tuple[int, int, List]]:
    """Find endpoint pairs that could be bridged."""
    max_dist = config["bridge_max_dist_px"]
    max_angle = config["bridge_max_angle_deg"]
    min_fg_frac = config["mask_min_fg_fraction"]
    sample_len = config["dir_sample_len"]
    
    degree = defaultdict(int)
    for e in edges:
        degree[e["u"]] += 1
        degree[e["v"]] += 1
    
    endpoints = [n for n in nodes 
                 if n["type"] == "endpoint" and degree[n["id"]] == 1]
    
    node_to_edge = {}
    for e in edges:
        if degree[e["u"]] == 1:
            node_to_edge[e["u"]] = e
        if degree[e["v"]] == 1:
            node_to_edge[e["v"]] = e
    
    candidates = []
    
    for i, n1 in enumerate(endpoints):
        for n2 in endpoints[i+1:]:
            c1 = np.array(n1["centroid"])
            c2 = np.array(n2["centroid"])
            dist = np.linalg.norm(c1 - c2)
            
            if dist > max_dist or dist < 1e-9:
                continue
            
            e1 = node_to_edge.get(n1["id"])
            e2 = node_to_edge.get(n2["id"])
            
            if e1 is None or e2 is None:
                continue
            
            n1_is_u = (e1["u"] == n1["id"])
            dir1 = get_edge_direction_at_node(e1["polyline"], n1_is_u, sample_len)
            
            n2_is_u = (e2["u"] == n2["id"])
            dir2 = get_edge_direction_at_node(e2["polyline"], n2_is_u, sample_len)
            
            gap_vec = c2 - c1
            gap_norm = gap_vec / dist
            
            dot1 = np.dot(dir1, gap_norm)
            angle1 = np.degrees(np.arccos(np.clip(abs(dot1), 0, 1)))
            
            dot2 = np.dot(dir2, -gap_norm)
            angle2 = np.degrees(np.arccos(np.clip(abs(dot2), 0, 1)))
            
            if angle1 > max_angle or angle2 > max_angle:
                continue
            
            if mask is not None:
                num_samples = max(3, int(dist))
                fg_count = 0
                for t in np.linspace(0, 1, num_samples):
                    px = int(c1[0] + t * gap_vec[0])
                    py = int(c1[1] + t * gap_vec[1])
                    if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]:
                        if mask[py, px] > 0:
                            fg_count += 1
                if fg_count / num_samples < min_fg_frac:
                    continue
            
            bridge_poly = []
            num_pts = max(2, int(dist))
            for t in np.linspace(0, 1, num_pts):
                px = int(c1[0] + t * gap_vec[0])
                py = int(c1[1] + t * gap_vec[1])
                bridge_poly.append([px, py])
            
            candidates.append((n1["id"], n2["id"], bridge_poly))
    
    return candidates


def apply_gap_bridges(nodes: List[dict], edges: List[dict],
                      bridges: List[Tuple[int, int, List]]) -> List[dict]:
    """Apply gap bridges by adding new edges."""
    next_edge_id = max(e["id"] for e in edges) + 1 if edges else 0
    
    new_edges = edges.copy()
    for n1_id, n2_id, polyline in bridges:
        new_edge = {
            "id": next_edge_id,
            "u": n1_id,
            "v": n2_id,
            "polyline": polyline,
            "length_px": compute_polyline_length(polyline),
            "bridge": True,
            "source_edge_ids": [],
        }
        new_edges.append(new_edge)
        next_edge_id += 1
    
    return new_edges


# ---------------------------------------------------------------------------
# Step 5: Polyline simplification
# ---------------------------------------------------------------------------

def simplify_polylines(edges: List[dict], config: dict) -> Tuple[List[dict], dict]:
    """Apply RDP simplification to all edge polylines."""
    epsilon = config["rdp_epsilon"]
    
    total_before = 0
    total_after = 0
    
    simplified_edges = []
    for edge in edges:
        poly = edge["polyline"]
        total_before += len(poly)
        
        simplified = rdp_simplify(poly, epsilon)
        total_after += len(simplified)
        
        new_edge = edge.copy()
        new_edge["polyline"] = simplified
        new_edge["length_px"] = compute_polyline_length(simplified)
        simplified_edges.append(new_edge)
    
    stats = {
        "points_before": total_before,
        "points_after": total_after,
        "reduction_ratio": 1.0 - (total_after / max(1, total_before)),
    }
    
    return simplified_edges, stats


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def visualize_nodes_on_image(nodes: List[dict], base_img: np.ndarray) -> np.ndarray:
    """Draw nodes on base image."""
    if len(base_img.shape) == 2:
        vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        vis = base_img.copy()
    
    vis = (vis.astype(np.float32) * 0.6).astype(np.uint8)
    
    for node in nodes:
        cx, cy = int(node["centroid"][0]), int(node["centroid"][1])
        if node["type"] == "endpoint":
            cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)
        else:
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
    
    return vis


def visualize_edges_on_image(edges: List[dict], nodes: List[dict],
                              base_img: np.ndarray) -> np.ndarray:
    """Draw edges on base image with colored polylines."""
    if len(base_img.shape) == 2:
        vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        vis = base_img.copy()
    
    vis = (vis.astype(np.float32) * 0.5).astype(np.uint8)
    
    palette = deterministic_color_palette(max(len(edges), 1))
    
    for i, edge in enumerate(edges):
        color = palette[i % len(palette)]
        pts = np.array(edge["polyline"], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=1)
    
    for node in nodes:
        cx, cy = int(node["centroid"][0]), int(node["centroid"][1])
        if node["type"] == "endpoint":
            cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)
        else:
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
    
    return vis


def visualize_cluster_membership(nodes: List[dict], h: int, w: int) -> np.ndarray:
    """Show cluster membership with colors."""
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    clusters = defaultdict(list)
    for node in nodes:
        sources = node.get("source_node_ids", [node["id"]])
        if len(sources) > 1:
            cluster_key = tuple(sorted(sources))
            clusters[cluster_key].append(node)
    
    palette = deterministic_color_palette(max(len(clusters), 1) + 1)
    
    for i, (key, members) in enumerate(sorted(clusters.items())):
        color = palette[i % len(palette)]
        for node in members:
            cx, cy = int(node["centroid"][0]), int(node["centroid"][1])
            cv2.circle(vis, (cx, cy), 5, color, -1)
    
    for node in nodes:
        sources = node.get("source_node_ids", [node["id"]])
        if len(sources) == 1:
            cx, cy = int(node["centroid"][0]), int(node["centroid"][1])
            cv2.circle(vis, (cx, cy), 2, (100, 100, 100), -1)
    
    return vis


def create_placeholder_image(h: int, w: int, text: str) -> np.ndarray:
    """Create placeholder image with text."""
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(vis, text, (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (128, 128, 128), 2)
    return vis


# ---------------------------------------------------------------------------
# Main cleanup function
# ---------------------------------------------------------------------------

def graph_cleanup(
    graph_path: str,
    mask_path: str = None,
    ridge_path: str = None,
    runs_root: str = "runs",
    debug: bool = True,
    config_path: str = None,
) -> Tuple[Path, Path, dict]:
    """Perform graph cleanup on raw graph."""
    config = load_config(config_path)
    
    graph_data = load_graph_raw(graph_path)
    img_info = graph_data["image"]
    h, w = img_info["height"], img_info["width"]
    
    raw_nodes = graph_data["nodes"]
    raw_edges = graph_data["edges"]
    
    mask = None
    if mask_path and Path(mask_path).exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    run_dir = infer_run_dir(graph_path, runs_root)
    
    original_input_path = find_original_input(run_dir)
    original_img = None
    if original_input_path:
        original_img = cv2.imread(str(original_input_path))
        if original_img is not None and original_img.shape[:2] != (h, w):
            original_img = cv2.resize(original_img, (w, h))
    
    base_img = original_img if original_img is not None else (
        mask if mask is not None else np.zeros((h, w, 3), dtype=np.uint8))
    
    artifacts = StageArtifacts(run_dir, 50, "graph_clean", debug=debug)
    
    report = {"config": config, "steps": {}}
    
    # Debug 1: Raw nodes
    if debug:
        vis = visualize_nodes_on_image(raw_nodes, base_img)
        artifacts.save_debug_image("nodes_raw_on_input", vis)
    
    # Step 1: Merge node clusters
    current_nodes, current_edges, merge_stats = merge_node_clusters(
        raw_nodes, raw_edges, config)
    report["steps"]["merge_clusters"] = merge_stats
    
    if debug:
        vis = visualize_nodes_on_image(current_nodes, base_img)
        artifacts.save_debug_image("nodes_merged_on_input", vis)
        vis = visualize_cluster_membership(current_nodes, h, w)
        artifacts.save_debug_image("merge_clusters_vis", vis)
    
    # Step 2: Spur pruning
    current_nodes, current_edges, prune_stats = prune_spurs(
        current_nodes, current_edges, config)
    report["steps"]["spur_pruning"] = prune_stats
    
    if debug:
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for edge in current_edges:
            pts = np.array(edge["polyline"], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(vis, [pts], False, (100, 100, 100), 1)
        artifacts.save_debug_image("spurs_removed_overlay", vis)
    
    # Step 3: Chain merge
    if debug:
        vis = visualize_edges_on_image(current_edges, current_nodes, base_img)
        artifacts.save_debug_image("before_chain_merge_edges", vis)
    
    current_nodes, current_edges, chain_stats = merge_collinear_chains(
        current_nodes, current_edges, config)
    report["steps"]["chain_merge"] = chain_stats
    
    if debug:
        vis = visualize_edges_on_image(current_edges, current_nodes, base_img)
        artifacts.save_debug_image("after_chain_merge_edges", vis)
        
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for edge in current_edges:
            sources = edge.get("source_edge_ids", [])
            color = (255, 255, 0) if len(sources) > 1 else (80, 80, 80)
            pts = np.array(edge["polyline"], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(vis, [pts], False, color, 1)
        artifacts.save_debug_image("chain_merge_diff", vis)
    
    # Step 4: Gap bridging
    bridge_stats = {"enabled": config["enable_gap_bridge"], "bridges_applied": 0}
    
    if config["enable_gap_bridge"]:
        candidates = find_gap_bridge_candidates(current_nodes, current_edges, mask, config)
        
        if debug:
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            for n1_id, n2_id, poly in candidates:
                n1 = next((n for n in current_nodes if n["id"] == n1_id), None)
                n2 = next((n for n in current_nodes if n["id"] == n2_id), None)
                if n1 and n2:
                    p1 = (int(n1["centroid"][0]), int(n1["centroid"][1]))
                    p2 = (int(n2["centroid"][0]), int(n2["centroid"][1]))
                    cv2.line(vis, p1, p2, (0, 255, 255), 1)
            artifacts.save_debug_image("gap_bridge_candidates", vis)
        
        if candidates:
            current_edges = apply_gap_bridges(current_nodes, current_edges, candidates)
            bridge_stats["bridges_applied"] = len(candidates)
        
        if debug:
            vis = visualize_edges_on_image(current_edges, current_nodes, base_img)
            artifacts.save_debug_image("gap_bridges_applied", vis)
    else:
        if debug:
            vis = create_placeholder_image(h, w, "Gap bridging disabled")
            artifacts.save_debug_image("gap_bridge_candidates", vis)
            artifacts.save_debug_image("gap_bridges_applied", vis)
    
    report["steps"]["gap_bridging"] = bridge_stats
    
    # Step 4b: Post-bridge chain merge (re-run chain merge after gap bridging)
    # Bridge edges often create new degree-2 nodes that can be merged with adjacent edges
    if config["enable_gap_bridge"] and bridge_stats["bridges_applied"] > 0:
        current_nodes, current_edges, post_chain_stats = merge_collinear_chains(
            current_nodes, current_edges, config)
        report["steps"]["post_bridge_chain_merge"] = post_chain_stats
        
        if debug:
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            for edge in current_edges:
                sources = edge.get("source_edge_ids", [])
                color = (0, 255, 128) if len(sources) > 1 else (80, 80, 80)
                pts = np.array(edge["polyline"], dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(vis, [pts], False, color, 1)
            artifacts.save_debug_image("post_bridge_chain_merge", vis)
    
    # Step 5: Simplification
    if debug:
        vis = visualize_edges_on_image(current_edges, current_nodes, base_img)
        artifacts.save_debug_image("simplify_before", vis)
    
    current_edges, simplify_stats = simplify_polylines(current_edges, config)
    report["steps"]["simplification"] = simplify_stats
    
    if debug:
        vis = visualize_edges_on_image(current_edges, current_nodes, base_img)
        artifacts.save_debug_image("simplify_after", vis)
    
    # Final output
    final_nodes = []
    old_to_new = {}
    for i, node in enumerate(sorted(current_nodes, key=lambda x: x["id"])):
        old_id = node["id"]
        new_node = node.copy()
        new_node["id"] = i
        old_to_new[old_id] = i
        final_nodes.append(new_node)
    
    final_edges = []
    for i, edge in enumerate(current_edges):
        new_edge = edge.copy()
        new_edge["id"] = i
        new_edge["u"] = old_to_new.get(edge["u"], edge["u"])
        new_edge["v"] = old_to_new.get(edge["v"], edge["v"])
        final_edges.append(new_edge)
    
    if debug:
        vis = visualize_edges_on_image(final_edges, final_nodes, base_img)
        artifacts.save_debug_image("final_graph_on_input", vis)
    
    output_graph = {
        "image": img_info,
        "params": graph_data.get("params", {}),
        "cleanup_params": config,
        "nodes": final_nodes,
        "edges": final_edges,
    }
    
    graph_clean_path = artifacts.path_out("graph_clean.json")
    graph_clean_path.write_text(json.dumps(output_graph, indent=2, cls=NumpyEncoder))
    
    clean_edge_lengths = [e["length_px"] for e in final_edges]
    raw_edge_lengths = [e.get("length_px", compute_polyline_length(e["polyline"])) 
                        for e in raw_edges]
    
    G_raw = build_nx_multigraph(raw_nodes, raw_edges)
    G_clean = build_nx_multigraph(final_nodes, final_edges)
    
    # Compute edge length quality metrics
    sorted_lengths = sorted(clean_edge_lengths, reverse=True)
    top_k = 20
    top_k_lengths = sorted_lengths[:min(top_k, len(sorted_lengths))]
    
    total_length = sum(clean_edge_lengths) if clean_edge_lengths else 0
    long_edge_threshold = 30.0
    long_edge_length = sum(l for l in clean_edge_lengths if l > long_edge_threshold)
    long_edge_fraction = long_edge_length / max(1, total_length)
    
    metrics = {
        "raw_node_count": len(raw_nodes),
        "raw_edge_count": len(raw_edges),
        "clean_node_count": len(final_nodes),
        "clean_edge_count": len(final_edges),
        "median_edge_length_raw": float(np.median(raw_edge_lengths)) if raw_edge_lengths else 0,
        "median_edge_length_clean": float(np.median(clean_edge_lengths)) if clean_edge_lengths else 0,
        "endpoint_count_raw": sum(1 for n in raw_nodes if n["type"] == "endpoint"),
        "endpoint_count_clean": sum(1 for n in final_nodes if n["type"] == "endpoint"),
        "junction_count_raw": sum(1 for n in raw_nodes if n["type"] == "junction"),
        "junction_count_clean": sum(1 for n in final_nodes if n["type"] == "junction"),
        "avg_polyline_points_raw": float(np.mean([len(e["polyline"]) for e in raw_edges])) if raw_edges else 0,
        "avg_polyline_points_clean": float(np.mean([len(e["polyline"]) for e in final_edges])) if final_edges else 0,
        "connected_components_raw": nx.number_connected_components(G_raw),
        "connected_components_clean": nx.number_connected_components(G_clean),
        # Edge length quality metrics
        "total_edge_length_px": float(total_length),
        "top_20_edge_lengths": [float(l) for l in top_k_lengths],
        "long_edge_threshold_px": long_edge_threshold,
        "long_edge_length_px": float(long_edge_length),
        "long_edge_fraction": float(long_edge_fraction),
        "sanity_flags": {},
    }
    
    junction_drop = 1.0 - (metrics["junction_count_clean"] / max(1, metrics["junction_count_raw"]))
    length_increase = metrics["median_edge_length_clean"] / max(1, metrics["median_edge_length_raw"])
    
    if junction_drop > 0.7 and length_increase < 1.5:
        metrics["sanity_flags"]["over_merge_suspected"] = True
    
    if metrics["endpoint_count_clean"] > 500 or metrics["median_edge_length_clean"] < 10:
        metrics["sanity_flags"]["still_fragmented"] = True
    
    artifacts.write_metrics(metrics)
    artifacts.save_json("report", report)
    
    return run_dir, graph_clean_path, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Graph cleanup for sketch-to-SVG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python graph_cleanup.py runs/clean/40_graph_raw/out/graph_raw.json --debug
    python graph_cleanup.py runs/clean/40_graph_raw/out/graph_raw.json --mask runs/clean/10_preprocess/out/output_mask.png --debug
        """
    )
    
    parser.add_argument("graph_raw_path", type=str, help="Path to graph_raw.json")
    parser.add_argument("--runs_root", type=str, default="runs")
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--ridge", type=str, default=None)
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no_debug", dest="debug", action="store_false")
    parser.add_argument("--config", type=str, default=None)
    parser.set_defaults(debug=True)
    
    args = parser.parse_args()
    
    print(f"Processing graph: {args.graph_raw_path}")
    
    run_dir, graph_path, metrics = graph_cleanup(
        args.graph_raw_path,
        mask_path=args.mask,
        ridge_path=args.ridge,
        runs_root=args.runs_root,
        debug=args.debug,
        config_path=args.config,
    )
    
    print(f"Run directory: {run_dir}")
    print(f"Graph saved to: {graph_path}")
    print(f"Nodes: {metrics['raw_node_count']} -> {metrics['clean_node_count']}")
    print(f"Edges: {metrics['raw_edge_count']} -> {metrics['clean_edge_count']}")
    print(f"Median edge length: {metrics['median_edge_length_raw']:.1f} -> {metrics['median_edge_length_clean']:.1f} px")
    print(f"Components: {metrics['connected_components_raw']} -> {metrics['connected_components_clean']}")
    
    if metrics["sanity_flags"]:
        print(f"Sanity flags: {metrics['sanity_flags']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
