"""
Skeleton graph construction for Patent Draw.

Build a graph representation of the skeleton for topology-aware stroke extraction.
Uses 8-connectivity for pixel neighborhoods.
"""

import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize

from patentdraw.io.save_artifacts import DebugArtifactWriter, draw_overlay
from patentdraw.tracer import get_tracer, trace


# 8-connectivity neighborhood offsets
NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


@trace(label="build_skeleton_graph")
def build_skeleton_graph(binary_img, debug_writer=None):
    """
    Build skeleton and graph representation from binary image.
    
    Returns:
        skeleton: uint8 image with skeleton pixels = 255
        graph: networkx graph with nodes at each skeleton pixel
        endpoints: list of (y, x) coordinates for degree-1 nodes
        junctions: list of (y, x) coordinates for degree-3+ nodes
    """
    tracer = get_tracer()
    
    # Skeletonize
    with tracer.span("skeletonize", module="skeleton_graph"):
        # skimage expects binary as bool
        binary_bool = binary_img > 0
        skeleton_bool = skeletonize(binary_bool)
        skeleton = (skeleton_bool.astype(np.uint8)) * 255
        
        skeleton_pixels = np.sum(skeleton > 0)
        tracer.event(f"Skeleton pixels: {skeleton_pixels}")
    
    # Build graph
    with tracer.span("build_graph", module="skeleton_graph"):
        graph = nx.Graph()
        height, width = skeleton.shape
        
        # Find all skeleton pixels
        ys, xs = np.where(skeleton > 0)
        
        # Add nodes
        for y, x in zip(ys, xs):
            graph.add_node((y, x))
        
        # Add edges for 8-connected neighbors
        for y, x in zip(ys, xs):
            for dy, dx in NEIGHBORS_8:
                ny, nx_coord = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx_coord < width:
                    if skeleton[ny, nx_coord] > 0:
                        if (ny, nx_coord) in graph:
                            graph.add_edge((y, x), (ny, nx_coord))
        
        tracer.event(f"Graph: nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")
    
    # Classify nodes
    with tracer.span("classify_nodes", module="skeleton_graph"):
        endpoints = []
        junctions = []
        
        for node in graph.nodes():
            degree = graph.degree(node)
            if degree == 1:
                endpoints.append(node)
            elif degree >= 3:
                junctions.append(node)
        
        tracer.event(f"Endpoints: {len(endpoints)}, Junctions: {len(junctions)}")
    
    # Save debug artifacts
    if debug_writer:
        debug_writer.save_image(skeleton, "stage3", "01_skeleton.png")
        
        # Draw endpoints and junctions overlay
        overlay_img = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
        
        # Endpoints in green
        endpoint_points = [((x, y), 3) for y, x in endpoints]
        # Junctions in red
        junction_points = [((x, y), 4) for y, x in junctions]
        
        overlay = draw_overlay(
            overlay_img,
            points=endpoint_points + junction_points,
            point_color=(0, 255, 0),  # endpoints green
        )
        # Redraw junctions in different color
        for y, x in junctions:
            cv2.circle(overlay, (x, y), 4, (255, 0, 0), -1)
        
        debug_writer.save_image(overlay, "stage3", "02_endpoints_junctions_overlay.png")
        
        metrics = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_endpoints": len(endpoints),
            "num_junctions": len(junctions),
        }
        debug_writer.save_json(metrics, "stage3", "stage3_metrics.json")
    
    return skeleton, graph, endpoints, junctions


def get_node_neighbors(graph, node):
    """Get ordered neighbors of a node for path tracing."""
    return list(graph.neighbors(node))


def compute_neighbor_count_matrix(skeleton):
    """
    Compute neighbor count for each skeleton pixel.
    
    Useful for quick endpoint/junction detection without building full graph.
    """
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ], dtype=np.uint8)
    
    skeleton_bool = (skeleton > 0).astype(np.uint8)
    neighbor_counts = cv2.filter2D(skeleton_bool, -1, kernel)
    
    # Mask to only skeleton pixels
    neighbor_counts = neighbor_counts * skeleton_bool
    
    return neighbor_counts
