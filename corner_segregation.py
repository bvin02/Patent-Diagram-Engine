"""
Corner and Junction Detection Module

Detects corners, endpoints, and junctions in binarized sketch lines.
These points become anchors for SVG path generation downstream.
"""

# python corner_segregation.py example_good_sketch_binary_preprocessed.png

import cv2
import numpy as np
from pathlib import Path


def detect_corners(
    input_path: str,
    output_path: str = None,
    corner_quality: float = 0.05,
    min_distance: int = 15,
    max_corners: int = 500,
    use_skeleton: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Detect corners and junctions in a binary sketch image.
    
    Uses Shi-Tomasi corner detection (goodFeaturesToTrack) with NMS
    on skeletonized lines to find key structural points.
    
    Args:
        input_path: Path to binary image
        output_path: Path for visualization output
        corner_quality: Minimum quality of corners (0-1)
        min_distance: Minimum distance between corners (NMS radius)
        max_corners: Maximum number of corners to detect
        use_skeleton: Whether to skeletonize first (recommended)
    
    Returns:
        Tuple of (skeleton_image, points_dict)
        - skeleton_image: The skeletonized line image
        - points_dict: Dict with 'corners', 'endpoints', 'junctions' lists
    """
    # Load binary image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")
    
    print(f"Loaded image: {input_path} ({img.shape[1]}x{img.shape[0]})")
    
    # Invert for processing (white lines on black)
    inverted = cv2.bitwise_not(img)
    
    # Skeletonize to get 1-pixel wide lines
    if use_skeleton:
        skeleton = skeletonize(inverted)
        print("Applied skeletonization")
    else:
        skeleton = inverted
    
    # Shi-Tomasi corner detection (better than Harris for this use case)
    corners_raw = cv2.goodFeaturesToTrack(
        skeleton,
        maxCorners=max_corners,
        qualityLevel=corner_quality,
        minDistance=min_distance,
    )
    
    corner_points = []
    if corners_raw is not None:
        corner_points = [(int(c[0][0]), int(c[0][1])) for c in corners_raw]
    print(f"Detected {len(corner_points)} corner points")
    
    # Detect endpoints and junctions from skeleton topology
    endpoints = detect_endpoints(skeleton, min_distance=min_distance // 2)
    junctions = detect_junctions(skeleton, min_distance=min_distance // 2)
    print(f"Detected {len(endpoints)} endpoints, {len(junctions)} junctions")
    
    # Compile all points
    all_points = {
        "corners": corner_points,
        "endpoints": endpoints,
        "junctions": junctions,
    }
    
    # Create visualization
    if output_path is None:
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.stem}_corners.png")
    
    vis = create_corner_visualization(img, all_points, skeleton)
    cv2.imwrite(output_path, vis)
    print(f"Saved corner visualization to: {output_path}")
    
    return skeleton, all_points


def skeletonize(binary_img: np.ndarray) -> np.ndarray:
    """
    Reduce binary shapes to 1-pixel wide skeleton using morphological thinning.
    """
    skeleton = np.zeros_like(binary_img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    temp = binary_img.copy()
    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        subset = cv2.subtract(temp, opened)
        skeleton = cv2.bitwise_or(skeleton, subset)
        temp = eroded.copy()
        
        if cv2.countNonZero(temp) == 0:
            break
    
    return skeleton


def count_neighbors(skeleton: np.ndarray, x: int, y: int) -> int:
    """Count 8-connected neighbors of a skeleton pixel."""
    h, w = skeleton.shape
    if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
        return 0
    
    # Extract 3x3 neighborhood
    neighborhood = skeleton[y-1:y+2, x-1:x+2]
    # Count non-zero pixels minus center
    return int(np.count_nonzero(neighborhood)) - (1 if skeleton[y, x] > 0 else 0)


def detect_endpoints(skeleton: np.ndarray, min_distance: int = 5) -> list[tuple[int, int]]:
    """
    Detect line endpoints in skeleton (pixels with only 1 neighbor).
    Applies NMS to avoid clusters.
    """
    raw_endpoints = []
    h, w = skeleton.shape
    
    # Find all pixels with exactly 1 neighbor
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 0:
                continue
            if count_neighbors(skeleton, x, y) == 1:
                raw_endpoints.append((x, y))
    
    # Apply non-maximum suppression
    return nms_points(raw_endpoints, min_distance)


def detect_junctions(skeleton: np.ndarray, min_distance: int = 5) -> list[tuple[int, int]]:
    """
    Detect junction points in skeleton (pixels with 3+ neighbors).
    Applies NMS to reduce clusters to single points.
    """
    raw_junctions = []
    h, w = skeleton.shape
    
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 0:
                continue
            if count_neighbors(skeleton, x, y) >= 3:
                raw_junctions.append((x, y))
    
    # Apply non-maximum suppression
    return nms_points(raw_junctions, min_distance)


def nms_points(points: list[tuple[int, int]], min_distance: int) -> list[tuple[int, int]]:
    """
    Non-maximum suppression for point clusters.
    Keeps one representative point per cluster.
    """
    if not points:
        return []
    
    points = list(points)
    kept = []
    
    while points:
        # Take first point
        current = points.pop(0)
        kept.append(current)
        
        # Remove all points within min_distance
        points = [
            p for p in points
            if abs(p[0] - current[0]) > min_distance or abs(p[1] - current[1]) > min_distance
        ]
    
    return kept


def create_corner_visualization(
    original: np.ndarray,
    points: dict,
    skeleton: np.ndarray = None
) -> np.ndarray:
    """Create visualization with detected points marked."""
    # Convert to color
    if len(original.shape) == 2:
        vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        vis = original.copy()
    
    # Overlay skeleton in light blue if available
    if skeleton is not None:
        vis[skeleton > 0] = [255, 200, 200]
    
    # Define colors and sizes for point types
    styles = {
        "corners": {"color": (0, 255, 0), "radius": 4},      # Green
        "endpoints": {"color": (0, 0, 255), "radius": 5},    # Red  
        "junctions": {"color": (255, 0, 255), "radius": 5},  # Magenta
    }
    
    # Draw points
    for point_type, point_list in points.items():
        style = styles.get(point_type, {"color": (255, 255, 0), "radius": 3})
        for (x, y) in point_list:
            cv2.circle(vis, (x, y), style["radius"], style["color"], -1)
            cv2.circle(vis, (x, y), style["radius"] + 1, (0, 0, 0), 1)  # Black outline
    
    return vis


def get_all_keypoints(points: dict) -> list[tuple[int, int]]:
    """Flatten all detected points into a single list."""
    all_pts = []
    for pt_list in points.values():
        all_pts.extend(pt_list)
    return list(set(all_pts))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Detect corners and junctions in binarized sketch"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="example_good_sketch_binary.png",
        help="Input binary image"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output visualization path"
    )
    parser.add_argument(
        "--quality",
        type=float,
        default=0.05,
        help="Corner quality threshold 0-1 (default: 0.05)"
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=15,
        help="Minimum distance between points (default: 15)"
    )
    parser.add_argument(
        "--max-corners",
        type=int,
        default=500,
        help="Maximum corners to detect (default: 500)"
    )
    parser.add_argument(
        "--no-skeleton",
        action="store_true",
        help="Skip skeletonization step"
    )
    
    args = parser.parse_args()
    
    skeleton, points = detect_corners(
        args.input,
        args.output,
        corner_quality=args.quality,
        min_distance=args.min_distance,
        max_corners=args.max_corners,
        use_skeleton=not args.no_skeleton,
    )
    
    print(f"\nTotal keypoints: {len(get_all_keypoints(points))}")


if __name__ == "__main__":
    main()
