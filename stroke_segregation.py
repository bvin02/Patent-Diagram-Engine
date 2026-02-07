"""
Stroke Segregation Module

Separates individual strokes from a binarized sketch image.
Each connected stroke region gets a unique label/mask.
"""

# python stroke_segregation.py example_good_sketch_binary_preprocessed.png --stats

import cv2
import numpy as np
from pathlib import Path
import random


def segregate_strokes(
    input_path: str,
    output_path: str = None,
    min_stroke_area: int = 10,
    connectivity: int = 8,
) -> tuple[np.ndarray, int]:
    """
    Separate individual strokes from a binary sketch image.
    
    Uses connected component analysis to identify distinct stroke regions.
    Each stroke gets a unique label for downstream processing.
    
    Args:
        input_path: Path to binary image (black lines on white background)
        output_path: Path for visualization output. If None, derived from input.
        min_stroke_area: Minimum pixel area for a valid stroke (filters noise)
        connectivity: 4 or 8 connected neighbors
    
    Returns:
        Tuple of (labeled_image, num_strokes)
        - labeled_image: Each pixel contains its stroke ID (0 = background)
        - num_strokes: Total number of detected strokes
    """
    # Load binary image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")
    
    print(f"Loaded binary image: {input_path} ({img.shape[1]}x{img.shape[0]})")
    
    # Invert: we need white strokes on black for connected components
    # Binary image has black lines (0) on white background (255)
    inverted = cv2.bitwise_not(img)
    
    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inverted, connectivity=connectivity
    )
    
    print(f"Found {num_labels - 1} raw components (including noise)")
    
    # Filter out small noise components
    valid_strokes = 0
    filtered_labels = np.zeros_like(labels)
    
    for label_id in range(1, num_labels):  # Skip 0 (background)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_stroke_area:
            valid_strokes += 1
            filtered_labels[labels == label_id] = valid_strokes
    
    print(f"Filtered to {valid_strokes} valid strokes (min_area={min_stroke_area})")
    
    # Generate visualization with random colors per stroke
    if output_path is None:
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.stem}_strokes.png")
    
    vis = create_stroke_visualization(filtered_labels, valid_strokes)
    cv2.imwrite(output_path, vis)
    print(f"Saved stroke visualization to: {output_path}")
    
    return filtered_labels, valid_strokes


def create_stroke_visualization(labels: np.ndarray, num_strokes: int) -> np.ndarray:
    """Create a color visualization where each stroke has a unique color."""
    # Generate distinct colors for each stroke
    random.seed(42)  # Reproducible colors
    colors = [(255, 255, 255)]  # Background = white
    for _ in range(num_strokes):
        colors.append((
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        ))
    
    # Map labels to colors
    vis = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label_id in range(num_strokes + 1):
        vis[labels == label_id] = colors[label_id]
    
    return vis


def get_stroke_mask(labels: np.ndarray, stroke_id: int) -> np.ndarray:
    """Extract a binary mask for a specific stroke."""
    return (labels == stroke_id).astype(np.uint8) * 255


def get_stroke_stats(labels: np.ndarray, num_strokes: int) -> list[dict]:
    """Get bounding box and area stats for each stroke."""
    stats = []
    for stroke_id in range(1, num_strokes + 1):
        mask = labels == stroke_id
        points = np.where(mask)
        if len(points[0]) == 0:
            continue
        
        y_min, y_max = points[0].min(), points[0].max()
        x_min, x_max = points[1].min(), points[1].max()
        
        stats.append({
            "id": stroke_id,
            "area": int(mask.sum()),
            "bbox": (int(x_min), int(y_min), int(x_max), int(y_max)),
            "centroid": (int(points[1].mean()), int(points[0].mean())),
        })
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Segregate strokes from binarized sketch"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="example_good_sketch_binary.png",
        help="Input binary image (default: example_good_sketch_binary.png)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output visualization path"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=10,
        help="Minimum stroke area in pixels (default: 10)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics for each stroke"
    )
    
    args = parser.parse_args()
    
    labels, num_strokes = segregate_strokes(
        args.input,
        args.output,
        min_stroke_area=args.min_area,
    )
    
    if args.stats:
        print("\n--- Stroke Statistics ---")
        stats = get_stroke_stats(labels, num_strokes)
        for s in stats:
            print(f"Stroke {s['id']}: area={s['area']}, bbox={s['bbox']}")


if __name__ == "__main__":
    main()
