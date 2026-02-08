"""
Stage 2: Distance Transform for Sketch-to-SVG Pipeline

Computes the distance transform of a binary stroke mask to find stroke
interiors and estimate stroke width. This prepares the data for ridge
extraction in the next stage.

Output:
- dt.npy: float32 distance transform array
- stroke_width.json: robust stroke width estimate
- Visualization images for debugging
"""

# python distance_transform.py runs/<run>/10_preprocess/out/output_mask.png --debug
# python distance_transform.py examples/clean.png --debug --from_raw
# python distance_transform.py examples/clean.png --debug --from_raw --runs_root runs

import argparse
import json
import cv2
import numpy as np
from pathlib import Path

from utils.artifacts import make_run_dir, StageArtifacts
from utils.io import read_image, ensure_uint8


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "min_dt_for_sampling": 1.5,      # minimum DT value to consider for sampling
    "max_dt_percentile": 99.0,       # exclude DT values above this percentile
    "num_samples": 5000,             # number of samples for width estimation
    "dt_vis_clip_percentile": 99.0,  # percentile for clipping DT visualization
}


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


def validate_binary_mask(img: np.ndarray) -> tuple:
    """
    Validate that input is a binary mask.
    
    Args:
        img: Input image array.
        
    Returns:
        Tuple of (mask, is_binary, was_thresholded).
        
    Raises:
        ValueError: If input appears to be a photo rather than binary mask.
    """
    # Check if color image
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Could be BGR, check if it looks like a photo
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        unique_vals = np.unique(gray)
        if len(unique_vals) > 10:
            raise ValueError(
                "Stage 2 expects a binary mask (strokes=255, background=0). "
                "This input appears to be a photo with many gray values. "
                "Run preprocess.py first to create a binary mask."
            )
        img = gray
    
    # Ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check unique values
    unique_vals = np.unique(img)
    
    # Ideal binary: only 0 and 255
    is_binary = set(unique_vals).issubset({0, 255})
    
    if is_binary:
        return img, True, False
    
    # Check if close to binary (maybe some anti-aliasing)
    if len(unique_vals) <= 10:
        # Threshold at 128
        _, mask = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        return mask, False, True
    
    # Too many values, likely a photo
    if len(unique_vals) > 50:
        raise ValueError(
            "Stage 2 expects a binary mask (strokes=255, background=0). "
            f"This input has {len(unique_vals)} unique values. "
            "Run preprocess.py first to create a binary mask."
        )
    
    # Threshold and warn
    _, mask = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return mask, False, True


def infer_run_dir(input_path: str, runs_root: str) -> Path:
    """
    Infer or create run directory from input path.
    
    If input is inside runs/<run>/..., returns that run directory.
    Otherwise creates a new run directory.
    
    Args:
        input_path: Path to input file.
        runs_root: Root directory for runs.
        
    Returns:
        Path to run directory.
    """
    input_p = Path(input_path).resolve()
    runs_root_p = Path(runs_root).resolve()
    
    # Check if input is inside runs directory
    try:
        rel_path = input_p.relative_to(runs_root_p)
        # First component is the run name
        run_name = rel_path.parts[0]
        return runs_root_p / run_name
    except ValueError:
        # Not inside runs directory, create new run
        return make_run_dir(input_path, runs_root)


def compute_distance_transform(mask: np.ndarray) -> np.ndarray:
    """
    Compute distance transform of binary mask.
    
    DT is 0 in background, positive inside strokes.
    Value at each stroke pixel = distance to nearest background pixel.
    
    Args:
        mask: Binary mask with strokes=255, background=0.
        
    Returns:
        Float32 distance transform array.
    """
    # cv2.distanceTransform expects non-zero pixels as foreground
    # Our mask already has strokes as 255 (non-zero)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, maskSize=5)
    return dt.astype(np.float32)


def estimate_stroke_width(dt: np.ndarray, config: dict) -> dict:
    """
    Estimate stroke width from distance transform values.
    
    Samples interior pixels (avoiding junctions) to get robust median.
    
    Args:
        dt: Distance transform array.
        config: Configuration dictionary.
        
    Returns:
        Dictionary with width statistics.
    """
    min_dt = config["min_dt_for_sampling"]
    max_percentile = config["max_dt_percentile"]
    num_samples = config["num_samples"]
    
    # Find candidate pixels: DT >= min threshold
    candidates_mask = dt >= min_dt
    candidate_values = dt[candidates_mask]
    
    if len(candidate_values) == 0:
        # No interior pixels found
        return {
            "stroke_radius_px": 0.0,
            "stroke_width_px": 0.0,
            "sample_count": 0,
            "percentiles": {},
            "dt_max": float(dt.max()),
        }
    
    # Compute percentile threshold to exclude junction blobs
    dt_cap = np.percentile(candidate_values, max_percentile)
    
    # Filter to pixels below the cap
    filtered_mask = (dt >= min_dt) & (dt <= dt_cap)
    filtered_values = dt[filtered_mask]
    
    if len(filtered_values) == 0:
        filtered_values = candidate_values  # fallback
    
    # Sample uniformly
    if len(filtered_values) > num_samples:
        indices = np.random.choice(len(filtered_values), num_samples, replace=False)
        samples = filtered_values[indices]
    else:
        samples = filtered_values
    
    # Compute robust statistics
    p25 = float(np.percentile(samples, 25))
    p50 = float(np.percentile(samples, 50))  # median
    p75 = float(np.percentile(samples, 75))
    p90 = float(np.percentile(samples, 90))
    p95 = float(np.percentile(samples, 95))
    p99 = float(np.percentile(samples, 99))
    
    stroke_radius = p50
    stroke_width = 2.0 * stroke_radius
    
    return {
        "stroke_radius_px": float(stroke_radius),
        "stroke_width_px": float(stroke_width),
        "sample_count": len(samples),
        "percentiles": {
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "p95": p95,
            "p99": p99,
        },
        "dt_max": float(dt.max()),
        "dt_cap_used": float(dt_cap),
    }


def get_sample_coordinates(dt: np.ndarray, config: dict) -> np.ndarray:
    """
    Get coordinates of sampled pixels for visualization.
    
    Args:
        dt: Distance transform array.
        config: Configuration dictionary.
        
    Returns:
        Nx2 array of (row, col) coordinates.
    """
    min_dt = config["min_dt_for_sampling"]
    max_percentile = config["max_dt_percentile"]
    num_samples = config["num_samples"]
    
    # Find candidate pixels
    candidates_mask = dt >= min_dt
    candidate_values = dt[candidates_mask]
    
    if len(candidate_values) == 0:
        return np.array([]).reshape(0, 2)
    
    # Compute cap
    dt_cap = np.percentile(candidate_values, max_percentile)
    
    # Filter
    filtered_mask = (dt >= min_dt) & (dt <= dt_cap)
    coords = np.argwhere(filtered_mask)
    
    # Sample
    if len(coords) > num_samples:
        indices = np.random.choice(len(coords), num_samples, replace=False)
        coords = coords[indices]
    
    return coords


def visualize_dt(dt: np.ndarray, clip_percentile: float = None) -> np.ndarray:
    """
    Create visualization of distance transform.
    
    Args:
        dt: Distance transform array.
        clip_percentile: If provided, clip DT to this percentile before normalizing.
        
    Returns:
        Uint8 visualization image.
    """
    if clip_percentile is not None:
        # Clip to percentile
        fg_values = dt[dt > 0]
        if len(fg_values) > 0:
            clip_val = np.percentile(fg_values, clip_percentile)
        else:
            clip_val = dt.max()
        dt_clipped = np.clip(dt, 0, clip_val)
    else:
        dt_clipped = dt
    
    # Normalize to 0-255
    dt_max = dt_clipped.max()
    if dt_max > 0:
        vis = (dt_clipped / dt_max * 255).astype(np.uint8)
    else:
        vis = np.zeros_like(dt, dtype=np.uint8)
    
    return vis





def visualize_samples(mask: np.ndarray, coords: np.ndarray, width_stats: dict) -> np.ndarray:
    """
    Create visualization showing sampled points on mask.
    
    Args:
        mask: Binary stroke mask.
        coords: Nx2 array of sample coordinates.
        width_stats: Width statistics dictionary.
        
    Returns:
        BGR visualization image.
    """
    # Convert mask to BGR
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Draw sample points as small colored dots
    for row, col in coords:
        cv2.circle(vis, (int(col), int(row)), 1, (0, 255, 0), -1)  # green
    
    # Add text with statistics
    radius = width_stats.get("stroke_radius_px", 0)
    width = width_stats.get("stroke_width_px", 0)
    count = width_stats.get("sample_count", 0)
    
    text_lines = [
        f"Median radius: {radius:.2f} px",
        f"Stroke width: {width:.2f} px",
        f"Samples: {count}",
    ]
    
    y_offset = 30
    for line in text_lines:
        cv2.putText(vis, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)  # yellow text
        y_offset += 25
    
    return vis


# ---------------------------------------------------------------------------
# Main distance transform function
# ---------------------------------------------------------------------------

def distance_transform(
    input_path: str,
    runs_root: str = "runs",
    debug: bool = True,
    config_path: str = None,
) -> tuple:
    """
    Compute distance transform and stroke width from binary mask.
    
    Args:
        input_path: Path to binary mask image.
        runs_root: Root directory for runs.
        debug: Whether to save debug images.
        config_path: Optional path to config JSON.
        
    Returns:
        Tuple of (run_dir, dt_path, width_stats).
    """
    # Load configuration
    config = load_config(config_path)
    
    # Read and validate input
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    mask, is_binary, was_thresholded = validate_binary_mask(img)
    h, w = mask.shape[:2]
    
    # Infer run directory
    run_dir = infer_run_dir(input_path, runs_root)
    
    # Create artifacts manager
    artifacts = StageArtifacts(run_dir, 20, "distance_transform", debug=debug)
    
    # Debug image 1: input mask
    artifacts.save_debug_image("input_mask", mask)
    
    # Debug image 2: stroke foreground
    stroke_fg = mask.copy()
    artifacts.save_debug_image("stroke_fg", stroke_fg)
    
    # Compute distance transform
    dt = compute_distance_transform(mask)
    
    # Debug image 3: raw DT visualization
    dt_raw_vis = visualize_dt(dt)
    artifacts.save_debug_image("dt_raw_vis", dt_raw_vis)
    
    # Debug image 4: clipped DT visualization
    dt_clipped_vis = visualize_dt(dt, clip_percentile=config["dt_vis_clip_percentile"])
    artifacts.save_debug_image("dt_clipped_vis", dt_clipped_vis)
    

    
    # Estimate stroke width
    width_stats = estimate_stroke_width(dt, config)
    
    # Get sample coordinates for visualization
    sample_coords = get_sample_coordinates(dt, config)
    
    # Debug image 5: width samples visualization
    samples_vis = visualize_samples(mask, sample_coords, width_stats)
    artifacts.save_debug_image("width_samples_vis", samples_vis)
    
    # Save DT array
    dt_path = artifacts.path_out("dt.npy")
    np.save(str(dt_path), dt)
    
    # Save stroke width JSON
    stroke_width_path = artifacts.path_out("stroke_width.json")
    stroke_width_data = {
        "stroke_radius_px": width_stats["stroke_radius_px"],
        "stroke_width_px": width_stats["stroke_width_px"],
        "sample_count": width_stats["sample_count"],
        "percentiles": width_stats["percentiles"],
        "dt_max": width_stats["dt_max"],
    }
    stroke_width_path.write_text(json.dumps(stroke_width_data, indent=2, sort_keys=True))
    
    # Save DT visualization as output
    artifacts.save_output_image("dt_vis", dt_clipped_vis)
    
    # Compute additional metrics
    fg_mask = mask > 0
    dt_mean_fg = float(dt[fg_mask].mean()) if np.any(fg_mask) else 0.0
    
    # Save metrics
    metrics = {
        "input_path": str(input_path),
        "image_w": w,
        "image_h": h,
        "is_binary_input": is_binary,
        "thresholded_input": was_thresholded,
        "dt_max": float(dt.max()),
        "dt_mean_over_fg": dt_mean_fg,
        "sampling_count": width_stats["sample_count"],
        "stroke_radius_px_median": width_stats["stroke_radius_px"],
        "stroke_width_px": width_stats["stroke_width_px"],
        "dt_sample_percentiles": width_stats["percentiles"],
        "config_used": config,
    }
    artifacts.write_metrics(metrics)
    
    return run_dir, dt_path, width_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute distance transform and stroke width from binary mask"
    )
    parser.add_argument(
        "input_path",
        help="Path to binary mask image"
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
    parser.add_argument(
        "--from_raw",
        action="store_true",
        default=False,
        help="Flag for raw photo input (will error with instructions)"
    )
    
    args = parser.parse_args()
    
    # Handle --from_raw flag
    if args.from_raw:
        print("Error: --from_raw is not supported in distance_transform.py.")
        print("Please run preprocess.py first to create a binary mask:")
        print(f"  python preprocess.py {args.input_path} --debug")
        print("Then run distance_transform on the output mask:")
        print("  python distance_transform.py runs/<run>/10_preprocess/out/output_mask.png --debug")
        return 1
    
    # Verify input exists
    input_p = Path(args.input_path)
    if not input_p.exists():
        print(f"Error: Input file not found: {args.input_path}")
        return 1
    
    print(f"Processing: {args.input_path}")
    
    try:
        run_dir, dt_path, width_stats = distance_transform(
            args.input_path,
            runs_root=args.runs_root,
            debug=args.debug,
            config_path=args.config,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    print(f"Run directory: {run_dir}")
    print(f"DT saved to: {dt_path}")
    print(f"Stroke radius (median): {width_stats['stroke_radius_px']:.2f} px")
    print(f"Stroke width: {width_stats['stroke_width_px']:.2f} px")
    print(f"Samples used: {width_stats['sample_count']}")
    
    return 0


if __name__ == "__main__":
    exit(main())


# ---------------------------------------------------------------------------
# Notes on Debug Artifacts
# ---------------------------------------------------------------------------
# Run
#    python preprocess.py examples/clean.png --debug
#    python distance_transform.py runs/<run>/10_preprocess/out/output_mask.png --debug
#
# Open runs/<run>/20_distance_transform/debug/ folder
#    01_input_mask.png - the binary mask input
#    02_stroke_fg.png - same as input (strokes = 255)
#    03_dt_raw_vis.png - raw DT normalized 0-255
#    04_dt_clipped_vis.png - DT clipped at p99 then normalized
#    05_width_samples_vis.png - sampled points with stats overlay
#
# Check runs/<run>/20_distance_transform/out/stroke_width.json
#    - stroke_radius_px_median should match visual stroke thickness / 2
#    - stroke_width_px should be approximately the line thickness in pixels
#
# Confirm dt_clipped_vis.png
#    - Background should be black (DT = 0)
#    - Stroke interiors should be brightest along centerlines
#
# Confirm width_samples_vis.png
#    - Green dots should be spread across stroke interiors
#    - Should NOT appear in background areas
#
# If width estimate is wrong
#    - Too small: decrease min_dt_for_sampling or check if mask is too thin
#    - Too large: increase min_dt_for_sampling or check if mask has merged strokes
#    - Very high variance: check max_dt_percentile, may need to lower it