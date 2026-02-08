"""
Stage 3: Ridge Extraction for Sketch-to-SVG Pipeline

Extracts centerline candidates from the distance transform using local maxima
detection. This produces a ridge map that stays continuous on thin strokes
and handles DT plateaus correctly.

Output:
- ridge.png: uint8 mask with ridge pixels=255
- coverage.json: quantitative coverage metrics
- metrics.json: full diagnostic metrics
"""

# python ridge_extraction.py runs/<run>/10_preprocess/out/output_mask.png runs/<run>/20_distance_transform/out/dt.npy --debug
# python ridge_extraction.py runs/<run>/10_preprocess/out/output_mask.png runs/<run>/20_distance_transform/out/dt.npy --debug --config configs/ridge.json

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import maximum_filter, label, distance_transform_edt
from skimage.morphology import thin

from utils.artifacts import make_run_dir, StageArtifacts
from utils.io import read_image, ensure_uint8


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "min_dt": 0.5,                    # minimum DT value to consider as ridge
    "eps": 1e-6,                      # tolerance for local max comparison
    "min_ridge_component_area": 2,    # remove ridge components smaller than this
    "dilate_radius": 2,               # radius for dilating sparse local maxima
    "apply_thinning": True,           # thin to 1-pixel skeleton after dilation
    "spur_prune_length": 1,           # remove spurs shorter than this (pixels)
    "plateau_thin": False,            # optional plateau thinning (before dilation)
    "dt_vis_clip_percentile": 99.0,   # percentile for clipping DT visualization
    "ridge_distance_clip": 5.0,       # clip distance to ridge for visualization
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


def infer_run_dir(mask_path: str, runs_root: str) -> Path:
    """
    Infer run directory from mask path.
    
    If mask_path is under runs/<run>/..., returns that run directory.
    Otherwise creates a new run directory.
    """
    mask_p = Path(mask_path).resolve()
    runs_root_p = Path(runs_root).resolve()
    
    try:
        rel_path = mask_p.relative_to(runs_root_p)
        run_name = rel_path.parts[0]
        return runs_root_p / run_name
    except ValueError:
        return make_run_dir(mask_path, runs_root)


def validate_binary_mask(img: np.ndarray) -> tuple:
    """
    Validate and return binary mask.
    
    Returns:
        Tuple of (mask, was_thresholded).
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    unique_vals = np.unique(img)
    is_binary = set(unique_vals).issubset({0, 255})
    
    if is_binary:
        return img, False
    
    _, mask = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return mask, True


def visualize_dt_clipped(dt: np.ndarray, fg: np.ndarray, percentile: float) -> np.ndarray:
    """Create clipped DT visualization."""
    fg_values = dt[fg]
    if len(fg_values) == 0:
        return np.zeros_like(dt, dtype=np.uint8)
    
    clip_val = np.percentile(fg_values, percentile)
    dt_clipped = np.clip(dt, 0, clip_val)
    
    if clip_val > 0:
        vis = (dt_clipped / clip_val * 255).astype(np.uint8)
    else:
        vis = np.zeros_like(dt, dtype=np.uint8)
    
    return vis


# ---------------------------------------------------------------------------
# Core ridge extraction functions
# ---------------------------------------------------------------------------

def compute_local_maxima(dt: np.ndarray, fg: np.ndarray, config: dict) -> np.ndarray:
    """
    Compute local maxima of DT using maximum_filter.
    
    A pixel is a local maximum if:
    - It is in the foreground
    - Its DT value >= min_dt
    - Its DT value is within eps of the neighborhood maximum
    
    This handles plateaus correctly by allowing equality.
    
    Args:
        dt: Distance transform array (float32).
        fg: Boolean foreground mask.
        config: Configuration dictionary.
        
    Returns:
        Boolean array of local maximum pixels.
    """
    min_dt = config["min_dt"]
    eps = config["eps"]
    
    # Compute local maximum in 3x3 neighborhood
    dt_max = maximum_filter(dt, size=3, mode="constant", cval=0.0)
    
    # Pixel is local max if it equals the neighborhood max (within eps)
    localmax = fg & (dt >= min_dt) & (np.abs(dt - dt_max) <= eps)
    
    return localmax


def thin_plateaus(localmax: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """
    Optionally thin plateaus by keeping pixels with at least one 4-neighbor
    having strictly lower DT.
    
    This is a simple deterministic thinning that reduces 2-pixel wide ridges
    while preserving connectivity.
    
    Args:
        localmax: Boolean local maximum mask.
        dt: Distance transform array.
        
    Returns:
        Thinned boolean mask.
    """
    # 4-connectivity neighbors (up, down, left, right)
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    h, w = dt.shape
    thinned = np.zeros_like(localmax)
    
    # Get coordinates of local max pixels
    coords = np.argwhere(localmax)
    
    for r, c in coords:
        current_dt = dt[r, c]
        has_lower_neighbor = False
        
        for dr, dc in shifts:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if dt[nr, nc] < current_dt - 1e-9:
                    has_lower_neighbor = True
                    break
        
        # Keep pixel if it has a strictly lower neighbor, or keep all if none do
        if has_lower_neighbor:
            thinned[r, c] = True
    
    # For pixels in plateaus with no lower neighbors, keep all of them
    # to preserve connectivity
    plateau_pixels = localmax & ~thinned
    
    # Check if any plateau pixel has a neighbor that was kept
    for r, c in np.argwhere(plateau_pixels):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if thinned[nr, nc]:
                    thinned[r, c] = True
                    break
    
    # If thinning removed too much, fall back to original
    if thinned.sum() < localmax.sum() * 0.3:
        return localmax
    
    return thinned


def remove_small_components(mask: np.ndarray, min_area: int) -> tuple:
    """
    Remove connected components smaller than min_area.
    
    Args:
        mask: Boolean or uint8 mask.
        min_area: Minimum component area to keep.
        
    Returns:
        Tuple of (filtered_mask, num_components_before, num_components_after).
    """
    binary = mask.astype(np.uint8)
    labeled, num_components = label(binary, structure=np.ones((3, 3)))
    
    if num_components == 0:
        return mask.astype(bool), 0, 0
    
    # Get component sizes
    component_sizes = np.bincount(labeled.ravel())
    # component 0 is background
    
    # Create mask of components to keep
    keep_mask = np.zeros_like(labeled, dtype=bool)
    num_kept = 0
    
    for i in range(1, num_components + 1):
        if component_sizes[i] >= min_area:
            keep_mask |= (labeled == i)
            num_kept += 1
    
    return keep_mask, num_components, num_kept


def dilate_and_thin(localmax: np.ndarray, fg: np.ndarray, config: dict) -> tuple:
    """
    Bridge gaps in sparse local maxima using dilation, then thin to 1-pixel.
    
    This handles the fragmentation issue on thin strokes where local maxima
    are sparse discrete points rather than continuous ridges.
    
    Steps:
    1. Dilate local maxima to connect nearby points
    2. AND with foreground to stay inside strokes
    3. Apply morphological thinning to get 1-pixel skeleton
    4. AND with foreground again (thinning can sometimes expand slightly)
    
    Args:
        localmax: Boolean local maximum mask (sparse points).
        fg: Boolean foreground mask.
        config: Configuration dictionary.
        
    Returns:
        Tuple of (connected_ridge, ridge_before_thin).
    """
    dilate_radius = config["dilate_radius"]
    apply_thinning = config["apply_thinning"]
    
    # Step 1: Dilate to connect sparse local maxima
    if dilate_radius > 0:
        kernel_size = 2 * dilate_radius + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        dilated = cv2.dilate(
            localmax.astype(np.uint8) * 255, 
            kernel, 
            iterations=1
        ) > 0
    else:
        dilated = localmax
    
    # Step 2: Constrain to foreground
    connected = dilated & fg
    
    ridge_before_thin = connected.copy()
    
    # Step 3: Thin to 1-pixel skeleton
    if apply_thinning:
        # skimage.morphology.thin expects boolean array
        connected = thin(connected)
        # Step 4: Enforce ridge stays inside foreground after thinning
        # (thinning can occasionally produce boundary artifacts)
        connected = connected & fg
    
    return connected, ridge_before_thin


def prune_spurs(ridge: np.ndarray, max_spur_length: int) -> np.ndarray:
    """
    Remove short spurs (branches ending in an endpoint) from ridge.
    
    Iteratively removes endpoint pixels if their removal does not disconnect
    the ridge significantly. This removes small artifacts from thinning.
    
    Args:
        ridge: Boolean ridge mask (1-pixel skeleton).
        max_spur_length: Maximum length of spurs to remove.
        
    Returns:
        Pruned ridge mask.
    """
    if max_spur_length <= 0:
        return ridge
    
    pruned = ridge.copy()
    
    # Iteratively remove endpoints up to max_spur_length times
    for _ in range(max_spur_length):
        # Find endpoints (pixels with exactly 1 neighbor)
        binary = pruned.astype(np.float32)
        kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=np.float32)
        neighbor_count = cv2.filter2D(binary, -1, kernel)
        
        endpoints = pruned & (neighbor_count.astype(np.int32) == 1)
        
        if not endpoints.any():
            break
        
        # Remove endpoints
        pruned = pruned & ~endpoints
    
    return pruned


def cleanup_ridge(ridge: np.ndarray, config: dict) -> tuple:
    """
    Clean up ridge mask by pruning spurs and removing small components.
    
    Args:
        ridge: Boolean ridge mask.
        config: Configuration dictionary.
        
    Returns:
        Tuple of (cleaned_mask, stats_dict).
    """
    min_area = config["min_ridge_component_area"]
    spur_length = config.get("spur_prune_length", 3)
    
    # Prune short spurs (artifacts from thinning)
    ridge_pruned = prune_spurs(ridge, spur_length)
    
    # Remove small components
    ridge_clean, count_raw, count_final = remove_small_components(ridge_pruned, min_area)
    
    stats = {
        "component_count_raw": count_raw,
        "component_count_final": count_final,
        "spur_prune_length": spur_length,
    }
    
    return ridge_clean, stats


# ---------------------------------------------------------------------------
# Diagnostic functions
# ---------------------------------------------------------------------------

def compute_pixel_degrees(ridge: np.ndarray) -> np.ndarray:
    """
    Compute degree (neighbor count) for each ridge pixel.
    
    Uses 8-connectivity.
    
    Args:
        ridge: Boolean ridge mask.
        
    Returns:
        Array of same shape with degree values (0 for non-ridge pixels).
    """
    binary = ridge.astype(np.float32)
    
    # Count 8-neighbors using convolution
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.float32)
    
    neighbor_count = cv2.filter2D(binary, -1, kernel)
    
    # Only keep values for ridge pixels
    degrees = np.where(ridge, neighbor_count.astype(np.int32), 0)
    
    return degrees


def find_endpoints_junctions(ridge: np.ndarray) -> tuple:
    """
    Find endpoints (degree=1) and junctions (degree>=3) on ridge.
    
    Args:
        ridge: Boolean ridge mask.
        
    Returns:
        Tuple of (endpoints_mask, junctions_mask, endpoint_count, junction_count).
    """
    degrees = compute_pixel_degrees(ridge)
    
    endpoints = (degrees == 1)
    junctions = (degrees >= 3)
    
    endpoint_count = int(endpoints.sum())
    junction_count = int(junctions.sum())
    
    return endpoints, junctions, endpoint_count, junction_count


def create_endpoints_junctions_vis(mask: np.ndarray, endpoints: np.ndarray, 
                                    junctions: np.ndarray) -> np.ndarray:
    """
    Create visualization showing endpoints (red) and junctions (blue) on mask.
    
    Args:
        mask: Grayscale stroke mask.
        endpoints: Boolean endpoints mask.
        junctions: Boolean junctions mask.
        
    Returns:
        BGR visualization image.
    """
    # Convert mask to grayscale BGR
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    vis = (vis * 0.5).astype(np.uint8)  # dim the background
    
    # Draw endpoints in red (BGR: 0, 0, 255)
    vis[endpoints] = [0, 0, 255]
    
    # Draw junctions in blue (BGR: 255, 0, 0)
    vis[junctions] = [255, 0, 0]
    
    return vis


def compute_coverage_metrics(ridge: np.ndarray, fg: np.ndarray, dt: np.ndarray) -> dict:
    """
    Compute coverage metrics for ridge quality assessment.
    
    Args:
        ridge: Boolean ridge mask.
        fg: Boolean foreground mask.
        dt: Original distance transform.
        
    Returns:
        Dictionary with coverage metrics.
    """
    # Distance from each pixel to nearest ridge pixel
    dt_ridge = distance_transform_edt(~ridge)
    
    # Coverage: fraction of stroke pixels within N px of ridge
    fg_count = fg.sum()
    if fg_count == 0:
        return {
            "coverage_1px": 0.0,
            "coverage_2px": 0.0,
            "ridge_dt_p10": 0.0,
            "ridge_dt_p50": 0.0,
            "ridge_dt_p90": 0.0,
        }
    
    within_1px = (dt_ridge <= 1.0) & fg
    within_2px = (dt_ridge <= 2.0) & fg
    
    coverage_1px = float(within_1px.sum() / fg_count)
    coverage_2px = float(within_2px.sum() / fg_count)
    
    # Ridge centeredness: DT values at ridge pixels
    ridge_dt_values = dt[ridge]
    if len(ridge_dt_values) > 0:
        ridge_dt_p10 = float(np.percentile(ridge_dt_values, 10))
        ridge_dt_p50 = float(np.percentile(ridge_dt_values, 50))
        ridge_dt_p90 = float(np.percentile(ridge_dt_values, 90))
    else:
        ridge_dt_p10 = ridge_dt_p50 = ridge_dt_p90 = 0.0
    
    return {
        "coverage_1px": coverage_1px,
        "coverage_2px": coverage_2px,
        "ridge_dt_p10": ridge_dt_p10,
        "ridge_dt_p50": ridge_dt_p50,
        "ridge_dt_p90": ridge_dt_p90,
        "dt_ridge": dt_ridge,  # for visualization
    }


def create_ridge_distance_vis(dt_ridge: np.ndarray, clip_val: float) -> np.ndarray:
    """
    Create visualization of distance to ridge.
    
    Args:
        dt_ridge: Distance to nearest ridge pixel.
        clip_val: Maximum distance to show.
        
    Returns:
        Uint8 visualization (darker = closer to ridge).
    """
    clipped = np.clip(dt_ridge, 0, clip_val)
    # Invert so closer = brighter
    vis = ((clip_val - clipped) / clip_val * 255).astype(np.uint8)
    return vis


def create_stroke_coverage_vis(dt_ridge: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """
    Create visualization of stroke pixels within 1px of ridge.
    
    Args:
        dt_ridge: Distance to nearest ridge pixel.
        fg: Boolean foreground mask.
        
    Returns:
        Uint8 visualization (white = within 1px, dark = farther).
    """
    within_1px = (dt_ridge <= 1.0) & fg
    vis = np.zeros_like(fg, dtype=np.uint8)
    vis[within_1px] = 255
    vis[fg & ~within_1px] = 64  # dim for farther pixels
    return vis


def create_ridge_overlay(base_img: np.ndarray, ridge: np.ndarray, 
                          color: tuple = (0, 0, 255)) -> np.ndarray:
    """
    Create overlay visualization with ridge pixels colored.
    
    Args:
        base_img: Grayscale or BGR base image.
        ridge: Boolean ridge mask.
        color: BGR color for ridge pixels.
        
    Returns:
        BGR visualization image.
    """
    if len(base_img.shape) == 2:
        vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        vis = base_img.copy()
    
    vis[ridge] = color
    return vis


def find_original_input(run_dir: Path) -> Path:
    """
    Try to locate original input image in run directory.
    
    Args:
        run_dir: Path to run directory.
        
    Returns:
        Path to input image or None if not found.
    """
    input_dir = run_dir / "00_input"
    if not input_dir.exists():
        return None
    
    # Look for 01_input.* files
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = input_dir / f"01_input{ext}"
        if candidate.exists():
            return candidate
    
    return None


# ---------------------------------------------------------------------------
# Main ridge extraction function
# ---------------------------------------------------------------------------

def ridge_extraction(
    mask_path: str,
    dt_path: str,
    runs_root: str = "runs",
    debug: bool = True,
    config_path: str = None,
) -> tuple:
    """
    Extract ridge (centerline candidates) from distance transform.
    
    Args:
        mask_path: Path to binary mask image.
        dt_path: Path to distance transform .npy file.
        runs_root: Root directory for runs.
        debug: Whether to save debug images.
        config_path: Optional path to config JSON.
        
    Returns:
        Tuple of (run_dir, ridge_path, metrics).
    """
    # Load configuration
    config = load_config(config_path)
    
    # Read mask
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_img is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    
    mask, thresholded_input = validate_binary_mask(mask_img)
    h, w = mask.shape[:2]
    
    # Load DT
    dt = np.load(str(dt_path)).astype(np.float32)
    if dt.shape != mask.shape:
        raise ValueError(f"DT shape {dt.shape} does not match mask shape {mask.shape}")
    
    # Foreground
    fg = mask > 0
    
    # Validate DT background
    corrected_dt_bg = False
    if np.any(dt[~fg] != 0):
        dt[~fg] = 0
        corrected_dt_bg = True
    
    # Infer run directory
    run_dir = infer_run_dir(mask_path, runs_root)
    
    # Create artifacts manager
    artifacts = StageArtifacts(run_dir, 30, "ridge", debug=debug)
    
    # Debug 1: input mask
    artifacts.save_debug_image("input_mask", mask)
    
    # Debug 2: DT clipped visualization
    dt_clipped_vis = visualize_dt_clipped(dt, fg, config["dt_vis_clip_percentile"])
    artifacts.save_debug_image("dt_clipped_vis", dt_clipped_vis)
    
    # Compute local maxima
    localmax = compute_local_maxima(dt, fg, config)
    localmax_raw_count = int(localmax.sum())
    
    # Optional plateau thinning
    if config["plateau_thin"]:
        localmax = thin_plateaus(localmax, dt)
    
    # Debug 3: local max raw
    localmax_uint8 = (localmax.astype(np.uint8) * 255)
    artifacts.save_debug_image("localmax_raw", localmax_uint8)
    
    # Debug 4: local max on mask
    localmax_on_mask = create_ridge_overlay(mask, localmax, color=(0, 0, 255))
    artifacts.save_debug_image("localmax_on_mask", localmax_on_mask)
    
    # Connect sparse local maxima using dilation + thinning
    ridge_connected, ridge_before_thin = dilate_and_thin(localmax, fg, config)
    
    # Debug 5: ridge after dilation (before thinning)
    ridge_before_thin_vis = (ridge_before_thin.astype(np.uint8) * 255)
    artifacts.save_debug_image("ridge_after_dilate", ridge_before_thin_vis)
    
    # Debug 6: ridge after thinning
    ridge_after_thin_vis = (ridge_connected.astype(np.uint8) * 255)
    artifacts.save_debug_image("ridge_after_thin", ridge_after_thin_vis)
    
    # Cleanup ridge (remove small components)
    ridge_final, cleanup_stats = cleanup_ridge(ridge_connected, config)
    ridge_final_count = int(ridge_final.sum())
    
    # Debug 7: ridge after cleanup
    ridge_final_vis = (ridge_final.astype(np.uint8) * 255)
    artifacts.save_debug_image("ridge_final", ridge_final_vis)
    
    # Debug 8: ridge overlay on mask
    ridge_overlay = create_ridge_overlay(mask, ridge_final, color=(0, 0, 255))
    artifacts.save_debug_image("ridge_overlay_on_mask", ridge_overlay)
    
    # Compute endpoints and junctions
    endpoints, junctions, endpoint_count, junction_count = find_endpoints_junctions(ridge_final)
    
    # Debug 9: endpoints/junctions visualization
    ej_vis = create_endpoints_junctions_vis(mask, endpoints, junctions)
    artifacts.save_debug_image("endpoints_junctions_vis", ej_vis)
    
    # Compute coverage metrics
    coverage = compute_coverage_metrics(ridge_final, fg, dt)
    dt_ridge = coverage.pop("dt_ridge")
    
    # Debug 10: ridge distance visualization
    ridge_dist_vis = create_ridge_distance_vis(dt_ridge, config["ridge_distance_clip"])
    artifacts.save_debug_image("ridge_distance_vis", ridge_dist_vis)
    
    # Debug 11: stroke coverage visualization
    stroke_cov_vis = create_stroke_coverage_vis(dt_ridge, fg)
    artifacts.save_debug_image("stroke_coverage_vis", stroke_cov_vis)
    
    # Debug 12: ridge overlay on original input (if exists)
    original_input_path = find_original_input(run_dir)
    if original_input_path is not None:
        original_img = cv2.imread(str(original_input_path))
        if original_img is not None:
            # Resize ridge to match original if needed
            if original_img.shape[:2] != ridge_final.shape:
                ridge_resized = cv2.resize(
                    ridge_final.astype(np.uint8) * 255,
                    (original_img.shape[1], original_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ) > 0
            else:
                ridge_resized = ridge_final
            
            input_overlay = create_ridge_overlay(original_img, ridge_resized, color=(0, 0, 255))
            artifacts.save_debug_image("ridge_overlay_on_input", input_overlay)
    
    # Save ridge output
    ridge_path = artifacts.path_out("ridge.png")
    cv2.imwrite(str(ridge_path), ridge_final_vis)
    
    # Save coverage.json
    coverage_data = {
        "coverage_1px": coverage["coverage_1px"],
        "coverage_2px": coverage["coverage_2px"],
        "ridge_dt_percentiles": {
            "p10": coverage["ridge_dt_p10"],
            "p50": coverage["ridge_dt_p50"],
            "p90": coverage["ridge_dt_p90"],
        },
        "endpoint_count": endpoint_count,
        "junction_count": junction_count,
    }
    coverage_path = artifacts.path_out("coverage.json")
    coverage_path.write_text(json.dumps(coverage_data, indent=2, sort_keys=True))
    
    # Build metrics
    metrics = {
        "mask_path": str(mask_path),
        "dt_path": str(dt_path),
        "image_w": w,
        "image_h": h,
        "thresholded_input": thresholded_input,
        "corrected_dt_background": corrected_dt_bg,
        "min_dt_used": config["min_dt"],
        "eps_used": config["eps"],
        "ridge_pixels_raw": localmax_raw_count,
        "ridge_pixels_final": ridge_final_count,
        "component_count_raw": cleanup_stats["component_count_raw"],
        "component_count_final": cleanup_stats["component_count_final"],
        "endpoint_count": endpoint_count,
        "junction_count": junction_count,
        "dilate_radius": config["dilate_radius"],
        "apply_thinning": config["apply_thinning"],
        "min_ridge_component_area": config["min_ridge_component_area"],
        "coverage_1px": coverage["coverage_1px"],
        "coverage_2px": coverage["coverage_2px"],
        "ridge_dt_percentiles": {
            "p10": coverage["ridge_dt_p10"],
            "p50": coverage["ridge_dt_p50"],
            "p90": coverage["ridge_dt_p90"],
        },
        "config_used": config,
    }
    artifacts.write_metrics(metrics)
    
    return run_dir, ridge_path, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract ridge (centerlines) from distance transform"
    )
    parser.add_argument(
        "mask_path",
        help="Path to binary mask image"
    )
    parser.add_argument(
        "dt_path",
        help="Path to distance transform .npy file"
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
    mask_p = Path(args.mask_path)
    dt_p = Path(args.dt_path)
    
    if not mask_p.exists():
        print(f"Error: Mask file not found: {args.mask_path}")
        return 1
    
    if not dt_p.exists():
        print(f"Error: DT file not found: {args.dt_path}")
        return 1
    
    print(f"Processing mask: {args.mask_path}")
    print(f"Processing DT: {args.dt_path}")
    
    try:
        run_dir, ridge_path, metrics = ridge_extraction(
            args.mask_path,
            args.dt_path,
            runs_root=args.runs_root,
            debug=args.debug,
            config_path=args.config,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    print(f"Run directory: {run_dir}")
    print(f"Ridge saved to: {ridge_path}")
    print(f"Ridge pixels: {metrics['ridge_pixels_raw']} -> {metrics['ridge_pixels_final']}")
    print(f"Endpoints: {metrics['endpoint_count']}, Junctions: {metrics['junction_count']}")
    print(f"Coverage (1px): {metrics['coverage_1px']:.2%}")
    print(f"Coverage (2px): {metrics['coverage_2px']:.2%}")
    
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
#
# Open runs/<run>/30_ridge/debug/
#
# ridge_overlay_on_mask.png
#    - Ridge should stay centered and continuous along long strokes
#    - Junctions can have small blobs, this is expected
#    - Ridge should never leave the stroke region
#
# endpoints_junctions_vis.png
#    - Red = endpoints, Blue = junctions
#    - Endpoints should appear mostly at true line ends
#    - Endpoint count should not explode (hundreds) for clean images
#
# ridge_distance_vis.png
#    - Brighter = closer to ridge
#    - Most stroke area should be bright (within 1-2 px of ridge)
#
# stroke_coverage_vis.png
#    - White = stroke pixels within 1px of ridge
#    - Dark gray = stroke pixels farther than 1px
#    - Most stroke area should be white
#
# checks in out/coverage.json:
#    - coverage_1px should be high:
#    - coverage_2px >= ~90%
