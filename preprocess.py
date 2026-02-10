"""
Stage 1: Preprocessing for Sketch-to-SVG Pipeline

Converts a photo of a pencil mechanical line diagram into a clean binary
stroke mask. Handles shadows, uneven lighting, and paper texture through
illumination flattening before adaptive thresholding.

Output: binary mask with strokes=255, background=0
"""

# python preprocess.py examples/clean.png --debug
# python preprocess.py examples/detailed.png --debug
# python preprocess.py examples/lowres.jpg --debug
# python preprocess.py examples/clean.png --runs_root runs --no_debug

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
    # Illumination flattening
    "flatten_enabled": True,         # set False if input is already clean
    "bg_blur_ksize": 51,             # kernel size for Gaussian blur (odd, large)
    
    # Pre-denoise: edge-preserving filter BEFORE illumination flattening
    # This preserves stroke edges during the division step
    "pre_denoise_enabled": True,     # apply bilateral before illumination flattening
    "pre_denoise_d": 5,              # diameter for pre-flatten bilateral
    "pre_denoise_sigma_color": 50,   # sigma color for pre-flatten bilateral
    "pre_denoise_sigma_space": 50,   # sigma space for pre-flatten bilateral
    
    # Post-denoise: filter AFTER illumination flattening (before threshold)
    "denoise_method": "bilateral",   # "bilateral", "gaussian", or "none"
    "bilateral_d": 5,                # diameter for bilateral filter
    "bilateral_sigma_color": 50,     # sigma color for bilateral
    "bilateral_sigma_space": 50,     # sigma space for bilateral
    "gaussian_ksize": 3,             # kernel size for gaussian denoise (odd)
    
    # Thresholding
    "threshold_method": "adaptive_gaussian",  # "adaptive_gaussian" or "otsu"
    "adaptive_block_size": 21,       # block size for adaptive threshold (odd)
    "adaptive_c": 8,                 # constant subtracted from mean
    
    # Cleanup
    "min_component_area": 30,        # minimum connected component area to keep
    "close_ksize": 3,                # morphological close kernel size (odd)
    "open_ksize": 0,                 # morphological open kernel size (0 to disable)
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def validate_odd(value: int, name: str, minimum: int = 3) -> int:
    """Ensure kernel size is odd and at least minimum."""
    if value < minimum:
        value = minimum
    if value % 2 == 0:
        value += 1
    return value


def load_config(config_path: str = None) -> dict:
    """Load config from JSON and merge with defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                user_config = json.load(f)
            config.update(user_config)
    
    # Validate kernel sizes are odd
    config["bg_blur_ksize"] = validate_odd(config["bg_blur_ksize"], "bg_blur_ksize")
    config["gaussian_ksize"] = validate_odd(config["gaussian_ksize"], "gaussian_ksize")
    config["adaptive_block_size"] = validate_odd(config["adaptive_block_size"], "adaptive_block_size")
    config["close_ksize"] = validate_odd(config["close_ksize"], "close_ksize")
    if config["open_ksize"] > 0:
        config["open_ksize"] = validate_odd(config["open_ksize"], "open_ksize")
    
    return config


def flatten_illumination(gray: np.ndarray, config: dict) -> np.ndarray:
    """
    Flatten illumination by dividing by a blurred background estimate.
    
    This handles uneven lighting from photos of paper sketches.
    For already-clean digital images, this is essentially a no-op.
    
    Args:
        gray: Grayscale input image.
        config: Configuration dictionary.
        
    Returns:
        Illumination-flattened grayscale image.
    """
    if not config.get("flatten_enabled", True):
        return gray
    
    ksize = config["bg_blur_ksize"]
    
    # Estimate background with heavy Gaussian blur
    bg = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    
    # Avoid division by zero
    bg_safe = np.maximum(bg, 1).astype(np.float32)
    gray_f = gray.astype(np.float32)
    
    # Divide gray by background and rescale
    # This normalizes out large-scale illumination variations
    flat = (gray_f / bg_safe) * 128.0
    
    # Clip and convert to uint8
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    
    return flat


def denoise_image(img: np.ndarray, config: dict) -> np.ndarray:
    """
    Apply denoising to reduce texture and noise.
    
    Args:
        img: Input grayscale image.
        config: Configuration dictionary.
        
    Returns:
        Denoised grayscale image.
    """
    method = config["denoise_method"]
    
    if method == "bilateral":
        d = config["bilateral_d"]
        sigma_c = config["bilateral_sigma_color"]
        sigma_s = config["bilateral_sigma_space"]
        denoised = cv2.bilateralFilter(img, d, sigma_c, sigma_s)
    elif method == "gaussian":
        ksize = config["gaussian_ksize"]
        denoised = cv2.GaussianBlur(img, (ksize, ksize), 0)
    else:  # none
        denoised = img
    
    return denoised


def threshold_image(img: np.ndarray, config: dict) -> np.ndarray:
    """
    Apply thresholding to create binary mask.
    
    Pencil strokes are dark, so we use THRESH_BINARY_INV to make strokes white.
    
    Args:
        img: Denoised grayscale image.
        config: Configuration dictionary.
        
    Returns:
        Binary mask with strokes=255, background=0.
    """
    method = config["threshold_method"]
    
    if method == "adaptive_gaussian":
        block_size = config["adaptive_block_size"]
        c = config["adaptive_c"]
        binary = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, c
        )
    else:  # otsu
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return binary


def count_components(mask: np.ndarray) -> int:
    """Count number of connected components in binary mask."""
    num_labels, _ = cv2.connectedComponents(mask)
    return num_labels - 1  # subtract background


def filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove connected components smaller than min_area.
    
    Args:
        mask: Binary mask (strokes=255).
        min_area: Minimum component area to keep.
        
    Returns:
        Filtered binary mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    result = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            result[labels == i] = 255
    
    return result


def cleanup_mask(mask: np.ndarray, config: dict) -> np.ndarray:
    """
    Clean up binary mask with morphological operations.
    
    Args:
        mask: Binary mask from thresholding.
        config: Configuration dictionary.
        
    Returns:
        Cleaned binary mask.
    """
    result = mask.copy()
    
    # Morphological close to fill small holes in strokes
    close_ksize = config["close_ksize"]
    if close_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    # Morphological open to remove small isolated noise (optional)
    open_ksize = config["open_ksize"]
    if open_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    
    # Filter small connected components
    min_area = config["min_component_area"]
    if min_area > 0:
        result = filter_small_components(result, min_area)
    
    return result


def create_overlay(original_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create overlay showing mask on original image.
    
    Draws strokes in green semi-transparent overlay for visual inspection.
    
    Args:
        original_bgr: Original color image (BGR).
        mask: Binary mask (strokes=255).
        
    Returns:
        Overlay image (BGR).
    """
    overlay = original_bgr.copy()
    
    # Create green overlay for stroke areas
    green_overlay = np.zeros_like(original_bgr)
    green_overlay[:, :, 1] = mask  # green channel
    
    # Blend with original
    overlay = cv2.addWeighted(overlay, 0.7, green_overlay, 0.3, 0)
    
    # Also draw contours for edge visibility
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)  # red edges
    
    return overlay


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------


def infer_run_dir(input_path: str, runs_root: str) -> Path:
    """
    Infer run directory from input path.
    
    If input_path is already under runs/<run>/..., reuse that run directory.
    Otherwise create a new run directory.
    """
    input_p = Path(input_path).resolve()
    runs_root_p = Path(runs_root).resolve()
    
    try:
        rel_path = input_p.relative_to(runs_root_p)
        run_name = rel_path.parts[0]
        return runs_root_p / run_name
    except ValueError:
        return make_run_dir(input_path, runs_root)


def preprocess(
    input_path: str,
    runs_root: str = "runs",
    debug: bool = True,
    config_path: str = None,
    save_overlay: bool = True,
) -> tuple:
    """
    Preprocess a sketch image to create a binary stroke mask.
    
    Args:
        input_path: Path to input image.
        runs_root: Root directory for runs.
        debug: Whether to save debug images.
        config_path: Optional path to config JSON.
        save_overlay: Whether to save overlay image.
        
    Returns:
        Tuple of (run_dir, output_mask_path, metrics).
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create run directory and artifacts manager
    run_dir = infer_run_dir(input_path, runs_root)
    artifacts = StageArtifacts(run_dir, 10, "preprocess", debug=debug)
    
    # Read input image
    input_bgr = read_image(input_path)
    h, w = input_bgr.shape[:2]
    
    # 1) Save input color for reference
    artifacts.save_debug_image("input_color", input_bgr)
    
    # 2) Convert to grayscale
    gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
    artifacts.save_debug_image("gray", gray)
    
    # 3) Pre-denoise: edge-preserving bilateral BEFORE illumination flattening
    # This preserves stroke edges during the division step and reduces paper texture
    if config.get("pre_denoise_enabled", True):
        pre_d = config.get("pre_denoise_d", 5)
        pre_sigma_c = config.get("pre_denoise_sigma_color", 50)
        pre_sigma_s = config.get("pre_denoise_sigma_space", 50)
        gray_predenoise = cv2.bilateralFilter(gray, pre_d, pre_sigma_c, pre_sigma_s)
        artifacts.save_debug_image("pre_denoise", gray_predenoise)
    else:
        gray_predenoise = gray
    
    # 4) Illumination flattening (saves debug even if disabled)
    bg = cv2.GaussianBlur(gray_predenoise, (config["bg_blur_ksize"], config["bg_blur_ksize"]), 0)
    artifacts.save_debug_image("illumination_bg", bg)
    
    gray_flat = flatten_illumination(gray_predenoise, config)
    artifacts.save_debug_image("illumination_flat", gray_flat)
    
    # 5) Post-denoise (after flattening, before threshold)
    denoised = denoise_image(gray_flat, config)
    artifacts.save_debug_image("denoise", denoised)
    
    # 6) Threshold
    threshold_raw = threshold_image(denoised, config)
    artifacts.save_debug_image("threshold_raw", threshold_raw)
    
    # Count components before cleanup
    num_components_before = count_components(threshold_raw)
    
    # 6) Cleanup
    mask_clean = cleanup_mask(threshold_raw, config)
    artifacts.save_debug_image("mask_clean", mask_clean)
    
    # Count components after cleanup
    num_components_after = count_components(mask_clean)
    
    # Verify mask orientation (strokes should be white, minority of pixels)
    foreground_ratio = np.sum(mask_clean > 0) / mask_clean.size
    
    # Save inverted check for visual verification
    artifacts.save_debug_image("mask_inverted_check", mask_clean)
    
    # 9) Optional overlay
    if save_overlay and debug:
        overlay = create_overlay(input_bgr, mask_clean)
        artifacts.save_debug_image("overlay_on_input", overlay)
    
    # Save final output mask
    output_mask_path = artifacts.save_output_image("output_mask", mask_clean)
    
    # Compute and save metrics
    metrics = {
        "input_path": str(input_path),
        "image_w": w,
        "image_h": h,
        "foreground_ratio": float(foreground_ratio),
        "num_components_before_filter": num_components_before,
        "num_components_after_filter": num_components_after,
        "min_component_area_used": config["min_component_area"],
        "threshold_method_used": config["threshold_method"],
        "config_used": config,
    }
    artifacts.write_metrics(metrics)
    
    return run_dir, output_mask_path, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess sketch image to binary stroke mask"
    )
    parser.add_argument(
        "input_path",
        help="Path to input sketch image"
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
        "--save_overlay",
        action="store_true",
        dest="save_overlay",
        default=True,
        help="Save overlay image (default: True)"
    )
    parser.add_argument(
        "--no_overlay",
        action="store_false",
        dest="save_overlay",
        help="Disable overlay image saving"
    )
    
    args = parser.parse_args()
    
    # Verify input exists
    input_p = Path(args.input_path)
    if not input_p.exists():
        print(f"Error: Input file not found: {args.input_path}")
        return 1
    
    print(f"Processing: {args.input_path}")
    
    run_dir, output_path, metrics = preprocess(
        args.input_path,
        runs_root=args.runs_root,
        debug=args.debug,
        config_path=args.config,
        save_overlay=args.save_overlay,
    )
    
    print(f"Run directory: {run_dir}")
    print(f"Output mask: {output_path}")
    print(f"Image size: {metrics['image_w']}x{metrics['image_h']}")
    print(f"Foreground ratio: {metrics['foreground_ratio']:.4f}")
    print(f"Components: {metrics['num_components_before_filter']} -> {metrics['num_components_after_filter']}")
    
    return 0


if __name__ == "__main__":
    exit(main())


# ---------------------------------------------------------------------------
# Notes on Debug Artifacts
# ---------------------------------------------------------------------------
# Run
#    python preprocess.py examples/clean.png --debug
#    python preprocess.py examples/detailed.png --debug
#    python preprocess.py examples/lowres.jpg --debug
#
# Open runs/<run>/10_preprocess/debug/ folder
#    01_input_color.png - original input
#    02_gray.png - grayscale conversion
#    03_illumination_bg.png - estimated background
#    04_illumination_flat.png - flattened illumination
#    05_denoise.png - after denoising
#    06_threshold_raw.png - raw threshold result
#    07_mask_clean.png - after cleanup
#    08_mask_inverted_check.png - verify strokes are white
#    09_overlay_on_input.png - mask overlaid on original
#
# Check runs/<run>/10_preprocess/out/metrics.json
#    - foreground_ratio should be reasonable (typically 0.01 to 0.3)
#    - If near 0, strokes may not be detected
#    - If near 1, mask may be inverted or threshold is wrong
#
# Confirm runs/<run>/10_preprocess/out/output_mask.png
#    - Should have strokes as white (255) on black (0) background
#    - This mask is ready for distance transform in the next stage
