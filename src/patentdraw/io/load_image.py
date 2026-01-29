"""
Image loading utilities for Patent Draw.

Stage 0 stub: assumes inputs are already image files.
Future: will support PDF extraction and image preprocessing.
"""

import os

import cv2
import numpy as np

from patentdraw.tracer import get_tracer, trace


@trace(label="load_image")
def load_image(path):
    """
    Load an image from disk.
    
    Returns a tuple of (image, metadata) where:
    - image: RGB numpy array (H, W, 3)
    - metadata: dict with width, height, dpi_estimate, source_path
    
    Raises FileNotFoundError if path does not exist.
    Raises ValueError if image cannot be loaded.
    """
    tracer = get_tracer()
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    # Load with OpenCV (BGR format)
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {path}")
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    height, width = img_rgb.shape[:2]
    
    # Estimate DPI from image metadata if available
    dpi_estimate = _estimate_dpi(path, width, height)
    
    tracer.event(f"Loaded image: {width}x{height}, dpi={dpi_estimate}")
    
    metadata = {
        "width": width,
        "height": height,
        "dpi_estimate": dpi_estimate,
        "source_path": os.path.abspath(path),
    }
    
    return img_rgb, metadata


def _estimate_dpi(path, width, height):
    """
    Estimate DPI from image dimensions or metadata.
    
    Uses heuristics based on typical document sizes:
    - If width suggests 8.5" page at ~300 DPI -> 300
    - Otherwise default to 150
    """
    # For a typical 8.5x11 inch document at 300 DPI:
    # width ~= 2550, height ~= 3300
    # At 150 DPI: width ~= 1275, height ~= 1650
    
    if width >= 2000:
        return 300
    elif width >= 1200:
        return 200
    else:
        return 150


def load_images(paths):
    """
    Load multiple images from disk.
    
    Returns a list of (image, metadata) tuples.
    """
    results = []
    for path in paths:
        img, meta = load_image(path)
        results.append((img, meta))
    return results


def validate_image_inputs(paths):
    """
    Validate that all input paths exist and are readable images.
    
    Returns a list of error messages (empty if all valid).
    """
    errors = []
    
    for path in paths:
        if not os.path.exists(path):
            errors.append(f"File not found: {path}")
            continue
        
        ext = os.path.splitext(path)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"]:
            errors.append(f"Unsupported image format: {path}")
            continue
        
        # Try to read header only
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                errors.append(f"Cannot read image: {path}")
        except Exception as e:
            errors.append(f"Error reading {path}: {str(e)}")
    
    return errors
