"""
IO Utilities for Sketch-to-SVG Pipeline

Minimal helpers for reading images and normalizing array types.
"""

import cv2
import numpy as np
from pathlib import Path


def read_image(path: str) -> np.ndarray:
    """
    Read an image from disk in BGR color format.
    
    Args:
        path: Path to the image file.
        
    Returns:
        Image as numpy array in BGR format.
        
    Raises:
        ValueError: If the image cannot be read.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert an image array to uint8 format for saving.
    
    Handles:
        - bool: maps False->0, True->255
        - float: normalizes min..max to 0..255
        - uint8: returns as-is
        - other int: clips to 0..255 and casts
    
    Args:
        img: Input image array.
        
    Returns:
        Image as uint8 numpy array.
    """
    if img.dtype == np.uint8:
        return img
    
    if img.dtype == bool:
        return (img.astype(np.uint8) * 255)
    
    if np.issubdtype(img.dtype, np.floating):
        # Handle constant arrays
        img_min = img.min()
        img_max = img.max()
        if img_max == img_min:
            # Constant array: if zero return zeros, else return 255s
            if img_max == 0:
                return np.zeros(img.shape, dtype=np.uint8)
            else:
                return np.full(img.shape, 255, dtype=np.uint8)
        # Normalize to 0..255
        normalized = (img - img_min) / (img_max - img_min)
        return (normalized * 255).astype(np.uint8)
    
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(img, 0, 255).astype(np.uint8)
    
    # Fallback: try direct cast
    return img.astype(np.uint8)
