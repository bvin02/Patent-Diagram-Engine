"""
Stage 2: Image binarization for Patent Draw.

Converts input RGB images to binary (black/white) for stroke extraction.
Optimized for A-tier inputs: clean scans or photos on white background.
"""

import cv2
import numpy as np

from patentdraw.io.save_artifacts import DebugArtifactWriter, create_binary_overlay
from patentdraw.tracer import get_tracer, trace


@trace(label="stage2_binarize")
def binarize(rgb_img, config, debug_writer=None):
    """
    Convert RGB image to binary.
    
    Returns a uint8 binary image with 0 for background (white) and 255 for strokes (black).
    
    The pipeline uses inverted binary internally (strokes=255) for morphology and skeletonization.
    """
    tracer = get_tracer()
    
    # Convert to grayscale
    with tracer.span("grayscale", module="stage2_binarize"):
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        if debug_writer:
            debug_writer.save_image(gray, "stage2", "01_gray.png")
    
    # Optional denoising
    with tracer.span("denoise", module="stage2_binarize"):
        if config.binarization.denoise_kernel > 0:
            denoised = cv2.medianBlur(gray, config.binarization.denoise_kernel)
        else:
            denoised = gray
    
    # Thresholding
    with tracer.span("threshold", module="stage2_binarize"):
        if config.binarization.method == "adaptive":
            binary = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                config.binarization.adaptive_block_size,
                config.binarization.adaptive_c,
            )
        else:
            # Otsu's method (default)
            _, binary = cv2.threshold(
                denoised,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
        
        tracer.event(f"Threshold method: {config.binarization.method}")
    
    # Morphological operations to clean up
    with tracer.span("morphology", module="stage2_binarize"):
        kernel_size = config.binarization.morph_kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        
        # Open to remove small noise
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Close to fill small gaps
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Calculate metrics
    total_pixels = cleaned.size
    black_pixels = np.sum(cleaned > 0)
    black_ratio = black_pixels / total_pixels
    
    tracer.event(f"Binary result: black_ratio={black_ratio:.3f}")
    
    # Save debug artifacts
    if debug_writer:
        debug_writer.save_image(rgb_img, "stage2", "00_input_preview.png")
        debug_writer.save_image(cleaned, "stage2", "02_binary.png")
        
        # Create overlay
        overlay = create_binary_overlay(rgb_img, cleaned, color=(255, 0, 0))
        debug_writer.save_image(overlay, "stage2", "03_overlay_binary_on_input.png")
        
        # Save metrics
        metrics = {
            "threshold_method": config.binarization.method,
            "denoise_kernel": config.binarization.denoise_kernel,
            "morph_kernel": config.binarization.morph_kernel,
            "black_pixel_ratio": round(black_ratio, 4),
            "image_width": rgb_img.shape[1],
            "image_height": rgb_img.shape[0],
        }
        debug_writer.save_json(metrics, "stage2", "stage2_metrics.json")
    
    return cleaned


def get_black_ratio(binary_img):
    """Calculate the ratio of black (stroke) pixels in a binary image."""
    return np.sum(binary_img > 0) / binary_img.size
