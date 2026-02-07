"""
Sketch to Binary Image Converter

Converts hand-drawn pencil sketches (photos of paper drawings) into crisp
binary images with white paper and black lines, ready for SVG conversion.
"""

# python binarization.py example_good_sketch.png

import cv2
import numpy as np
from pathlib import Path


def sketch_to_binary(
    input_path: str,
    output_path: str = None,
    block_size: int = 21,
    c_value: int = 10,
    denoise_strength: int = 10,
    morph_kernel_size: int = 1,
    invert: bool = False,
) -> np.ndarray:
    """
    Convert a hand-drawn pencil sketch image to a crisp binary image.
    
    This function processes photos of pencil drawings on paper to produce
    clean black lines on white background, suitable for SVG vectorization.
    
    Args:
        input_path: Path to the input image (PNG, JPG, etc.)
        output_path: Path for the output binary image. If None, derived from input.
        block_size: Size of pixel neighborhood for adaptive thresholding.
                   Must be odd. Larger = smoother, smaller = more detail.
        c_value: Constant subtracted from mean. Higher = more aggressive
                background removal (more white), lower = keep more gray lines.
        denoise_strength: Strength of denoising (0 to disable). Higher = smoother.
        morph_kernel_size: Size for morphological operations to clean up.
                          0 to disable, 1-2 for light cleanup.
        invert: If True, invert the final result (white lines on black).
    
    Returns:
        The processed binary image as a numpy array.
    """
    # Load the image
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")
    
    print(f"Loaded image: {input_path} ({img.shape[1]}x{img.shape[0]})")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply denoising to reduce paper texture and noise
    if denoise_strength > 0:
        gray = cv2.fastNlMeansDenoising(gray, h=denoise_strength)
        print(f"Applied denoising with strength {denoise_strength}")
    
    # Apply adaptive thresholding
    # This works well for varying lighting conditions common in photos
    # ADAPTIVE_THRESH_GAUSSIAN_C uses weighted sum of neighborhood values
    binary = cv2.adaptiveThreshold(
        gray,
        255,  # max value (white)
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,  # neighborhood size
        c_value  # constant subtracted from mean
    )
    print(f"Applied adaptive threshold (block={block_size}, C={c_value})")
    
    # Optional morphological operations to clean up the result
    if morph_kernel_size > 0:
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        
        # Close small gaps in lines
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise spots
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        print(f"Applied morphological cleanup (kernel={morph_kernel_size})")
    
    # Invert if requested (for white lines on black)
    if invert:
        binary = cv2.bitwise_not(binary)
        print("Inverted result")
    
    # Determine output path
    if output_path is None:
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.stem}_binary.png")
    
    # Save the result
    cv2.imwrite(output_path, binary)
    print(f"Saved binary image to: {output_path}")
    
    return binary


def process_with_multiple_settings(
    input_path: str,
    output_dir: str = None,
) -> dict:
    """
    Process the image with multiple threshold settings for comparison.
    
    Useful for finding the best parameters for your specific sketch style.
    
    Args:
        input_path: Path to the input image.
        output_dir: Directory for output images. If None, uses input directory.
    
    Returns:
        Dictionary mapping setting names to output paths.
    """
    input_p = Path(input_path)
    if output_dir is None:
        output_dir = input_p.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Different settings to try
    settings = {
        "soft": {
            "block_size": 31,
            "c_value": 5,
            "denoise_strength": 5,
            "morph_kernel_size": 0,
        },
        "medium": {
            "block_size": 21,
            "c_value": 10,
            "denoise_strength": 10,
            "morph_kernel_size": 1,
        },
        "crisp": {
            "block_size": 15,
            "c_value": 15,
            "denoise_strength": 15,
            "morph_kernel_size": 1,
        },
        "aggressive": {
            "block_size": 11,
            "c_value": 20,
            "denoise_strength": 20,
            "morph_kernel_size": 2,
        },
    }
    
    outputs = {}
    for name, params in settings.items():
        output_path = str(output_dir / f"{input_p.stem}_{name}.png")
        print(f"\n--- Processing with '{name}' settings ---")
        sketch_to_binary(input_path, output_path, **params)
        outputs[name] = output_path
    
    return outputs


def main():
    """Main entry point for processing the example sketch."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert hand-drawn sketches to crisp binary images"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="example_good_sketch.png",
        help="Input image path (default: example_good_sketch.png)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output image path (default: <input>_binary.png)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate multiple versions with different settings for comparison"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=15,
        help="Block size for adaptive threshold (odd number, default: 15)"
    )
    parser.add_argument(
        "--c-value",
        type=int,
        default=15,
        help="C constant for adaptive threshold (default: 15)"
    )
    parser.add_argument(
        "--denoise",
        type=int,
        default=15,
        help="Denoising strength, 0 to disable (default: 15)"
    )
    parser.add_argument(
        "--morph",
        type=int,
        default=1,
        help="Morphological kernel size, 0 to disable (default: 1)"
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert result (white lines on black)"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        print(f"Processing {args.input} with multiple settings for comparison...")
        outputs = process_with_multiple_settings(args.input)
        print("\n=== Comparison complete ===")
        print("Generated files:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")
    else:
        sketch_to_binary(
            args.input,
            args.output,
            block_size=args.block_size,
            c_value=args.c_value,
            denoise_strength=args.denoise,
            morph_kernel_size=args.morph,
            invert=args.invert,
        )


if __name__ == "__main__":
    main()
