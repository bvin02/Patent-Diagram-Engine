"""
Sketch to Binary Image Converter

Converts hand-drawn pencil sketches (photos of paper drawings) into crisp
binary images with white paper and black lines, ready for SVG conversion.
"""

# python binarization.py example_good_sketch.png
# python binarization.py example_good_sketch.png --debug

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
    debug: bool = False,
) -> np.ndarray:
    """
    Convert a hand-drawn pencil sketch image to a crisp binary image.
    
    Args:
        input_path: Path to the input image (PNG, JPG, etc.)
        output_path: Path for the output binary image. If None, derived from input.
        block_size: Size of pixel neighborhood for adaptive thresholding.
        c_value: Constant subtracted from mean.
        denoise_strength: Strength of denoising (0 to disable).
        morph_kernel_size: Size for morphological operations.
        invert: If True, invert the final result.
        debug: If True, save intermediate step images.
    
    Returns:
        The processed binary image as a numpy array.
    """
    input_p = Path(input_path)
    debug_dir = input_p.parent / "debug_binarization"
    if debug:
        debug_dir.mkdir(exist_ok=True)
        print(f"Debug output directory: {debug_dir}")
    
    # Load the image
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")
    
    print(f"[1/5] Loaded image: {input_path} ({img.shape[1]}x{img.shape[0]})")
    if debug:
        cv2.imwrite(str(debug_dir / "01_input.png"), img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"[2/5] Converted to grayscale")
    if debug:
        cv2.imwrite(str(debug_dir / "02_grayscale.png"), gray)
    
    # Apply denoising
    if denoise_strength > 0:
        denoised = cv2.fastNlMeansDenoising(gray, h=denoise_strength)
        print(f"[3/5] Applied denoising (strength={denoise_strength})")
    else:
        denoised = gray
        print(f"[3/5] Skipped denoising")
    if debug:
        cv2.imwrite(str(debug_dir / "03_denoised.png"), denoised)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value
    )
    print(f"[4/5] Applied adaptive threshold (block={block_size}, C={c_value})")
    if debug:
        cv2.imwrite(str(debug_dir / "04_threshold.png"), binary)
    
    # Morphological cleanup
    if morph_kernel_size > 0:
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        print(f"[5/5] Applied morphological cleanup (kernel={morph_kernel_size})")
    else:
        print(f"[5/5] Skipped morphological cleanup")
    
    if debug:
        cv2.imwrite(str(debug_dir / "05_morphology.png"), binary)
    
    # Invert if requested
    if invert:
        binary = cv2.bitwise_not(binary)
        print("Inverted result")
        if debug:
            cv2.imwrite(str(debug_dir / "06_inverted.png"), binary)
    
    # Determine output path
    if output_path is None:
        output_path = str(input_p.parent / f"{input_p.stem}_binary.png")
    
    cv2.imwrite(output_path, binary)
    print(f"Saved binary image to: {output_path}")
    
    if debug:
        # Copy final to debug dir
        cv2.imwrite(str(debug_dir / "final_output.png"), binary)
        print(f"\nDebug images saved to: {debug_dir}/")
    
    return binary


def process_with_multiple_settings(
    input_path: str,
    output_dir: str = None,
) -> dict:
    """Process the image with multiple threshold settings for comparison."""
    input_p = Path(input_path)
    if output_dir is None:
        output_dir = input_p.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    settings = {
        "soft": {"block_size": 31, "c_value": 5, "denoise_strength": 5, "morph_kernel_size": 0},
        "medium": {"block_size": 21, "c_value": 10, "denoise_strength": 10, "morph_kernel_size": 1},
        "crisp": {"block_size": 15, "c_value": 15, "denoise_strength": 15, "morph_kernel_size": 1},
        "aggressive": {"block_size": 11, "c_value": 20, "denoise_strength": 20, "morph_kernel_size": 2},
    }
    
    outputs = {}
    for name, params in settings.items():
        output_path = str(output_dir / f"{input_p.stem}_{name}.png")
        print(f"\n--- Processing with '{name}' settings ---")
        sketch_to_binary(input_path, output_path, **params)
        outputs[name] = output_path
    
    return outputs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert hand-drawn sketches to crisp binary images"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="example_good_sketch.png",
        help="Input image path"
    )
    parser.add_argument("-o", "--output", help="Output image path")
    parser.add_argument("--compare", action="store_true", help="Generate multiple versions")
    parser.add_argument("--block-size", type=int, default=15, help="Block size (default: 15)")
    parser.add_argument("--c-value", type=int, default=20, help="C constant (default: 20)")
    parser.add_argument("--denoise", type=int, default=25, help="Denoise strength (default: 25)")
    parser.add_argument("--morph", type=int, default=1, help="Morph kernel size (default: 1)")
    parser.add_argument("--invert", action="store_true", help="Invert result")
    parser.add_argument("--debug", action="store_true", help="Save intermediate images to debug_binarization/")
    
    args = parser.parse_args()
    
    if args.compare:
        print(f"Processing {args.input} with multiple settings...")
        outputs = process_with_multiple_settings(args.input)
        print("\n=== Comparison complete ===")
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
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
