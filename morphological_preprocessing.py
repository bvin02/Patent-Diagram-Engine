"""
Morphological Preprocessing Module

Fixes small pixel gaps in binary line art before further processing.
Bridges discontinuities while preserving line structure.
"""

# python morphological_preprocessing.py example_good_sketch_binary.png --directional

import cv2
import numpy as np
from pathlib import Path


def fix_line_gaps(
    input_path: str,
    output_path: str = None,
    gap_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Fix small gaps in binary line art using morphological closing.
    
    Closing = dilate then erode. This bridges small gaps without
    significantly changing line thickness.
    
    Args:
        input_path: Path to binary image (black lines on white)
        output_path: Path for output. If None, derived from input.
        gap_size: Maximum gap size to bridge (kernel size)
        iterations: Number of close operations to apply
    
    Returns:
        Processed binary image
    """
    # Load binary image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")
    
    print(f"Loaded image: {input_path} ({img.shape[1]}x{img.shape[0]})")
    
    # Invert: morphological ops work on white foreground
    # Binary has black lines (0) on white (255), we need opposite
    inverted = cv2.bitwise_not(img)
    
    # Create kernel for closing
    # Ellipse kernel works well for line art (connects in all directions)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (gap_size, gap_size)
    )
    
    # Apply morphological closing (dilate then erode)
    # This bridges gaps smaller than kernel size
    closed = cv2.morphologyEx(
        inverted, 
        cv2.MORPH_CLOSE, 
        kernel, 
        iterations=iterations
    )
    print(f"Applied closing (kernel={gap_size}, iterations={iterations})")
    
    # Invert back to black lines on white
    result = cv2.bitwise_not(closed)
    
    # Determine output path
    if output_path is None:
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.stem}_preprocessed.png")
    
    cv2.imwrite(output_path, result)
    print(f"Saved preprocessed image to: {output_path}")
    
    return result


def fix_line_gaps_directional(
    input_path: str,
    output_path: str = None,
    gap_size: int = 2,
) -> np.ndarray:
    """
    Fix gaps using directional kernels for better line preservation.
    
    Applies closing with horizontal, vertical, and diagonal kernels
    separately, then combines. Better for technical drawings.
    
    Args:
        input_path: Path to binary image
        output_path: Path for output
        gap_size: Maximum gap to bridge
    
    Returns:
        Processed binary image
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")
    
    print(f"Loaded image: {input_path} ({img.shape[1]}x{img.shape[0]})")
    
    inverted = cv2.bitwise_not(img)
    
    # Directional kernels
    kernels = {
        "horizontal": cv2.getStructuringElement(cv2.MORPH_RECT, (gap_size, 1)),
        "vertical": cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap_size)),
        "diag1": np.eye(gap_size, dtype=np.uint8),
        "diag2": np.fliplr(np.eye(gap_size, dtype=np.uint8)),
    }
    
    # Apply closing with each kernel and combine
    result = inverted.copy()
    for name, kernel in kernels.items():
        closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
        result = cv2.bitwise_or(result, closed)
        print(f"Applied {name} closing")
    
    # Invert back
    result = cv2.bitwise_not(result)
    
    if output_path is None:
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.stem}_preprocessed.png")
    
    cv2.imwrite(output_path, result)
    print(f"Saved preprocessed image to: {output_path}")
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix pixel gaps in binary line art"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="example_good_sketch_binary.png",
        help="Input binary image"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path"
    )
    parser.add_argument(
        "--gap-size",
        type=int,
        default=3,
        help="Maximum gap size to bridge in pixels (default: 3)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of closing iterations (default: 1)"
    )
    parser.add_argument(
        "--directional",
        action="store_true",
        help="Use directional kernels (better for technical drawings)"
    )
    
    args = parser.parse_args()
    
    if args.directional:
        fix_line_gaps_directional(
            args.input,
            args.output,
            gap_size=max(1,args.gap_size-1),
        )
    else:
        fix_line_gaps(
            args.input,
            args.output,
            gap_size=args.gap_size,
            iterations=args.iterations,
        )


if __name__ == "__main__":
    main()
