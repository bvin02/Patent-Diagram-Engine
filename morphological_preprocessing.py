"""
Morphological Preprocessing Module

Fixes small pixel gaps in binary line art before further processing.
Bridges discontinuities while preserving line structure.
"""

# python morphological_preprocessing.py example_good_sketch_binary.png --directional
# python morphological_preprocessing.py example_good_sketch_binary.png --directional --debug

import cv2
import numpy as np
from pathlib import Path


def fix_line_gaps(
    input_path: str,
    output_path: str = None,
    gap_size: int = 3,
    iterations: int = 1,
    debug: bool = False,
) -> np.ndarray:
    """
    Fix small gaps in binary line art using morphological closing.
    
    Args:
        input_path: Path to binary image (black lines on white)
        output_path: Path for output. If None, derived from input.
        gap_size: Maximum gap size to bridge (kernel size)
        iterations: Number of close operations to apply
        debug: If True, save intermediate images
    
    Returns:
        Processed binary image
    """
    input_p = Path(input_path)
    debug_dir = input_p.parent / "debug_morphological_preprocessing"
    if debug:
        debug_dir.mkdir(exist_ok=True)
        print(f"Debug output directory: {debug_dir}")
    
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")
    
    print(f"[1/4] Loaded image: {input_path} ({img.shape[1]}x{img.shape[0]})")
    if debug:
        cv2.imwrite(str(debug_dir / "01_input.png"), img)
    
    # Invert for processing
    inverted = cv2.bitwise_not(img)
    print(f"[2/4] Inverted (white lines on black)")
    if debug:
        cv2.imwrite(str(debug_dir / "02_inverted.png"), inverted)
    
    # Create kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_size, gap_size))
    
    # Apply closing
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    print(f"[3/4] Applied closing (kernel={gap_size}, iterations={iterations})")
    if debug:
        cv2.imwrite(str(debug_dir / "03_closed.png"), closed)
    
    # Invert back
    result = cv2.bitwise_not(closed)
    print(f"[4/4] Inverted back (black lines on white)")
    if debug:
        cv2.imwrite(str(debug_dir / "04_result.png"), result)
    
    if output_path is None:
        output_path = str(input_p.parent / f"{input_p.stem}_preprocessed.png")
    
    cv2.imwrite(output_path, result)
    print(f"Saved preprocessed image to: {output_path}")
    
    if debug:
        cv2.imwrite(str(debug_dir / "final_output.png"), result)
        print(f"\nDebug images saved to: {debug_dir}/")
    
    return result


def fix_line_gaps_directional(
    input_path: str,
    output_path: str = None,
    gap_size: int = 2,
    debug: bool = False,
) -> np.ndarray:
    """
    Fix gaps using directional kernels for better line preservation.
    
    Args:
        input_path: Path to binary image
        output_path: Path for output
        gap_size: Maximum gap to bridge
        debug: If True, save intermediate images
    
    Returns:
        Processed binary image
    """
    input_p = Path(input_path)
    debug_dir = input_p.parent / "debug_morphological_preprocessing"
    if debug:
        debug_dir.mkdir(exist_ok=True)
        print(f"Debug output directory: {debug_dir}")
    
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")
    
    print(f"[1/6] Loaded image: {input_path} ({img.shape[1]}x{img.shape[0]})")
    if debug:
        cv2.imwrite(str(debug_dir / "01_input.png"), img)
    
    inverted = cv2.bitwise_not(img)
    print(f"[2/6] Inverted (white lines on black)")
    if debug:
        cv2.imwrite(str(debug_dir / "02_inverted.png"), inverted)
    
    # Directional kernels
    kernels = {
        "horizontal": cv2.getStructuringElement(cv2.MORPH_RECT, (gap_size, 1)),
        "vertical": cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap_size)),
        "diag1": np.eye(gap_size, dtype=np.uint8),
        "diag2": np.fliplr(np.eye(gap_size, dtype=np.uint8)),
    }
    
    result = inverted.copy()
    step = 3
    for name, kernel in kernels.items():
        closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
        result = cv2.bitwise_or(result, closed)
        print(f"[{step}/6] Applied {name} closing")
        if debug:
            cv2.imwrite(str(debug_dir / f"0{step}_{name}.png"), cv2.bitwise_not(result))
        step += 1
    
    result = cv2.bitwise_not(result)
    
    if output_path is None:
        output_path = str(input_p.parent / f"{input_p.stem}_preprocessed.png")
    
    cv2.imwrite(output_path, result)
    print(f"Saved preprocessed image to: {output_path}")
    
    if debug:
        cv2.imwrite(str(debug_dir / "final_output.png"), result)
        print(f"\nDebug images saved to: {debug_dir}/")
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix pixel gaps in binary line art")
    parser.add_argument("input", nargs="?", default="example_good_sketch_binary.png", help="Input binary image")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--gap-size", type=int, default=3, help="Maximum gap size (default: 3)")
    parser.add_argument("--iterations", type=int, default=1, help="Closing iterations (default: 1)")
    parser.add_argument("--directional", action="store_true", help="Use directional kernels")
    parser.add_argument("--debug", action="store_true", help="Save intermediate images")
    
    args = parser.parse_args()
    
    if args.directional:
        fix_line_gaps_directional(
            args.input,
            args.output,
            gap_size=max(1, args.gap_size - 1),
            debug=args.debug,
        )
    else:
        fix_line_gaps(
            args.input,
            args.output,
            gap_size=args.gap_size,
            iterations=args.iterations,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
