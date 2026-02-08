"""
Stage 0: Initialize Run Directory

Creates a run-scoped output folder for the sketch-to-SVG pipeline.
This stage sets up the artifact structure before any processing begins.
"""

# python stage0_init_run.py example_good_sketch.png
# python stage0_init_run.py example_good_sketch.png --runs_root runs

import argparse
from pathlib import Path

from utils.artifacts import make_run_dir, StageArtifacts


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a run directory for the sketch-to-SVG pipeline"
    )
    parser.add_argument(
        "input_path",
        help="Path to the input sketch image"
    )
    parser.add_argument(
        "--runs_root",
        default="runs",
        help="Root directory for all runs (default: runs)"
    )
    
    args = parser.parse_args()
    
    # Verify input exists
    input_p = Path(args.input_path)
    if not input_p.exists():
        print(f"Error: Input file not found: {args.input_path}")
        return 1
    
    # Create run directory
    run_dir = make_run_dir(args.input_path, args.runs_root)
    
    # Print results
    print(f"Created run directory: {run_dir}")
    print(f"Input copied to: {run_dir / '00_input' / ('01_input' + input_p.suffix)}")
    
    # Note for future stages:
    # -----------------------
    # To use this harness in processing stages, do:
    #
    #   from utils.artifacts import make_run_dir, StageArtifacts
    #
    #   run_dir = make_run_dir("input.png")
    #   artifacts = StageArtifacts(run_dir, 10, "preprocess", debug=True)
    #
    #   # Save debug images (auto-numbered: 01_*, 02_*, etc.)
    #   artifacts.save_debug_image("grayscale", gray_img)
    #   artifacts.save_debug_image("threshold", binary_img)
    #
    #   # Save final outputs (no numbering)
    #   artifacts.save_output_image("result", final_img)
    #   artifacts.save_npy("contours", contour_array)
    #   artifacts.write_metrics({"lines": 42, "corners": 8})
    
    return 0


if __name__ == "__main__":
    exit(main())
