#!/usr/bin/env python3
"""
Run the full sketch-to-SVG pipeline end-to-end.

Usage:
    python run_pipeline.py examples/clean.png
    python run_pipeline.py examples/clean.png --no-debug
    python run_pipeline.py examples/clean.png --only-stages 1 2 3
    python run_pipeline.py examples/clean.png --from-stage 4
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

STAGES = [
    {
        "id": "0",
        "name": "Init Run",
        "script": "stage0_init_run.py",
        "build_args": lambda ctx: [ctx["input_image"]],
    },
    {
        "id": "1",
        "name": "Preprocess",
        "script": "preprocess.py",
        # Use input copy inside run dir so preprocess reuses the same run dir
        "build_args": lambda ctx: [ctx.get("run_input", ctx["input_image"])],
    },
    {
        "id": "2",
        "name": "Distance Transform",
        "script": "distance_transform.py",
        "build_args": lambda ctx: [
            str(ctx["run_dir"] / "10_preprocess" / "out" / "output_mask.png"),
        ],
    },
    {
        "id": "3",
        "name": "Ridge Extraction",
        "script": "ridge_extraction.py",
        "build_args": lambda ctx: [
            str(ctx["run_dir"] / "10_preprocess" / "out" / "output_mask.png"),
            str(ctx["run_dir"] / "20_distance_transform" / "out" / "dt.npy"),
        ],
    },
    {
        "id": "4",
        "name": "Graph Build",
        "script": "graph_build.py",
        "build_args": lambda ctx: [
            str(ctx["run_dir"] / "30_ridge" / "out" / "ridge.png"),
            "--mask", str(ctx["run_dir"] / "10_preprocess" / "out" / "output_mask.png"),
        ],
    },
    {
        "id": "5",
        "name": "Graph Cleanup",
        "script": "graph_cleanup.py",
        "build_args": lambda ctx: [
            str(ctx["run_dir"] / "40_graph_raw" / "out" / "graph_raw.json"),
            "--mask", str(ctx["run_dir"] / "10_preprocess" / "out" / "output_mask.png"),
            "--ridge", str(ctx["run_dir"] / "30_ridge" / "out" / "ridge.png"),
        ],
    },
    {
        "id": "6",
        "name": "Primitive Fitting",
        "script": "fit_primitives.py",
        "build_args": lambda ctx: [
            str(ctx["run_dir"] / "50_graph_clean" / "out" / "graph_clean.json"),
            "--mask", str(ctx["run_dir"] / "10_preprocess" / "out" / "output_mask.png"),
        ],
    },
    {
        "id": "7",
        "name": "SVG Emission",
        "script": "emit_svg.py",
        "build_args": lambda ctx: [
            str(ctx["run_dir"] / "60_fit" / "out" / "primitives.json"),
            "--mask", str(ctx["run_dir"] / "10_preprocess" / "out" / "output_mask.png"),
        ],
    },
    {
        "id": "8",
        "name": "Label Identify",
        "script": "label_identify.py",
        "build_args": lambda ctx: [
            str(ctx["run_dir"] / "10_preprocess" / "out" / "output_mask.png"),
            "--svg", str(ctx["run_dir"] / "70_svg" / "out" / "output.svg"),
        ],
    },
    {
        "id": "9",
        "name": "Label Leaders",
        "script": "label_leaders.py",
        "build_args": lambda ctx: [
            str(ctx["run_dir"] / "80_label_identify" / "out" / "components.json"),
            "--mask", str(ctx["run_dir"] / "10_preprocess" / "out" / "output_mask.png"),
        ],
    },
    {
        "id": "10",
        "name": "Label Place",
        "script": "label_place.py",
        "build_args": lambda ctx: [
            str(ctx["run_dir"] / "90_label_leaders" / "out" / "leaders.json"),
            "--svg", str(ctx["run_dir"] / "70_svg" / "out" / "output.svg"),
            "--mask", str(ctx["run_dir"] / "10_preprocess" / "out" / "output_mask.png"),
        ],
    },
]


def infer_run_dir(input_image: str, runs_root: str = "runs") -> Path:
    """Infer the run directory from the input image path."""
    input_p = Path(input_image).resolve()
    runs_root_p = Path(runs_root).resolve()

    # If already inside a run dir, use that
    try:
        rel = input_p.relative_to(runs_root_p)
        return runs_root_p / rel.parts[0]
    except ValueError:
        pass

    # Derive run name from input filename
    slug = input_p.stem.lower().replace(" ", "_")
    return runs_root_p / slug


def run_stage(stage: dict, ctx: dict, debug: bool) -> bool:
    """Run a single pipeline stage. Returns True on success."""
    script = Path(__file__).parent / stage["script"]

    if not script.exists():
        print(f"  ⚠  Script not found: {stage['script']} — skipping")
        return True  # Non-fatal; stage may be optional

    args = [sys.executable, str(script)] + stage["build_args"](ctx)

    # Add common flags (stage 0 has no debug flag)
    if stage["id"] != "0":
        args.append("--debug" if debug else "--no_debug")

    result = subprocess.run(args, capture_output=True, text=True,
                            cwd=str(Path(__file__).parent))
    # Print stdout/stderr passthrough
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    # After stage 0, discover the actual run directory it created
    if stage["id"] == "0" and result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith("Created run directory:"):
                actual_run_dir = Path(line.split(":", 1)[1].strip()).resolve()
                ctx["run_dir"] = actual_run_dir
                # Find the input copy for stage 1
                input_dir = actual_run_dir / "00_input"
                copies = list(input_dir.glob("01_input.*"))
                if copies:
                    ctx["run_input"] = str(copies[0])
                break

    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full sketch-to-SVG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py examples/clean.png
    python run_pipeline.py examples/clean.png --no-debug
    python run_pipeline.py examples/clean.png --from-stage 4
    python run_pipeline.py examples/clean.png --only-stages 1 2 3
        """,
    )

    parser.add_argument("input_image", type=str, help="Path to input sketch image")
    parser.add_argument("--runs-root", type=str, default="runs",
                        help="Root directory for run outputs (default: runs)")
    parser.add_argument("--debug", dest="debug", action="store_true", default=True,
                        help="Enable debug output (default)")
    parser.add_argument("--no-debug", dest="debug", action="store_false",
                        help="Disable debug output")
    parser.add_argument("--skip-stages", nargs="+", default=[],
                        help="Stage IDs to skip (e.g. 6.5)")
    parser.add_argument("--only-stages", nargs="+", default=None,
                        help="Run ONLY these stage IDs")
    parser.add_argument("--from-stage", type=str, default=None,
                        help="Start from this stage ID (skip earlier stages)")

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"Error: Input image not found: {args.input_image}")
        return 1

    # Build context
    run_dir = infer_run_dir(args.input_image, args.runs_root)
    ctx = {
        "input_image": str(input_path),
        "run_dir": run_dir,
        "runs_root": args.runs_root,
    }

    # Determine which stages to run
    skip_ids = set(args.skip_stages)
    only_ids = set(args.only_stages) if args.only_stages else None
    from_stage = args.from_stage

    stages_to_run = []
    past_from = from_stage is None  # If no --from-stage, start immediately

    for stage in STAGES:
        sid = stage["id"]

        if not past_from:
            if sid == from_stage:
                past_from = True
            else:
                continue

        if sid in skip_ids:
            continue
        if only_ids is not None and sid not in only_ids:
            continue

        stages_to_run.append(stage)

    if not stages_to_run:
        print("No stages selected to run.")
        return 1

    # Run pipeline
    print(f"{'='*60}")
    print(f"  Patent Diagram Pipeline")
    print(f"  Input:    {args.input_image}")
    print(f"  Run dir:  {run_dir}")
    print(f"  Stages:   {', '.join(s['id'] for s in stages_to_run)}")
    print(f"  Debug:    {'on' if args.debug else 'off'}")
    print(f"{'='*60}")

    total_start = time.time()
    failed = []

    for i, stage in enumerate(stages_to_run, 1):
        header = f"[{i}/{len(stages_to_run)}] Stage {stage['id']}: {stage['name']}"
        print(f"\n{'─'*60}")
        print(f"  {header}")
        print(f"{'─'*60}")

        t0 = time.time()
        success = run_stage(stage, ctx, args.debug)
        elapsed = time.time() - t0

        if success:
            print(f"  ✓  {stage['name']} completed in {elapsed:.1f}s")
        else:
            print(f"  ✗  {stage['name']} FAILED after {elapsed:.1f}s")
            failed.append(stage)
            # Stop on failure
            print(f"\nPipeline stopped due to failure in stage {stage['id']}.")
            break

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    if not failed:
        print(f"  ✓  Pipeline completed successfully in {total_elapsed:.1f}s")
        # Show labelled SVG if available, otherwise plain SVG
        labelled_path = run_dir / "100_label_place" / "out" / "labelled.svg"
        svg_path = run_dir / "70_svg" / "out" / "output.svg"
        if labelled_path.exists():
            print(f"  Output:  {labelled_path}")
        elif svg_path.exists():
            print(f"  Output:  {svg_path}")
    else:
        print(f"  ✗  Pipeline failed at stage {failed[0]['id']}: {failed[0]['name']}")
    print(f"{'='*60}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
