"""
Command-line interface for Patent Draw.

Provides commands for running the pipeline and applying operations.
"""

import argparse
import sys

from patentdraw.config import load_config, save_default_config
from patentdraw.tracer import configure_tracer, get_tracer


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Patent Draw: Convert hand-drawn sketches to editable SVG line art",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument(
        "--inputs", "-i",
        nargs="+",
        required=True,
        help="Input image files",
    )
    run_parser.add_argument(
        "--out", "-o",
        required=True,
        help="Output directory",
    )
    run_parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML configuration file",
    )
    run_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug artifact generation",
    )
    run_parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable runtime tracing",
    )
    run_parser.add_argument(
        "--trace-level",
        default="INFO",
        choices=["ERROR", "WARN", "INFO", "DEBUG"],
        help="Trace log level",
    )
    run_parser.add_argument(
        "--trace-file",
        default=None,
        help="Path to write trace logs",
    )
    run_parser.add_argument(
        "--trace-json",
        action="store_true",
        help="Enable JSON trace output",
    )
    
    # Ops command
    ops_parser = subparsers.add_parser("ops", help="Operations subcommands")
    ops_subparsers = ops_parser.add_subparsers(dest="ops_command")
    
    apply_parser = ops_subparsers.add_parser("apply", help="Apply operations to existing scene")
    apply_parser.add_argument(
        "--scene",
        required=True,
        help="Path to scene.json from previous run",
    )
    apply_parser.add_argument(
        "--ops",
        required=True,
        help="Path to operations JSON file",
    )
    apply_parser.add_argument(
        "--out", "-o",
        required=True,
        help="Output directory",
    )
    apply_parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML configuration file",
    )
    apply_parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable runtime tracing",
    )
    
    # Init config command
    init_parser = subparsers.add_parser("init-config", help="Create default config file")
    init_parser.add_argument(
        "--out", "-o",
        default="patentdraw_config.yaml",
        help="Output path for config file",
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Handle commands
    if args.command == "run":
        return handle_run(args)
    elif args.command == "ops":
        if args.ops_command == "apply":
            return handle_ops_apply(args)
        else:
            ops_parser.print_help()
            return 0
    elif args.command == "init-config":
        return handle_init_config(args)
    
    return 0


def handle_run(args):
    """Handle the run command."""
    # Configure tracing
    configure_tracer(
        enabled=args.trace,
        level=args.trace_level,
        file_path=args.trace_file,
        json_output=args.trace_json,
    )
    
    tracer = get_tracer()
    
    try:
        from patentdraw.pipeline import run_pipeline
        
        with tracer.span("cli_run", module="cli"):
            document = run_pipeline(
                input_paths=args.inputs,
                out_dir=args.out,
                config_path=args.config,
                debug=args.debug,
            )
        
        # Print summary
        print(f"\nPipeline completed successfully.")
        print(f"  Views processed: {len(document.views)}")
        print(f"  Components detected: {len(document.component_registry)}")
        print(f"  Labels created: {len(document.label_registry)}")
        print(f"  Validation errors: {document.validation.error_count}")
        print(f"  Validation warnings: {document.validation.warning_count}")
        print(f"\nOutputs saved to: {args.out}/")
        print(f"  - final.svg")
        print(f"  - final.pdf")
        print(f"  - scene.json")
        print(f"  - validation_report.json")
        
        if document.validation.has_errors:
            print(f"\n[!] Validation errors detected. Review validation_report.json")
            return 1
        
        return 0
        
    except Exception as e:
        tracer.event(f"Pipeline failed: {str(e)}", level="ERROR")
        print(f"\nError: {str(e)}", file=sys.stderr)
        return 1


def handle_ops_apply(args):
    """Handle the ops apply command."""
    # Configure tracing
    configure_tracer(enabled=args.trace, level="INFO")
    
    tracer = get_tracer()
    
    try:
        from patentdraw.pipeline import apply_ops_pipeline
        
        with tracer.span("cli_ops_apply", module="cli"):
            document = apply_ops_pipeline(
                scene_path=args.scene,
                ops_path=args.ops,
                out_dir=args.out,
                config_path=args.config,
            )
        
        print(f"\nOperations applied successfully.")
        print(f"  Components: {len(document.component_registry)}")
        print(f"\nOutputs saved to: {args.out}/")
        
        return 0
        
    except Exception as e:
        tracer.event(f"Operations failed: {str(e)}", level="ERROR")
        print(f"\nError: {str(e)}", file=sys.stderr)
        return 1


def handle_init_config(args):
    """Handle the init-config command."""
    save_default_config(args.out)
    print(f"Default configuration saved to: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
