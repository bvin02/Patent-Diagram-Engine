"""
Main pipeline orchestrator for Patent Draw.

Runs stages 2-8 sequentially and manages the document scene graph.
"""

import json
import os
from datetime import datetime

from patentdraw.components.grouping import group_strokes_baseline, create_component_overlay
from patentdraw.components.operations import apply_operations
from patentdraw.components.proposals_ml import get_ml_proposals
from patentdraw.config import PipelineConfig, load_config
from patentdraw.export.pdf_layout import generate_pdf
from patentdraw.export.svg_package import generate_svg_package
from patentdraw.io.load_image import load_image, validate_image_inputs
from patentdraw.io.save_artifacts import DebugArtifactWriter, ensure_dir, save_json
from patentdraw.labels.label_propose import propose_labels
from patentdraw.labels.leader_route import route_leaders
from patentdraw.labels.numbering import assign_numbering
from patentdraw.models import (
    Document, InputMeta, View, ImageMeta, ViewLabel,
    generate_doc_id, generate_view_id,
)
from patentdraw.preprocess.stage2_binarize import binarize
from patentdraw.strokes.bezier_fit import fit_bezier_to_polylines
from patentdraw.strokes.polyline_trace import trace_polylines, merge_short_segments
from patentdraw.strokes.simplify import simplify_polylines
from patentdraw.strokes.skeleton_graph import build_skeleton_graph
from patentdraw.strokes.svg_emit import (
    create_strokes_from_beziers, emit_strokes_svg_with_ids, create_debug_bbox_overlay
)
from patentdraw.tracer import get_tracer, trace, configure_tracer
from patentdraw.validate.report import generate_report
from patentdraw.validate.rules import run_validation


@trace(label="run_pipeline")
def run_pipeline(input_paths, out_dir, config=None, config_path=None, debug=False):
    """
    Run the full pipeline from Stage 2 to Stage 8.
    
    Args:
        input_paths: list of input image file paths
        out_dir: output directory
        config: PipelineConfig object (optional)
        config_path: path to YAML config file (optional)
        debug: enable debug artifact generation
    
    Returns:
        Document object with all results
    """
    tracer = get_tracer()
    
    # Load configuration
    if config is None:
        config = load_config(config_path)
    
    # Configure debug
    config.debug.enabled = debug
    
    # Validate inputs
    errors = validate_image_inputs(input_paths)
    if errors:
        for error in errors:
            tracer.event(error, level="ERROR")
        raise ValueError(f"Input validation failed: {errors}")
    
    # Create output directory
    ensure_dir(out_dir)
    
    # Create document
    doc_id = generate_doc_id(input_paths)
    document = Document(
        doc_id=doc_id,
        created_at=datetime.now().isoformat(),
    )
    
    # Process each input image
    for idx, input_path in enumerate(input_paths):
        with tracer.span(f"process_view_{idx}", module="pipeline"):
            view = process_single_image(
                input_path, idx, out_dir, config, document
            )
            document.views.append(view)
            
            # Add input metadata
            document.inputs.append(InputMeta(
                input_id=f"input_{idx}",
                source_path=input_path,
                view_label=view.view_label,
            ))
    
    # Stage 5: Component grouping
    with tracer.span("stage5_grouping", module="pipeline"):
        for view in document.views:
            debug_writer = DebugArtifactWriter(
                out_dir, view.view_id, 
                enabled=config.debug.enabled,
                max_edge=config.debug.max_edge_scale,
            ) if config.debug.enabled else None
            
            components = group_strokes_baseline(view.strokes, config, debug_writer)
            
            for comp in components:
                document.component_registry[comp.component_id] = comp
                view.component_ids.append(comp.component_id)
            
            # Optional ML proposals
            if config.ml_enabled:
                ml_proposals = get_ml_proposals(None, view.strokes, config)
                for comp in ml_proposals:
                    if comp.component_id not in document.component_registry:
                        document.component_registry[comp.component_id] = comp
            
            # Save component overlay
            if debug_writer:
                orig_img, _ = load_image(document.inputs[0].source_path)
                overlay = create_component_overlay(components, view.strokes, orig_img)
                debug_writer.save_image(overlay, "stage5", "01_components_overlay.png")
    
    # Stage 6: Label proposals
    with tracer.span("stage6_labels", module="pipeline"):
        all_strokes = []
        for view in document.views:
            all_strokes.extend(view.strokes)
        
        for view in document.views:
            view_components = [
                document.component_registry[cid] 
                for cid in view.component_ids 
                if cid in document.component_registry
            ]
            
            labels = propose_labels(
                view_components, view.strokes,
                view.image_meta.width, view.image_meta.height,
                config,
            )
            
            labels, crossing_stats = route_leaders(labels, view.strokes, config)
            
            for label in labels:
                document.label_registry[label.label_id] = label
                view.label_ids.append(label.label_id)
            
            # Debug overlay
            if config.debug.enabled:
                debug_writer = DebugArtifactWriter(
                    out_dir, view.view_id,
                    enabled=True,
                    max_edge=config.debug.max_edge_scale,
                )
                debug_writer.save_json(crossing_stats, "stage6", "stage6_metrics.json")
    
    # Stage 7: Numbering
    with tracer.span("stage7_numbering", module="pipeline"):
        document = assign_numbering(document, config)
        
        # Debug output
        if config.debug.enabled:
            for view in document.views:
                debug_writer = DebugArtifactWriter(
                    out_dir, view.view_id,
                    enabled=True,
                    max_edge=config.debug.max_edge_scale,
                )
                debug_writer.save_json(
                    dict(document.numbering_registry), 
                    "stage7", 
                    "numbering_registry.json"
                )
    
    # Stage 8: Validation and export
    with tracer.span("stage8_validate_export", module="pipeline"):
        # Validation
        document.validation = run_validation(document, config)
        
        # Generate SVG package
        generate_svg_package(document, out_dir, config)
        
        # Generate PDF
        generate_pdf(document, out_dir, config)
        
        # Generate reports
        debug_writer = DebugArtifactWriter(
            out_dir, "global",
            enabled=config.debug.enabled,
            max_edge=config.debug.max_edge_scale,
        ) if config.debug.enabled else None
        
        generate_report(document, out_dir, debug_writer)
    
    # Save scene graph
    scene_path = os.path.join(out_dir, "scene.json")
    save_json(document.model_dump(), scene_path)
    
    tracer.event(f"Pipeline complete: {len(document.views)} views, {len(document.component_registry)} components")
    
    return document


def process_single_image(input_path, idx, out_dir, config, document):
    """
    Process a single input image through Stages 2-4.
    
    Returns a View object with strokes.
    """
    tracer = get_tracer()
    
    # Load image
    rgb_img, metadata = load_image(input_path)
    
    view_id = generate_view_id(input_path, idx)
    
    debug_writer = DebugArtifactWriter(
        out_dir, view_id,
        enabled=config.debug.enabled,
        max_edge=config.debug.max_edge_scale,
    ) if config.debug.enabled else None
    
    # Stage 2: Binarization
    with tracer.span("stage2_binarize", module="pipeline"):
        binary = binarize(rgb_img, config, debug_writer)
    
    # Stage 3: Skeleton graph
    with tracer.span("stage3_skeleton", module="pipeline"):
        skeleton, graph, endpoints, junctions = build_skeleton_graph(binary, debug_writer)
        polylines = trace_polylines(graph, endpoints, junctions, debug_writer, skeleton)
        polylines = merge_short_segments(polylines, config.grouping.angle_continuity_threshold)
    
    # Stage 4: Vectorization
    with tracer.span("stage4_vectorize", module="pipeline"):
        # Simplify
        simplified = simplify_polylines(polylines, config.simplify.rdp_epsilon)
        
        # Fit Beziers
        bezier_lists = fit_bezier_to_polylines(
            simplified, 
            config.bezier.error_tolerance,
            config.bezier.max_iterations,
        )
        
        # Create stroke objects
        strokes = create_strokes_from_beziers(view_id, simplified, bezier_lists, config)
        
        # Emit SVG and debug artifacts
        emit_strokes_svg_with_ids(
            strokes, metadata["width"], metadata["height"],
            config, debug_writer,
        )
        
        # Debug: bbox overlay
        if debug_writer:
            overlay = create_debug_bbox_overlay(strokes, binary, config.debug.max_strokes)
            debug_writer.save_image(overlay, "stage4", "02_stroke_bboxes_overlay.png")
    
    # Create view
    view = View(
        view_id=view_id,
        view_label=ViewLabel.UNKNOWN,
        image_meta=ImageMeta(
            width=metadata["width"],
            height=metadata["height"],
            dpi_estimate=metadata["dpi_estimate"],
            source_path=input_path,
        ),
        strokes=strokes,
    )
    
    return view


@trace(label="apply_ops_pipeline")
def apply_ops_pipeline(scene_path, ops_path, out_dir, config_path=None):
    """
    Apply operations to an existing scene and re-run downstream stages.
    
    Args:
        scene_path: path to scene.json from previous run
        ops_path: path to JSON file with operations
        out_dir: output directory for updated results
        config_path: optional config file path
    
    Returns:
        Updated Document object
    """
    tracer = get_tracer()
    
    # Load scene
    with open(scene_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)
    
    document = Document.model_validate(scene_data)
    
    # Load operations
    with open(ops_path, "r", encoding="utf-8") as f:
        operations = json.load(f)
    
    config = load_config(config_path)
    
    # Apply operations
    with tracer.span("apply_operations", module="pipeline"):
        document = apply_operations(document, operations)
    
    # Re-run downstream stages
    from patentdraw.labels.numbering import renumber_after_operations
    
    with tracer.span("renumber", module="pipeline"):
        document = renumber_after_operations(document, config)
    
    with tracer.span("revalidate", module="pipeline"):
        document.validation = run_validation(document, config)
    
    with tracer.span("re_export", module="pipeline"):
        ensure_dir(out_dir)
        generate_svg_package(document, out_dir, config)
        generate_pdf(document, out_dir, config)
        generate_report(document, out_dir, None)
    
    # Save updated scene
    scene_out_path = os.path.join(out_dir, "scene.json")
    save_json(document.model_dump(), scene_out_path)
    
    tracer.event(f"Operations applied, output saved to {out_dir}")
    
    return document
