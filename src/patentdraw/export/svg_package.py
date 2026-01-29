"""
SVG package generation for Patent Draw.

Creates final SVG files with strokes, leader lines, and labels.
"""

import os

import svgwrite

from patentdraw.io.save_artifacts import ensure_dir, save_svg, render_svg_to_png
from patentdraw.tracer import get_tracer, trace


@trace(label="generate_svg_package")
def generate_svg_package(document, out_dir, config):
    """
    Generate final SVG files.
    
    Creates:
    - One SVG per view: {view_id}.svg
    - Combined SVG: combined.svg
    - final.svg in output root
    """
    tracer = get_tracer()
    
    svg_dir = os.path.join(out_dir, "svg")
    ensure_dir(svg_dir)
    
    view_svgs = []
    
    for view in document.views:
        svg_path = os.path.join(svg_dir, f"{view.view_id}.svg")
        
        dwg = create_view_svg(
            view, 
            document.label_registry, 
            document.numbering_registry,
            config,
        )
        save_svg(dwg, svg_path)
        view_svgs.append(svg_path)
        
        tracer.event(f"Created SVG for view {view.view_id}")
    
    # Create combined SVG
    if document.views:
        combined_path = os.path.join(svg_dir, "combined.svg")
        final_path = os.path.join(out_dir, "final.svg")
        
        combined = create_combined_svg(document, config)
        save_svg(combined, combined_path)
        save_svg(combined, final_path)
        
        tracer.event(f"Created combined SVG with {len(document.views)} views")
    
    return view_svgs


def create_view_svg(view, label_registry, numbering_registry, config):
    """Create SVG for a single view with strokes, leaders, and labels."""
    width = view.image_meta.width
    height = view.image_meta.height
    
    dwg = svgwrite.Drawing(size=(f"{width}px", f"{height}px"))
    dwg.viewbox(0, 0, width, height)
    
    # Add styles
    dwg.defs.add(dwg.style("""
        .stroke { stroke-linecap: round; stroke-linejoin: round; }
        .leader { stroke-linecap: round; }
        .label-text { font-family: Arial, sans-serif; }
    """))
    
    # Strokes group
    stroke_group = dwg.g(
        id="strokes",
        fill="none",
        stroke=config.stroke.color,
        stroke_width=config.stroke.width,
        class_="stroke",
    )
    
    for stroke in view.strokes:
        if stroke.svg_path:
            path = dwg.path(d=stroke.svg_path, id=stroke.stroke_id)
            stroke_group.add(path)
    
    dwg.add(stroke_group)
    
    # Leaders group
    leader_group = dwg.g(
        id="leaders",
        fill="none",
        stroke="black",
        stroke_width=config.label.leader_stroke_width,
        class_="leader",
    )
    
    for label_id in view.label_ids:
        if label_id in label_registry:
            label = label_registry[label_id]
            if len(label.leader_path) >= 2:
                points = " ".join(f"{p[0]},{p[1]}" for p in label.leader_path)
                polyline = dwg.polyline(points=label.leader_path, id=f"leader_{label_id}")
                leader_group.add(polyline)
    
    dwg.add(leader_group)
    
    # Labels group
    label_group = dwg.g(id="labels", class_="label-text")
    
    for label_id in view.label_ids:
        if label_id in label_registry:
            label = label_registry[label_id]
            if label.text:
                text = dwg.text(
                    label.text,
                    insert=(label.text_pos[0], label.text_pos[1]),
                    font_size=f"{config.label.label_font_size}px",
                    fill="black",
                    id=f"text_{label_id}",
                )
                label_group.add(text)
    
    dwg.add(label_group)
    
    return dwg


def create_combined_svg(document, config):
    """Create a combined SVG with all views arranged vertically."""
    if not document.views:
        return svgwrite.Drawing(size=("100px", "100px"))
    
    # Calculate total dimensions
    total_height = 0
    max_width = 0
    spacing = 50
    
    for view in document.views:
        total_height += view.image_meta.height + spacing
        max_width = max(max_width, view.image_meta.width)
    
    total_height -= spacing  # Remove last spacing
    
    dwg = svgwrite.Drawing(size=(f"{max_width}px", f"{total_height}px"))
    dwg.viewbox(0, 0, max_width, total_height)
    
    # Add styles
    dwg.defs.add(dwg.style("""
        .stroke { stroke-linecap: round; stroke-linejoin: round; }
        .leader { stroke-linecap: round; }
        .label-text { font-family: Arial, sans-serif; }
        .fig-title { font-family: Arial, sans-serif; font-weight: bold; }
    """))
    
    y_offset = 0
    
    for fig_num, view in enumerate(document.views, start=1):
        view_group = dwg.g(
            id=f"view_{view.view_id}",
            transform=f"translate(0, {y_offset})",
        )
        
        # Figure title
        title = dwg.text(
            f"FIG. {fig_num}",
            insert=(view.image_meta.width / 2, -10),
            font_size="14px",
            text_anchor="middle",
            class_="fig-title",
        )
        view_group.add(title)
        
        # Strokes
        stroke_group = dwg.g(
            fill="none",
            stroke=config.stroke.color,
            stroke_width=config.stroke.width,
            class_="stroke",
        )
        for stroke in view.strokes:
            if stroke.svg_path:
                stroke_group.add(dwg.path(d=stroke.svg_path))
        view_group.add(stroke_group)
        
        # Leaders
        leader_group = dwg.g(
            fill="none",
            stroke="black",
            stroke_width=config.label.leader_stroke_width,
            class_="leader",
        )
        for label_id in view.label_ids:
            if label_id in document.label_registry:
                label = document.label_registry[label_id]
                if len(label.leader_path) >= 2:
                    leader_group.add(dwg.polyline(points=label.leader_path))
        view_group.add(leader_group)
        
        # Labels
        label_group = dwg.g(class_="label-text")
        for label_id in view.label_ids:
            if label_id in document.label_registry:
                label = document.label_registry[label_id]
                if label.text:
                    text = dwg.text(
                        label.text,
                        insert=(label.text_pos[0], label.text_pos[1]),
                        font_size=f"{config.label.label_font_size}px",
                        fill="black",
                    )
                    label_group.add(text)
        view_group.add(label_group)
        
        dwg.add(view_group)
        y_offset += view.image_meta.height + spacing
    
    return dwg
