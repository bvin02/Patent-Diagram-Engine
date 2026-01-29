"""
PDF layout and export for Patent Draw.

Creates patent-compliant PDF with proper page size, margins, and figure layout.
"""

import os

from patentdraw.io.save_artifacts import ensure_dir, save_svg, render_svg_to_png
from patentdraw.tracer import get_tracer, trace


@trace(label="generate_pdf")
def generate_pdf(document, out_dir, config):
    """
    Generate final PDF with all views laid out on pages.
    
    Uses 8.5x11 inch page size with configurable margins.
    Views are scaled to fit within margins while preserving aspect ratio.
    """
    tracer = get_tracer()
    
    try:
        import cairosvg
    except ImportError:
        tracer.event("cairosvg not available, skipping PDF generation", level="WARN")
        return None
    
    pdf_path = os.path.join(out_dir, "final.pdf")
    ensure_dir(os.path.dirname(pdf_path))
    
    # Page dimensions in points (1 inch = 72 points)
    page_width_pt = config.pdf.page_width_inches * 72
    page_height_pt = config.pdf.page_height_inches * 72
    margin_pt = config.pdf.margin_inches * 72
    
    content_width = page_width_pt - 2 * margin_pt
    content_height = page_height_pt - 2 * margin_pt
    
    # For MVP, create PDF from combined SVG
    svg_path = os.path.join(out_dir, "final.svg")
    
    if not os.path.exists(svg_path):
        tracer.event("final.svg not found, cannot create PDF", level="WARN")
        return None
    
    try:
        # Scale SVG to fit within content area
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        
        # Create scaled SVG for PDF
        scaled_svg = _create_scaled_svg(
            svg_content,
            document.views,
            config,
            content_width,
            content_height,
            margin_pt,
        )
        
        # Save temporary scaled SVG
        temp_svg_path = os.path.join(out_dir, "_temp_pdf.svg")
        with open(temp_svg_path, "w", encoding="utf-8") as f:
            f.write(scaled_svg)
        
        # Convert to PDF
        cairosvg.svg2pdf(
            url=temp_svg_path,
            write_to=pdf_path,
        )
        
        # Clean up temp file
        os.remove(temp_svg_path)
        
        tracer.event(f"PDF saved: {pdf_path}")
        
    except Exception as e:
        tracer.event(f"PDF generation failed: {str(e)}", level="ERROR")
        return None
    
    return pdf_path


def _create_scaled_svg(svg_content, views, config, content_width, content_height, margin):
    """
    Create an SVG scaled to fit the PDF content area.
    """
    if not views:
        return svg_content
    
    # Calculate original dimensions from views
    total_height = 0
    max_width = 0
    spacing = 50
    
    for view in views:
        total_height += view.image_meta.height + spacing
        max_width = max(max_width, view.image_meta.width)
    
    total_height -= spacing
    
    if max_width == 0 or total_height == 0:
        return svg_content
    
    # Calculate scale factor
    scale_x = content_width / max_width
    scale_y = content_height / total_height
    scale = min(scale_x, scale_y)
    
    scaled_width = max_width * scale
    scaled_height = total_height * scale
    
    # Add page dimensions and margins
    page_width = config.pdf.page_width_inches * 72
    page_height = config.pdf.page_height_inches * 72
    
    # Center content
    offset_x = (page_width - scaled_width) / 2
    offset_y = margin + 20  # Add space for title
    
    # Create wrapper SVG
    wrapper = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     width="{page_width}pt" height="{page_height}pt"
     viewBox="0 0 {page_width} {page_height}">
  <rect width="100%" height="100%" fill="white"/>
  <g transform="translate({offset_x}, {offset_y}) scale({scale})">
'''
    
    # Extract inner content from original SVG
    import re
    
    # Find the inner content (everything between <svg> and </svg>)
    match = re.search(r'<svg[^>]*>(.*)</svg>', svg_content, re.DOTALL)
    if match:
        inner = match.group(1)
        wrapper += inner
    
    wrapper += '''
  </g>
</svg>'''
    
    return wrapper


def layout_views_on_pages(views, content_width, content_height, spacing=20):
    """
    Calculate layout for views across multiple pages.
    
    Returns list of pages, each containing list of (view, x, y, scale).
    """
    pages = []
    current_page = []
    current_y = 0
    
    for view in views:
        view_width = view.image_meta.width
        view_height = view.image_meta.height
        
        # Calculate scale to fit width
        scale = min(1.0, content_width / view_width)
        scaled_height = view_height * scale
        
        # Check if view fits on current page
        if current_y + scaled_height > content_height and current_page:
            # Start new page
            pages.append(current_page)
            current_page = []
            current_y = 0
        
        # Add view to current page
        current_page.append({
            "view": view,
            "x": 0,
            "y": current_y,
            "scale": scale,
        })
        
        current_y += scaled_height + spacing
    
    if current_page:
        pages.append(current_page)
    
    return pages
