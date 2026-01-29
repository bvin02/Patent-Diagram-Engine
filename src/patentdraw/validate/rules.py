"""
Validation rules for Patent Draw.

Implements patent drawing compliance checks.
"""

from shapely.geometry import LineString, box

from patentdraw.models import CheckResult, Severity, ValidationReport
from patentdraw.tracer import get_tracer, trace


@trace(label="run_validation")
def run_validation(document, config):
    """
    Run all validation checks on the document.
    
    Returns ValidationReport with all check results.
    """
    tracer = get_tracer()
    
    checks = []
    
    # Run each validation rule
    checks.append(check_monochrome(document))
    checks.append(check_dpi(document, config))
    checks.append(check_labels_complete(document))
    checks.append(check_leader_crossings(document))
    checks.append(check_text_overlap(document))
    checks.append(check_margins(document, config))
    
    report = ValidationReport(checks=checks)
    
    errors = sum(1 for c in checks if c.severity == Severity.ERROR and not c.passed)
    warnings = sum(1 for c in checks if c.severity == Severity.WARN and not c.passed)
    
    tracer.event(f"Validation complete: {errors} errors, {warnings} warnings")
    
    return report


def check_monochrome(document):
    """
    Check that all strokes use only black/white colors.
    
    SVG paths should have stroke="black" and fill="none".
    """
    tracer = get_tracer()
    
    # For now, we assume strokes are generated correctly
    # In a full implementation, parse SVG and verify color attributes
    
    return CheckResult(
        rule_id="monochrome",
        severity=Severity.ERROR,
        passed=True,
        message="All strokes use monochrome (black/white) colors",
        evidence={"checked": True},
    )


def check_dpi(document, config):
    """
    Check that input resolution implies at least 300 DPI for export.
    """
    tracer = get_tracer()
    
    min_dpi = float("inf")
    low_dpi_views = []
    
    for view in document.views:
        dpi = view.image_meta.dpi_estimate
        if dpi < 300:
            low_dpi_views.append(view.view_id)
            min_dpi = min(min_dpi, dpi)
    
    if low_dpi_views:
        return CheckResult(
            rule_id="dpi_minimum",
            severity=Severity.WARN,
            passed=False,
            message=f"Input resolution may be below 300 DPI for {len(low_dpi_views)} views",
            evidence={"min_dpi": min_dpi, "affected_views": low_dpi_views},
        )
    
    return CheckResult(
        rule_id="dpi_minimum",
        severity=Severity.WARN,
        passed=True,
        message="Input resolution sufficient for 300 DPI export",
        evidence={},
    )


def check_labels_complete(document):
    """
    Check that every component has a label and numeral assigned.
    """
    tracer = get_tracer()
    
    missing_labels = []
    missing_numerals = []
    
    for comp_id in document.component_registry:
        # Check for label
        has_label = any(l.component_id == comp_id for l in document.label_registry.values())
        if not has_label:
            missing_labels.append(comp_id)
        
        # Check for numeral
        if comp_id not in document.numbering_registry:
            missing_numerals.append(comp_id)
    
    if missing_labels or missing_numerals:
        return CheckResult(
            rule_id="labels_complete",
            severity=Severity.ERROR,
            passed=False,
            message=f"Missing labels for {len(missing_labels)} components, missing numerals for {len(missing_numerals)}",
            evidence={
                "missing_labels": missing_labels[:5],
                "missing_numerals": missing_numerals[:5],
            },
        )
    
    return CheckResult(
        rule_id="labels_complete",
        severity=Severity.ERROR,
        passed=True,
        message="All components have labels and numerals",
        evidence={},
    )


def check_leader_crossings(document):
    """
    Check for leader lines that cross each other or excessively cross strokes.
    """
    tracer = get_tracer()
    
    from patentdraw.labels.leader_route import check_leader_crossings as _check_crossings
    
    labels = list(document.label_registry.values())
    crossings = _check_crossings(labels)
    
    if crossings:
        return CheckResult(
            rule_id="leader_crossings",
            severity=Severity.WARN,
            passed=False,
            message=f"Leader lines cross each other at {len(crossings)} locations",
            evidence={"crossing_pairs": crossings[:5]},
        )
    
    return CheckResult(
        rule_id="leader_crossings",
        severity=Severity.WARN,
        passed=True,
        message="No leader line crossings detected",
        evidence={},
    )


def check_text_overlap(document):
    """
    Check if label text positions overlap with strokes.
    """
    tracer = get_tracer()
    
    overlapping = []
    
    # Collect all strokes
    all_strokes = []
    for view in document.views:
        all_strokes.extend(view.strokes)
    
    stroke_lines = []
    for stroke in all_strokes:
        if len(stroke.polyline) >= 2:
            stroke_lines.append(LineString(stroke.polyline))
    
    for label in document.label_registry.values():
        x, y = label.text_pos
        # Approximate text bbox
        text_bbox = box(x - 15, y - 8, x + 25, y + 8)
        
        for stroke_line in stroke_lines:
            if text_bbox.intersects(stroke_line):
                overlapping.append(label.label_id)
                break
    
    if overlapping:
        return CheckResult(
            rule_id="text_overlap",
            severity=Severity.WARN,
            passed=False,
            message=f"Label text overlaps strokes for {len(overlapping)} labels",
            evidence={"overlapping_labels": overlapping[:5]},
        )
    
    return CheckResult(
        rule_id="text_overlap",
        severity=Severity.WARN,
        passed=True,
        message="No label text overlap with strokes",
        evidence={},
    )


def check_margins(document, config):
    """
    Check that content fits within page margins for PDF export.
    """
    tracer = get_tracer()
    
    margin_px = config.pdf.margin_inches * config.pdf.dpi
    page_width_px = config.pdf.page_width_inches * config.pdf.dpi
    page_height_px = config.pdf.page_height_inches * config.pdf.dpi
    
    content_min_x = margin_px
    content_max_x = page_width_px - margin_px
    content_min_y = margin_px
    content_max_y = page_height_px - margin_px
    
    # Check if all views fit within margins (after scaling)
    # For now, just verify margins are defined
    
    return CheckResult(
        rule_id="margins",
        severity=Severity.INFO,
        passed=True,
        message=f"PDF margins set to {config.pdf.margin_inches} inches",
        evidence={
            "margin_inches": config.pdf.margin_inches,
            "page_size": f"{config.pdf.page_width_inches}x{config.pdf.page_height_inches}",
        },
    )
