"""
Pydantic data models for the Patent Draw scene graph.

All scene data flows through these validated models to ensure consistency.
Content-based ID generation provides deterministic outputs.
"""

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ViewLabel(str, Enum):
    """Standard view labels for patent drawings."""
    FRONT = "front"
    SIDE = "side"
    TOP = "top"
    ISOMETRIC = "isometric"
    UNKNOWN = "unknown"


class LabelStatus(str, Enum):
    """Status of a label in the editing workflow."""
    PROPOSED = "proposed"
    EDITED = "edited"
    FINAL = "final"


class Severity(str, Enum):
    """Severity levels for validation checks."""
    ERROR = "error"
    WARN = "warn"
    INFO = "info"


class CubicBezier(BaseModel):
    """A single cubic Bezier curve segment."""
    p0: List[float] = Field(..., min_length=2, max_length=2)  # start point
    p1: List[float] = Field(..., min_length=2, max_length=2)  # control point 1
    p2: List[float] = Field(..., min_length=2, max_length=2)  # control point 2
    p3: List[float] = Field(..., min_length=2, max_length=2)  # end point


class Stroke(BaseModel):
    """A single editable stroke in the drawing."""
    stroke_id: str
    polyline: List[List[float]] = Field(default_factory=list)
    bezier_segments: List[CubicBezier] = Field(default_factory=list)
    svg_path: str = ""
    bbox: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    
    model_config = ConfigDict(extra="forbid")


class Component(BaseModel):
    """A detected component grouping multiple strokes."""
    component_id: str
    stroke_ids: List[str] = Field(default_factory=list)
    bbox: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    centroid: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    proposal_sources: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    
    model_config = ConfigDict(extra="forbid")


class Label(BaseModel):
    """A label with leader line for a component."""
    label_id: str
    component_id: str
    anchor_point: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    text_pos: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    leader_path: List[List[float]] = Field(default_factory=list)
    text: str = ""
    status: LabelStatus = LabelStatus.PROPOSED
    
    model_config = ConfigDict(extra="forbid")


class ImageMeta(BaseModel):
    """Metadata for an input image."""
    width: int
    height: int
    dpi_estimate: int = 150
    source_path: str = ""
    
    model_config = ConfigDict(extra="forbid")


class InputMeta(BaseModel):
    """Metadata for a single input file."""
    input_id: str
    source_path: str
    view_label: ViewLabel = ViewLabel.UNKNOWN
    
    model_config = ConfigDict(extra="forbid")


class View(BaseModel):
    """A single view of the drawing (one input image processed)."""
    view_id: str
    view_label: ViewLabel = ViewLabel.UNKNOWN
    image_meta: ImageMeta
    strokes: List[Stroke] = Field(default_factory=list)
    component_ids: List[str] = Field(default_factory=list)
    label_ids: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")


class CheckResult(BaseModel):
    """Result of a single validation check."""
    rule_id: str
    severity: Severity
    passed: bool
    message: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(extra="forbid")


class ValidationReport(BaseModel):
    """Collection of validation check results."""
    checks: List[CheckResult] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")
    
    @property
    def has_errors(self):
        """Check if any errors exist."""
        return any(c.severity == Severity.ERROR and not c.passed for c in self.checks)
    
    @property
    def error_count(self):
        """Count of failed error-level checks."""
        return sum(1 for c in self.checks if c.severity == Severity.ERROR and not c.passed)
    
    @property
    def warning_count(self):
        """Count of failed warning-level checks."""
        return sum(1 for c in self.checks if c.severity == Severity.WARN and not c.passed)


class Document(BaseModel):
    """Root document containing all views and registries."""
    doc_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    inputs: List[InputMeta] = Field(default_factory=list)
    views: List[View] = Field(default_factory=list)
    component_registry: Dict[str, Component] = Field(default_factory=dict)
    label_registry: Dict[str, Label] = Field(default_factory=dict)
    numbering_registry: Dict[str, int] = Field(default_factory=dict)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    
    model_config = ConfigDict(extra="forbid")


# ID generation functions for deterministic outputs

def generate_stroke_id(polyline, view_id, round_digits=2):
    """
    Generate deterministic stroke ID from polyline coordinates.
    
    Rounds coordinates to avoid floating point instability.
    """
    if not polyline:
        return f"stroke_{view_id}_empty"
    
    rounded = [[round(p[0], round_digits), round(p[1], round_digits)] for p in polyline]
    data = f"{view_id}:{rounded}"
    h = hashlib.sha256(data.encode()).hexdigest()[:12]
    return f"stroke_{h}"


def generate_component_id(stroke_ids):
    """
    Generate deterministic component ID from sorted stroke IDs.
    """
    if not stroke_ids:
        return "comp_empty"
    
    sorted_ids = sorted(stroke_ids)
    data = ":".join(sorted_ids)
    h = hashlib.sha256(data.encode()).hexdigest()[:12]
    return f"comp_{h}"


def generate_label_id(component_id, anchor_point, text_pos, round_digits=1):
    """
    Generate deterministic label ID from component and positions.
    """
    anchor_rounded = [round(anchor_point[0], round_digits), round(anchor_point[1], round_digits)]
    text_rounded = [round(text_pos[0], round_digits), round(text_pos[1], round_digits)]
    data = f"{component_id}:{anchor_rounded}:{text_rounded}"
    h = hashlib.sha256(data.encode()).hexdigest()[:12]
    return f"label_{h}"


def generate_doc_id(input_paths):
    """
    Generate deterministic document ID from input file paths.
    """
    sorted_paths = sorted(input_paths)
    data = ":".join(sorted_paths)
    h = hashlib.sha256(data.encode()).hexdigest()[:16]
    return f"doc_{h}"


def generate_view_id(source_path, index):
    """
    Generate deterministic view ID from source path and index.
    """
    data = f"{source_path}:{index}"
    h = hashlib.sha256(data.encode()).hexdigest()[:12]
    return f"view_{h}"


def compute_bbox(points):
    """
    Compute bounding box from a list of [x, y] points.
    
    Returns [min_x, min_y, max_x, max_y].
    """
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def compute_centroid(points):
    """
    Compute centroid of a list of [x, y] points.
    """
    if not points:
        return [0.0, 0.0]
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def compute_bbox_from_bboxes(bboxes):
    """
    Compute combined bounding box from multiple bboxes.
    
    Each bbox is [min_x, min_y, max_x, max_y].
    """
    if not bboxes:
        return [0.0, 0.0, 0.0, 0.0]
    
    min_x = min(b[0] for b in bboxes)
    min_y = min(b[1] for b in bboxes)
    max_x = max(b[2] for b in bboxes)
    max_y = max(b[3] for b in bboxes)
    return [min_x, min_y, max_x, max_y]
