"""
Configuration management for Patent Draw.

Loads YAML configuration with sensible defaults for all pipeline stages.
"""

import os
from dataclasses import dataclass, field

import yaml


@dataclass
class BinarizationConfig:
    """Configuration for Stage 2 binarization."""
    method: str = "otsu"  # "otsu" or "adaptive"
    denoise_kernel: int = 3
    morph_kernel: int = 3
    adaptive_block_size: int = 11
    adaptive_c: int = 2


@dataclass
class SimplifyConfig:
    """Configuration for polyline simplification."""
    rdp_epsilon: float = 1.5


@dataclass
class BezierConfig:
    """Configuration for Bezier curve fitting."""
    error_tolerance: float = 2.0
    max_iterations: int = 4


@dataclass
class StrokeConfig:
    """Configuration for stroke rendering."""
    width: float = 1.5
    color: str = "black"


@dataclass
class GroupingConfig:
    """Configuration for component grouping."""
    endpoint_distance_threshold: float = 5.0
    bbox_overlap_threshold: float = 0.1
    angle_continuity_threshold: float = 30.0  # degrees


@dataclass
class LabelConfig:
    """Configuration for label placement."""
    text_offset: float = 30.0
    label_font_size: float = 12.0
    leader_stroke_width: float = 0.5
    max_crossings_for_bend: int = 2


@dataclass
class PDFConfig:
    """Configuration for PDF export."""
    page_width_inches: float = 8.5
    page_height_inches: float = 11.0
    margin_inches: float = 1.0
    dpi: int = 300


@dataclass
class NumberingConfig:
    """Configuration for numeral assignment."""
    start_number: int = 10
    increment: int = 2


@dataclass
class TracingConfig:
    """Configuration for runtime tracing."""
    enabled: bool = False
    level: str = "INFO"
    file_path: str = None
    json_output: bool = False


@dataclass
class DebugConfig:
    """Configuration for debug artifact generation."""
    enabled: bool = False
    max_strokes: int = 50
    max_edge_scale: int = 1600


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    binarization: BinarizationConfig = field(default_factory=BinarizationConfig)
    simplify: SimplifyConfig = field(default_factory=SimplifyConfig)
    bezier: BezierConfig = field(default_factory=BezierConfig)
    stroke: StrokeConfig = field(default_factory=StrokeConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    numbering: NumberingConfig = field(default_factory=NumberingConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    ml_enabled: bool = False


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Falls back to defaults for any missing values.
    """
    config = PipelineConfig()
    
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        
        config = _merge_config(config, yaml_data)
    
    return config


def _merge_config(config, yaml_data):
    """Merge YAML data into config dataclass."""
    if "binarization" in yaml_data:
        for key, value in yaml_data["binarization"].items():
            if hasattr(config.binarization, key):
                setattr(config.binarization, key, value)
    
    if "simplify" in yaml_data:
        for key, value in yaml_data["simplify"].items():
            if hasattr(config.simplify, key):
                setattr(config.simplify, key, value)
    
    if "bezier" in yaml_data:
        for key, value in yaml_data["bezier"].items():
            if hasattr(config.bezier, key):
                setattr(config.bezier, key, value)
    
    if "stroke" in yaml_data:
        for key, value in yaml_data["stroke"].items():
            if hasattr(config.stroke, key):
                setattr(config.stroke, key, value)
    
    if "grouping" in yaml_data:
        for key, value in yaml_data["grouping"].items():
            if hasattr(config.grouping, key):
                setattr(config.grouping, key, value)
    
    if "label" in yaml_data:
        for key, value in yaml_data["label"].items():
            if hasattr(config.label, key):
                setattr(config.label, key, value)
    
    if "pdf" in yaml_data:
        for key, value in yaml_data["pdf"].items():
            if hasattr(config.pdf, key):
                setattr(config.pdf, key, value)
    
    if "numbering" in yaml_data:
        for key, value in yaml_data["numbering"].items():
            if hasattr(config.numbering, key):
                setattr(config.numbering, key, value)
    
    if "tracing" in yaml_data:
        for key, value in yaml_data["tracing"].items():
            if hasattr(config.tracing, key):
                setattr(config.tracing, key, value)
    
    if "debug" in yaml_data:
        for key, value in yaml_data["debug"].items():
            if hasattr(config.debug, key):
                setattr(config.debug, key, value)
    
    if "ml_enabled" in yaml_data:
        config.ml_enabled = yaml_data["ml_enabled"]
    
    return config


def save_default_config(path):
    """Save default configuration to YAML file for reference."""
    config = PipelineConfig()
    
    yaml_data = {
        "binarization": {
            "method": config.binarization.method,
            "denoise_kernel": config.binarization.denoise_kernel,
            "morph_kernel": config.binarization.morph_kernel,
            "adaptive_block_size": config.binarization.adaptive_block_size,
            "adaptive_c": config.binarization.adaptive_c,
        },
        "simplify": {
            "rdp_epsilon": config.simplify.rdp_epsilon,
        },
        "bezier": {
            "error_tolerance": config.bezier.error_tolerance,
            "max_iterations": config.bezier.max_iterations,
        },
        "stroke": {
            "width": config.stroke.width,
            "color": config.stroke.color,
        },
        "grouping": {
            "endpoint_distance_threshold": config.grouping.endpoint_distance_threshold,
            "bbox_overlap_threshold": config.grouping.bbox_overlap_threshold,
            "angle_continuity_threshold": config.grouping.angle_continuity_threshold,
        },
        "label": {
            "text_offset": config.label.text_offset,
            "label_font_size": config.label.label_font_size,
            "leader_stroke_width": config.label.leader_stroke_width,
            "max_crossings_for_bend": config.label.max_crossings_for_bend,
        },
        "pdf": {
            "page_width_inches": config.pdf.page_width_inches,
            "page_height_inches": config.pdf.page_height_inches,
            "margin_inches": config.pdf.margin_inches,
            "dpi": config.pdf.dpi,
        },
        "numbering": {
            "start_number": config.numbering.start_number,
            "increment": config.numbering.increment,
        },
        "tracing": {
            "enabled": config.tracing.enabled,
            "level": config.tracing.level,
        },
        "debug": {
            "enabled": config.debug.enabled,
            "max_strokes": config.debug.max_strokes,
            "max_edge_scale": config.debug.max_edge_scale,
        },
        "ml_enabled": config.ml_enabled,
    }
    
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
