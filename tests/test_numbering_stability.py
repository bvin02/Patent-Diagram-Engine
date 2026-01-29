"""Tests for numbering stability."""

import pytest

from patentdraw.models import (
    Document, View, Stroke, Component, Label, ImageMeta,
    generate_component_id, ViewLabel, LabelStatus,
)


def create_consistent_document():
    """Create a document with reproducible structure."""
    # Create strokes with deterministic IDs
    strokes = [
        Stroke(
            stroke_id="stroke_001",
            polyline=[[10, 10], [20, 20], [30, 30]],
            bbox=[10, 10, 30, 30],
        ),
        Stroke(
            stroke_id="stroke_002",
            polyline=[[50, 50], [60, 60], [70, 70]],
            bbox=[50, 50, 70, 70],
        ),
        Stroke(
            stroke_id="stroke_003",
            polyline=[[100, 100], [110, 110], [120, 120]],
            bbox=[100, 100, 120, 120],
        ),
    ]
    
    # Create components with deterministic IDs
    comp1 = Component(
        component_id=generate_component_id(["stroke_001"]),
        stroke_ids=["stroke_001"],
        bbox=[10, 10, 30, 30],
        centroid=[20, 20],
    )
    comp2 = Component(
        component_id=generate_component_id(["stroke_002"]),
        stroke_ids=["stroke_002"],
        bbox=[50, 50, 70, 70],
        centroid=[60, 60],
    )
    comp3 = Component(
        component_id=generate_component_id(["stroke_003"]),
        stroke_ids=["stroke_003"],
        bbox=[100, 100, 120, 120],
        centroid=[110, 110],
    )
    
    doc = Document(
        doc_id="test_doc",
        views=[
            View(
                view_id="view_1",
                view_label=ViewLabel.FRONT,
                image_meta=ImageMeta(width=200, height=200),
                strokes=strokes,
                component_ids=[comp1.component_id, comp2.component_id, comp3.component_id],
            )
        ],
    )
    
    doc.component_registry[comp1.component_id] = comp1
    doc.component_registry[comp2.component_id] = comp2
    doc.component_registry[comp3.component_id] = comp3
    
    return doc


class TestNumberingStability:
    """Tests for numbering determinism and stability."""
    
    def test_same_input_same_numbering(self, default_config):
        """Test that identical documents get identical numbering."""
        from patentdraw.labels.numbering import assign_numbering
        
        doc1 = create_consistent_document()
        doc2 = create_consistent_document()
        
        doc1 = assign_numbering(doc1, default_config)
        doc2 = assign_numbering(doc2, default_config)
        
        assert doc1.numbering_registry == doc2.numbering_registry
    
    def test_numbering_order_by_centroid(self, default_config):
        """Test that numbering follows centroid order (y, then x)."""
        from patentdraw.labels.numbering import assign_numbering
        
        doc = create_consistent_document()
        doc = assign_numbering(doc, default_config)
        
        # Get components sorted by numeral
        numbered = sorted(
            doc.numbering_registry.items(),
            key=lambda x: x[1]
        )
        
        # First numeral should be for component with lowest centroid y
        first_comp_id = numbered[0][0]
        first_comp = doc.component_registry[first_comp_id]
        
        for other_id, _ in numbered[1:]:
            other = doc.component_registry[other_id]
            # First should have lower or equal y
            assert first_comp.centroid[1] <= other.centroid[1]
    
    def test_numbering_starts_at_config_value(self, default_config):
        """Test that numbering starts at configured start number."""
        from patentdraw.labels.numbering import assign_numbering
        
        default_config.numbering.start_number = 100
        default_config.numbering.increment = 5
        
        doc = create_consistent_document()
        doc = assign_numbering(doc, default_config)
        
        numerals = list(doc.numbering_registry.values())
        
        assert min(numerals) == 100
    
    def test_preserved_numbering_after_merge(self, default_config):
        """Test that merging preserves the lowest numeral."""
        from patentdraw.labels.numbering import assign_numbering
        from patentdraw.components.operations import merge_components
        
        doc = create_consistent_document()
        doc = assign_numbering(doc, default_config)
        
        # Get the first two component IDs
        comp_ids = list(doc.component_registry.keys())[:2]
        original_numerals = [doc.numbering_registry[cid] for cid in comp_ids]
        lowest = min(original_numerals)
        
        # Merge them
        new_id, doc = merge_components(doc, comp_ids)
        
        # New component should have the lowest numeral
        assert doc.numbering_registry[new_id] == lowest
    
    def test_multiple_runs_identical(self, default_config):
        """Test that running numbering multiple times gives same result."""
        from patentdraw.labels.numbering import assign_numbering
        
        doc = create_consistent_document()
        
        doc = assign_numbering(doc, default_config)
        first_result = dict(doc.numbering_registry)
        
        doc = assign_numbering(doc, default_config)
        second_result = dict(doc.numbering_registry)
        
        assert first_result == second_result


class TestDeterministicIds:
    """Tests for deterministic ID generation."""
    
    def test_stroke_id_deterministic(self):
        """Test that stroke ID is deterministic for same input."""
        from patentdraw.models import generate_stroke_id
        
        polyline = [[10, 20], [30, 40], [50, 60]]
        view_id = "view_123"
        
        id1 = generate_stroke_id(polyline, view_id)
        id2 = generate_stroke_id(polyline, view_id)
        
        assert id1 == id2
    
    def test_component_id_deterministic(self):
        """Test that component ID is deterministic for same strokes."""
        from patentdraw.models import generate_component_id
        
        stroke_ids = ["stroke_a", "stroke_b", "stroke_c"]
        
        id1 = generate_component_id(stroke_ids)
        id2 = generate_component_id(stroke_ids)
        
        assert id1 == id2
    
    def test_component_id_order_independent(self):
        """Test that component ID is same regardless of stroke order."""
        from patentdraw.models import generate_component_id
        
        id1 = generate_component_id(["a", "b", "c"])
        id2 = generate_component_id(["c", "a", "b"])
        
        assert id1 == id2
