"""Tests for component grouping and operations."""

import pytest

from patentdraw.models import (
    Document, View, Stroke, Component, Label, ImageMeta, 
    generate_component_id, ViewLabel, LabelStatus,
)


def create_test_strokes():
    """Create test strokes for grouping tests."""
    # Two strokes close together (should group)
    stroke1 = Stroke(
        stroke_id="stroke_a",
        polyline=[[0, 0], [10, 10], [20, 20]],
        bbox=[0, 0, 20, 20],
    )
    stroke2 = Stroke(
        stroke_id="stroke_b",
        polyline=[[20, 20], [30, 30], [40, 40]],  # Shares endpoint with stroke1
        bbox=[20, 20, 40, 40],
    )
    
    # Third stroke far away (should not group)
    stroke3 = Stroke(
        stroke_id="stroke_c",
        polyline=[[100, 100], [110, 110], [120, 120]],
        bbox=[100, 100, 120, 120],
    )
    
    return [stroke1, stroke2, stroke3]


class TestGrouping:
    """Tests for baseline component grouping."""
    
    def test_connected_strokes_grouped(self, default_config):
        """Test that strokes sharing endpoints are grouped together."""
        from patentdraw.components.grouping import group_strokes_baseline
        
        strokes = create_test_strokes()
        components = group_strokes_baseline(strokes, default_config)
        
        # Should produce 2 components: (stroke_a + stroke_b) and (stroke_c)
        assert len(components) == 2
    
    def test_disconnected_strokes_separate(self, default_config):
        """Test that distant strokes form separate components."""
        from patentdraw.components.grouping import group_strokes_baseline
        
        strokes = create_test_strokes()
        components = group_strokes_baseline(strokes, default_config)
        
        # Find the component with stroke_c
        stroke_c_comp = next(c for c in components if "stroke_c" in c.stroke_ids)
        
        # It should only contain stroke_c
        assert stroke_c_comp.stroke_ids == ["stroke_c"]
    
    def test_component_id_deterministic(self, default_config):
        """Test that component IDs are deterministic."""
        from patentdraw.components.grouping import group_strokes_baseline
        
        strokes = create_test_strokes()
        
        components1 = group_strokes_baseline(strokes, default_config)
        components2 = group_strokes_baseline(strokes, default_config)
        
        ids1 = sorted(c.component_id for c in components1)
        ids2 = sorted(c.component_id for c in components2)
        
        assert ids1 == ids2


class TestOperations:
    """Tests for merge and split operations."""
    
    def create_test_document(self):
        """Create a test document with components."""
        from patentdraw.components.grouping import group_strokes_baseline
        from patentdraw.config import PipelineConfig
        
        strokes = create_test_strokes()
        config = PipelineConfig()
        components = group_strokes_baseline(strokes, config)
        
        doc = Document(
            doc_id="test_doc",
            views=[
                View(
                    view_id="view_1",
                    view_label=ViewLabel.FRONT,
                    image_meta=ImageMeta(width=200, height=200),
                    strokes=strokes,
                    component_ids=[c.component_id for c in components],
                )
            ],
        )
        
        for c in components:
            doc.component_registry[c.component_id] = c
            doc.numbering_registry[c.component_id] = 10 + len(doc.numbering_registry) * 2
        
        return doc
    
    def test_merge_combines_strokes(self):
        """Test that merge creates component with combined strokes."""
        from patentdraw.components.operations import merge_components
        
        doc = self.create_test_document()
        comp_ids = list(doc.component_registry.keys())
        
        new_id, doc = merge_components(doc, comp_ids)
        
        # Should now have 1 component
        assert len(doc.component_registry) == 1
        
        # New component should have all strokes
        new_comp = doc.component_registry[new_id]
        assert len(new_comp.stroke_ids) == 3
    
    def test_merge_preserves_lowest_numeral(self):
        """Test that merge keeps the lowest numeral."""
        from patentdraw.components.operations import merge_components
        
        doc = self.create_test_document()
        comp_ids = list(doc.component_registry.keys())
        
        new_id, doc = merge_components(doc, comp_ids)
        
        # Should have the lowest numeral (10)
        assert doc.numbering_registry[new_id] == 10
    
    def test_split_creates_two_components(self):
        """Test that split creates two separate components."""
        from patentdraw.components.operations import merge_components, split_component_by_strokes
        
        doc = self.create_test_document()
        
        # First merge all
        comp_ids = list(doc.component_registry.keys())
        merged_id, doc = merge_components(doc, comp_ids)
        
        # Then split
        new_ids, doc = split_component_by_strokes(
            doc, 
            merged_id, 
            [["stroke_a", "stroke_b"], ["stroke_c"]]
        )
        
        assert len(new_ids) == 2
        assert len(doc.component_registry) == 2
    
    def test_split_updates_view(self):
        """Test that split updates view component list."""
        from patentdraw.components.operations import merge_components, split_component_by_strokes
        
        doc = self.create_test_document()
        
        # Merge and split
        comp_ids = list(doc.component_registry.keys())
        merged_id, doc = merge_components(doc, comp_ids)
        new_ids, doc = split_component_by_strokes(
            doc, merged_id, [["stroke_a"], ["stroke_b", "stroke_c"]]
        )
        
        # View should have the new component IDs
        view_comp_ids = set(doc.views[0].component_ids)
        assert set(new_ids) == view_comp_ids
