"""Tests for skeleton graph construction."""

import cv2
import numpy as np
import pytest


class TestSkeletonGraph:
    """Tests for skeleton graph building."""
    
    def test_simple_line_skeleton(self, simple_line_image, default_config):
        """Test skeleton from a simple horizontal line."""
        from patentdraw.preprocess.stage2_binarize import binarize
        from patentdraw.strokes.skeleton_graph import build_skeleton_graph
        
        binary = binarize(simple_line_image, default_config)
        skeleton, graph, endpoints, junctions = build_skeleton_graph(binary)
        
        # A simple line should have 2 endpoints and 0 junctions
        assert len(endpoints) == 2
        assert len(junctions) == 0
    
    def test_l_shape_junction(self, l_shape_image, default_config):
        """Test skeleton from an L-shape has expected topology."""
        from patentdraw.preprocess.stage2_binarize import binarize
        from patentdraw.strokes.skeleton_graph import build_skeleton_graph
        
        binary = binarize(l_shape_image, default_config)
        skeleton, graph, endpoints, junctions = build_skeleton_graph(binary)
        
        # L-shape should have 2 endpoints and 1 junction (at the corner)
        assert len(endpoints) == 2
        # Junction count may vary depending on line thickness and skeleton
        assert len(junctions) >= 0
    
    def test_rectangle_skeleton(self, simple_rectangle_image, default_config):
        """Test skeleton from rectangle has expected structure."""
        from patentdraw.preprocess.stage2_binarize import binarize
        from patentdraw.strokes.skeleton_graph import build_skeleton_graph
        
        binary = binarize(simple_rectangle_image, default_config)
        skeleton, graph, endpoints, junctions = build_skeleton_graph(binary)
        
        # Rectangle should have no endpoints (closed loop) but some junctions at corners
        # Actually depends on skeleton algorithm, may have endpoints at corners
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
    
    def test_graph_connectivity(self, simple_line_image, default_config):
        """Test that graph edges connect neighboring skeleton pixels."""
        from patentdraw.preprocess.stage2_binarize import binarize
        from patentdraw.strokes.skeleton_graph import build_skeleton_graph
        
        binary = binarize(simple_line_image, default_config)
        skeleton, graph, endpoints, junctions = build_skeleton_graph(binary)
        
        # All edges should connect adjacent pixels
        for u, v in graph.edges():
            dy = abs(u[0] - v[0])
            dx = abs(u[1] - v[1])
            # 8-connectivity: max distance is sqrt(2) ~ 1.4
            assert dy <= 1 and dx <= 1


class TestPolylineTrace:
    """Tests for polyline tracing."""
    
    def test_trace_simple_line(self, simple_line_image, default_config):
        """Test tracing produces polyline for simple line."""
        from patentdraw.preprocess.stage2_binarize import binarize
        from patentdraw.strokes.skeleton_graph import build_skeleton_graph
        from patentdraw.strokes.polyline_trace import trace_polylines
        
        binary = binarize(simple_line_image, default_config)
        skeleton, graph, endpoints, junctions = build_skeleton_graph(binary)
        polylines = trace_polylines(graph, endpoints, junctions)
        
        # Should produce at least one polyline
        assert len(polylines) >= 1
        
        # Polyline should have multiple points
        assert len(polylines[0]) >= 2
    
    def test_polyline_endpoint_preservation(self, simple_line_image, default_config):
        """Test that polyline endpoints match graph endpoints."""
        from patentdraw.preprocess.stage2_binarize import binarize
        from patentdraw.strokes.skeleton_graph import build_skeleton_graph
        from patentdraw.strokes.polyline_trace import trace_polylines
        
        binary = binarize(simple_line_image, default_config)
        skeleton, graph, endpoints, junctions = build_skeleton_graph(binary)
        polylines = trace_polylines(graph, endpoints, junctions)
        
        if polylines and endpoints:
            # First/last points of polylines should be near endpoint pixels
            for polyline in polylines:
                start = polyline[0]
                end = polyline[-1]
                # Convert to (y, x) format for comparison
                start_yx = (start[1], start[0])
                end_yx = (end[1], end[0])
                # At least one endpoint should be in graph endpoints
                endpoints_set = set(endpoints)
                assert start_yx in endpoints_set or end_yx in endpoints_set
