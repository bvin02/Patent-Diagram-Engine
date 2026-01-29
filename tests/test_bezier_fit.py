"""Tests for Bezier curve fitting."""

import numpy as np
import pytest

from patentdraw.strokes.bezier_fit import (
    fit_cubic_beziers, evaluate_bezier, bezier_to_svg_path,
    _line_to_bezier,
)


class TestBezierFit:
    """Tests for Bezier curve fitting."""
    
    def test_straight_line_fit(self):
        """Test fitting a straight line produces valid Bezier."""
        points = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]])
        
        beziers = fit_cubic_beziers(points, error_tolerance=1.0)
        
        # Should produce at least one Bezier segment
        assert len(beziers) >= 1
        
        # First segment should start at first point
        assert beziers[0].p0 == [0, 0]
        
        # Last segment should end at last point
        assert beziers[-1].p3 == [40, 0]
    
    def test_arc_fit_error_threshold(self):
        """Test that arc is fit within error tolerance."""
        # Create quarter circle points
        t = np.linspace(0, np.pi/2, 20)
        radius = 50
        points = np.column_stack([radius * np.cos(t), radius * np.sin(t)])
        
        error_tolerance = 2.0
        beziers = fit_cubic_beziers(points, error_tolerance=error_tolerance)
        
        # Verify error is below threshold for sampled points
        for bez in beziers:
            for i in range(11):
                t_val = i / 10.0
                fitted = evaluate_bezier(bez, t_val)
                # Distance from origin should be approximately radius
                dist = np.sqrt(fitted[0]**2 + fitted[1]**2)
                # Allow some tolerance
                assert abs(dist - radius) < error_tolerance * 3
    
    def test_endpoint_preservation(self):
        """Test that endpoints are exactly preserved."""
        points = np.array([[5.5, 10.2], [15, 20], [25, 15], [35.7, 22.3]])
        
        beziers = fit_cubic_beziers(points, error_tolerance=1.0)
        
        # Start point should match exactly
        assert beziers[0].p0[0] == pytest.approx(5.5, abs=0.01)
        assert beziers[0].p0[1] == pytest.approx(10.2, abs=0.01)
        
        # End point should match exactly
        assert beziers[-1].p3[0] == pytest.approx(35.7, abs=0.01)
        assert beziers[-1].p3[1] == pytest.approx(22.3, abs=0.01)
    
    def test_two_point_line(self):
        """Test fitting with only two points."""
        points = np.array([[0, 0], [100, 100]])
        
        beziers = fit_cubic_beziers(points, error_tolerance=1.0)
        
        assert len(beziers) == 1
        assert beziers[0].p0 == [0, 0]
        assert beziers[0].p3 == [100, 100]
    
    def test_bezier_to_svg_path(self):
        """Test SVG path generation."""
        bez = _line_to_bezier(np.array([0, 0]), np.array([30, 0]))
        
        svg_path = bezier_to_svg_path([bez])
        
        assert svg_path.startswith("M 0.00 0.00")
        assert "C" in svg_path
        assert "30.00 0.00" in svg_path
    
    def test_empty_input(self):
        """Test handling of empty input."""
        beziers = fit_cubic_beziers(np.array([]), error_tolerance=1.0)
        
        assert beziers == []
    
    def test_single_point(self):
        """Test handling of single point input."""
        points = np.array([[10, 20]])
        
        beziers = fit_cubic_beziers(points, error_tolerance=1.0)
        
        assert beziers == []


class TestSimplify:
    """Tests for polyline simplification."""
    
    def test_rdp_reduces_points(self):
        """Test that RDP reduces the number of points."""
        from patentdraw.strokes.simplify import rdp_simplify
        
        # Create a line with many collinear points
        points = [[i, 0] for i in range(100)]
        
        simplified = rdp_simplify(points, epsilon=1.0)
        
        # Should reduce to just 2 points (start and end)
        assert len(simplified) == 2
        assert simplified[0] == [0, 0]
        assert simplified[-1] == [99, 0]
    
    def test_rdp_preserves_endpoints(self):
        """Test that RDP preserves exact endpoints."""
        from patentdraw.strokes.simplify import rdp_simplify
        
        points = [[5.5, 10.2], [10, 15], [15, 10], [20.3, 5.7]]
        
        simplified = rdp_simplify(points, epsilon=1.0)
        
        assert simplified[0] == [5.5, 10.2]
        assert simplified[-1] == [20.3, 5.7]
    
    def test_rdp_preserves_corners(self):
        """Test that RDP preserves significant corners."""
        from patentdraw.strokes.simplify import rdp_simplify
        
        # L-shape
        points = [[0, 0], [0, 10], [0, 20], [0, 30], [0, 40], [0, 50],
                  [10, 50], [20, 50], [30, 50], [40, 50], [50, 50]]
        
        simplified = rdp_simplify(points, epsilon=1.0)
        
        # Should preserve at least the corner at [0, 50]
        assert len(simplified) >= 3
        corner_found = any(
            abs(p[0] - 0) < 2 and abs(p[1] - 50) < 2 
            for p in simplified
        )
        assert corner_found
