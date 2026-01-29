"""Tests for Stage 2 binarization."""

import cv2
import numpy as np
import pytest


class TestBinarize:
    """Tests for the binarize function."""
    
    def test_grayscale_conversion(self, simple_rectangle_image, default_config):
        """Test that RGB input is converted to grayscale."""
        from patentdraw.preprocess.stage2_binarize import binarize
        
        binary = binarize(simple_rectangle_image, default_config)
        
        # Result should be 2D (grayscale/binary)
        assert len(binary.shape) == 2
    
    def test_binary_output_values(self, simple_rectangle_image, default_config):
        """Test that output only contains 0 and 255."""
        from patentdraw.preprocess.stage2_binarize import binarize
        
        binary = binarize(simple_rectangle_image, default_config)
        
        unique_values = np.unique(binary)
        assert set(unique_values).issubset({0, 255})
    
    def test_rectangle_detected(self, simple_rectangle_image, default_config):
        """Test that a rectangle is detected as foreground pixels."""
        from patentdraw.preprocess.stage2_binarize import binarize
        
        binary = binarize(simple_rectangle_image, default_config)
        
        # Should have some foreground (255) pixels
        foreground_count = np.sum(binary > 0)
        assert foreground_count > 100
    
    def test_otsu_method(self, simple_rectangle_image, default_config):
        """Test Otsu thresholding method."""
        from patentdraw.preprocess.stage2_binarize import binarize
        
        default_config.binarization.method = "otsu"
        binary = binarize(simple_rectangle_image, default_config)
        
        assert binary is not None
        assert binary.dtype == np.uint8
    
    def test_adaptive_method(self, simple_rectangle_image, default_config):
        """Test adaptive thresholding method."""
        from patentdraw.preprocess.stage2_binarize import binarize
        
        default_config.binarization.method = "adaptive"
        binary = binarize(simple_rectangle_image, default_config)
        
        assert binary is not None
        assert binary.dtype == np.uint8
    
    def test_morphology_removes_noise(self, default_config):
        """Test that morphology operations remove small noise."""
        from patentdraw.preprocess.stage2_binarize import binarize
        
        # Create image with small noise dots
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # Add a line
        cv2.line(img, (10, 50), (90, 50), (0, 0, 0), 2)
        # Add small noise dot
        img[20, 20] = [0, 0, 0]
        
        default_config.binarization.morph_kernel = 3
        binary = binarize(img, default_config)
        
        # The line should be preserved
        line_region = binary[45:55, 10:90]
        assert np.sum(line_region > 0) > 50
