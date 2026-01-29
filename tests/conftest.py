"""Pytest fixtures for Patent Draw tests."""

import os
import tempfile

import cv2
import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_rectangle_image():
    """Create a simple white image with a black rectangle."""
    img = np.ones((200, 300, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (250, 150), (0, 0, 0), 2)
    return img


@pytest.fixture
def simple_line_image():
    """Create a simple white image with a single black line."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    cv2.line(img, (20, 50), (180, 50), (0, 0, 0), 2)
    return img


@pytest.fixture
def l_shape_image():
    """Create an L-shaped line drawing."""
    img = np.ones((150, 150, 3), dtype=np.uint8) * 255
    # Vertical line
    cv2.line(img, (50, 20), (50, 100), (0, 0, 0), 2)
    # Horizontal line
    cv2.line(img, (50, 100), (130, 100), (0, 0, 0), 2)
    return img


@pytest.fixture
def two_rectangles_image():
    """Create two separate rectangles for testing component grouping."""
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (30, 50), (130, 150), (0, 0, 0), 2)
    cv2.rectangle(img, (250, 50), (350, 150), (0, 0, 0), 2)
    return img


@pytest.fixture
def default_config():
    """Create default pipeline configuration."""
    from patentdraw.config import PipelineConfig
    return PipelineConfig()


@pytest.fixture
def synthetic_input_file(temp_dir, simple_rectangle_image):
    """Create a synthetic input file for integration tests."""
    path = os.path.join(temp_dir, "test_input.png")
    cv2.imwrite(path, cv2.cvtColor(simple_rectangle_image, cv2.COLOR_RGB2BGR))
    return path
