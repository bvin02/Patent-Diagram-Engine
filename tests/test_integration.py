"""Integration tests for the full pipeline."""

import os

import cv2
import numpy as np
import pytest


class TestIntegration:
    """Integration tests that run the full pipeline."""
    
    def test_pipeline_creates_debug_artifacts(self, temp_dir):
        """Test that pipeline creates expected debug files."""
        from patentdraw.config import PipelineConfig
        from patentdraw.pipeline import run_pipeline
        
        # Create synthetic input
        img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 50), (150, 100), (0, 0, 0), 2)
        cv2.rectangle(img, (200, 50), (350, 100), (0, 0, 0), 2)
        
        input_path = os.path.join(temp_dir, "test_input.png")
        cv2.imwrite(input_path, img)
        
        out_dir = os.path.join(temp_dir, "output")
        
        config = PipelineConfig()
        config.debug.enabled = True
        
        # Run pipeline
        document = run_pipeline(
            input_paths=[input_path],
            out_dir=out_dir,
            config=config,
            debug=True,
        )
        
        # Check outputs exist
        assert os.path.exists(os.path.join(out_dir, "scene.json"))
        assert os.path.exists(os.path.join(out_dir, "final.svg"))
        assert os.path.exists(os.path.join(out_dir, "validation_report.json"))
        
        # Check debug directory structure
        debug_dir = os.path.join(out_dir, "debug")
        assert os.path.isdir(debug_dir)
        
        # Should have at least one view directory
        view_dirs = [d for d in os.listdir(debug_dir) if os.path.isdir(os.path.join(debug_dir, d))]
        assert len(view_dirs) >= 1
        
        view_dir = os.path.join(debug_dir, view_dirs[0])
        
        # Stage 2 artifacts
        stage2_dir = os.path.join(view_dir, "stage2")
        if os.path.isdir(stage2_dir):
            expected_stage2 = ["01_gray.png", "02_binary.png"]
            for f in expected_stage2:
                assert os.path.exists(os.path.join(stage2_dir, f)), f"Missing {f}"
    
    def test_pipeline_produces_document(self, temp_dir):
        """Test that pipeline produces valid document structure."""
        from patentdraw.config import PipelineConfig
        from patentdraw.pipeline import run_pipeline
        
        # Create simple input
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        cv2.line(img, (50, 100), (250, 100), (0, 0, 0), 2)
        
        input_path = os.path.join(temp_dir, "line_input.png")
        cv2.imwrite(input_path, img)
        
        out_dir = os.path.join(temp_dir, "output")
        
        document = run_pipeline(
            input_paths=[input_path],
            out_dir=out_dir,
            debug=False,
        )
        
        # Verify document structure
        assert document.doc_id is not None
        assert len(document.views) == 1
        assert len(document.inputs) == 1
        
        # Should have detected at least one component
        assert len(document.component_registry) >= 1
        
        # Should have numbering
        assert len(document.numbering_registry) >= 1
    
    def test_pipeline_deterministic(self, temp_dir):
        """Test that running pipeline twice produces identical results."""
        from patentdraw.config import PipelineConfig
        from patentdraw.pipeline import run_pipeline
        import json
        
        # Create input
        img = np.ones((150, 200, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (30, 30), (170, 120), (0, 0, 0), 2)
        
        input_path = os.path.join(temp_dir, "rect_input.png")
        cv2.imwrite(input_path, img)
        
        out_dir1 = os.path.join(temp_dir, "run1")
        out_dir2 = os.path.join(temp_dir, "run2")
        
        config = PipelineConfig()
        
        doc1 = run_pipeline([input_path], out_dir1, config=config, debug=False)
        doc2 = run_pipeline([input_path], out_dir2, config=config, debug=False)
        
        # Compare key properties
        assert doc1.doc_id == doc2.doc_id
        assert list(doc1.component_registry.keys()) == list(doc2.component_registry.keys())
        assert doc1.numbering_registry == doc2.numbering_registry
    
    def test_multiple_inputs(self, temp_dir):
        """Test pipeline with multiple input images."""
        from patentdraw.pipeline import run_pipeline
        
        # Create two inputs
        img1 = np.ones((100, 150, 3), dtype=np.uint8) * 255
        cv2.circle(img1, (75, 50), 30, (0, 0, 0), 2)
        
        img2 = np.ones((100, 150, 3), dtype=np.uint8) * 255
        cv2.rectangle(img2, (30, 20), (120, 80), (0, 0, 0), 2)
        
        path1 = os.path.join(temp_dir, "input1.png")
        path2 = os.path.join(temp_dir, "input2.png")
        cv2.imwrite(path1, img1)
        cv2.imwrite(path2, img2)
        
        out_dir = os.path.join(temp_dir, "multi_output")
        
        document = run_pipeline([path1, path2], out_dir, debug=False)
        
        assert len(document.views) == 2
        assert len(document.inputs) == 2
