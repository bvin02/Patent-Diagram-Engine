"""
Artifact Manager for Sketch-to-SVG Pipeline

Provides run-scoped output management so artifacts never overwrite across
different inputs. Each stage writes to its own numbered subfolder.

Usage:
    run_dir = make_run_dir("input.png")
    artifacts = StageArtifacts(run_dir, 10, "preprocess", debug=True)
    artifacts.save_debug_image("threshold", binary_img)
    artifacts.save_output_image("result", final_img)
    artifacts.write_metrics({"lines": 42, "corners": 8})
"""

import json
import shutil
import cv2
import numpy as np
from pathlib import Path

from .io import ensure_uint8


def slugify(name: str) -> str:
    """
    Turn any filename or run name into a safe folder name.
    
    Converts to lowercase, keeps only alphanumeric and underscores.
    Replaces spaces and dashes with underscores. Collapses consecutive underscores.
    
    Args:
        name: Input string to slugify.
        
    Returns:
        Safe folder name string.
    """
    # Lowercase
    result = name.lower()
    
    # Replace spaces and dashes with underscores
    result = result.replace(" ", "_").replace("-", "_")
    
    # Keep only alphanumeric and underscores
    result = "".join(c if c.isalnum() or c == "_" else "_" for c in result)
    
    # Collapse consecutive underscores
    while "__" in result:
        result = result.replace("__", "_")
    
    # Strip leading/trailing underscores
    result = result.strip("_")
    
    return result if result else "unnamed"


def make_run_dir(input_path: str, runs_root: str = "runs") -> Path:
    """
    Create a unique run directory based on the input filename.
    
    Creates: runs/<stem_slug>/
    If exists, appends _2, _3, etc.
    Copies input image to runs/<run>/00_input/ for traceability.
    
    Args:
        input_path: Path to the input image.
        runs_root: Root directory for all runs (default: "runs").
        
    Returns:
        Path to the created run directory.
    """
    input_p = Path(input_path)
    runs_root_p = Path(runs_root)
    
    # Create base slug from input stem
    base_slug = slugify(input_p.stem)
    
    # Find unique run directory name
    run_dir = runs_root_p / base_slug
    if run_dir.exists():
        suffix = 2
        while True:
            run_dir = runs_root_p / f"{base_slug}_{suffix}"
            if not run_dir.exists():
                break
            suffix += 1
    
    # Create run directory and 00_input subfolder
    input_dir = run_dir / "00_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy input image to 00_input/01_input.<ext>
    input_copy_path = input_dir / f"01_input{input_p.suffix}"
    shutil.copy2(str(input_p), str(input_copy_path))
    
    return run_dir


class StageArtifacts:
    """
    Manages artifact output for a single pipeline stage.
    
    Creates a stage-specific directory with debug/ and out/ subfolders.
    Provides methods to save numbered debug images, output images, JSON, and numpy arrays.
    """
    
    def __init__(
        self,
        run_dir: Path,
        stage_id: int,
        stage_name: str,
        debug: bool = True,
    ):
        """
        Initialize stage artifacts manager.
        
        Args:
            run_dir: Path to the run directory.
            stage_id: Numeric ID for ordering stages (e.g., 10, 20, 30).
            stage_name: Human-readable stage name.
            debug: If True, debug images are written. If False, save_debug_image is a no-op.
        """
        self.run_dir = Path(run_dir)
        self.stage_id = stage_id
        self.stage_name = stage_name
        self.debug_enabled = debug
        
        # Create stage directory: <run_dir>/<stage_id:02d>_<stage_name_slug>/
        stage_slug = slugify(stage_name)
        self.stage_dir = self.run_dir / f"{stage_id:02d}_{stage_slug}"
        
        # Create subdirectories
        self.debug_dir = self.stage_dir / "debug"
        self.out_dir = self.stage_dir / "out"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Counter for debug image numbering
        self._debug_counter = 0
    
    def save_debug_image(self, name: str, img: np.ndarray) -> Path:
        """
        Save a debug image with auto-incrementing numbering.
        
        If debug is disabled, returns the would-be path without writing.
        If img is float, normalizes to 0..255 uint8 for visualization.
        
        Args:
            name: Base name for the image file.
            img: Image array to save.
            
        Returns:
            Path where the image was (or would be) saved.
        """
        self._debug_counter += 1
        filename = f"{self._debug_counter:02d}_{name}.png"
        save_path = self.debug_dir / filename
        
        if self.debug_enabled:
            img_uint8 = ensure_uint8(img)
            cv2.imwrite(str(save_path), img_uint8)
        
        return save_path
    
    def save_output_image(self, name: str, img: np.ndarray) -> Path:
        """
        Save an output image (no numbering).
        
        Args:
            name: Filename for the image (without extension).
            img: Image array to save.
            
        Returns:
            Path where the image was saved.
        """
        save_path = self.out_dir / f"{name}.png"
        img_uint8 = ensure_uint8(img)
        cv2.imwrite(str(save_path), img_uint8)
        return save_path
    
    def save_json(self, name: str, data: dict, debug: bool = False) -> Path:
        """
        Save a dictionary as JSON.
        
        Args:
            name: Filename for the JSON (without extension).
            data: Dictionary to serialize.
            debug: If True, save to debug/ with numbering. Else save to out/.
            
        Returns:
            Path where the JSON was saved.
        """
        if debug:
            self._debug_counter += 1
            filename = f"{self._debug_counter:02d}_{name}.json"
            save_path = self.debug_dir / filename
        else:
            save_path = self.out_dir / f"{name}.json"
        
        json_str = json.dumps(data, indent=2, sort_keys=True)
        save_path.write_text(json_str)
        
        return save_path
    
    def save_npy(self, name: str, arr: np.ndarray) -> Path:
        """
        Save a numpy array to the output directory.
        
        Args:
            name: Filename for the array (without extension).
            arr: Numpy array to save.
            
        Returns:
            Path where the array was saved.
        """
        save_path = self.out_dir / f"{name}.npy"
        np.save(str(save_path), arr)
        return save_path
    
    def write_metrics(self, metrics: dict) -> Path:
        """
        Save metrics to out/metrics.json.
        
        Args:
            metrics: Dictionary of metrics to save.
            
        Returns:
            Path where metrics.json was saved.
        """
        save_path = self.out_dir / "metrics.json"
        json_str = json.dumps(metrics, indent=2, sort_keys=True)
        save_path.write_text(json_str)
        return save_path
    
    def path_out(self, filename: str) -> Path:
        """
        Get the full path for a file in the output directory.
        
        Args:
            filename: Filename to append to out/.
            
        Returns:
            Full path to out/<filename>.
        """
        return self.out_dir / filename
