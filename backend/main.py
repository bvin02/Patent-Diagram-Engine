"""
Patent Diagram Generator — FastAPI Backend

Serves the React frontend, Method-Draw editor, and provides API
endpoints that wrap the existing Python pipeline stages.

Endpoints:
    POST /api/vectorize        Upload image → run stages 0-7 → return SVG
    POST /api/label             SVG + anchors → run stages 9-10 → return labeled SVG
    GET  /api/runs/{run_id}/... Browse run artifacts
"""

import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
RUNS_DIR = PROJECT_ROOT / "runs"
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "dist"
METHOD_DRAW_DIR = PROJECT_ROOT / "Method-Draw" / "src"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Patent Diagram Generator", version="1.0.0")

# CORS — allow local dev (Vite dev server on :5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_pipeline_stages(
    input_image: Path,
    stages: str,
    *,
    debug: bool = True,
    from_stage: Optional[str] = None,
) -> Path:
    """
    Run the pipeline via subprocess, return the run directory.

    ``stages`` is a comma-separated list like "0,1,2,3,4,5,6,7"
    or None to run all.
    """
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_pipeline.py"),
        str(input_image),
    ]
    if stages:
        cmd += ["--only-stages"] + stages.split(",")
    if from_stage:
        cmd += ["--from-stage", from_stage]
    if not debug:
        cmd.append("--no-debug")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed (exit {result.returncode}):\n{result.stderr}\n{result.stdout}"
        )

    # Parse run directory from stage-0 output.
    # The path printed is relative to PROJECT_ROOT (the subprocess cwd),
    # so resolve it against PROJECT_ROOT, not the backend's own cwd.
    for line in result.stdout.splitlines():
        if line.startswith("Created run directory:"):
            raw = line.split(":", 1)[1].strip()
            resolved = (PROJECT_ROOT / raw).resolve()
            if resolved.exists():
                return resolved

    # Fallback: infer from input name
    slug = input_image.stem.lower().replace(" ", "_")
    candidate = RUNS_DIR / slug
    if candidate.exists():
        return candidate

    raise RuntimeError(
        f"Could not determine run directory from pipeline output.\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )


def _find_run_dir(run_id: str) -> Path:
    """Resolve a run_id to an existing run directory."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run_dir


def _run_label_stages(
    run_dir: Path,
    components_override: Optional[dict] = None,
):
    """
    Run label pipeline (stages 8-10) inside an existing run directory.

    If ``components_override`` is given, write it to the stage-8 output
    directory so stages 9-10 pick it up.
    """
    mask_path = run_dir / "10_preprocess" / "out" / "output_mask.png"
    svg_path = run_dir / "70_svg" / "out" / "output.svg"

    if not mask_path.exists():
        raise HTTPException(status_code=400, detail="Run is missing preprocessed mask")
    if not svg_path.exists():
        raise HTTPException(status_code=400, detail="Run is missing output.svg")

    # --- Stage 8: label identify ---------------------------------------------------
    cmd_8 = [
        sys.executable,
        str(PROJECT_ROOT / "label_identify.py"),
        str(mask_path),
        "--svg", str(svg_path),
        "--debug",
    ]
    r8 = subprocess.run(cmd_8, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if r8.returncode != 0:
        raise RuntimeError(f"Stage 8 failed:\n{r8.stderr}\n{r8.stdout}")

    # If caller supplied custom anchor positions, patch components.json
    # while preserving the metadata (image size, paths, config) that
    # label_leaders.py needs.
    components_json = run_dir / "80_label_identify" / "out" / "components.json"
    if components_override is not None:
        existing = {}
        if components_json.exists():
            existing = json.loads(components_json.read_text())
        existing["components"] = (
            components_override if isinstance(components_override, list)
            else components_override.get("components", components_override)
        )
        components_json.write_text(json.dumps(existing, indent=2))

    # --- Stage 9: leader routing ---------------------------------------------------
    cmd_9 = [
        sys.executable,
        str(PROJECT_ROOT / "label_leaders.py"),
        str(components_json),
        "--mask", str(mask_path),
        "--debug",
    ]
    r9 = subprocess.run(cmd_9, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if r9.returncode != 0:
        raise RuntimeError(f"Stage 9 failed:\n{r9.stderr}\n{r9.stdout}")

    # --- Stage 10: label placement -------------------------------------------------
    leaders_json = run_dir / "90_label_leaders" / "out" / "leaders.json"
    cmd_10 = [
        sys.executable,
        str(PROJECT_ROOT / "label_place.py"),
        str(leaders_json),
        "--svg", str(svg_path),
        "--mask", str(mask_path),
        "--debug",
    ]
    r10 = subprocess.run(cmd_10, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if r10.returncode != 0:
        raise RuntimeError(f"Stage 10 failed:\n{r10.stderr}\n{r10.stdout}")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/vectorize")
async def vectorize(image: UploadFile = File(...)):
    """
    Upload an image and run the vectorization pipeline (stages 0-7).

    Returns JSON with ``run_id`` and ``svg`` (the output SVG string).
    """
    # Save upload to a temp file
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await image.read()
        tmp.write(content)
        tmp.close()

        run_dir = _run_pipeline_stages(
            Path(tmp.name),
            stages="0,1,2,3,4,5,6,7",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    # Read the output SVG
    svg_path = run_dir / "70_svg" / "out" / "output.svg"
    if not svg_path.exists():
        raise HTTPException(status_code=500, detail="Pipeline produced no SVG")

    svg_text = svg_path.read_text()
    run_id = run_dir.name

    return JSONResponse({
        "run_id": run_id,
        "svg": svg_text,
    })


@app.post("/api/save-edited-svg")
async def save_edited_svg(run_id: str = Form(...), svg: str = Form(...)):
    """
    Save the user-edited SVG back into the run directory so that
    subsequent labelling stages use the edited version.
    """
    run_dir = _find_run_dir(run_id)
    svg_out = run_dir / "70_svg" / "out" / "output.svg"
    svg_out.write_text(svg)

    return JSONResponse({"status": "ok"})


@app.post("/api/identify-components")
async def identify_components(run_id: str = Form(...)):
    """
    Run stage 8 (component identification) and return the list of
    detected anchor points so the frontend can show them for editing.
    """
    run_dir = _find_run_dir(run_id)
    mask_path = run_dir / "10_preprocess" / "out" / "output_mask.png"
    svg_path = run_dir / "70_svg" / "out" / "output.svg"

    if not mask_path.exists():
        raise HTTPException(status_code=400, detail="Missing preprocessed mask")
    if not svg_path.exists():
        raise HTTPException(status_code=400, detail="Missing output.svg")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "label_identify.py"),
        str(mask_path),
        "--svg", str(svg_path),
        "--debug",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if r.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Stage 8 failed:\n{r.stderr}")

    components_json = run_dir / "80_label_identify" / "out" / "components.json"
    if not components_json.exists():
        raise HTTPException(status_code=500, detail="No components.json produced")

    raw = json.loads(components_json.read_text())
    # components.json wraps the list in {"components": [...]};
    # unwrap so the response is {"components": [...]}, not double-nested.
    comp_list = raw["components"] if isinstance(raw, dict) and "components" in raw else raw
    return JSONResponse({"components": comp_list})


@app.post("/api/label")
async def label(
    run_id: str = Form(...),
    anchors: str = Form(...),
):
    """
    Run the full labelling pipeline (stages 9-10) with user-supplied
    anchor positions.

    ``anchors`` is a JSON string: list of {id, anchor_x, anchor_y, ...}
    matching the components.json schema.

    Returns the final labeled SVG and a labels-only SVG.
    """
    run_dir = _find_run_dir(run_id)

    try:
        anchor_data = json.loads(anchors)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid anchors JSON")

    try:
        _run_label_stages(run_dir, components_override=anchor_data)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    labelled_svg = run_dir / "100_label_place" / "out" / "labelled.svg"
    labels_only = run_dir / "100_label_place" / "out" / "labels_only.svg"

    if not labelled_svg.exists():
        raise HTTPException(status_code=500, detail="No labelled.svg produced")

    result = {
        "labelled_svg": labelled_svg.read_text(),
    }
    if labels_only.exists():
        result["labels_only_svg"] = labels_only.read_text()

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Run artifact browsing
# ---------------------------------------------------------------------------

@app.get("/api/runs")
async def list_runs():
    """List all run directories."""
    if not RUNS_DIR.exists():
        return JSONResponse({"runs": []})
    runs = sorted(
        [d.name for d in RUNS_DIR.iterdir() if d.is_dir()],
        reverse=True,
    )
    return JSONResponse({"runs": runs})


@app.get("/api/runs/{run_id}/files")
async def list_run_files(run_id: str):
    """List all files in a run directory (recursive)."""
    run_dir = _find_run_dir(run_id)
    files = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            files.append(str(p.relative_to(run_dir)))
    return JSONResponse({"files": files})


@app.get("/api/runs/{run_id}/file/{file_path:path}")
async def get_run_file(run_id: str, file_path: str):
    """Serve a single file from a run directory."""
    run_dir = _find_run_dir(run_id)
    target = (run_dir / file_path).resolve()

    # Security: must be inside run_dir
    if not str(target).startswith(str(run_dir)):
        raise HTTPException(status_code=403, detail="Path traversal denied")
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(target)


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

# Serve Method Draw at /editor/method-draw/
if METHOD_DRAW_DIR.exists():
    app.mount(
        "/editor/method-draw",
        StaticFiles(directory=str(METHOD_DRAW_DIR), html=True),
        name="method-draw",
    )

# Serve React frontend at / (must be last mount)
if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)],
    )
