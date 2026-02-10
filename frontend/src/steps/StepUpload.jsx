import React, { useRef, useState, useCallback } from "react";

export default function StepUpload({
  onVectorized,
  setLoading,
  setLoadingMsg,
  setError,
}) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setError(null);
  }, [setError]);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files?.[0];
      if (f && f.type.startsWith("image/")) handleFile(f);
    },
    [handleFile]
  );

  const handleConvert = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setLoadingMsg("Vectorizing sketch — this may take 30-60 seconds…");
    setError(null);

    try {
      const form = new FormData();
      form.append("image", file);

      const res = await fetch("/api/vectorize", { method: "POST", body: form });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      onVectorized(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [file, onVectorized, setLoading, setLoadingMsg, setError]);

  return (
    <div className="step-panel">
      {/* Left: Patent drawing checklist */}
      <aside className="patent-checklist">
        <h3>Utility Patent Drawing Checklist</h3>
        <p className="checklist-intro">
          Include enough views to clearly show how the invention is structured
          and how it works. At a minimum, your drawings should display every
          claimed feature.
        </p>

        <h4>Required in most cases</h4>
        <ul>
          <li>Front view</li>
          <li>Side view</li>
          <li>Top view</li>
        </ul>

        <h4>Include when applicable</h4>
        <ul>
          <li>Bottom view (only if it shows features not visible elsewhere)</li>
          <li>Isometric view (recommended for clear 3D understanding)</li>
          <li>Exploded view (if the invention has multiple parts)</li>
          <li>Section or cross-section views (if internal components affect function)</li>
          <li>Detail views (for small or complex features)</li>
        </ul>

        <h4>Drawing standards</h4>
        <ul>
          <li>Black-and-white line drawings</li>
          <li>Consistent reference numbers across all figures</li>
          <li>No unnecessary shading or perspective distortion</li>
          <li>Each figure labeled and referenced in the description</li>
        </ul>

        <div className="checklist-rule">
          <strong>Rule of thumb:</strong> If a feature is claimed, it must appear
          clearly in at least one drawing.
        </div>
      </aside>

      {/* Right: Upload area */}
      <div className="upload-area">
        <h2>Upload Patent Sketch</h2>
        <p>Upload a pencil sketch (PNG or JPG) to convert to a clean vector SVG.</p>

        <div
          className={`drop-zone ${dragging ? "dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
          {preview ? (
            <img src={preview} alt="Preview" className="preview-thumb" />
          ) : (
            <div className="label">
              Drag &amp; drop or <strong>browse</strong>
            </div>
          )}
        </div>

        {file && (
          <div className="upload-actions">
            <button
              className="btn btn-secondary"
              onClick={() => {
                setFile(null);
                setPreview(null);
              }}
            >
              Clear
            </button>
            <button className="btn btn-primary" onClick={handleConvert}>
              Convert to Vector
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
