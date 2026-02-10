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
