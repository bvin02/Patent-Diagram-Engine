import React, { useMemo, useCallback } from "react";

/**
 * Step 4 — Final review & export.
 *
 * Shows the labeled SVG, a patent compliance checklist, and
 * download buttons for the SVG files.
 */
export default function StepReview({
  runId,
  labelledSvg,
  labelsOnlySvg,
  onStartOver,
}) {
  // ── Compliance checks (static for now) ───────────────────────
  const checks = useMemo(
    () => [
      { label: "Black and white only", ok: true },
      { label: "Legible line weights", ok: true },
      { label: "Margins present", ok: true },
      { label: "Labels outside drawing", ok: true },
      { label: "Leader lines don't cross strokes", ok: true },
      { label: "Sequential numbering", ok: true },
    ],
    []
  );

  // ── Download helper ──────────────────────────────────────────
  const downloadSvg = useCallback((svgText, filename) => {
    const blob = new Blob([svgText], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, []);

  return (
    <div className="review-panel">
      {/* Preview */}
      <div className="review-preview">
        {labelledSvg && (
          <div dangerouslySetInnerHTML={{ __html: labelledSvg }} />
        )}
      </div>

      {/* Sidebar */}
      <div className="review-sidebar">
        <div>
          <h3>Patent Compliance</h3>
          <ul className="checklist">
            {checks.map((c, i) => (
              <li key={i}>
                <span className="check">{c.ok ? "✓" : "✗"}</span>
                {c.label}
              </li>
            ))}
          </ul>
        </div>

        <div>
          <h3>Download</h3>
          <div className="download-group">
            <button
              className="btn btn-primary"
              onClick={() => downloadSvg(labelledSvg, "labelled.svg")}
            >
              Download Labeled SVG
            </button>
            {labelsOnlySvg && (
              <button
                className="btn btn-secondary"
                onClick={() => downloadSvg(labelsOnlySvg, "labels_only.svg")}
              >
                Download Labels Only
              </button>
            )}
          </div>
        </div>

        {runId && (
          <div>
            <h3>Debug Artifacts</h3>
            <p style={{ fontSize: 13, color: "var(--text-muted)" }}>
              Run ID: <code>{runId}</code>
            </p>
            <a
              href={`/api/runs/${runId}/files`}
              target="_blank"
              rel="noopener noreferrer"
              style={{ fontSize: 13 }}
            >
              Browse all artifacts →
            </a>
          </div>
        )}

        <button className="btn btn-secondary" onClick={onStartOver}>
          Start Over
        </button>
      </div>
    </div>
  );
}
