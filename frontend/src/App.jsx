import React, { useState, useCallback } from "react";
import StepUpload from "./steps/StepUpload.jsx";
import StepEdit from "./steps/StepEdit.jsx";
import StepAnchors from "./steps/StepAnchors.jsx";
import StepReview from "./steps/StepReview.jsx";

const STEPS = [
  { id: 1, label: "Upload" },
  { id: 2, label: "Edit SVG" },
  { id: 3, label: "Anchors" },
  { id: 4, label: "Review" },
];

export default function App() {
  // ── wizard state ──────────────────────────────────────────────
  const [step, setStep] = useState(1);
  const [runId, setRunId] = useState(null);
  const [vectorSvg, setVectorSvg] = useState(null);
  const [editedSvg, setEditedSvg] = useState(null);
  const [components, setComponents] = useState(null);
  const [labelledSvg, setLabelledSvg] = useState(null);
  const [labelsOnlySvg, setLabelsOnlySvg] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");
  const [error, setError] = useState(null);

  // ── Step 1 → 2 callback ──────────────────────────────────────
  const onVectorized = useCallback((data) => {
    setRunId(data.run_id);
    setVectorSvg(data.svg);
    setEditedSvg(null);
    setComponents(null);
    setLabelledSvg(null);
    setLabelsOnlySvg(null);
    setStep(2);
  }, []);

  // ── Step 2 → 3 callback ──────────────────────────────────────
  const onDoneEditing = useCallback((svg) => {
    setEditedSvg(svg);
    setStep(3);
  }, []);

  // ── Step 3 → 4 callback ──────────────────────────────────────
  const onLabelled = useCallback((data) => {
    setLabelledSvg(data.labelled_svg);
    setLabelsOnlySvg(data.labels_only_svg || null);
    setStep(4);
  }, []);

  // ── Reset (start over) ───────────────────────────────────────
  const reset = useCallback(() => {
    setStep(1);
    setRunId(null);
    setVectorSvg(null);
    setEditedSvg(null);
    setComponents(null);
    setLabelledSvg(null);
    setLabelsOnlySvg(null);
    setError(null);
  }, []);

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <h1>Patent Diagram Generator</h1>
        {runId && <span className="run-id">Run: {runId}</span>}
      </header>

      {/* Stepper */}
      <nav className="stepper">
        {STEPS.map((s, i) => (
          <React.Fragment key={s.id}>
            {i > 0 && <div className="step-connector" />}
            <div
              className={`step-dot ${
                step === s.id ? "active" : step > s.id ? "done" : ""
              }`}
            >
              <span className="step-number">
                {step > s.id ? "✓" : s.id}
              </span>
              <span>{s.label}</span>
            </div>
          </React.Fragment>
        ))}
      </nav>

      {/* Loading overlay */}
      {loading && (
        <div className="spinner-overlay">
          <div className="spinner" />
          <p>{loadingMsg || "Processing…"}</p>
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div style={{ padding: "0 24px", marginTop: 12 }}>
          <div className="error-banner">
            {error}
            <button
              className="btn btn-sm btn-secondary"
              style={{ marginLeft: 12 }}
              onClick={() => setError(null)}
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Step panels */}
      <div className="app-body">
        {step === 1 && (
          <StepUpload
            onVectorized={onVectorized}
            setLoading={setLoading}
            setLoadingMsg={setLoadingMsg}
            setError={setError}
          />
        )}
        {step === 2 && (
          <StepEdit
            runId={runId}
            vectorSvg={vectorSvg}
            onDone={onDoneEditing}
            setLoading={setLoading}
            setLoadingMsg={setLoadingMsg}
            setError={setError}
          />
        )}
        {step === 3 && (
          <StepAnchors
            runId={runId}
            editedSvg={editedSvg}
            components={components}
            setComponents={setComponents}
            onLabelled={onLabelled}
            setLoading={setLoading}
            setLoadingMsg={setLoadingMsg}
            setError={setError}
            onBack={() => setStep(2)}
          />
        )}
        {step === 4 && (
          <StepReview
            runId={runId}
            labelledSvg={labelledSvg}
            labelsOnlySvg={labelsOnlySvg}
            onStartOver={reset}
          />
        )}
      </div>

      <footer className="disclaimer">
        This tool is provided as-is for reference purposes only and does not replace the expertise of a professional patent drafter.
      </footer>
    </div>
  );
}
