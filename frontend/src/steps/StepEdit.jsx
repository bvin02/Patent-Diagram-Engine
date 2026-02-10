import React, { useRef, useEffect, useState, useCallback } from "react";

/**
 * Step 2 — Embedded Method Draw editor.
 *
 * Loads the pipeline-generated SVG into Method Draw via the postMessage
 * bridge, lets the user edit freely, and extracts the SVG when they click
 * "Done Editing".
 */
export default function StepEdit({
  runId,
  vectorSvg,
  onDone,
  setLoading,
  setLoadingMsg,
  setError,
}) {
  const iframeRef = useRef(null);
  const [bridgeReady, setBridgeReady] = useState(false);
  const pendingResolveRef = useRef(null);

  // ── Listen for messages from the iframe ──────────────────────
  useEffect(() => {
    function onMessage(e) {
      if (!e.data || typeof e.data.type !== "string") return;

      if (e.data.type === "BRIDGE_READY") {
        setBridgeReady(true);
      }

      if (e.data.type === "CURRENT_SVG") {
        if (pendingResolveRef.current) {
          pendingResolveRef.current(e.data.svg);
          pendingResolveRef.current = null;
        }
      }
    }

    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);

  // ── Once the bridge is ready, push SVG into the editor ───────
  useEffect(() => {
    if (bridgeReady && vectorSvg && iframeRef.current) {
      iframeRef.current.contentWindow.postMessage(
        { type: "LOAD_SVG", svg: vectorSvg },
        "*"
      );
    }
  }, [bridgeReady, vectorSvg]);

  // ── Request SVG back from the editor ─────────────────────────
  const requestSvg = useCallback(() => {
    return new Promise((resolve, reject) => {
      if (!iframeRef.current) return reject(new Error("No iframe"));
      pendingResolveRef.current = resolve;
      iframeRef.current.contentWindow.postMessage(
        { type: "REQUEST_SVG" },
        "*"
      );
      // Timeout after 5 seconds
      setTimeout(() => {
        if (pendingResolveRef.current) {
          pendingResolveRef.current = null;
          reject(new Error("Timed out waiting for SVG from editor"));
        }
      }, 5000);
    });
  }, []);

  // ── "Done Editing" button handler ────────────────────────────
  const handleDone = useCallback(async () => {
    setError(null);
    setLoading(true);
    setLoadingMsg("Extracting SVG from editor…");

    try {
      const svg = await requestSvg();

      // Save the edited SVG back to the run
      const form = new FormData();
      form.append("run_id", runId);
      form.append("svg", svg);

      const res = await fetch("/api/save-edited-svg", {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || "Failed to save edited SVG");
      }

      onDone(svg);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [runId, requestSvg, onDone, setLoading, setLoadingMsg, setError]);

  return (
    <div className="editor-panel">
      <div className="editor-toolbar">
        <p>
          {bridgeReady
            ? "Edit your SVG in Method Draw below. Click Done when finished."
            : "Loading editor…"}
        </p>
        <button
          className="btn btn-primary"
          disabled={!bridgeReady}
          onClick={handleDone}
        >
          Done Editing →
        </button>
      </div>
      <div className="editor-iframe-wrap">
        <iframe
          ref={iframeRef}
          src="/editor/method-draw/index.html"
          title="Method Draw Editor"
        />
      </div>
    </div>
  );
}
