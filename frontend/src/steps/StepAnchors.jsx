import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
  useMemo,
} from "react";

/**
 * Step 3 — Anchor point selection.
 *
 * Shows the edited SVG with draggable anchor markers overlaid.
 * Users can drag, delete, and add anchor points before running the
 * labelling pipeline (stages 9-10).
 */
export default function StepAnchors({
  runId,
  editedSvg,
  components,
  setComponents,
  onLabelled,
  setLoading,
  setLoadingMsg,
  setError,
  onBack,
}) {
  const containerRef = useRef(null);
  const [anchors, setAnchors] = useState([]);
  const [svgViewBox, setSvgViewBox] = useState(null);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [dragging, setDragging] = useState(null); // index being dragged
  const [panStart, setPanStart] = useState(null);
  const nextIdRef = useRef(100);

  // ── Parse SVG viewBox ────────────────────────────────────────
  useEffect(() => {
    if (!editedSvg) return;
    const match = editedSvg.match(/viewBox=["']([^"']+)["']/);
    if (match) {
      const [x, y, w, h] = match[1].split(/[\s,]+/).map(Number);
      setSvgViewBox({ x, y, w, h });
    } else {
      // Fallback: try width/height
      const wm = editedSvg.match(/width=["'](\d+)/);
      const hm = editedSvg.match(/height=["'](\d+)/);
      const w = wm ? Number(wm[1]) : 800;
      const h = hm ? Number(hm[1]) : 600;
      setSvgViewBox({ x: 0, y: 0, w, h });
    }
  }, [editedSvg]);

  // ── Load components from backend (stage 8) ───────────────────
  useEffect(() => {
    if (components) {
      // Already have components from a previous visit
      setAnchors(
        components.map((c) => ({
          id: c.id,
          x: c.anchor_x,
          y: c.anchor_y,
          area: c.area,
        }))
      );
      return;
    }

    let cancelled = false;
    async function identify() {
      setLoading(true);
      setLoadingMsg("Identifying label anchor points…");
      setError(null);

      try {
        const form = new FormData();
        form.append("run_id", runId);

        const res = await fetch("/api/identify-components", {
          method: "POST",
          body: form,
        });
        if (!res.ok) {
          const d = await res.json().catch(() => ({}));
          throw new Error(d.detail || "Component identification failed");
        }
        const data = await res.json();
        if (!cancelled) {
          setComponents(data.components);
          setAnchors(
            data.components.map((c) => ({
              id: c.id,
              x: c.anchor_x,
              y: c.anchor_y,
              area: c.area,
            }))
          );
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    identify();
    return () => { cancelled = true; };
  }, [runId, components, setComponents, setLoading, setLoadingMsg, setError]);

  // ── Coordinate conversion helpers ────────────────────────────
  const screenToSvg = useCallback(
    (clientX, clientY) => {
      if (!containerRef.current || !svgViewBox) return { x: 0, y: 0 };
      const rect = containerRef.current.getBoundingClientRect();
      const sx = (clientX - rect.left - pan.x) / zoom;
      const sy = (clientY - rect.top - pan.y) / zoom;
      // Map from pixel space to SVG viewBox space
      const scale = svgViewBox.w / (rect.width / zoom);
      return {
        x: svgViewBox.x + sx * scale,
        y: svgViewBox.y + sy * scale,
      };
    },
    [pan, zoom, svgViewBox]
  );

  // ── Zoom via scroll ──────────────────────────────────────────
  const handleWheel = useCallback(
    (e) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setZoom((z) => Math.max(0.1, Math.min(10, z * delta)));
    },
    []
  );

  // ── Pan via middle-click / shift-click drag ──────────────────
  const handleMouseDown = useCallback(
    (e) => {
      // Right-click or shift-click: pan
      if (e.button === 1 || e.shiftKey) {
        e.preventDefault();
        setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
        return;
      }

      // Double-click: add new anchor
      if (e.detail === 2 && svgViewBox) {
        const pos = screenToSvg(e.clientX, e.clientY);
        setAnchors((prev) => [
          ...prev,
          { id: nextIdRef.current++, x: pos.x, y: pos.y, area: 0 },
        ]);
        return;
      }
    },
    [pan, svgViewBox, screenToSvg]
  );

  const handleMouseMove = useCallback(
    (e) => {
      if (panStart) {
        setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
        return;
      }

      if (dragging !== null && svgViewBox) {
        const pos = screenToSvg(e.clientX, e.clientY);
        setAnchors((prev) =>
          prev.map((a, i) => (i === dragging ? { ...a, x: pos.x, y: pos.y } : a))
        );
      }
    },
    [panStart, dragging, svgViewBox, screenToSvg]
  );

  const handleMouseUp = useCallback(() => {
    setPanStart(null);
    setDragging(null);
  }, []);

  // ── Delete an anchor ─────────────────────────────────────────
  const deleteAnchor = useCallback((idx) => {
    setAnchors((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  // ── Submit to label endpoint ─────────────────────────────────
  const handleSubmit = useCallback(async () => {
    setLoading(true);
    setLoadingMsg("Generating labels and leader lines…");
    setError(null);

    try {
      // Build components payload matching the schema of components.json
      const payload = anchors.map((a, i) => ({
        id: a.id,
        anchor_x: Math.round(a.x),
        anchor_y: Math.round(a.y),
        area: a.area || 0,
        bbox: [0, 0, 0, 0],
        boundary_edges: [],
      }));

      const form = new FormData();
      form.append("run_id", runId);
      form.append("anchors", JSON.stringify(payload));

      const res = await fetch("/api/label", { method: "POST", body: form });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || "Labelling failed");
      }
      const data = await res.json();
      onLabelled(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [anchors, runId, onLabelled, setLoading, setLoadingMsg, setError]);

  // ── Render the SVG + overlay ─────────────────────────────────
  // We render the base SVG as a background <img> via data URI, then
  // overlay an interactive SVG with the anchor markers.
  const svgDataUri = useMemo(() => {
    if (!editedSvg) return null;
    return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(editedSvg);
  }, [editedSvg]);

  const viewBoxStr = svgViewBox
    ? `${svgViewBox.x} ${svgViewBox.y} ${svgViewBox.w} ${svgViewBox.h}`
    : "0 0 800 600";

  return (
    <div className="anchor-panel">
      <div className="anchor-toolbar">
        <div>
          <p>
            <strong>{anchors.length}</strong> anchors.{" "}
            <em>Drag to move · Double-click to add · Right-click to delete · Shift-drag to pan · Scroll to zoom</em>
          </p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-secondary" onClick={onBack}>
            ← Back
          </button>
          <button
            className="btn btn-primary"
            disabled={anchors.length === 0}
            onClick={handleSubmit}
          >
            Generate Labels →
          </button>
        </div>
      </div>

      <div
        className="anchor-viewer"
        ref={containerRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {/* Base SVG as image background */}
        <svg
          viewBox={viewBoxStr}
          style={{
            width: svgViewBox ? svgViewBox.w : 800,
            height: svgViewBox ? svgViewBox.h : 600,
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          }}
        >
          {/* Render the diagram SVG inline */}
          <image
            href={svgDataUri}
            x={svgViewBox?.x || 0}
            y={svgViewBox?.y || 0}
            width={svgViewBox?.w || 800}
            height={svgViewBox?.h || 600}
          />

          {/* Anchor markers */}
          {anchors.map((a, i) => (
            <g
              key={a.id}
              className="anchor-marker"
              onMouseDown={(e) => {
                e.stopPropagation();
                if (e.button === 0) setDragging(i);
              }}
              onContextMenu={(e) => {
                e.preventDefault();
                deleteAnchor(i);
              }}
            >
              <circle cx={a.x} cy={a.y} r={Math.max(8, (svgViewBox?.w || 800) * 0.012)} />
              <text
                x={a.x}
                y={a.y - Math.max(10, (svgViewBox?.w || 800) * 0.015)}
                textAnchor="middle"
              >
                {i + 1}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}
