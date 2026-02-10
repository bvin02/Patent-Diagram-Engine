# Project Direction and Technical Rationale

This project intentionally takes a deterministic computer-vision approach rather than a generative AI approach. The decision was made to better align with the real constraints of patent drafting, IP sensitivity, and current technical feasibility.

## Why I did not use generative AI

### 1. IP privacy and data control are non-negotiable
Patent diagrams contain unreleased intellectual property. Sending sketches or product images to third-party AI APIs introduces unacceptable ambiguity around data retention, model training, and ownership. Even with self-hosted models, patent users expect full local control, traceability, and auditability rather than probabilistic inference.

### 2. Patent drawings require determinism, not probability
USPTO drawings must be precise, repeatable, and consistent across runs. Current generative image models are inherently probabilistic and cannot reliably produce stable vector geometry, consistent line topology, or deterministic labeling. This makes them unsuitable for compliance-driven technical drawings.

### 3. Inferring unseen components is invalid for patents
The original prompt suggests generating additional views or components from an image. For patents, this is conceptually incorrect. Mechanical patent objects are non-generic and IP-specific. Inferring unseen geometry or hidden components is equivalent to inventing parts of the patent, which makes such output unusable in practice.

## The user reality this project optimizes for

Patent filers generally fall into three categories:

1. Users who cannot draw diagrams at all and require professional drafting from scratch.
2. Users who can draw accurate diagrams on paper but struggle to convert them into clean, digital drawings.
3. Users who are already fluent with digital illustration tools and do not need automation.

This project explicitly targets the second group, where automation is both feasible and highest impact.

## The implemented approach

Instead of probabilistic generation, the system uses a deterministic raster-to-vector pipeline optimized for mechanical line diagrams. No general solutions existed online so I built the pipeline form scratch, and in doing so I was able to optimize it specifically for line art. This pipeline does the following:

- Converts hand-drawn pencil sketches or photos into clean binary stroke masks
- Extracts true stroke centerlines
- Reconstructs drawing topology as a graph
- Fits explicit geometric primitives (lines, arcs, Beziers, polylines)
- Emits fully editable SVG output
- Adds patent-style numbered labels and leader lines
- Keeps the user in the loop via an embedded SVG editor

No geometry is invented. No components are inferred beyond what is explicitly drawn. Every stage is reproducible and debuggable.

Full technical details of the pipeline, stages, and architecture are documented in the attached README.

## Why this better satisfies the original goal

The stated goal is to reduce the cost and friction of producing USPTO-compliant technical drawings. This approach eliminates generative uncertainty, preserves IP integrity, matches real patent-filing workflows, and produces compliance-ready vector output while allowing final human control.

## Summary

This project constrains the original problem to what is technically correct, legally safe, and immediately useful. It demonstrates a realistic path to automating the most expensive and time-consuming step in patent drafting: converting accurate hand-drawn diagrams into clean, digital, labeled technical drawings without inventing or altering the underlying IP.
