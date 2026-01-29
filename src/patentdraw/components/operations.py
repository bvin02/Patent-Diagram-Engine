"""
Deterministic component operations for Patent Draw.

Provides merge and split operations that maintain consistency
across the scene graph (components, labels, numbering).
"""

from patentdraw.models import (
    Component, Label, Document,
    generate_component_id, generate_label_id,
    compute_bbox_from_bboxes, compute_centroid,
)
from patentdraw.tracer import get_tracer, trace


@trace(label="merge_components")
def merge_components(document, component_ids):
    """
    Merge multiple components into one.
    
    Creates a new component with combined stroke IDs.
    Updates labels to point to new component.
    Preserves the lowest numeral for the merged component.
    
    Args:
        document: Document object
        component_ids: list of component IDs to merge
    
    Returns:
        tuple of (new_component_id, updated_document)
    """
    tracer = get_tracer()
    
    if len(component_ids) < 2:
        tracer.event("Merge requires at least 2 components", level="WARN")
        return None, document
    
    # Collect all stroke IDs
    combined_stroke_ids = []
    combined_bboxes = []
    combined_points = []
    
    for comp_id in component_ids:
        if comp_id not in document.component_registry:
            tracer.event(f"Component not found: {comp_id}", level="WARN")
            continue
        
        comp = document.component_registry[comp_id]
        combined_stroke_ids.extend(comp.stroke_ids)
        combined_bboxes.append(comp.bbox)
        # Approximate points from bbox corners
        combined_points.extend([
            [comp.bbox[0], comp.bbox[1]],
            [comp.bbox[2], comp.bbox[3]],
        ])
    
    if not combined_stroke_ids:
        return None, document
    
    # Create new component
    new_comp = Component(
        component_id=generate_component_id(combined_stroke_ids),
        stroke_ids=sorted(set(combined_stroke_ids)),
        bbox=compute_bbox_from_bboxes(combined_bboxes),
        centroid=compute_centroid(combined_points),
        proposal_sources=["merge"],
        confidence=0.9,
    )
    
    # Find lowest numeral among merged components
    numerals = []
    for comp_id in component_ids:
        if comp_id in document.numbering_registry:
            numerals.append(document.numbering_registry[comp_id])
    
    # Update document
    # Remove old components
    for comp_id in component_ids:
        if comp_id in document.component_registry:
            del document.component_registry[comp_id]
        if comp_id in document.numbering_registry:
            del document.numbering_registry[comp_id]
    
    # Add new component
    document.component_registry[new_comp.component_id] = new_comp
    
    # Assign numeral (lowest of merged or new)
    if numerals:
        document.numbering_registry[new_comp.component_id] = min(numerals)
    
    # Update view component lists
    for view in document.views:
        new_comp_ids = []
        merged_found = False
        for cid in view.component_ids:
            if cid in component_ids:
                if not merged_found:
                    new_comp_ids.append(new_comp.component_id)
                    merged_found = True
            else:
                new_comp_ids.append(cid)
        view.component_ids = new_comp_ids
    
    # Update labels
    new_labels = {}
    for label_id, label in document.label_registry.items():
        if label.component_id in component_ids:
            # Update to point to new component
            label.component_id = new_comp.component_id
        new_labels[label_id] = label
    
    # Remove duplicate labels for same component
    seen_components = set()
    final_labels = {}
    for label_id, label in new_labels.items():
        if label.component_id not in seen_components:
            final_labels[label_id] = label
            seen_components.add(label.component_id)
    
    document.label_registry = final_labels
    
    tracer.event(f"Merged {len(component_ids)} components into {new_comp.component_id}")
    
    return new_comp.component_id, document


@trace(label="split_component")
def split_component_by_strokes(document, component_id, stroke_sets):
    """
    Split a component into multiple components by stroke assignment.
    
    Args:
        document: Document object
        component_id: ID of component to split
        stroke_sets: list of lists of stroke IDs (one per new component)
    
    Returns:
        tuple of (list of new_component_ids, updated_document)
    """
    tracer = get_tracer()
    
    if component_id not in document.component_registry:
        tracer.event(f"Component not found: {component_id}", level="WARN")
        return [], document
    
    if len(stroke_sets) < 2:
        tracer.event("Split requires at least 2 stroke sets", level="WARN")
        return [], document
    
    old_comp = document.component_registry[component_id]
    old_numeral = document.numbering_registry.get(component_id)
    
    # Create new components
    new_components = []
    new_ids = []
    
    for i, stroke_ids in enumerate(stroke_sets):
        if not stroke_ids:
            continue
        
        # Get stroke data for bbox/centroid calculation
        stroke_points = []
        for view in document.views:
            for stroke in view.strokes:
                if stroke.stroke_id in stroke_ids:
                    stroke_points.extend(stroke.polyline)
        
        new_comp = Component(
            component_id=generate_component_id(stroke_ids),
            stroke_ids=sorted(stroke_ids),
            bbox=compute_bbox_from_bboxes([old_comp.bbox]),  # Approximation
            centroid=compute_centroid(stroke_points) if stroke_points else old_comp.centroid,
            proposal_sources=["split"],
            confidence=0.85,
        )
        new_components.append(new_comp)
        new_ids.append(new_comp.component_id)
    
    # Remove old component
    del document.component_registry[component_id]
    if component_id in document.numbering_registry:
        del document.numbering_registry[component_id]
    
    # Add new components
    for new_comp in new_components:
        document.component_registry[new_comp.component_id] = new_comp
    
    # Assign numerals - first gets old numeral, rest get new
    if old_numeral is not None and new_ids:
        document.numbering_registry[new_ids[0]] = old_numeral
        # Others will get new numbers in next numbering pass
    
    # Update view component lists
    for view in document.views:
        new_comp_ids = []
        for cid in view.component_ids:
            if cid == component_id:
                new_comp_ids.extend(new_ids)
            else:
                new_comp_ids.append(cid)
        view.component_ids = new_comp_ids
    
    # Update labels - first new component gets label
    for label_id, label in document.label_registry.items():
        if label.component_id == component_id and new_ids:
            label.component_id = new_ids[0]
    
    tracer.event(f"Split {component_id} into {len(new_ids)} components")
    
    return new_ids, document


@trace(label="apply_operations")
def apply_operations(document, operations):
    """
    Apply a list of operations to a document.
    
    Operations format:
    [
        {"op": "merge_components", "component_ids": ["id1", "id2"]},
        {"op": "split_component", "component_id": "id", "stroke_sets": [["s1"], ["s2"]]},
        {"op": "move_label", "label_id": "id", "text_pos": [x, y]},
    ]
    
    Returns updated document.
    """
    tracer = get_tracer()
    
    for i, op in enumerate(operations):
        op_type = op.get("op")
        
        with tracer.span(f"operation_{i}", module="operations", op_type=op_type):
            if op_type == "merge_components":
                _, document = merge_components(document, op.get("component_ids", []))
            
            elif op_type == "split_component":
                _, document = split_component_by_strokes(
                    document, 
                    op.get("component_id", ""),
                    op.get("stroke_sets", []),
                )
            
            elif op_type == "move_label":
                label_id = op.get("label_id")
                text_pos = op.get("text_pos")
                if label_id in document.label_registry and text_pos:
                    document.label_registry[label_id].text_pos = text_pos
                    document.label_registry[label_id].status = "edited"
            
            elif op_type == "relabel_view":
                view_id = op.get("view_id")
                view_label = op.get("view_label")
                for view in document.views:
                    if view.view_id == view_id:
                        view.view_label = view_label
            
            else:
                tracer.event(f"Unknown operation: {op_type}", level="WARN")
    
    return document
