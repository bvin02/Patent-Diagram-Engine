"""
Numeral assignment for Patent Draw.

Assigns reference numerals to components in a stable, deterministic order.
"""

from patentdraw.models import LabelStatus
from patentdraw.tracer import get_tracer, trace


@trace(label="assign_numbering")
def assign_numbering(document, config):
    """
    Assign reference numerals to all components.
    
    Numbering is stable and deterministic:
    1. Sort components by centroid (y, then x) within each view
    2. Assign numerals starting from start_number, incrementing by increment
    3. Preserve existing numbers for unchanged component_ids
    
    Updates document.numbering_registry and label.text for each label.
    
    Returns updated document.
    """
    tracer = get_tracer()
    
    start = config.numbering.start_number
    increment = config.numbering.increment
    
    # Get components sorted by position
    sorted_components = sort_components_by_position(document)
    
    # Preserve existing numbering for unchanged components
    existing = dict(document.numbering_registry)
    used_numbers = set(existing.values())
    
    # Assign new numbers
    new_assignments = 0
    preserved = 0
    
    next_number = start
    
    for comp_id in sorted_components:
        if comp_id in existing:
            preserved += 1
            continue
        
        # Find next unused number
        while next_number in used_numbers:
            next_number += increment
        
        document.numbering_registry[comp_id] = next_number
        used_numbers.add(next_number)
        next_number += increment
        new_assignments += 1
    
    # Update label text
    for label_id, label in document.label_registry.items():
        comp_id = label.component_id
        if comp_id in document.numbering_registry:
            label.text = str(document.numbering_registry[comp_id])
            if label.status == LabelStatus.PROPOSED:
                label.status = LabelStatus.FINAL
    
    tracer.event(f"Numbering: {preserved} preserved, {new_assignments} new")
    
    return document


def sort_components_by_position(document):
    """
    Sort component IDs by position for stable numbering.
    
    Order: by view order, then by (centroid_y, centroid_x).
    """
    component_positions = {}
    
    for view_idx, view in enumerate(document.views):
        for comp_id in view.component_ids:
            if comp_id in document.component_registry:
                comp = document.component_registry[comp_id]
                # Position key: (view_index, y, x)
                key = (view_idx, comp.centroid[1], comp.centroid[0])
                component_positions[comp_id] = key
    
    sorted_ids = sorted(component_positions.keys(), key=lambda cid: component_positions[cid])
    
    return sorted_ids


def renumber_after_operations(document, config):
    """
    Renumber components after merge/split operations.
    
    Preserves numbers where possible, assigns new numbers to new components.
    """
    tracer = get_tracer()
    
    # Get all current component IDs
    current_ids = set(document.component_registry.keys())
    
    # Remove defunct entries from numbering registry
    defunct = [cid for cid in document.numbering_registry if cid not in current_ids]
    for cid in defunct:
        del document.numbering_registry[cid]
    
    # Reassign numbering
    document = assign_numbering(document, config)
    
    tracer.event(f"Renumbered after ops, removed {len(defunct)} defunct entries")
    
    return document
