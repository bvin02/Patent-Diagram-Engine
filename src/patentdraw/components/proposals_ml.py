"""
ML proposal interface for Patent Draw.

Provides a stub implementation and interface for integrating ML-based
component proposals (e.g., SAM-style segmentation masks).
"""

from abc import ABC, abstractmethod

from patentdraw.models import Component, generate_component_id, compute_bbox, compute_centroid
from patentdraw.tracer import get_tracer, trace


class ProposalProvider(ABC):
    """Abstract interface for ML component proposal providers."""
    
    @abstractmethod
    def get_proposals(self, image, strokes):
        """
        Generate component proposals from image and strokes.
        
        Args:
            image: RGB numpy array of the original image
            strokes: list of Stroke objects
        
        Returns:
            list of Component objects with proposal_sources=["ml"]
        """
        pass
    
    @abstractmethod
    def is_available(self):
        """Check if this provider is ready to use."""
        pass


class StubProvider(ProposalProvider):
    """
    Stub implementation that returns no proposals.
    
    Placeholder for future ML integration.
    """
    
    def get_proposals(self, image, strokes):
        """Return empty list - no ML proposals available."""
        return []
    
    def is_available(self):
        """Stub is always available but provides no proposals."""
        return True


class MaskToComponentMapper:
    """
    Utility to convert segmentation masks to component proposals.
    
    Maps strokes to masks based on overlap area.
    """
    
    def __init__(self, overlap_threshold=0.5):
        self.overlap_threshold = overlap_threshold
    
    def map_strokes_to_masks(self, strokes, masks):
        """
        Map strokes to segmentation masks.
        
        Args:
            strokes: list of Stroke objects
            masks: list of binary masks (H, W arrays)
        
        Returns:
            list of Component objects
        """
        tracer = get_tracer()
        
        if not masks:
            return []
        
        import numpy as np
        
        components = []
        
        for mask_idx, mask in enumerate(masks):
            matched_stroke_ids = []
            matched_points = []
            
            for stroke in strokes:
                overlap = self._compute_overlap(stroke.polyline, mask)
                if overlap >= self.overlap_threshold:
                    matched_stroke_ids.append(stroke.stroke_id)
                    matched_points.extend(stroke.polyline)
            
            if matched_stroke_ids:
                component = Component(
                    component_id=generate_component_id(matched_stroke_ids),
                    stroke_ids=matched_stroke_ids,
                    bbox=compute_bbox(matched_points),
                    centroid=compute_centroid(matched_points),
                    proposal_sources=["ml"],
                    confidence=0.8,  # ML proposals get higher base confidence
                )
                components.append(component)
        
        tracer.event(f"Mapped {len(masks)} masks to {len(components)} components")
        
        return components
    
    def _compute_overlap(self, polyline, mask):
        """Compute fraction of polyline points that fall within mask."""
        if not polyline:
            return 0.0
        
        import numpy as np
        
        h, w = mask.shape
        count = 0
        
        for x, y in polyline:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < h:
                if mask[iy, ix] > 0:
                    count += 1
        
        return count / len(polyline)


def get_provider(config):
    """
    Factory to get the appropriate proposal provider.
    
    Returns StubProvider if ML is not enabled.
    """
    tracer = get_tracer()
    
    if not config.ml_enabled:
        tracer.event("ML disabled, using stub provider")
        return StubProvider()
    
    # Future: check for SAM availability and return appropriate provider
    tracer.event("ML enabled but no provider available, using stub")
    return StubProvider()


@trace(label="get_ml_proposals")
def get_ml_proposals(image, strokes, config):
    """
    Get ML-based component proposals if available.
    
    Returns empty list if ML is disabled or unavailable.
    """
    tracer = get_tracer()
    
    provider = get_provider(config)
    
    if not provider.is_available():
        tracer.event("ML provider not available")
        return []
    
    return provider.get_proposals(image, strokes)
