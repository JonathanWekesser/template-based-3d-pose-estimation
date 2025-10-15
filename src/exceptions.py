# exceptions.py
class PipelineError(Exception):
    """Base class for pipeline-related errors."""


class SegmentationError(PipelineError):
    """Generic error in the segmentation stage."""


class InvalidInputImage(SegmentationError):
    """Input image has an unexpected shape/dtype or is None."""


class NoDetectionsError(SegmentationError):
    """Model returned no detections for the current frame."""


class TargetNotFoundError(SegmentationError):
    """The requested target class was not detected in the frame."""
