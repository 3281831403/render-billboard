"""Camera-facing billboard package."""

from .facing import (
    BillboardError,
    align_up,
    billboard_batch,
    clamp_axis,
    cross,
    depth_shade,
    face_camera,
    normalize,
    screen_scale,
    validate,
)

__all__ = [
    "BillboardError",
    "align_up",
    "billboard_batch",
    "clamp_axis",
    "cross",
    "depth_shade",
    "face_camera",
    "normalize",
    "screen_scale",
    "validate",
]
