"""Camera-facing billboard contracts.

This module intentionally contains only stubs.  The shared data model is:

* A 3-vector is a finite ``(x, y, z)`` tuple of numbers.
* ``base`` is the billboard anchor in world space.
* A billboard basis is ``(right, up, normal)``: three unit vectors with
  ``cross(right, up) == normal``.  ``normal`` points from the billboard toward
  the camera, so the visible front face is the one whose normal faces the
  camera.  This is a right-handed frame in world space.
* A camera is ``{"pos": (x, y, z), "up": (x, y, z),
  "forward": (x, y, z)}``.  ``up`` is the camera's screen-up direction and
  ``forward`` is its unit viewing direction.  Cameras are never mutated.

All geometric comparisons use a tolerance of ``1e-6``.  No function reads the
wall clock, uses global random state, or performs I/O.
"""

from __future__ import annotations

import math

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

EPS = 1e-6

_ORIENTATION_MODES = ("full", "yaw", "axial")
_SCALE_MODES = ("perspective", "screen", "world")
_WORLD_UP = (0.0, 1.0, 0.0)
_FALLBACK_UP = (0.0, 0.0, 1.0)


class BillboardError(ValueError):
    """Raised when billboard input is malformed or a batch is rejected."""


def _is_real(value: object) -> bool:
    """True for finite real numbers, excluding booleans and non-finite floats."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _is_vector(value: object) -> bool:
    """True for finite 3-vectors (tuples or lists of three real numbers)."""

    return (
        isinstance(value, (tuple, list))
        and len(value) == 3
        and all(_is_real(component) for component in value)
    )


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(s, v):
    return (s * v[0], s * v[1], s * v[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross_raw(a, b):
    """Cross product without input or zero-length validation (internal use)."""

    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return the right-handed cross product ``a x b``.

    The result is ``(a_y*b_z - a_z*b_y, a_z*b_x - a_x*b_z,
    a_x*b_y - a_y*b_x)``.  For example, ``cross((1, 0, 0), (0, 1, 0))`` is
    ``(0, 0, 1)``; reversing the operands negates the result.  This ordering
    makes ``cross(right, up)`` point along the outward billboard normal.

    Both arguments must be finite 3-vectors.  A zero-length ``a`` or ``b``
    makes the direction undefined and raises :class:`BillboardError`.
    Results are compared with a tolerance of ``1e-6``.
    """

    if not _is_vector(a) or not _is_vector(b):
        raise BillboardError("cross requires two finite 3-vectors")
    result = _cross_raw(a, b)
    if math.sqrt(_dot(result, result)) <= EPS:
        raise BillboardError("cross product of parallel or zero vectors is undefined")
    return result


def normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return ``v`` scaled to unit length.

    A zero-length vector has no direction and raises :class:`BillboardError`.
    Non-finite input and values that are not 3-vectors also raise
    :class:`BillboardError`.  The returned tuple is already normalized, so
    callers must not skip this call before building a basis; otherwise the
    basis length drifts with the camera distance.  Results are compared with a
    tolerance of ``1e-6``.
    """

    if not _is_vector(v):
        raise BillboardError("normalize requires a finite 3-vector")
    length = math.sqrt(_dot(v, v))
    if length <= EPS:
        raise BillboardError("cannot normalize a zero-length vector")
    return (v[0] / length, v[1] / length, v[2] / length)


def _yaw_frame(offset):
    """Build the upright, Y-locked frame for a camera offset vector."""

    horizontal = (offset[0], 0.0, offset[2])
    normal = normalize(horizontal)
    up = _WORLD_UP
    right = normalize(cross(up, normal))
    return right, up, normal


def face_camera(
    base: tuple[float, float, float],
    cam_pos: tuple[float, float, float],
    mode: str,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Build a camera-facing basis ``(right, up, normal)`` at ``base``.

    ``normal`` always points from ``base`` toward ``cam_pos``.  The three
    supported orientation modes are:

    * ``"full"`` -- face the camera in all three axes.  Let ``n`` be the unit
      direction from ``base`` to the camera and let ``r`` be
      ``normalize(cross((0, 1, 0), n))``; then ``up = cross(n, r)``.  When the
      view direction is nearly parallel to world up, ``(0, 0, 1)`` is used as
      the reference up instead so the basis does not collapse.
    * ``"yaw"`` -- rotate only around the world Y axis.  The horizontal part of
      ``cam_pos - base`` gives ``normal``, ``up`` is exactly ``(0, 1, 0)``,
      and ``right = normalize(cross(up, normal))``.  Camera pitch never tilts
      the billboard.
    * ``"axial"`` -- the axis-locked variant of ``"yaw"``.  With the only
      available lock axis being world Y, it returns the same upright,
      horizontally-facing frame as ``"yaw"`` and never rolls or pitches.

    A mode outside those three names raises :class:`BillboardError` instead of
    silently falling back to ``"full"``.  A camera position that coincides
    with ``base`` (distance at most ``1e-6``) raises :class:`BillboardError`;
    a yaw/axial orientation whose horizontal offset is zero also raises,
    because the rotation around Y is undefined.  The returned basis is
    orthonormal and satisfies ``cross(right, up) == normal``.  Results are
    compared with a tolerance of ``1e-6``.
    """

    if mode not in _ORIENTATION_MODES:
        raise BillboardError(f"unknown mode: {mode}")
    if not _is_vector(base) or not _is_vector(cam_pos):
        raise BillboardError("base and camera position must be finite 3-vectors")

    offset = _sub(cam_pos, base)
    if math.sqrt(_dot(offset, offset)) <= EPS:
        raise BillboardError("camera coincides with base")

    if mode in ("yaw", "axial"):
        return _yaw_frame(offset)

    normal = normalize(offset)
    reference = _WORLD_UP
    candidate = _cross_raw(reference, normal)
    if math.sqrt(_dot(candidate, candidate)) <= EPS:
        reference = _FALLBACK_UP
    right = normalize(_cross_raw(reference, normal))
    up = normalize(_cross_raw(normal, right))
    return right, up, normal


def screen_scale(
    distance: float,
    fov: float,
    base_size: float,
    mode: str,
) -> float:
    """Return the world-space scale for a billboard at ``distance``.

    ``fov`` is the vertical field of view in radians and must satisfy
    ``0 < fov < pi``.  ``distance`` is the positive distance from the camera
    to the billboard.  ``base_size`` is the billboard size in the units named
    by the mode.  The supported scaling modes are:

    * ``"perspective"`` -- project the size through the pinhole camera:
      ``base_size / (2 * distance * tan(fov / 2))``.  The result shrinks as
      the distance grows, so near billboards are larger than far ones.
    * ``"screen"`` -- return ``base_size`` unchanged.  ``base_size`` is already
      a screen-space size, so distance and fov do not change it.
    * ``"world"`` -- return ``base_size`` unchanged.  ``base_size`` is a
      literal world-space size, so distance and fov do not change it.

    ``distance`` must be finite and greater than zero; a non-positive
    distance would project through a division by zero.  ``base_size`` must be
    finite and non-negative.  An unknown mode or invalid argument raises
    :class:`BillboardError`.  Results are compared with a tolerance of
    ``1e-6``.
    """

    if mode not in _SCALE_MODES:
        raise BillboardError(f"unknown scale mode: {mode}")
    if not _is_real(distance) or distance <= 0.0:
        raise BillboardError("distance must be finite and greater than zero")
    if not _is_real(fov) or not 0.0 < fov < math.pi:
        raise BillboardError("fov must satisfy 0 < fov < pi")
    if not _is_real(base_size) or base_size < 0.0:
        raise BillboardError("base_size must be finite and non-negative")

    if mode == "perspective":
        return base_size / (2.0 * distance * math.tan(fov / 2.0))
    return base_size


def align_up(
    basis: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    cam_up: tuple[float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Re-align a basis's up vector to the camera's up vector.

    The plane normal ``basis[2]`` is preserved (up to normalization).  Let
    ``p = cam_up - dot(cam_up, normal) * normal``.  ``up`` is ``normalize(p)``
    and ``right = normalize(cross(up, normal))``.  This Gram-Schmidt step
    keeps the up axis in the billboard plane instead of leaving the raw
    camera up tilted toward the normal, which is what makes a pitched camera
    appear to flip the sprite.

    ``cam_up`` must be a finite 3-vector.  If it is zero, or if it is parallel
    to the normal so that ``p`` has length at most ``1e-6``, the roll is
    undefined and :class:`BillboardError` is raised.  ``basis`` must contain
    three finite 3-vectors; malformed input raises :class:`BillboardError`.
    The returned basis is orthonormal with
    ``cross(right, up) == normal``.  Results are compared with a tolerance of
    ``1e-6``.
    """

    if (
        not isinstance(basis, (tuple, list))
        or len(basis) != 3
        or not all(_is_vector(axis) for axis in basis)
    ):
        raise BillboardError("basis must contain three finite 3-vectors")
    if not _is_vector(cam_up):
        raise BillboardError("camera up must be a finite 3-vector")

    normal = normalize(basis[2])
    projected = _sub(cam_up, _scale(_dot(cam_up, normal), normal))
    if math.sqrt(_dot(projected, projected)) <= EPS:
        raise BillboardError("camera up is parallel to the billboard normal")
    up = normalize(projected)
    right = normalize(cross(up, normal))
    return right, up, normal


def _validate_frame(basis):
    """Check that a supplied basis is a finite orthonormal right-handed frame."""

    if (
        not isinstance(basis, (tuple, list))
        or len(basis) != 3
        or not all(_is_vector(axis) for axis in basis)
    ):
        raise BillboardError("basis must contain three finite 3-vectors")
    right, up, normal = basis
    for axis in (right, up, normal):
        if abs(math.sqrt(_dot(axis, axis)) - 1.0) > EPS:
            raise BillboardError("basis vectors must be unit length")
    for a, b in ((right, up), (right, normal), (up, normal)):
        if abs(_dot(a, b)) > EPS:
            raise BillboardError("basis vectors must be mutually orthogonal")
    expected = cross(right, up)
    if any(abs(expected[i] - normal[i]) > EPS for i in range(3)):
        raise BillboardError("basis must satisfy cross(right, up) == normal")


def clamp_axis(
    basis: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    cam_pos: tuple[float, float, float],
    base: tuple[float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Lock a billboard to the world Y axis and yaw it toward the camera.

    The horizontal direction from ``base`` to ``cam_pos`` becomes the new
    plane normal, so only rotation about Y is allowed.  The returned up axis
    is exactly ``(0, 1, 0)`` and
    ``right = normalize(cross(up, normal))``.  Any pitch or roll present in
    ``basis`` is discarded rather than rotating the object about the wrong
    axis; the input basis is used to validate the supplied frame and is never
    mutated.

    A camera that coincides with ``base``, or whose offset is purely vertical
    so that the horizontal direction has length at most ``1e-6``, raises
    :class:`BillboardError` because the yaw angle is undefined.  A malformed
    ``basis`` also raises :class:`BillboardError`.  Results are compared with
    a tolerance of ``1e-6``.
    """

    _validate_frame(basis)
    if not _is_vector(base) or not _is_vector(cam_pos):
        raise BillboardError("base and camera position must be finite 3-vectors")

    offset = _sub(cam_pos, base)
    if math.sqrt(_dot(offset, offset)) <= EPS:
        raise BillboardError("camera coincides with base")
    return _yaw_frame(offset)


def depth_shade(distance: float) -> float:
    """Return a distance fade in ``(0, 1]`` for a billboard.

    The fade is ``1.0 / (1.0 + distance)``.  At distance zero it is exactly
    ``1.0``, so a billboard touching the camera is fully visible rather than
    disappearing.  It decreases monotonically with distance but is always
    strictly positive, so far billboards fade toward zero without ever being
    cut to zero.  ``distance`` must be finite and non-negative; invalid input
    raises :class:`BillboardError`.  Results are compared with a tolerance of
    ``1e-6``.
    """

    if not _is_real(distance) or distance < 0.0:
        raise BillboardError("distance must be finite and non-negative")
    return 1.0 / (1.0 + distance)


def validate(
    base: tuple[float, float, float],
    cam_pos: tuple[float, float, float],
    mode: str,
) -> list[str]:
    """Return the problems found in a billboard request, or ``[]``.

    ``base`` and ``cam_pos`` are finite 3-vectors and ``mode`` is one of
    ``"full"``, ``"yaw"`` and ``"axial"``.  The checks, in this order, report
    these exact strings:

    * ``"base must be a finite 3-vector"`` for a malformed base;
    * ``"base vector is zero"`` when every base component is zero;
    * ``"camera position must be a finite 3-vector"`` for a malformed
      camera position;
    * ``"unknown mode: <mode>"`` for an unsupported mode (the supplied value
      is appended verbatim);
    * ``"camera coincides with base"`` when their distance is at most
      ``1e-6``.

    The function is a reporting helper: billboard problems are returned in
    the list and do not raise.  It is used by :func:`billboard_batch` before
    any output is produced, so callers can collect every problem at once
    instead of silently treating an unknown mode as ``"full"``.  Results are
    compared with a tolerance of ``1e-6``.
    """

    issues: list[str] = []
    if not _is_vector(base):
        issues.append("base must be a finite 3-vector")
    elif all(component == 0.0 for component in base):
        issues.append("base vector is zero")

    if not _is_vector(cam_pos):
        issues.append("camera position must be a finite 3-vector")

    if mode not in _ORIENTATION_MODES:
        issues.append(f"unknown mode: {mode}")

    if _is_vector(base) and _is_vector(cam_pos):
        offset = _sub(cam_pos, base)
        if math.sqrt(_dot(offset, offset)) <= EPS:
            issues.append("camera coincides with base")

    return issues


def _check_camera(camera: dict) -> None:
    if not isinstance(camera, dict):
        raise BillboardError("camera must be a dictionary")
    for key in ("pos", "up", "forward"):
        if key not in camera or not _is_vector(camera[key]):
            raise BillboardError(f"camera {key} must be a finite 3-vector")
    for key in ("up", "forward"):
        if math.sqrt(_dot(camera[key], camera[key])) <= EPS:
            raise BillboardError(f"camera {key} must be non-zero")


def _check_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise BillboardError("config must be a dictionary")
    settings = {
        "mode": config.get("mode", "full"),
        "scale_mode": config.get("scale_mode", "perspective"),
        "fov": config.get("fov", math.pi / 2.0),
        "base_size": config.get("base_size", 1.0),
    }
    if settings["mode"] not in _ORIENTATION_MODES:
        raise BillboardError(f"unknown mode: {settings['mode']}")
    if settings["scale_mode"] not in _SCALE_MODES:
        raise BillboardError(f"unknown scale mode: {settings['scale_mode']}")
    if not _is_real(settings["fov"]) or not 0.0 < settings["fov"] < math.pi:
        raise BillboardError("fov must satisfy 0 < fov < pi")
    if not _is_real(settings["base_size"]) or settings["base_size"] < 0.0:
        raise BillboardError("base_size must be finite and non-negative")
    return settings


def _validate_instance(instance, camera, settings) -> list[str]:
    if not isinstance(instance, dict):
        return ["instance must be a dictionary"]

    issues: list[str] = []
    base = instance.get("base")
    if not _is_vector(base):
        issues.append("base must be a finite 3-vector")
    elif all(component == 0.0 for component in base):
        issues.append("base vector is zero")

    mode = instance.get("mode", settings["mode"])
    if mode not in _ORIENTATION_MODES:
        issues.append(f"unknown mode: {mode}")

    scale_mode = instance.get("scale_mode", settings["scale_mode"])
    if scale_mode not in _SCALE_MODES:
        issues.append(f"unknown scale mode: {scale_mode}")

    size = instance.get("size", settings["base_size"])
    if not _is_real(size) or size < 0.0:
        issues.append("size must be finite and non-negative")

    if issues:
        return issues

    issues.extend(validate(base, camera["pos"], mode))
    if issues:
        return issues

    distance = math.sqrt(_dot(_sub(camera["pos"], base), _sub(camera["pos"], base)))
    try:
        screen_scale(distance, settings["fov"], size, scale_mode)
    except BillboardError as exc:
        issues.append(str(exc))
    return issues


def billboard_batch(
    instances: list[dict],
    camera: dict,
    config: dict,
) -> dict:
    """Build billboards atomically, with an independent basis per instance.

    ``instances`` is a sequence of dictionaries.  Each requires a finite,
    non-zero ``"base"`` 3-vector and may override the defaults with
    ``"mode"``, ``"scale_mode"`` and ``"size"``.  ``camera`` is
    ``{"pos": ..., "up": ..., "forward": ...}``; all three entries must be
    finite 3-vectors, and ``up``/``forward`` must be non-zero.  ``config``
    supplies defaults and projection parameters:

    * ``"mode"`` -- orientation mode, default ``"full"``;
    * ``"scale_mode"`` -- ``"perspective"``, ``"screen"`` or ``"world"``,
      default ``"perspective"``;
    * ``"fov"`` -- vertical field of view in radians, default ``pi / 2``;
    * ``"base_size"`` -- default billboard size, default ``1.0``.

    Every instance is validated before any billboard is built.  Validation
    uses :func:`validate`, :func:`screen_scale` and the size rules above.  If
    any instance is invalid, nothing is partially built and the result is::

        {"ok": False, "billboards": [], "errors": {index: [issue, ...]}}

    with one entry for every invalid instance, in input order.  On success the
    result is::

        {"ok": True,
         "billboards": [
             {"base": base,
              "mode": mode,
              "size": size,
              "basis": (right, up, normal),
              "distance": distance,
              "scale": scale,
              "shade": shade}
         ],
         "errors": {}}

    Each billboard calls :func:`face_camera` with its own base, so no basis
    is shared between instances.  For ``"full"`` the basis is then passed
    through :func:`align_up` with ``camera["up"]``; yaw and axial billboards
    stay locked upright.  ``distance`` is the Euclidean distance from
    ``camera["pos"]`` to the instance base, ``scale`` comes from
    :func:`screen_scale` and ``shade`` from :func:`depth_shade`.  Input order
    is preserved, empty input succeeds with an empty list, and the input
    objects are never mutated.

    A malformed ``camera``, ``config`` or top-level ``instances`` value raises
    :class:`BillboardError`; instance-level problems are reported through the
    atomic failure dictionary.  Results are compared with a tolerance of
    ``1e-6``.
    """

    if not isinstance(instances, (list, tuple)):
        raise BillboardError("instances must be a sequence of dictionaries")
    _check_camera(camera)
    settings = _check_config(config)

    errors: dict = {}
    for index, instance in enumerate(instances):
        instance_issues = _validate_instance(instance, camera, settings)
        if instance_issues:
            errors[index] = instance_issues
    if errors:
        return {"ok": False, "billboards": [], "errors": errors}

    billboards = []
    for instance in instances:
        base = instance["base"]
        mode = instance.get("mode", settings["mode"])
        scale_mode = instance.get("scale_mode", settings["scale_mode"])
        size = instance.get("size", settings["base_size"])

        basis = face_camera(base, camera["pos"], mode)
        if mode == "full":
            basis = align_up(basis, camera["up"])

        offset = _sub(camera["pos"], base)
        distance = math.sqrt(_dot(offset, offset))
        scale = screen_scale(distance, settings["fov"], size, scale_mode)
        shade = depth_shade(distance)

        billboards.append(
            {
                "base": base,
                "mode": mode,
                "size": size,
                "basis": basis,
                "distance": distance,
                "scale": scale,
                "shade": shade,
            }
        )

    return {"ok": True, "billboards": billboards, "errors": {}}
