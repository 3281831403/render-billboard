"""Red tests for the camera-facing billboard bugfix task.

Every test intentionally calls an unimplemented stub from
``billboard.facing``.  The initial suite therefore reports exactly
``15 failed`` with no collection, import, or syntax errors.

A basis is ``(right, up, normal)`` and satisfies
``cross(right, up) == normal`` with ``normal`` pointing toward the camera.
A camera is ``{"pos": ..., "up": ..., "forward": ...}``.  All geometric
comparisons use a tolerance of ``1e-6``.
"""

import math

import pytest

from billboard.facing import (
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


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(_dot(a, a))


# Defect 1: cross(a, b) is evaluated as cross(b, a), so the normal points into
# the billboard instead of toward the camera.
def test_defect_01_cross_is_right_handed_and_points_outward():
    assert cross((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == pytest.approx(
        (0.0, 0.0, 1.0), abs=1e-6
    )
    assert cross((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 0.0, -1.0), abs=1e-6
    )


# Defect 2: raw camera-offset vectors are copied into the basis, so the axes
# are not unit length and apparent scale drifts with distance.
def test_defect_02_face_camera_normalizes_every_basis_vector():
    base = (1.0, 2.0, 3.0)
    camera = (4.0, 6.0, 15.0)  # offset (3, 4, 12), length 13

    right, up, normal = face_camera(base, camera, "full")

    for axis in (right, up, normal):
        assert _norm(axis) == pytest.approx(1.0, abs=1e-6)
    assert normal == pytest.approx((3.0 / 13.0, 4.0 / 13.0, 12.0 / 13.0), abs=1e-6)
    assert abs(_dot(right, normal)) <= 1e-6
    assert abs(_dot(up, normal)) <= 1e-6
    assert abs(_dot(right, up)) <= 1e-6


# Defect 3: yaw mode takes the full 3D camera direction, tilting the character
# when the camera looks down.
def test_defect_03_yaw_mode_stays_upright():
    base = (1.0, 2.0, 3.0)
    camera = (1.0, 12.0, 13.0)  # steeply above, but straight ahead in X/Z

    right, up, normal = face_camera(base, camera, "yaw")

    assert up == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert normal == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)
    assert right == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)


# Defect 4: screen scaling returns the world size, so near and far billboards
# have the same apparent size.
def test_defect_04_perspective_scale_shrinks_with_distance():
    near = screen_scale(5.0, math.pi / 2.0, 2.0, "perspective")
    far = screen_scale(20.0, math.pi / 2.0, 2.0, "perspective")

    assert near == pytest.approx(0.2, abs=1e-6)
    assert far == pytest.approx(0.05, abs=1e-6)
    assert far < near
    assert screen_scale(5.0, math.pi / 2.0, 2.0, "screen") == pytest.approx(
        2.0, abs=1e-6
    )
    assert screen_scale(20.0, math.pi / 2.0, 2.0, "world") == pytest.approx(
        2.0, abs=1e-6
    )


# Defect 5: the raw camera up vector is used without removing its component
# along the normal, so a pitching camera makes the billboard roll/flip.
def test_defect_05_align_up_removes_the_tilt():
    basis = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.6, 0.8))

    right, up, normal = align_up(basis, (0.0, 1.0, 0.0))

    assert up == pytest.approx((0.0, 0.8, -0.6), abs=1e-6)
    assert up[1] > 0.0
    assert abs(_dot(up, normal)) <= 1e-6
    assert right == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    assert normal == pytest.approx((0.0, 0.6, 0.8), abs=1e-6)


# Defect 6: axial locking rotates about the X or Z axis instead of Y, so the
# up vector tilts even though the billboard should stay upright.
def test_defect_06_clamp_axis_rotates_only_around_y():
    # A valid frame whose normal starts on +X.
    basis = ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    base = (2.0, 3.0, 4.0)
    camera = (2.0, 3.0, 14.0)

    right, up, normal = clamp_axis(basis, camera, base)

    assert normal == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)
    assert up == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert right == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    assert cross(right, up) == pytest.approx(normal, abs=1e-6)


# Defect 7: distance attenuation is inverse-exponential or starts at zero, so
# a billboard touching the camera vanishes.
def test_defect_07_depth_shade_is_opaque_nearby_and_positive_far_away():
    assert depth_shade(0.0) == pytest.approx(1.0, abs=1e-6)
    assert depth_shade(1.0) > 0.0
    assert depth_shade(1000.0) > 0.0
    assert 0.0 < depth_shade(1000.0) < depth_shade(10.0) < 1.0


# Defect 8: an unknown orientation mode is silently treated as "full" instead
# of being reported.
def test_defect_08_validate_reports_unknown_mode():
    issues = validate((1.0, 2.0, 3.0), (1.0, 2.0, 13.0), "spherical")

    assert any("mode" in issue.lower() for issue in issues)


# Defect 9: the batch reuses one basis for every instance, so different
# billboards face the same direction and appear to contaminate each other.
def test_defect_09_batch_builds_independent_bases():
    config = {
        "mode": "full",
        "scale_mode": "perspective",
        "fov": math.pi / 2.0,
        "base_size": 1.0,
    }
    camera = {"pos": (0.0, 2.0, 13.0), "up": (0.0, 1.0, 0.0), "forward": (0.0, 0.0, -1.0)}
    instances = [
        {"base": (1.0, 2.0, 3.0)},
        {"base": (-1.0, 2.0, 3.0)},
    ]

    result = billboard_batch(instances, camera, config)

    assert result["ok"] is True
    first = result["billboards"][0]["basis"][2]
    second = result["billboards"][1]["basis"][2]
    root = math.sqrt(101.0)
    assert first == pytest.approx((-1.0 / root, 0.0, 10.0 / root), abs=1e-6)
    assert second == pytest.approx((1.0 / root, 0.0, 10.0 / root), abs=1e-6)
    assert first != second


# Defect 10: a camera exactly at the billboard is accepted and the direction
# calculation divides by zero.
def test_defect_10_coincident_camera_is_rejected():
    base = (1.0, 2.0, 3.0)
    issues = validate(base, base, "full")

    assert any("coincide" in issue.lower() for issue in issues)
    with pytest.raises(BillboardError):
        face_camera(base, base, "full")


# Combination: with a camera directly in front on +Z, the canonical basis is
# right=(1, 0, 0), up=(0, 1, 0), normal=(0, 0, 1).
def test_combo_front_camera_basis_vectors():
    right, up, normal = face_camera((0.0, 0.0, 1.0), (0.0, 0.0, 11.0), "full")

    assert right == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    assert up == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert normal == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)


# Combination: an unsupported mode is an error, not a fallback to full.
def test_combo_unknown_mode_raises():
    with pytest.raises(BillboardError):
        face_camera((1.0, 2.0, 3.0), (1.0, 2.0, 13.0), "spherical")


# Combination: zero-length vectors have no direction and must be rejected;
# validation also reports a zero base vector instead of accepting it.
def test_combo_zero_basis_vector_raises():
    with pytest.raises(BillboardError):
        normalize((0.0, 0.0, 0.0))
    with pytest.raises(BillboardError):
        cross((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    issues = validate((0.0, 0.0, 0.0), (0.0, 0.0, 10.0), "full")
    assert any("zero" in issue.lower() for issue in issues)


# Combination: a fully valid batch keeps the caller's instance order.
def test_combo_batch_preserves_order_on_success():
    config = {
        "mode": "full",
        "scale_mode": "perspective",
        "fov": math.pi / 2.0,
        "base_size": 1.0,
    }
    camera = {"pos": (0.0, 1.0, 10.0), "up": (0.0, 1.0, 0.0), "forward": (0.0, 0.0, -1.0)}
    instances = [
        {"base": (3.0, 1.0, 0.0), "mode": "full"},
        {"base": (2.0, 1.0, 0.0), "mode": "yaw"},
        {"base": (1.0, 1.0, 0.0), "mode": "full"},
    ]

    result = billboard_batch(instances, camera, config)

    assert result["ok"] is True
    assert [item["base"] for item in result["billboards"]] == [
        (3.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    ]
    assert [item["mode"] for item in result["billboards"]] == [
        "full",
        "yaw",
        "full",
    ]


# Combination: one bad instance invalidates the whole batch, leaving no
# partial billboards behind.
def test_combo_batch_rolls_back_on_invalid_instance():
    config = {
        "mode": "full",
        "scale_mode": "perspective",
        "fov": math.pi / 2.0,
        "base_size": 1.0,
    }
    camera = {"pos": (0.0, 1.0, 10.0), "up": (0.0, 1.0, 0.0), "forward": (0.0, 0.0, -1.0)}
    instances = [
        {"base": (2.0, 1.0, 3.0)},
        {"base": (3.0, 1.0, 3.0), "mode": "spherical"},
        {"base": (4.0, 1.0, 3.0)},
    ]

    result = billboard_batch(instances, camera, config)

    assert result["ok"] is False
    assert result["billboards"] == []
    assert list(result["errors"]) == [1]
    assert any("mode" in issue.lower() for issue in result["errors"][1])
