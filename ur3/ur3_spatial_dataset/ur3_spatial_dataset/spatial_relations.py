"""Pure geometry helpers for the canonical spatial scene representation."""

import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for a normalized quaternion."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def transform_point(point: Iterable[float], translation, quaternion) -> np.ndarray:
    """Apply a TF transform whose target frame is represented by the transform."""
    rotation = quaternion_matrix(*quaternion)
    return rotation @ np.asarray(point, dtype=np.float64) + np.asarray(
        translation, dtype=np.float64
    )


def project_point(point_camera, intrinsics: Mapping[str, float]) -> Optional[Tuple[float, float, float]]:
    """Project an optical-frame point to (u, v, depth); reject points behind camera."""
    x, y, z = [float(value) for value in point_camera]
    if not math.isfinite(z) or z <= 1e-6:
        return None
    u = float(intrinsics["fx"]) * x / z + float(intrinsics["cx"])
    v = float(intrinsics["fy"]) * y / z + float(intrinsics["cy"])
    return u, v, z


def surface_distance(object_a: Mapping, object_b: Mapping) -> float:
    """Euclidean separation of two axis-aligned bounding boxes (zero if touching)."""
    a = np.asarray(object_a["pose"]["position"], dtype=np.float64)
    b = np.asarray(object_b["pose"]["position"], dtype=np.float64)
    half_a = 0.5 * np.asarray(object_a["dimensions"], dtype=np.float64)
    half_b = 0.5 * np.asarray(object_b["dimensions"], dtype=np.float64)
    separation = np.maximum(np.abs(a - b) - half_a - half_b, 0.0)
    return float(np.linalg.norm(separation))


def is_inside(subject: Mapping, container: Mapping, tolerance_m: float = 0.005) -> bool:
    """Test full AABB containment using the container's usable inner dimensions."""
    inner = container.get("container_inner_dimensions")
    if inner is None:
        return False
    subject_center = np.asarray(subject["pose"]["position"], dtype=np.float64)
    subject_half = 0.5 * np.asarray(subject["dimensions"], dtype=np.float64)
    container_center = np.asarray(container["pose"]["position"], dtype=np.float64)
    container_half = 0.5 * np.asarray(inner, dtype=np.float64)
    return bool(np.all(
        np.abs(subject_center - container_center) + subject_half
        <= container_half + float(tolerance_m)
    ))


def compute_relations(
    objects: List[Mapping],
    projections: Optional[Dict[str, Tuple[float, float, float]]] = None,
    pixel_margin: float = 8.0,
    depth_margin_m: float = 0.015,
    near_threshold_m: float = 0.12,
) -> List[dict]:
    """Compute directed image-frame, optical-depth and metric relations.

    `left_of`, `right_of`, `above`, and `below` intentionally use the frozen
    VIEW_POSE image frame mandated by the research protocol.  `front_of` uses
    optical depth.  `inside` and `near` use metric AABBs in base_link.
    """
    projections = projections or {}
    relations = []
    for subject in objects:
        for reference in objects:
            if subject["id"] == reference["id"]:
                continue
            subject_id = subject["id"]
            reference_id = reference["id"]
            subject_uvz = projections.get(subject_id)
            reference_uvz = projections.get(reference_id)
            if subject_uvz is not None and reference_uvz is not None:
                du = float(reference_uvz[0] - subject_uvz[0])
                dv = float(reference_uvz[1] - subject_uvz[1])
                dz = float(reference_uvz[2] - subject_uvz[2])
                if du > pixel_margin:
                    relations.append(_relation(subject_id, "left_of", reference_id, du, "view_image"))
                elif du < -pixel_margin:
                    relations.append(_relation(subject_id, "right_of", reference_id, -du, "view_image"))
                if dv > pixel_margin:
                    relations.append(_relation(subject_id, "above", reference_id, dv, "view_image"))
                elif dv < -pixel_margin:
                    relations.append(_relation(subject_id, "below", reference_id, -dv, "view_image"))
                if dz > depth_margin_m:
                    relations.append(_relation(subject_id, "front_of", reference_id, dz, "camera_optical"))
                elif dz < -depth_margin_m:
                    relations.append(_relation(subject_id, "behind", reference_id, -dz, "camera_optical"))
            if is_inside(subject, reference):
                relations.append(_relation(subject_id, "inside", reference_id, 0.0, "base_link"))
            distance = surface_distance(subject, reference)
            if distance <= near_threshold_m:
                relations.append(_relation(subject_id, "near", reference_id, distance, "base_link"))
    relations.sort(key=lambda item: (item["subject_id"], item["predicate"], item["object_id"]))
    return relations


def _relation(subject_id: str, predicate: str, object_id: str, margin: float, frame: str) -> dict:
    return {
        "subject_id": subject_id,
        "predicate": predicate,
        "object_id": object_id,
        "margin": round(float(margin), 6),
        "relation_frame": frame,
    }
