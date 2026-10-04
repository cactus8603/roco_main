"""Shared RobustSpring-20 proxy adapter for stereo and optical-flow studies.

The pixel renderer is reused from the audited stereo track and is checksum
pinned.  This adapter adds a stable observation identity for arbitrary
stereo/temporal endpoints and transports vector ground truth through elastic
warps.  Weather and motion remain local proxies, not the private official
renderer.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import re
import sys
from typing import Any

import cv2
import numpy as np

from .robust20 import ROBUSTSPRING20


RENDERER_PATH = Path(
    "/ssd7/cactus8603/roco_spring/stereo-track-20260827/"
    "tracks/stereo/robust_proxy/corruptions.py"
)
RENDERER_SHA256 = "4f237537510041ee10eeb4a06bb4a58f214955f2070aa986f44a4d9f023688f4"
PROFILE_RE = re.compile(r"^robust20__(?P<condition>[a-z0-9_]+)__s(?P<severity>[123])$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def renderer():
    if not RENDERER_PATH.is_file():
        raise FileNotFoundError(RENDERER_PATH)
    actual = _sha256(RENDERER_PATH)
    if actual != RENDERER_SHA256:
        raise ValueError(f"Robust20 renderer checksum mismatch: {actual}")
    module_name = "stablebridge_audited_robust20_renderer"
    spec = importlib.util.spec_from_file_location(module_name, RENDERER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {RENDERER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if tuple(module.CORRUPTION_NAMES) != tuple(
        name for family in ("color", "blur", "noise", "quality", "weather")
        for name in module.CORRUPTION_NAMES if module.CORRUPTION_SPECS[name].family == family
    ):
        raise ValueError("Renderer corruption specification is internally inconsistent")
    if set(module.CORRUPTION_NAMES) != set(ROBUSTSPRING20):
        raise ValueError("Renderer does not cover exact RobustSpring-20")
    return module


def profile_name(condition: str, severity: int) -> str:
    if condition not in ROBUSTSPRING20 or severity not in (1, 2, 3):
        raise ValueError(f"Invalid Robust20 profile: {condition}, severity={severity}")
    return f"robust20__{condition}__s{severity}"


def robust20_profiles(severities=(1, 2, 3)) -> list[dict[str, Any]]:
    return [
        {"name": profile_name(condition, severity),
         "condition": condition, "severity": severity, "family": renderer().CORRUPTION_SPECS[condition].family}
        for severity in severities for condition in ROBUSTSPRING20
    ]


def expand_profile_spec(value: Any) -> list[dict[str, Any]]:
    """Expand the compact registered matrix while preserving legacy lists."""
    if isinstance(value, list):
        return value
    if not isinstance(value, dict) or value.get("kind") != "robustspring20_proxy":
        raise ValueError("profiles must be a list or a robustspring20_proxy specification")
    if set(value) != {"kind", "severities"}:
        raise ValueError("Robust20 profile spec accepts exactly kind and severities")
    severities = tuple(value["severities"])
    if not severities or len(set(severities)) != len(severities) or any(s not in (1, 2, 3) for s in severities):
        raise ValueError("Robust20 severities must be a nonempty unique subset of [1,2,3]")
    return robust20_profiles(severities)


def parse_profile(name: str) -> tuple[str, int] | None:
    match = PROFILE_RE.fullmatch(name)
    if match is None:
        return None
    condition, severity = match.group("condition"), int(match.group("severity"))
    if condition not in ROBUSTSPRING20:
        raise ValueError(f"Unknown RobustSpring condition in profile: {condition}")
    return condition, severity


def _observation_seed(seed: int, scene: str, frame: int, view: str, condition: str) -> tuple[int, dict[str, Any]]:
    module = renderer()
    spec = module.CORRUPTION_SPECS[condition]
    identity = {
        "base_seed": int(seed),
        "scene": f"{int(scene):04d}",
        "condition": condition,
        "frame": None if spec.time_consistent else int(frame),
        "view": None if spec.official_coupling == "shared" else view,
        "renderer_sha256": RENDERER_SHA256,
    }
    token = repr(sorted(identity.items())).encode()
    derived = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return derived, identity


def observation_maps(
    shape_hw: tuple[int, int], *, profile: str, seed: int, scene: str, frame: int, view: str
) -> tuple[np.ndarray, np.ndarray] | None:
    parsed = parse_profile(profile)
    if parsed is None or parsed[0] != "elastic_transform":
        return None
    module = renderer()
    condition, severity = parsed
    parameters = module._severity_parameters(condition, severity)
    derived, _ = _observation_seed(seed, scene, frame, view, condition)
    rng = np.random.default_rng(derived)
    return module._elastic_maps(
        shape_hw, rng, float(parameters["alpha"]), float(parameters["sigma"])
    )


def apply_observation(
    image: np.ndarray,
    *,
    profile: str,
    seed: int,
    scene: str,
    frame: int,
    view: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    parsed = parse_profile(profile)
    if parsed is None:
        raise ValueError(f"Not a Robust20 profile: {profile}")
    module = renderer()
    condition, severity = parsed
    parameters = module._severity_parameters(condition, severity)
    derived, identity = _observation_seed(seed, scene, frame, view, condition)
    output, maps = module._apply_one(
        np.asarray(image), condition, parameters, derived, disparity=None,
        elastic_maps=observation_maps(
            image.shape[:2], profile=profile, seed=seed, scene=scene, frame=frame, view=view
        ),
    )
    spec = module.CORRUPTION_SPECS[condition]
    metadata = {
        "version": "stablebridge_robust20_proxy_v1",
        "profile": profile,
        "condition": condition,
        "family": spec.family,
        "severity": severity,
        "parameters": parameters,
        "derived_seed": derived,
        "seed_identity": identity,
        "temporal_correlation": "persistent" if spec.time_consistent else "independent",
        "view_correlation": spec.official_coupling,
        "geometry_preserved": not spec.changes_geometry,
        "changes_geometry": spec.changes_geometry,
        "exact_official_renderer": spec.exact_official_renderer,
        "renderer_path": str(RENDERER_PATH),
        "renderer_sha256": RENDERER_SHA256,
        "applied_before_crop": True,
        "shape_hw": list(image.shape[:2]),
        "full_corrupted_rgb_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
    }
    if maps is not None:
        metadata["elastic_map_sha256"] = hashlib.sha256(
            np.ascontiguousarray(np.stack(maps)).tobytes()
        ).hexdigest()
    return output, metadata


def _sample(array: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        np.asarray(array, np.float32), x.astype(np.float32), y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def transform_vector_ground_truth(
    gt4: np.ndarray,
    source_maps: tuple[np.ndarray, np.ndarray],
    target_maps: tuple[np.ndarray, np.ndarray],
    *,
    inverse_iterations: int = 8,
) -> np.ndarray:
    """Transport four dense vector-GT branches through two output-to-input maps."""

    gt4 = np.asarray(gt4, np.float32)
    if gt4.ndim != 4 or gt4.shape[:2] != (4, 2):
        raise ValueError("Expected four vector GT branches with shape [4,2,H,W]")
    source_x, source_y = (np.asarray(value, np.float32) for value in source_maps)
    target_x, target_y = (np.asarray(value, np.float32) for value in target_maps)
    h, w = source_x.shape
    if gt4.shape[2:] != (h, w) or any(value.shape != (h, w) for value in (source_y, target_x, target_y)):
        raise ValueError("Elastic maps and GT grid differ")
    grid_y, grid_x = np.mgrid[:h, :w].astype(np.float32)
    target_dx, target_dy = target_x - grid_x, target_y - grid_y
    source_inside = ((source_x >= 0) & (source_x <= w - 1) &
                     (source_y >= 0) & (source_y <= h - 1))
    transformed = np.full_like(gt4, np.nan)
    for branch in range(4):
        clean_u = gt4[branch, 0]
        clean_v = gt4[branch, 1]
        clean_valid = np.isfinite(clean_u) & np.isfinite(clean_v)
        valid_weight = _sample(clean_valid.astype(np.float32), source_x, source_y)
        u = _sample(np.where(clean_valid, clean_u, 0.0), source_x, source_y)
        v = _sample(np.where(clean_valid, clean_v, 0.0), source_x, source_y)
        endpoint_x, endpoint_y = source_x + u, source_y + v
        output_x, output_y = endpoint_x.copy(), endpoint_y.copy()
        for _ in range(max(1, int(inverse_iterations))):
            output_x = endpoint_x - _sample(target_dx, output_x, output_y)
            output_y = endpoint_y - _sample(target_dy, output_x, output_y)
        valid = (source_inside & (valid_weight > 0.999) & np.isfinite(output_x) & np.isfinite(output_y)
                 & (endpoint_x >= 0) & (endpoint_x <= w - 1)
                 & (endpoint_y >= 0) & (endpoint_y <= h - 1)
                 & (output_x >= 0) & (output_x <= w - 1)
                 & (output_y >= 0) & (output_y <= h - 1))
        transformed[branch, 0, valid] = output_x[valid] - grid_x[valid]
        transformed[branch, 1, valid] = output_y[valid] - grid_y[valid]
    return transformed
