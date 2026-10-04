"""RobustSpring-20 data contract and read-only inventory checks.

This module deliberately excludes ``clean``.  It validates the public
corrupted-image panel; it does not claim that the public test images have
ground truth or that a local proxy renderer reproduces the official renderer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROBUSTSPRING20 = (
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "rain",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
)

ROBUSTSPRING_FAMILIES = {
    "color": ("brightness", "contrast", "saturate"),
    "blur": ("defocus_blur", "gaussian_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "noise": ("gaussian_noise", "impulse_noise", "speckle_noise", "shot_noise"),
    "quality": ("pixelate", "jpeg_compression", "elastic_transform"),
    "weather": ("spatter", "frost", "snow", "rain", "fog"),
}

EXPECTED_TEST_SCENES = (
    "0003",
    "0019",
    "0028",
    "0029",
    "0031",
    "0034",
    "0035",
    "0040",
    "0042",
    "0046",
)


def validate_condition_names(names: Iterable[str]) -> None:
    """Require the exact corrupted panel and reject accidental clean mixing."""

    observed = tuple(names)
    if "clean" in observed:
        raise ValueError("clean is forbidden in the S01-R robust-only panel")
    missing = sorted(set(ROBUSTSPRING20) - set(observed))
    unexpected = sorted(set(observed) - set(ROBUSTSPRING20))
    duplicates = sorted({name for name in observed if observed.count(name) > 1})
    if missing or unexpected or duplicates or len(observed) != len(ROBUSTSPRING20):
        raise ValueError(
            f"RobustSpring-20 mismatch: missing={missing}, "
            f"unexpected={unexpected}, duplicates={duplicates}"
        )


def audit_public_test_root(
    root: str | Path,
    *,
    expected_scenes: tuple[str, ...] = EXPECTED_TEST_SCENES,
    expected_frames_per_view: int = 1000,
) -> dict[str, Any]:
    """Inventory one public RobustSpring test root without reading GT."""

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    names = sorted(path.name for path in root.iterdir() if path.is_dir())
    validate_condition_names(names)

    conditions: dict[str, Any] = {}
    for name in ROBUSTSPRING20:
        test_root = root / name / "test"
        if not test_root.is_dir():
            raise FileNotFoundError(test_root)
        scenes = tuple(sorted(path.name for path in test_root.iterdir() if path.is_dir()))
        if scenes != expected_scenes:
            raise ValueError(f"{name}: scene mismatch: {scenes}")
        left_paths = tuple(test_root.glob("*/frame_left/*.png"))
        right_paths = tuple(test_root.glob("*/frame_right/*.png"))
        if len(left_paths) != expected_frames_per_view or len(right_paths) != expected_frames_per_view:
            raise ValueError(
                f"{name}: expected {expected_frames_per_view} frames/view, "
                f"found left={len(left_paths)} right={len(right_paths)}"
            )
        broken = [str(path) for path in (*left_paths, *right_paths) if not path.is_file()]
        if broken:
            raise FileNotFoundError(f"{name}: {len(broken)} broken image paths; first={broken[0]}")
        conditions[name] = {
            "scenes": list(scenes),
            "left_frames": len(left_paths),
            "right_frames": len(right_paths),
        }

    return {
        "schema": "stablebridge/robustspring20-public-test-inventory/v1",
        "root": str(root),
        "clean_included": False,
        "condition_order": list(ROBUSTSPRING20),
        "condition_count": len(conditions),
        "scene_count_per_condition": len(expected_scenes),
        "left_frames_total": sum(row["left_frames"] for row in conditions.values()),
        "right_frames_total": sum(row["right_frames"] for row in conditions.values()),
        "conditions": conditions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an exact RobustSpring-20 public test root")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_public_test_root(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
