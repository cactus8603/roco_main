"""Build the RGB-only Sintel manifest for U0 uncertainty development."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

from stablebridge.physical_repair.u2flow_augmentations import (
    U2FlowAugmentationProfileV1,
)
from stablebridge.physical_repair.u2flow_training_data import (
    build_u2flow_training_manifest_v1,
)
from stablebridge.util import save_json


def _safe_output(path: Path) -> Path:
    absolute = path.absolute()
    lexical = PurePosixPath(str(absolute))
    for forbidden in (PurePosixPath("/tmp"), PurePosixPath("/ssd8")):
        if lexical == forbidden or forbidden in lexical.parents:
            raise ValueError(f"output must not be inside {forbidden}")
    resolved = absolute.resolve(strict=False)
    resolved_posix = PurePosixPath(str(resolved))
    for forbidden in (PurePosixPath("/tmp"), PurePosixPath("/ssd8")):
        if resolved_posix == forbidden or forbidden in resolved_posix.parents:
            raise ValueError(f"resolved output must not be inside {forbidden}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stablebridge/u0_sintel_u2flow_data_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    augmentation = config["augmentation_recipe"]
    profile = U2FlowAugmentationProfileV1(**augmentation["profile"])
    if augmentation["profile_hash"] != profile.profile_hash:
        raise ValueError("configured augmentation profile hash drift")

    manifest = build_u2flow_training_manifest_v1(config)
    output = _safe_output(args.output)
    save_json(output, manifest.as_dict())
    role_rows = {
        role: sum(row.split_role.value == role for row in manifest.rows)
        for role in ("fit", "validation", "calibration", "evaluation")
    }
    print(json.dumps({
        "status": "PASS_U2FLOW_SINTEL_RGB_MANIFEST",
        "output": str(output),
        "manifest_hash": manifest.manifest_hash,
        "split_hash": manifest.split_hash,
        "augmentation_profile_hash": profile.profile_hash,
        "rows": len(manifest.rows),
        "scenes": len({row.scene_id for row in manifest.rows}),
        "role_rows": role_rows,
        "base_frame_count": manifest.base_frame_count,
        "fusion_context_enabled": manifest.fusion_context_enabled,
        "flow_ground_truth_decoded": False,
        "future_outcomes_joined": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
