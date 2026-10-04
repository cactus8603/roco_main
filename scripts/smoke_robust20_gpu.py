#!/usr/bin/env python3
"""Two-task GPU smoke for Robust20 rendering, elastic GT and CroCo inference."""

import json
import numpy as np
import torch

from stablebridge.backbones import CroCoAdapter
from stablebridge.data import SpringProvider
from stablebridge.evaluation import error_map
from stablebridge.util import ROOT


def main():
    registry_path = ROOT / "configs/stablebridge/data_models_v1.json"
    registry = json.loads(registry_path.read_text())
    provider = SpringProvider.from_registry(registry_path)
    device = "cuda:6"
    torch.cuda.set_device(device)
    output = []
    for task, context in (("stereo", (352, 704)), ("flow", (320, 384))):
        model = registry["models"]["main"][task]
        adapter = CroCoAdapter(task, model["local_path"], device, context, model["sha256"])
        for profile in ("robust20__zoom_blur__s3", "robust20__elastic_transform__s2"):
            pair = provider.read_pair(task, "0001", 50, context, profile=profile, seed=20260917)
            prediction = adapter.predict(pair["image0"], pair["image1"], pair["origin_xy"])
            gt = provider.read_corrupted_gt(task, "0001", 50, roi_xyhw=pair["roi_xyhw"],
                                            profile=profile, seed=20260917)
            error = error_map(prediction.displacement, gt)
            valid = np.isfinite(error)
            output.append({
                "task": task, "profile": profile, "valid_pixels": int(valid.sum()),
                "mean_error_px": float(error[valid].mean()),
                "prediction_shape": list(prediction.displacement.shape),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            })
        del adapter
        torch.cuda.empty_cache()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
