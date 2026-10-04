"""S04 inference caches and separately opened offline Spring labels.

The strong E0 bank is exactly the three S02 direct operations followed by its
nine jointly shifted re-estimates. No GT, query mask, or GT-selected candidate
enters :func:`infer_support_case`. A coordinatewise bank median is a locked,
observable support-processing rule, not a claim that support quality improves.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .data import file_sha256
from .evaluation import error_map, gt_valid_mask
from .pipeline import local_operator
from .quartet_experiment import restore_translation


CONTEXT_HW = (320, 384)
ORIGIN_XY = (768, 380)
OPERATORS = ("identity", "median3", "gaussian1")
REMATCH_SHIFTS_XY = ((8, 0), (-8, 0), (0, 8), (0, -8), (16, 0),
                     (-16, 0), (0, 16), (0, -16), (8, 8))
CACHE_VERSION = "s04_support_inference_v1"
INFERENCE_KEYS = frozenset({
    "image0", "image1", "source_features", "target_features", "initial",
    "raw_uncertainty", "token_centers_xy", "e0_bank", "e0_valid",
    "processed_support", "support_xy", "support_raw", "support_processed",
    "support_valid",
})


def fixed_support_grid(context_hw=CONTEXT_HW, *, spacing=32, offset=15):
    """Image-independent row-major native crop coordinates; default 120 points."""
    h, w = map(int, context_hw)
    if min(h, w, spacing) <= 0 or not 0 <= offset < min(h, w, spacing):
        raise ValueError("Invalid fixed support-grid geometry")
    yy, xx = np.meshgrid(np.arange(offset, h, spacing), np.arange(offset, w, spacing), indexing="ij")
    return np.stack((xx, yy), axis=-1).reshape(-1, 2).astype(np.float32)


def support_guard_mask(context_hw, support_xy, *, radius_px=8.):
    """GT-independent pixels within inclusive Euclidean radius of any support."""
    h, w = map(int, context_hw)
    support = np.asarray(support_xy, dtype=np.float32)
    if support.ndim != 2 or support.shape[1] != 2 or not np.isfinite(support).all():
        raise ValueError("Support coordinates must be finite Sx2")
    if not np.isfinite(radius_px) or radius_px < 0:
        raise ValueError("Guard radius must be finite and nonnegative")
    yy, xx = np.mgrid[:h, :w]
    guard = np.zeros((h, w), dtype=bool)
    for x, y in support:
        guard |= (xx-x)**2 + (yy-y)**2 <= radius_px**2
    return guard


def processed_support_median(bank, valid):
    """Median of jointly finite eligible vectors, independently per component.

    Original identity is the fallback at any location without eligible entries.
    The median can synthesize a new vector and need not improve its GT error.
    """
    bank = np.asarray(bank, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if bank.ndim != 4 or bank.shape[1] != 2 or valid.shape != (bank.shape[0], *bank.shape[2:]):
        raise ValueError("Expected bank Kx2xHxW and valid KxHxW")
    if bank.shape[0] < 1 or not np.isfinite(bank[0]).all():
        raise ValueError("Identity must be present and finite everywhere")
    eligible = valid & np.isfinite(bank).all(axis=1)
    # Identity fallback is added only to empty sets; it does not override a
    # caller's explicit validity mask for nonempty sets.
    eligible = eligible.copy()
    empty = ~eligible.any(axis=0)
    eligible[0] |= empty
    values = np.where(eligible[:, None], bank, np.nan)
    return np.nanmedian(values, axis=0).astype(np.float32)


def _support_indices(support_xy, context_hw):
    points = np.asarray(support_xy)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite support coordinates Sx2")
    if not np.equal(points, np.rint(points)).all():
        raise ValueError("Sparse support readout requires integer pixel centers")
    x, y = points.astype(np.int64).T
    h, w = context_hw
    if ((x < 0) | (x >= w) | (y < 0) | (y >= h)).any():
        raise ValueError("Support coordinates outside crop")
    return x, y


def _case_arguments(case):
    task = case["task"]
    if task not in ("stereo", "flow"):
        raise ValueError("Task must be stereo or flow")
    profile = case.get("profile", {"name": "clean"})
    if isinstance(profile, str):
        profile = {"name": profile}
    if not isinstance(profile, dict) or "name" not in profile:
        raise ValueError("Profile requires {'name': ..., 'options': {...}}")
    return task, profile, case.get("view", "left"), case.get("split", "train")


def _cuda_start(adapter):
    device = torch.device(adapter.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return device


def _cuda_measure(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
    return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}


def infer_support_case(adapter, provider, case, *, seed=0, context_hw=CONTEXT_HW,
                       origin_xy=ORIGIN_XY, support_spacing=32, support_offset=15):
    """Read the original pair and execute 12 frozen forwards, never opening GT.

    ``case`` has task, scene, frame and profile={name,options}; optional split,
    view and case_id fields only affect provenance/input selection. RGB remains
    uint8; encoder features are float32, frozen and unnormalized. Support
    coordinates are crop-local; token centers remain global as in CroCoAdapter.
    """
    start = time.perf_counter()
    task, profile, view, split = _case_arguments(case)
    if adapter.task != task or tuple(adapter.context_hw) != tuple(context_hw):
        raise ValueError("Adapter task/context differs from the case contract")
    if task == "stereo" and view != "left":
        raise ValueError("This S04 cache uses left-to-right stereo only")
    pair = provider.read_pair(task, case["scene"], case["frame"], context_hw,
                              view=view, split=split, profile=profile["name"], seed=seed,
                              corruption_options=profile.get("options"), origin_xy=origin_xy)
    if pair["metadata"].get("gt_read"):
        raise ValueError("Provider supplied a GT-contaminated pair")
    read_seconds = time.perf_counter()-start
    bank, valid, ledger, arrays = [], [], [], {}
    operations = [("operation", op) for op in OPERATORS] + [("shift", shift) for shift in REMATCH_SHIFTS_XY]
    for index, (kind, argument) in enumerate(operations):
        device = _cuda_start(adapter)
        op_start = time.perf_counter()
        im0, im1 = pair["image0"], pair["image1"]
        if kind == "operation" and argument != "identity":
            im0, im1 = local_operator(im0, argument), local_operator(im1, argument)
        elif kind == "shift":
            dx, dy = argument
            im0 = np.roll(im0, (dy, dx), axis=(0, 1))
            im1 = np.roll(im1, (dy, dx), axis=(0, 1))
        prediction = adapter.predict(im0, im1, origin_xy=origin_xy)
        field = np.asarray(prediction.displacement, dtype=np.float32)
        if field.shape != (2, *context_hw) or not np.isfinite(field).all():
            raise ValueError("Frozen predictor must return finite 2xHxW displacement")
        if kind == "shift":
            field, eligibility = restore_translation(field, argument)
        else:
            # Match S02 C0 semantics: direct candidates remain defined even
            # when endpoints leave the crop. Search readout handles that later.
            eligibility = np.isfinite(field).all(axis=0)
        memory = _cuda_measure(device)
        ledger.append({"index": index, "kind": kind, "operation": argument,
                       "forward_calls": 1, "charged_seconds": time.perf_counter()-op_start,
                       **memory, "prediction": prediction.metadata})
        bank.append(field.copy())
        valid.append(eligibility.copy())
        if index == 0:
            arrays.update({"image0": pair["image0"], "image1": pair["image1"],
                           "source_features": prediction.source_features.copy(),
                           "target_features": prediction.target_features.copy(),
                           "initial": field.copy(), "raw_uncertainty": prediction.raw_uncertainty.copy(),
                           "token_centers_xy": prediction.token_centers_xy.copy()})
        del prediction
    arrays["e0_bank"] = np.stack(bank)
    arrays["e0_valid"] = np.stack(valid)
    process_start = time.perf_counter()
    arrays["processed_support"] = processed_support_median(arrays["e0_bank"], arrays["e0_valid"])
    arrays["support_xy"] = fixed_support_grid(context_hw, spacing=support_spacing, offset=support_offset)
    sx, sy = _support_indices(arrays["support_xy"], context_hw)
    arrays["support_raw"] = arrays["initial"][:, sy, sx].T.copy()
    arrays["support_processed"] = arrays["processed_support"][:, sy, sx].T.copy()
    arrays["support_valid"] = (np.isfinite(arrays["support_raw"]).all(axis=1) &
                                 np.isfinite(arrays["support_processed"]).all(axis=1))
    processing_seconds = time.perf_counter()-process_start
    source_root = Path(__file__).parent
    metadata = {
        "cache_version": CACHE_VERSION, "case": dict(case), "seed": int(seed),
        "context_hw": list(context_hw), "origin_xy": list(origin_xy), "pair": pair["metadata"],
        "gt_read": False, "gt_used_for_candidate_or_support_selection": False,
        "feature_storage": "float32_native_post_enc_norm_stride16_offset7.5",
        "bank_order": [{"kind": k, "operation": a} for k, a in operations],
        "support": {"count": len(sx), "spacing_px": support_spacing, "offset_px": support_offset,
                    "coordinates": "crop_local_native_integer_centers",
                    "processed_rule": "coordinatewise_median_of_jointly_finite_valid_E0_bank",
                    "quality_improvement_guaranteed": False},
        "cost": {"forward_calls": len(ledger), "identity_forward_calls": 1,
                 "incremental_E0_processing_forward_calls": len(ledger)-1,
                 "image_read_seconds": read_seconds, "support_processing_seconds": processing_seconds,
                 "charged_forward_seconds": sum(row["charged_seconds"] for row in ledger),
                 "cache_build_wall_seconds": time.perf_counter()-start,
                 "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in ledger),
                 "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in ledger),
                 "process_lifetime_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                 "observation_wait_frames": 0,
                 "cache_storage_seconds_excluded": True},
        "ledger": ledger,
        "source_sha256": {name: file_sha256(source_root/name) for name in
                          ("support_data.py", "backbones.py", "data.py", "pipeline.py", "quartet_experiment.py")},
    }
    return {"arrays": arrays, "metadata": metadata}


def _atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


def save_support_cache(path, result):
    """Commit inference-only NPZ, then its hash/provenance JSON commit record."""
    path = Path(path)
    arrays = result["arrays"]
    if set(arrays) != INFERENCE_KEYS or result["metadata"].get("gt_read") is not False:
        raise ValueError("Inference cache schema is closed and forbids GT/query labels")
    _atomic_npz(path, arrays)
    metadata = {**result["metadata"], "artifact": {"path": str(path), "sha256": file_sha256(path)}}
    _atomic_json(path.with_suffix(".json"), metadata)
    return metadata


def load_support_cache(path, *, verify_hash=True):
    """Read only inference data; never discovers or opens a labels artifact."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata.get("cache_version") != CACHE_VERSION or metadata.get("gt_read") is not False:
        raise ValueError("Unsupported or contaminated inference cache")
    if verify_hash and file_sha256(path) != metadata["artifact"]["sha256"]:
        raise ValueError("Inference cache hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != INFERENCE_KEYS:
            raise ValueError("Inference cache schema mismatch; GT labels are forbidden")
        arrays = {key: archive[key].copy() for key in archive.files}
    return {"arrays": arrays, "metadata": metadata}


def read_case_labels(provider, case, *, roi_xyhw, seed=0):
    """Explicit offline/training-only GT read; keep result away from model kwargs."""
    task, profile, view, split = _case_arguments(case)
    gt4 = provider.read_corrupted_gt(
        task, case["scene"], case["frame"], view=view, split=split,
        roi_xyhw=roi_xyhw, profile=profile["name"], seed=seed)
    path = provider.gt_path(task, case["scene"], case["frame"], view=view, split=split)
    return {"gt4": gt4, "metadata": {"role": "training_targets_and_offline_evaluation_only",
            "gt_read": True, "path": str(path), "file_sha256": file_sha256(path),
            "roi_xyhw": list(roi_xyhw), "units": "native_pixel_vector_min4"}}


def sparse_gt_support(gt4, support_xy):
    """Privileged diagnostic only: fixed GT branch zero at fixed support points.

    This never chooses a branch by the query answer, or fills invalid support
    labels from another branch. The caller must reject support-to-query
    distances <=8px per query, including during full-image evaluation.
    """
    gt4 = np.asarray(gt4, dtype=np.float32)
    if gt4.ndim != 4 or gt4.shape[:2] != (4, 2):
        raise ValueError("Expected Spring GT 4x2xHxW")
    x, y = _support_indices(support_xy, gt4.shape[2:])
    values = gt4[0, :, y, x].copy()
    valid = np.isfinite(values).all(axis=1)
    return np.where(valid[:, None], values, 0).astype(np.float32), valid


def offline_labels(initial, bank, bank_valid, gt4, support_xy, *, threshold_px=1., guard_px=8.):
    """Strong-E0 fixed Q on ALL GT-valid pixels; support-guard subset is secondary."""
    initial, bank, gt4 = np.asarray(initial), np.asarray(bank), np.asarray(gt4)
    bank_valid = np.asarray(bank_valid, dtype=bool)
    if threshold_px <= 0 or not np.isfinite(threshold_px):
        raise ValueError("Error threshold must be finite and positive")
    if bank.ndim != 4 or bank.shape[1:] != initial.shape or bank.shape[0] != 12:
        raise ValueError("Strong E0 requires exact 12-candidate bank")
    if bank_valid.shape != (12, *initial.shape[1:]):
        raise ValueError("Strong E0 validity shape mismatch")
    if not np.array_equal(bank[0], initial) or not bank_valid[0].all():
        raise ValueError("Identity must be the first always-retained strong E0 candidate")
    valid = gt_valid_mask(gt4)
    initial_error = error_map(initial, gt4)
    candidate_errors = np.stack([error_map(candidate, gt4) for candidate in bank])
    candidate_errors = np.where(bank_valid, candidate_errors, np.inf)
    oracle_error = candidate_errors.min(axis=0)
    q = valid & (oracle_error > threshold_px)
    guard = support_guard_mask(initial.shape[1:], support_xy, radius_px=guard_px)
    return {"gt4": gt4.copy(), "gt_valid": valid, "initial_error": initial_error,
            "strong_e0_oracle_error": oracle_error, "fixed_q": q,
            "support_guard": guard, "fixed_q_outside_support_guard": q & ~guard,
            "initial_good": valid & (initial_error <= threshold_px)}


def save_label_cache(path, arrays, metadata):
    """Physically separate label archive; cannot be loaded as an inference cache."""
    path = Path(path)
    if not metadata.get("gt_read") or metadata.get("role") != "training_targets_and_offline_evaluation_only":
        raise ValueError("Label storage must declare privileged offline role")
    _atomic_npz(path, arrays)
    record = {**metadata, "artifact": {"path": str(path), "sha256": file_sha256(path)}}
    _atomic_json(path.with_suffix(".json"), record)
    return record
