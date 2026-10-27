"""Checked, portable feature datasets. This module never calls WAFT or reads GT."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA = "waft-repair-dataset"
FEATURE_SCHEMA = "waft-repair-features"
CHEAP_METHOD = "waft-visible-pull-rgb-v1"
EXPENSIVE_METHOD = "waft-dense-final-epe-512-v1"
_CHECKED: dict[tuple, tuple] = {}
_NORMALIZATION: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return
    temporary = path.with_name(path.name + ".writing-" + str(os.getpid()))
    try:
        write_json(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def native_mask(native_hw, padding, feature_hw) -> torch.Tensor:
    """Feature center support at WAFT's original half-resolution grid."""
    require(len(native_hw) == 2 and all(type(v) is int and v > 0 for v in native_hw), "native_hw must be positive [H,W]")
    require(len(padding) == 4 and all(type(v) is int and v >= 0 for v in padding), "padding must be nonnegative [left,right,top,bottom]")
    require(len(feature_hw) == 2 and all(type(v) is int and v > 1 for v in feature_hw), "feature grid must have H,W > 1")
    height, width = native_hw
    left, right, top, bottom = padding
    fh, fw = feature_hw
    require((2 * fh, 2 * fw) == (height + top + bottom, width + left + right), "Input and feature padded geometry differ")
    yy, xx = torch.meshgrid(torch.arange(fh), torch.arange(fw), indexing="ij")
    mask = ((2 * xx - left >= 0) & (2 * xx - left < width)
            & (2 * yy - top >= 0) & (2 * yy - top < height))[None, None, None]
    require(mask.any(), "Native feature support must be nonempty")
    return mask


def validate_features(value: torch.Tensor) -> None:
    require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == torch.float32,
            "Feature packets require CPU FP32 tensors")
    require(value.ndim == 5 and tuple(value.shape[:3]) == (1, 2, 64), "Features must have shape [1,2,64,H,W]")
    require(not value.requires_grad and value.grad_fn is None, "Feature packets must contain detached tensors")
    require(torch.isfinite(value).all(), "Every feature coordinate, including padding, must be finite")


@torch.no_grad()
def compute_sigma(original: torch.Tensor, native_hw, padding) -> tuple[torch.Tensor, torch.Tensor]:
    """Original-only paired native RMS, in FP64, floored before FP32 storage.

    The floor is 0.01 * lower median(channel RMS) + 1e-8. This is an
    uncentered RMS, not a standard deviation. Targets never enter this call.
    """
    validate_features(original)
    mask = native_mask(native_hw, padding, tuple(original.shape[-2:]))
    values = original[0, :, :, mask[0, 0, 0]].double()
    rms = values.square().mean(dim=(0, 2)).sqrt()
    sigma = rms.clamp_min(0.01 * rms.median() + 1e-8).float()
    require(torch.isfinite(sigma).all() and (sigma > 0).all(), "Finite positive RMS normalization required")
    return sigma, mask


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def load_packet(path: str | Path) -> dict:
    """Load tensors and basic metadata without pickle object execution."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            value = {}
            for key in archive.files:
                item = archive[key]
                if key == "metadata":
                    value.update(json.loads(str(_plain(item))))
                elif item.dtype.kind in "biufc" and item.ndim:
                    value[key] = torch.from_numpy(item.copy())
                else:
                    value[key] = _plain(item)
        # Small geometry arrays are metadata; never feature operands.
        for key in ("native_hw", "padding"):
            if isinstance(value.get(key), torch.Tensor):
                value[key] = value[key].tolist()
    elif path.suffix.lower() in (".pt", ".pth"):
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    else:
        raise ValueError("Feature files must be .npz or .pt/.pth")
    require(isinstance(value, dict), "Feature packet must be a mapping")
    return value


def packet_checkpoint_sha(packet: dict) -> str | None:
    value = packet.get("waft_checkpoint_sha256")
    if value is None:
        value = packet.get("model", {}).get("model_item", {}).get("sha256")
    if value is None:
        value = packet.get("provenance", {}).get("checkpoint", {}).get("sha256")
    return value


def packet_pair_id(packet: dict) -> str | None:
    return packet.get("pair_id", packet.get("case_id"))


def _domain(value: str) -> str:
    return value.replace("_", "-")


def _dense_contract(packet: dict, recipe: dict | None = None) -> bool:
    """Recognize the accepted compact cache by its numerical method contract."""
    method = packet.get("provenance", {}).get("fixed_method", {})
    if "common_recipe" in (recipe or {}):
        common = recipe["common_recipe"]
        optimizer = common.get("optimizer", {})
        return (packet.get("quality", {}).get("passed") is True
                and common.get("WAFT_updates") == 5 and common.get("total_updates") == 512
                and common.get("warmup") == 0 and common.get("early_75percent_stop") is False
                and optimizer == dict(name="AdamW", lr=0.001, betas=[0.9, 0.999], eps=1e-8,
                                      weight_decay=0.0, foreach=False, fused=False)
                and common.get("projection") == "hard-clamp delta to [-0.25,+0.25] and zero padding after every Adam update"
                and common.get("selection") == "first strict minimum final EPE including F0 and every retained candidate"
                and common.get("clipping") == "global FP64 gradient norm max1, original1e-6 denominator, copy to FP32"
                and common.get("backward") == "complete recurrent-flow and sampling-coordinate graph; only delta updates"
                and common.get("loss") == "mean vector EPE of only fifth native flow against finite-valid GT")
    expected = dict(maximum_updates=512, fixed_total_updates=512, normalized_delta_bound=0.25,
                    learning_rate=0.001, betas=[0.9, 0.999], epsilon=1e-8, weight_decay=0.0,
                    clip_norm=1.0, clip_precision="FP64", warmup_updates=0, stop_geometry=False,
                    objective="FP32_mean_final_fifth_flow_vector_EPE_on_fixed_finite_valid_GT",
                    projection="after_AdamW_hardclamp_delta_to_plusminus0.25_then_zero_padding",
                    selection="first_strict_minimum_search_EPE_including_original_F0")
    return (all(method.get(key) == value for key, value in expected.items())
            and packet.get("quality", {}).get("passed") is True)


def _geometry(packet: dict, row: dict, *, target: bool) -> None:
    hw = packet.get("native_hw")
    if hw is None:
        rgb = packet.get("teacher_rgb" if target else "rgb")
        if isinstance(rgb, torch.Tensor):
            require(tuple(rgb.shape[:3]) == (1, 2, 3), "Packet RGB must contain one complete pair")
            hw = list(rgb.shape[-2:])
    require(hw is not None and list(hw) == row["native_hw"], "Packet native geometry differs from the manifest")
    require(list(packet.get("padding", [])) == row["padding"], "Packet padding differs from the manifest")


def _file_ref(value: str | dict, base: Path, output_base: Path) -> dict:
    item = {"path": value} if isinstance(value, str) else dict(value)
    path = (base / item["path"]).resolve(strict=True)
    actual = sha256(path)
    if "sha256" in item:
        require(item["sha256"] == actual, "Bound feature file SHA256 changed: " + str(path))
    if "bytes" in item:
        require(item["bytes"] == path.stat().st_size, "Bound feature file size changed: " + str(path))
    return {"path": os.path.relpath(path, output_base), "sha256": actual, "bytes": path.stat().st_size}


def _check_file(value: dict, base: Path) -> Path:
    require(set(value) >= {"path", "sha256", "bytes"} and valid_sha(value["sha256"]), "A SHA256-bound feature file is required; run prepare/import-cache first")
    path = (base / value["path"]).resolve(strict=True)
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    key = (str(path), value["sha256"], value["bytes"])
    if _CHECKED.get(key) != identity:
        require(stat.st_size == value["bytes"] and sha256(path) == value["sha256"], "Bound feature file changed: " + str(path))
        _CHECKED[key] = identity
    return path


def _validate_manifest(value: dict) -> None:
    require(value.get("schema") == SCHEMA, "Unsupported dataset schema")
    require(valid_sha(value.get("waft_checkpoint_sha256")), "A WAFT checkpoint SHA256 is required")
    require(value.get("target_kind") in ("cheap", "expensive"), "target_kind must be cheap or expensive")
    rows = value.get("samples")
    require(isinstance(rows, list) and bool(rows), "At least one explicit feature sample is required")
    ids = []
    for row in rows:
        require(isinstance(row.get("pair_id"), str) and bool(row["pair_id"]), "Each sample requires a pair_id")
        require(row.get("domain") in ("kitti", "sintel-clean", "sintel-final", "synthetic"), "Unknown sample domain")
        require("input_file" in row and "target_file" in row, "Each training row needs input_file and target_file")
        if row["domain"] == "sintel-final" and value["target_kind"] == "cheap":
            require(isinstance(row.get("target_source_pair_id"), str) and bool(row["target_source_pair_id"]), "Cheap Final targets require an explicit paired Clean target_source_pair_id")
        ids.append(row["pair_id"])
    require(len(ids) == len(set(ids)), "Duplicate pair_id would silently change the fitting population")


def load_manifest(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    _validate_manifest(value)
    value["_base"] = str(path.parent)
    value["_sha256"] = sha256(path)
    return value


def load_sample(manifest: str | Path | dict, row: int | dict, *, require_target: bool = True) -> dict:
    """Return only feature learning operands plus identity/geometry metadata."""
    manifest = load_manifest(manifest) if not isinstance(manifest, dict) else manifest
    row = manifest["samples"][row] if isinstance(row, int) else row
    base = Path(manifest["_base"])
    for provenance_key in ("accepted_entry", "independent_audit", "target_method_file"):
        if provenance_key in row:
            _check_file(row[provenance_key], base)
    original_packet = load_packet(_check_file(row["input_file"], base))
    require(packet_pair_id(original_packet) == row["pair_id"], "Input pair identity differs from the manifest")
    require(packet_checkpoint_sha(original_packet) == manifest["waft_checkpoint_sha256"], "Input WAFT checkpoint differs from the dataset")
    _geometry(original_packet, row, target=False)
    if "domain" in original_packet:
        require(_domain(original_packet["domain"]) == row["domain"], "Input packet domain differs")
    original = original_packet[row.get("input_key", "features")]
    validate_features(original)
    normalization_key = (row["input_file"]["sha256"], row.get("input_key", "features"),
                         tuple(row["native_hw"]), tuple(row["padding"]))
    if normalization_key not in _NORMALIZATION:
        _NORMALIZATION[normalization_key] = compute_sigma(original, row["native_hw"], row["padding"])
    sigma, mask = _NORMALIZATION[normalization_key]
    value = dict(original=original, sigma=sigma, native_mask=mask, pair_id=row["pair_id"],
                 domain=row["domain"], native_hw=row["native_hw"], padding=row["padding"],
                 waft_checkpoint_sha256=manifest["waft_checkpoint_sha256"])
    if not require_target:
        return value
    target_path = _check_file(row["target_file"], base)
    input_path = (base / row["input_file"]["path"]).resolve()
    packet = original_packet if target_path == input_path else load_packet(target_path)
    source_id = row.get("target_source_pair_id", row["pair_id"])
    final_clean = row["domain"] == "sintel-final" and manifest["target_kind"] == "cheap"
    canonical_final = (final_clean and packet_pair_id(packet) == row["pair_id"]
                       and packet.get("target_source_pair_id") == source_id)
    require(packet_pair_id(packet) == source_id or canonical_final, "Target pair identity differs from the manifest")
    require(packet_checkpoint_sha(packet) == manifest["waft_checkpoint_sha256"], "Target WAFT checkpoint differs from the input")
    _geometry(packet, row, target=True)
    expected_kind = manifest["target_kind"]
    expected_method = CHEAP_METHOD if expected_kind == "cheap" else EXPENSIVE_METHOD
    method = packet.get("target_method", packet.get("method"))
    # Existing audited visible packets predate the portable metadata envelope.
    legacy_cheap = packet.get("arm") == {"name": "visible_pull_1", "kind": "pull", "alpha": 1.0, "encoder_calls": 1}
    recipe = None
    if expected_kind == "expensive" and "target_method_file" in row:
        bound_method = packet.get("provenance", {}).get("fixed_method", {})
        require(bound_method.get("sha256") == row["target_method_file"]["sha256"], "Dense packet and bound method document differ")
        recipe = json.loads(_check_file(row["target_method_file"], base).read_text(encoding="utf-8"))
    legacy_expensive = expected_kind == "expensive" and _dense_contract(packet, recipe)
    require(packet.get("target_kind") == expected_kind or (expected_kind == "cheap" and legacy_cheap)
            or legacy_expensive, "Target packet kind is missing or differs from the manifest")
    require(method == expected_method or (expected_kind == "cheap" and legacy_cheap) or legacy_expensive,
            "Target packet must identify the selected published method")
    target = packet[row.get("target_key", "teacher_features")]
    validate_features(target)
    require(target.shape == original.shape, "Target and input must cover the same complete padded pair")
    if expected_kind == "cheap" and row["domain"] != "sintel-final":
        require(torch.equal(original[:, 1].contiguous().view(torch.uint8), target[:, 1].contiguous().view(torch.uint8)), "Cheap target must preserve all second-frame feature bytes")
    if final_clean:
        teacher_domain = packet.get("teacher_domain", packet.get("domain"))
        require(teacher_domain in ("sintel-clean", "sintel_clean"), "Cheap Final target must explicitly identify the complete paired Clean field")
    value["target"] = target
    return value


def prepare_manifest(spec_path: str | Path, output_path: str | Path) -> dict:
    """Bind an explicit JSON sample specification to existing packet contents."""
    spec_path, output_path = Path(spec_path).resolve(strict=True), Path(output_path).resolve()
    require(not output_path.exists(), "Manifest output already exists")
    value = json.loads(spec_path.read_text(encoding="utf-8"))
    _validate_manifest(value)
    for row in value["samples"]:
        for key in ("input_file", "target_file"):
            row[key] = _file_ref(row[key], spec_path.parent, output_path.parent)
    # Verify data before creating any output. No model, images, or GT are loaded.
    value["_base"] = str(output_path.parent)
    for row in value["samples"]:
        load_sample(value, row)
    value.pop("_base")
    write_json(output_path, value)
    return value


def import_dense_cache(entries_dir: str | Path, output_path: str | Path) -> dict:
    """Import only accepted index entries, using each packet's original/target.

    Compact cache files are referenced in place. Every entry and audit binding
    is retained; unindexed or unqualified physical files are never imported.
    No target is selected based on a new GT evaluation.
    """
    entries_dir = Path(entries_dir).resolve(strict=True)
    output_path = Path(output_path).resolve()
    require(entries_dir.is_dir(), "entries_dir must be an accepted-entry directory")
    require(not output_path.exists(), "Manifest output already exists")
    rows, checkpoint_sha = [], None
    for entry_path in sorted(entries_dir.rglob("entry.json")):
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
        if entry.get("qualified") is not True:
            continue
        require(entry.get("teacher_updates") == 512 and type(entry.get("selected_step")) is int
                and 0 <= entry["selected_step"] <= 512, "Accepted entry must retain the complete 512-update method")
        packet_ref = _file_ref(entry["packet"], entry_path.parent, output_path.parent)
        packet = load_packet(output_path.parent / packet_ref["path"])
        method_ref, recipe = None, None
        method_value = packet.get("provenance", {}).get("fixed_method", {})
        if set(method_value) >= {"path", "sha256", "bytes"}:
            method_ref = _file_ref(method_value, entry_path.parent, output_path.parent)
            recipe = json.loads((output_path.parent / method_ref["path"]).read_text(encoding="utf-8"))
        require(_dense_contract(packet, recipe), "Accepted packet does not identify the fixed expensive method")
        pair_id = packet_pair_id(packet)
        require(pair_id == entry.get("case_id"), "Accepted entry and packet pair identity differ")
        require(_domain(packet["domain"]) == _domain(entry["domain"]), "Accepted entry and packet domain differ")
        current_sha = packet_checkpoint_sha(packet)
        require(valid_sha(current_sha) and (checkpoint_sha is None or checkpoint_sha == current_sha), "Accepted cache mixes frozen WAFT checkpoints")
        checkpoint_sha = current_sha
        if "selected_step" in packet["quality"]:
            require(packet["quality"]["selected_step"] == entry["selected_step"], "Accepted entry selection differs from the packet")
        row = dict(pair_id=pair_id, domain=_domain(packet["domain"]), native_hw=list(packet["native_hw"]),
                   padding=list(packet["padding"]), input_file=packet_ref, target_file=dict(packet_ref),
                   input_key="original", target_key="teacher_features", teacher_updates=512,
                   selected_step=entry["selected_step"],
                   accepted_entry=_file_ref(str(entry_path), entry_path.parent, output_path.parent),
                   independent_audit=_file_ref(entry["audit"], entry_path.parent, output_path.parent))
        if method_ref is not None:
            row["target_method_file"] = method_ref
        rows.append(row)
    value = dict(schema=SCHEMA, waft_checkpoint_sha256=checkpoint_sha, target_kind="expensive", samples=rows,
                 normalization="original native paired per-channel RMS; complete padded pair loss",
                 selection="all qualified indexed packets; unindexed and unfinished packets excluded")
    _validate_manifest(value)
    value["_base"] = str(output_path.parent)
    for row in rows:
        sample = load_sample(value, row)
        packet = load_packet(output_path.parent / row["input_file"]["path"])
        require(torch.equal(sample["sigma"].view(torch.uint8), packet["sigma"].view(torch.uint8)), "Stored original RMS differs from recomputed normalization")
        require(torch.equal(sample["native_mask"], packet["native_mask"]), "Stored native support differs from packet geometry")
    value.pop("_base")
    write_json(output_path, value)
    return value


def import_cache(pair_paths: list[str | Path], output_path: str | Path, *, target_kind: str = "cheap") -> dict:
    """Bind existing audited paired JSON records; do not copy their large tensors.

    This adapter accepts the saved original_source/teacher descriptors. Portable
    packets can be prepared directly without this compatibility adapter.
    """
    require(target_kind == "cheap", "The audited-pair adapter supports the selected cheap packets; prepare expensive portable packets explicitly")
    output_path = Path(output_path).resolve()
    require(not output_path.exists(), "Manifest output already exists")
    require(bool(pair_paths), "Provide at least one audited pair JSON")
    rows, checkpoint_sha = [], None
    for pair_path in pair_paths:
        pair_path = Path(pair_path).resolve(strict=True)
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        require(pair.get("method") == CHEAP_METHOD, "Only selected visible cheap targets can be imported")
        current_sha = pair["model"]["model_item"]["sha256"]
        require(checkpoint_sha is None or checkpoint_sha == current_sha, "Cache mixes WAFT checkpoints")
        checkpoint_sha = current_sha
        domain = pair.get("domain", "kitti" if pair["pair_id"].startswith("kitti") else None)
        require(domain is not None, "A non-KITTI legacy pair needs an explicit domain")
        row = dict(pair_id=pair["pair_id"], domain=domain, native_hw=pair["native_hw"], padding=pair["padding"],
                   input_file=_file_ref(pair["original_source"]["packet"], pair_path.parent, output_path.parent),
                   target_file=_file_ref(pair["teacher"], pair_path.parent, output_path.parent),
                   input_key=pair["original_source"].get("key", "features"))
        if "target_source_pair_id" in pair:
            row["target_source_pair_id"] = pair["target_source_pair_id"]
        rows.append(row)
    value = dict(schema=SCHEMA, waft_checkpoint_sha256=checkpoint_sha, target_kind=target_kind, samples=rows,
                 normalization="original native paired per-channel RMS; complete padded pair loss")
    _validate_manifest(value)
    value["_base"] = str(output_path.parent)
    for row in rows:
        load_sample(value, row)
    value.pop("_base")
    write_json(output_path, value)
    return value


def make_demo(output_dir: str | Path) -> Path:
    """Create a tiny deterministic CPU pipeline exercise, not research data."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(7)
    rows = []
    for index in range(2):
        pair_id = "example-" + str(index + 1)
        x = torch.randn(1, 2, 64, 4, 4, generator=generator)
        target = x.clone()
        target[:, 0] += 0.5
        metadata = dict(schema=FEATURE_SCHEMA, pair_id=pair_id, domain="synthetic", native_hw=[6, 6],
                        padding=[1, 1, 1, 1], waft_checkpoint_sha256="0" * 64)
        input_name, target_name = pair_id + "-input.pt", pair_id + "-target.pt"
        torch.save(dict(metadata, features=x), output / input_name)
        torch.save(dict(metadata, target_kind="cheap", target_method=CHEAP_METHOD, teacher_features=target), output / target_name)
        rows.append(dict(pair_id=pair_id, domain="synthetic", native_hw=metadata["native_hw"], padding=metadata["padding"],
                         input_file=input_name, target_file=target_name))
    spec = dict(schema=SCHEMA, waft_checkpoint_sha256="0" * 64, target_kind="cheap", samples=rows,
                note="Synthetic pipeline exercise; no pretrained WAFT or dataset required")
    write_json(output / "samples.json", spec)
    prepare_manifest(output / "samples.json", output / "manifest.json")
    return output / "manifest.json"
