"""Verify the frozen KITTI 2012/2015 archives and extracted U²Flow layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile
import zlib


_FORBIDDEN_ROOTS = (PurePosixPath("/tmp"), PurePosixPath("/ssd8"))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_KITTI_MULTIVIEW_FRAME = re.compile(r"^(\d{6})_(\d{2})\.png$")
_EXPECTED_SOURCE_IDS = frozenset({
    "kitti2012_benchmark",
    "kitti2012_multiview",
    "kitti2015_benchmark",
    "kitti2015_multiview",
})


def _safe_absolute_directory(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty canonical text")
    lexical = PurePosixPath(value)
    if not lexical.is_absolute():
        raise ValueError(f"{name} must be absolute")
    lexical = PurePosixPath(os.path.abspath(os.path.normpath(value)))
    for forbidden in _FORBIDDEN_ROOTS:
        if lexical == forbidden or forbidden in lexical.parents:
            raise ValueError(f"{name} must not be inside {forbidden}")
    path = Path(value).resolve(strict=True)
    resolved = PurePosixPath(str(path))
    for forbidden in _FORBIDDEN_ROOTS:
        if resolved == forbidden or forbidden in resolved.parents:
            raise ValueError(f"resolved {name} must not be inside {forbidden}")
    if not path.is_dir():
        raise ValueError(f"{name} must be a directory")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _crc32(path: Path) -> int:
    checksum = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF


def _safe_zip_member_path(name: str, source_id: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe ZIP member in {source_id}: {name!r}")
    return path.as_posix()


def _u2flow_pair_count(directory: Path, excluded_indices: set[int]) -> int:
    frames: dict[str, list[int]] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        match = _KITTI_MULTIVIEW_FRAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"noncanonical KITTI multiview filename: {path.name}")
        frames.setdefault(match.group(1), []).append(int(match.group(2)))
    pairs = 0
    for indices in frames.values():
        ordered = sorted(indices)
        if len(ordered) != len(set(ordered)):
            raise ValueError(f"duplicate KITTI multiview frame in {directory}")
        pairs += sum(
            second == first + 1
            and first not in excluded_indices
            and second not in excluded_indices
            for first, second in zip(ordered, ordered[1:])
        )
    return pairs


def _validated_sources(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("sources must be a list of objects")
    source_ids = [source.get("source_id") for source in value]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("KITTI source IDs must be unique")
    if set(source_ids) != _EXPECTED_SOURCE_IDS:
        raise ValueError(
            "KITTI sources must contain exactly the two benchmark and two "
            "multiview archives"
        )
    return [dict(item) for item in value]


def _verify_source(root: Path, source: object, *, verify_zip: bool) -> dict[str, object]:
    if not isinstance(source, dict):
        raise ValueError("each source must be an object")
    source_id = source["source_id"]
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be nonempty text")
    expected_size = source["size_bytes"]
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        raise ValueError(f"size_bytes must be positive for {source_id}")
    expected_digest = source["sha256"]
    if (
        not isinstance(expected_digest, str)
        or _SHA256.fullmatch(expected_digest) is None
    ):
        raise ValueError(f"sha256 must be a lowercase digest for {source_id}")
    archive = root / source["archive_relative_path"]
    archive = archive.resolve(strict=True)
    if root not in archive.parents:
        raise ValueError(f"archive escapes storage root: {source_id}")
    size = archive.stat().st_size
    if size != expected_size:
        raise ValueError(f"archive size drift: {source_id}: {size}")
    digest = _sha256(archive)
    if digest != expected_digest:
        raise ValueError(f"archive SHA-256 drift: {source_id}: {digest}")
    with zipfile.ZipFile(archive) as bundle:
        members: dict[str, zipfile.ZipInfo] = {}
        for item in bundle.infolist():
            if item.is_dir():
                continue
            member_path = _safe_zip_member_path(item.filename, source_id)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"symlink ZIP member in {source_id}: {member_path}")
            if member_path in members:
                raise ValueError(f"duplicate ZIP member in {source_id}: {member_path}")
            members[member_path] = item
        if verify_zip:
            first_bad_member = bundle.testzip()
            if first_bad_member is not None:
                raise ValueError(f"corrupt ZIP member in {source_id}: {first_bad_member}")
    extracted = (root / source["extracted_relative_path"]).resolve(strict=True)
    if root not in extracted.parents or not extracted.is_dir():
        raise ValueError(f"invalid extracted directory: {source_id}")
    raw_required_paths = source["required_relative_paths"]
    if (
        not isinstance(raw_required_paths, list)
        or not raw_required_paths
        or any(not isinstance(item, str) for item in raw_required_paths)
        or len(raw_required_paths) != len(set(raw_required_paths))
    ):
        raise ValueError(
            f"required_relative_paths must be a nonempty unique list for {source_id}"
        )
    required_paths = tuple(raw_required_paths)
    resolved_required: list[Path] = []
    for item in required_paths:
        if (
            not isinstance(item, str)
            or not item
            or PurePosixPath(item).is_absolute()
        ):
            raise ValueError(f"invalid required relative path for {source_id}: {item!r}")
        candidate = (extracted / item).resolve(strict=False)
        if candidate != extracted and extracted not in candidate.parents:
            raise ValueError(
                f"required path escapes extraction root: {source_id}: {item}"
            )
        resolved_required.append(candidate)
    missing_or_wrong_type = [
        item
        for item, path in zip(required_paths, resolved_required)
        if not path.is_dir()
    ]
    if missing_or_wrong_type:
        raise ValueError(
            f"missing extracted directories for {source_id}: "
            f"{missing_or_wrong_type}"
        )
    expected_png_counts = source.get("expected_png_counts")
    if not isinstance(expected_png_counts, dict) or not expected_png_counts:
        raise ValueError(f"expected_png_counts must be a nonempty object for {source_id}")
    observed_png_counts: dict[str, int] = {}
    for relative_directory, expected_count in expected_png_counts.items():
        if (
            not isinstance(relative_directory, str)
            or relative_directory not in required_paths
            or isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count <= 0
        ):
            raise ValueError(f"invalid expected PNG count for {source_id}")
        directory = extracted / relative_directory
        observed_count = sum(
            1
            for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() == ".png"
        )
        if observed_count != expected_count:
            raise ValueError(
                f"PNG count drift: {source_id}: {relative_directory}: "
                f"{observed_count} != {expected_count}"
            )
        observed_png_counts[relative_directory] = observed_count
    pair_contract = source.get("u2flow_pair_contract")
    observed_pair_counts: dict[str, int] = {}
    if pair_contract is not None:
        if not isinstance(pair_contract, dict):
            raise ValueError(f"invalid U²Flow pair contract for {source_id}")
        excluded = pair_contract.get("excluded_frame_indices")
        expected_pairs = pair_contract.get("expected_pairs")
        if (
            not isinstance(excluded, list)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in excluded)
            or not isinstance(expected_pairs, dict)
            or not expected_pairs
        ):
            raise ValueError(f"invalid U²Flow pair contract for {source_id}")
        excluded_set = set(excluded)
        for relative_directory, expected_count in expected_pairs.items():
            if (
                relative_directory not in required_paths
                or isinstance(expected_count, bool)
                or not isinstance(expected_count, int)
                or expected_count <= 0
            ):
                raise ValueError(f"invalid expected pair count for {source_id}")
            observed_count = _u2flow_pair_count(
                extracted / relative_directory, excluded_set
            )
            if observed_count != expected_count:
                raise ValueError(
                    f"U²Flow pair count drift: {source_id}: {relative_directory}: "
                    f"{observed_count} != {expected_count}"
                )
            observed_pair_counts[relative_directory] = observed_count
    observed: dict[str, Path] = {}
    for item in extracted.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"symlink in extracted tree: {source_id}: {item}")
        if item.is_file():
            observed[item.relative_to(extracted).as_posix()] = item
    missing_members = sorted(set(members) - set(observed))
    extra_members = sorted(set(observed) - set(members))
    if missing_members or extra_members:
        raise ValueError(
            f"extracted member path drift: {source_id}: "
            f"missing={missing_members[:10]}, extra={extra_members[:10]}"
        )
    for relative_path, info in members.items():
        path = observed[relative_path]
        if path.stat().st_size != info.file_size:
            raise ValueError(
                f"extracted member size drift: {source_id}: {relative_path}"
            )
        if verify_zip and _crc32(path) != info.CRC:
            raise ValueError(
                f"extracted member CRC drift: {source_id}: {relative_path}"
            )
    return {
        "source_id": source_id,
        "archive": str(archive),
        "size_bytes": size,
        "sha256": digest,
        "zip_file_members": len(members),
        "extracted": str(extracted),
        "extracted_files": len(observed),
        "extracted_crc_verified": verify_zip,
        "png_counts": observed_png_counts,
        "u2flow_pair_counts": observed_pair_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stablebridge/kitti_u2flow_data_v1.json"),
    )
    parser.add_argument(
        "--verify-zip",
        action="store_true",
        help="decompress every ZIP member in addition to checking size and SHA-256",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if config.get("schema") != "stablebridge-kitti-u2flow-data/v1":
        raise ValueError("unsupported KITTI U²Flow data schema")
    root = _safe_absolute_directory(config["storage_root"], "storage root")
    sources = _validated_sources(config.get("sources"))
    results = [
        _verify_source(root, source, verify_zip=args.verify_zip)
        for source in sources
    ]
    print(json.dumps(
        {
            "status": "PASS_KITTI_U2FLOW_DATA",
            "storage_root": str(root),
            "source_count": len(results),
            "total_archive_bytes": sum(item["size_bytes"] for item in results),
            "verified_zip_payloads": args.verify_zip,
            "sources": results,
            "kitti_raw_included": False,
            "sam_masks_included": False,
        },
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
