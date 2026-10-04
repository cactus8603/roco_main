from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import pytest


def _load_script_module():
    import importlib.util

    path = Path(__file__).parents[2] / "scripts" / "verify_kitti_u2flow_data.py"
    spec = importlib.util.spec_from_file_location("verify_kitti_u2flow_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_source_binds_archive_and_extracted_inventory(tmp_path: Path) -> None:
    module = _load_script_module()
    root = tmp_path / "kitti"
    downloads = root / "downloads"
    extracted = root / "extracted" / "kitti2015" / "benchmark"
    downloads.mkdir(parents=True)
    extracted.mkdir(parents=True)
    archive = downloads / "fixture.zip"
    payload = b"fixture-rgb"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("training/image_2/000000_10.png", payload)
    image = extracted / "training" / "image_2" / "000000_10.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(payload)
    source = {
        "source_id": "fixture",
        "archive_relative_path": "downloads/fixture.zip",
        "size_bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "extracted_relative_path": "extracted/kitti2015/benchmark",
        "required_relative_paths": ["training/image_2"],
        "expected_png_counts": {"training/image_2": 1},
    }

    result = module._verify_source(root.resolve(), source, verify_zip=True)

    assert result["source_id"] == "fixture"
    assert result["zip_file_members"] == 1
    assert result["extracted_files"] == 1


def test_safe_directory_rejects_ssd8_before_touching_it() -> None:
    module = _load_script_module()

    with pytest.raises(ValueError, match="must not be inside /ssd8"):
        module._safe_absolute_directory("/ssd8/cactus8603/kitti", "storage root")
    with pytest.raises(ValueError, match="must not be inside /ssd8"):
        module._safe_absolute_directory(
            "/ssd6/../ssd8/cactus8603/kitti", "storage root"
        )


def test_verify_source_rejects_required_path_escape(tmp_path: Path) -> None:
    module = _load_script_module()
    root = tmp_path / "kitti"
    archive = root / "downloads" / "fixture.zip"
    extracted = root / "extracted" / "fixture"
    archive.parent.mkdir(parents=True)
    extracted.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("image.png", b"rgb")
    (extracted / "image.png").write_bytes(b"rgb")
    source = {
        "source_id": "fixture",
        "archive_relative_path": "downloads/fixture.zip",
        "size_bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "extracted_relative_path": "extracted/fixture",
        "required_relative_paths": ["../../outside"],
    }

    with pytest.raises(ValueError, match="escapes extraction root"):
        module._verify_source(root.resolve(), source, verify_zip=False)


def test_verify_source_rejects_same_count_wrong_member_path(tmp_path: Path) -> None:
    module = _load_script_module()
    root = tmp_path / "kitti"
    archive = root / "downloads" / "fixture.zip"
    extracted = root / "extracted" / "fixture"
    archive.parent.mkdir(parents=True)
    (extracted / "training" / "image_2").mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("training/image_2/expected.png", b"rgb")
    (extracted / "training" / "image_2" / "replacement.png").write_bytes(b"rgb")
    source = {
        "source_id": "fixture",
        "archive_relative_path": "downloads/fixture.zip",
        "size_bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "extracted_relative_path": "extracted/fixture",
        "required_relative_paths": ["training/image_2"],
        "expected_png_counts": {"training/image_2": 1},
    }

    with pytest.raises(ValueError, match="member path drift"):
        module._verify_source(root.resolve(), source, verify_zip=False)


def test_zip_member_paths_fail_closed() -> None:
    module = _load_script_module()

    with pytest.raises(ValueError, match="unsafe ZIP member"):
        module._safe_zip_member_path("../escape.png", "fixture")
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        module._safe_zip_member_path("/absolute.png", "fixture")


def test_source_set_must_be_exact_and_unique() -> None:
    module = _load_script_module()
    with pytest.raises(ValueError, match="exactly"):
        module._validated_sources([])
    duplicate = [
        {"source_id": "kitti2012_benchmark"},
        {"source_id": "kitti2012_benchmark"},
    ]
    with pytest.raises(ValueError, match="unique"):
        module._validated_sources(duplicate)
    expected = [
        {"source_id": "kitti2012_benchmark"},
        {"source_id": "kitti2012_multiview"},
        {"source_id": "kitti2015_benchmark"},
        {"source_id": "kitti2015_multiview"},
    ]
    assert module._validated_sources(expected) == expected


def test_u2flow_pair_count_excludes_any_pair_touching_frames_9_to_12(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    directory = tmp_path / "image_2"
    directory.mkdir()
    for frame in range(21):
        (directory / f"000000_{frame:02d}.png").write_bytes(b"rgb")
    for frame in range(16):
        (directory / f"000001_{frame:02d}.png").write_bytes(b"rgb")

    assert module._u2flow_pair_count(directory, {9, 10, 11, 12}) == 25
