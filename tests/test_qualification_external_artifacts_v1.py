from __future__ import annotations

import json
import grp
from pathlib import Path
import pwd

import pytest

from stablebridge.physical_repair.qualification_external_artifacts_v1 import (
    QualificationExternalArtifactRootV1,
    canonical_sha256,
    mounted_filesystem_identity,
)


def _binding(external: Path) -> dict:
    stat = external.stat()
    payload = {
        "schema": "e272-external-artifact-root-binding/v1",
        "status": "FROZEN_BEFORE_E263_EXECUTION",
        "external_root": str(external),
        "external_root_realpath": str(external),
        "mount": mounted_filesystem_identity(external),
        "owner": {
            "user": pwd.getpwuid(stat.st_uid).pw_name,
            "group": grp.getgrgid(stat.st_gid).gr_name,
        },
        "allowed_top_level_outputs": ["matrix_receipts", "target_children"],
        "sentinel_name": ".binding.json",
        "authority": {
            "execution": False,
            "task_GT_decode": False,
            "scientific_qualification": False,
            "selector_admission": False,
            "production": False,
        },
    }
    return {**payload, "binding_sha256": canonical_sha256(payload)}


def _root(tmp_path: Path) -> tuple[QualificationExternalArtifactRootV1, Path, Path]:
    workspace = tmp_path / "workspace"
    package = workspace / "experiments" / "E263"
    external = tmp_path / "persistent" / "E263"
    package.mkdir(parents=True)
    external.mkdir(parents=True)
    binding = _binding(external)
    binding_path = workspace / "binding.json"
    binding_path.write_text(json.dumps(binding))
    (external / binding["sentinel_name"]).write_text(json.dumps(binding))
    for name in binding["allowed_top_level_outputs"]:
        (external / name).mkdir()
        (package / name).symlink_to(external / name, target_is_directory=True)
    value = QualificationExternalArtifactRootV1.load(
        workspace_root=workspace,
        logical_package_relative=Path("experiments/E263"),
        binding_path=binding_path,
    )
    return value, package, external


def test_exact_allowlisted_symlink_mapping_round_trip(tmp_path):
    root, package, external = _root(tmp_path)
    logical = package / "matrix_receipts" / "case.json"
    assert root.resolve_for_output(logical) == external / "matrix_receipts" / "case.json"
    physical = external / "matrix_receipts" / "case.json"
    physical.write_text("evidence")
    assert root.resolve_for_read(logical) == physical
    assert root.workspace_relative_logical_path(logical) == (
        "experiments/E263/matrix_receipts/case.json"
    )


def test_unlisted_namespace_fails_closed(tmp_path):
    root, package, _ = _root(tmp_path)
    with pytest.raises(RuntimeError, match="not allowlisted"):
        root.resolve_for_output(package / "other" / "case.json")


def test_wrong_symlink_target_fails_closed(tmp_path):
    root, package, _ = _root(tmp_path)
    logical_top = package / "matrix_receipts"
    logical_top.unlink()
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    logical_top.symlink_to(wrong, target_is_directory=True)
    with pytest.raises(RuntimeError, match="not mapped"):
        root.resolve_for_output(logical_top / "case.json")


def test_nested_external_symlink_escape_fails_closed(tmp_path):
    root, package, external = _root(tmp_path)
    escape = tmp_path / "escape"
    escape.mkdir()
    (external / "target_children" / "bad").symlink_to(
        escape, target_is_directory=True,
    )
    with pytest.raises(RuntimeError, match="escaped frozen root"):
        root.resolve_for_output(package / "target_children" / "bad" / "x.npy")


def test_sentinel_drift_fails_closed(tmp_path):
    _, _, external = _root(tmp_path)
    sentinel = external / ".binding.json"
    value = json.loads(sentinel.read_text())
    value["external_root"] = "/ssd8/forbidden"
    sentinel.write_text(json.dumps(value))
    workspace = tmp_path / "workspace"
    with pytest.raises(RuntimeError, match="sentinel drift"):
        QualificationExternalArtifactRootV1.load(
            workspace_root=workspace,
            logical_package_relative=Path("experiments/E263"),
            binding_path=workspace / "binding.json",
        )
