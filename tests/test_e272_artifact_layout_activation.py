from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import grp
import pwd
import sys

import pytest

from stablebridge.physical_repair import qualification_external_artifacts_v1 as mapper


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/E272_action_bank_v2_external_artifact_root_freeze_v1"
    / "activate_layout.py"
)
SPEC = importlib.util.spec_from_file_location("e272_activate_layout_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
activation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = activation
SPEC.loader.exec_module(activation)


def _binding(external: Path, namespaces=None) -> dict:
    stat = external.stat()
    payload = {
        "schema": "e272-external-artifact-root-binding/v1",
        "status": "FROZEN_BEFORE_E263_EXECUTION",
        "external_root": str(external),
        "external_root_realpath": str(external),
        "mount": mapper.mounted_filesystem_identity(external),
        "owner": {
            "user": pwd.getpwuid(stat.st_uid).pw_name,
            "group": grp.getgrgid(stat.st_gid).gr_name,
        },
        "allowed_top_level_outputs": list(
            activation.EXPECTED_NAMESPACES if namespaces is None else namespaces
        ),
        "sentinel_name": ".stablebridge_artifact_root_binding.json",
        "authority": {
            "execution": False,
            "task_GT_decode": False,
            "scientific_qualification": False,
            "selector_admission": False,
            "production": False,
        },
    }
    return {**payload, "binding_sha256": mapper.canonical_sha256(payload)}


def _layout(tmp_path: Path, namespaces=None):
    workspace = tmp_path / "workspace"
    package_relative = Path("experiments/E263")
    package = workspace / package_relative
    external = tmp_path / "persistent/E263"
    package.mkdir(parents=True)
    external.mkdir(parents=True)
    value = _binding(external, namespaces)
    binding_path = workspace / "binding.json"
    binding_path.write_text(json.dumps(value))
    (external / value["sentinel_name"]).write_text(json.dumps(value))
    kwargs = {
        "workspace_root": workspace,
        "logical_package_relative": package_relative,
        "binding_path": binding_path,
    }
    return kwargs, package, external, binding_path


def _activate(kwargs):
    return activation.reconcile_layout(**kwargs, activate=True)


def _compare(kwargs):
    return activation.reconcile_layout(**kwargs, activate=False)


def test_compare_only_requires_complete_layout_and_never_mutates(tmp_path):
    kwargs, package, external, _ = _layout(tmp_path)
    before_package = set(package.iterdir())
    before_external = set(external.iterdir())
    with pytest.raises(RuntimeError, match="layout is incomplete"):
        _compare(kwargs)
    assert set(package.iterdir()) == before_package
    assert set(external.iterdir()) == before_external


def test_activation_creates_exact_layout_is_idempotent_and_compares(tmp_path):
    kwargs, package, external, _ = _layout(tmp_path)
    first = _activate(kwargs)
    assert first["namespace_count"] == 7
    assert first["created_physical_namespaces"] == list(
        activation.EXPECTED_NAMESPACES
    )
    assert first["created_logical_symlinks"] == list(
        activation.EXPECTED_NAMESPACES
    )
    for name in activation.EXPECTED_NAMESPACES:
        physical = external / name
        logical = package / name
        assert physical.is_dir() and not physical.is_symlink()
        assert logical.is_symlink()
        assert logical.resolve(strict=True) == physical

    second = _activate(kwargs)
    assert second["created_physical_namespaces"] == []
    assert second["created_logical_symlinks"] == []
    compared = _compare(kwargs)
    assert compared["mode"] == "COMPARE_ONLY"
    assert compared["status"] == "ACTIVE_EXACT_SEVEN_NAMESPACE_LAYOUT"


def test_activation_rejects_ordinary_logical_directory_before_mutation(tmp_path):
    kwargs, package, external, _ = _layout(tmp_path)
    (package / activation.EXPECTED_NAMESPACES[0]).mkdir()
    with pytest.raises(RuntimeError, match="not a symlink"):
        _activate(kwargs)
    assert {entry.name for entry in external.iterdir()} == {
        ".stablebridge_artifact_root_binding.json"
    }


@pytest.mark.parametrize("broken", [False, True])
def test_activation_rejects_wrong_or_broken_logical_symlink(tmp_path, broken):
    kwargs, package, external, _ = _layout(tmp_path)
    name = activation.EXPECTED_NAMESPACES[0]
    if broken:
        target = tmp_path / "missing-target"
    else:
        target = tmp_path / "wrong-target"
        target.mkdir()
    (package / name).symlink_to(target, target_is_directory=True)
    message = "broken" if broken else "wrong target"
    with pytest.raises(RuntimeError, match=message):
        _activate(kwargs)
    assert not (external / name).exists()


def test_activation_rejects_external_symlink_and_extra_namespace(tmp_path):
    kwargs, _, external, _ = _layout(tmp_path)
    name = activation.EXPECTED_NAMESPACES[0]
    outside = tmp_path / "outside"
    outside.mkdir()
    (external / name).symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="not a real directory"):
        _activate(kwargs)

    (external / name).unlink()
    (external / "unlisted_namespace").mkdir()
    with pytest.raises(RuntimeError, match="unexpected external"):
        _activate(kwargs)


def test_activation_rejects_unlisted_logical_namespace_symlink(tmp_path):
    kwargs, package, _, _ = _layout(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (package / "unlisted_namespace").symlink_to(
        outside, target_is_directory=True,
    )
    with pytest.raises(RuntimeError, match="unexpected logical"):
        _activate(kwargs)


def test_compare_rejects_nested_symlink_escape(tmp_path):
    kwargs, _, external, _ = _layout(tmp_path)
    _activate(kwargs)
    outside = tmp_path / "outside"
    outside.mkdir()
    (external / activation.EXPECTED_NAMESPACES[0] / "escape").symlink_to(
        outside, target_is_directory=True,
    )
    with pytest.raises(RuntimeError, match="nested symlink"):
        _compare(kwargs)


def test_binding_must_contain_exact_ordered_seven_namespaces(tmp_path):
    short = activation.EXPECTED_NAMESPACES[:-1]
    kwargs, package, external, _ = _layout(tmp_path, namespaces=short)
    with pytest.raises(RuntimeError, match="exact seven"):
        _activate(kwargs)
    assert list(package.iterdir()) == []
    assert {entry.name for entry in external.iterdir()} == {
        ".stablebridge_artifact_root_binding.json"
    }


def test_mapper_binding_sentinel_mount_and_owner_checks_run_before_mutation(
    tmp_path, monkeypatch,
):
    kwargs, package, external, binding_path = _layout(tmp_path)

    binding = json.loads(binding_path.read_text())
    binding["external_root"] = "/tampered"
    binding_path.write_text(json.dumps(binding))
    with pytest.raises(RuntimeError, match="semantic hash drift"):
        _activate(kwargs)

    kwargs, package, external, binding_path = _layout(tmp_path / "sentinel")
    sentinel = external / ".stablebridge_artifact_root_binding.json"
    sentinel_value = json.loads(sentinel.read_text())
    sentinel_value["status"] = "DRIFTED"
    sentinel.write_text(json.dumps(sentinel_value))
    with pytest.raises(RuntimeError, match="sentinel drift"):
        _activate(kwargs)

    kwargs, package, external, binding_path = _layout(tmp_path / "mount")
    monkeypatch.setattr(
        mapper,
        "mounted_filesystem_identity",
        lambda path: {
            "source": "wrong", "target": "wrong", "filesystem": "wrong",
            "uuid": "wrong",
        },
    )
    with pytest.raises(RuntimeError, match="filesystem identity drift"):
        _activate(kwargs)

    monkeypatch.undo()
    kwargs, package, external, binding_path = _layout(tmp_path / "owner")
    value = json.loads(binding_path.read_text())
    value["owner"] = {"user": "definitely-wrong", "group": "definitely-wrong"}
    payload = {key: item for key, item in value.items() if key != "binding_sha256"}
    value["binding_sha256"] = mapper.canonical_sha256(payload)
    binding_path.write_text(json.dumps(value))
    (external / value["sentinel_name"]).write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="owner drift"):
        _activate(kwargs)
    assert list(package.iterdir()) == []


def test_cli_compare_mode_uses_no_implicit_activation(tmp_path, monkeypatch):
    kwargs, package, external, binding_path = _layout(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT), "--compare",
            "--workspace-root", str(kwargs["workspace_root"]),
            "--logical-package-relative", str(kwargs["logical_package_relative"]),
            "--binding", str(binding_path),
        ],
    )
    with pytest.raises(RuntimeError, match="layout is incomplete"):
        activation.main()
    assert list(package.iterdir()) == []
    assert {entry.name for entry in external.iterdir()} == {
        ".stablebridge_artifact_root_binding.json"
    }
