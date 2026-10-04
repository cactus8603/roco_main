"""Fail-closed logical-to-physical mapping for qualification artifacts.

Large qualification arrays retain their stable workspace-relative logical
paths while an explicitly frozen binding places their bytes on a persistent
external filesystem.  A symlink is merely the transport: trust comes from the
binding semantic hash, identical external sentinel, allowlisted namespace, and
exact logical/physical path correspondence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import grp
import pwd
import subprocess
from typing import Any, Mapping


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _lexical_absolute(path: Path) -> Path:
    """Normalize ``..`` without following symlinks."""
    return Path(os.path.abspath(os.fspath(path)))


def mounted_filesystem_identity(path: Path) -> dict[str, str]:
    try:
        output = subprocess.check_output(
            [
                "findmnt", "--target", str(path.resolve(strict=True)),
                "-no", "SOURCE,TARGET,FSTYPE,UUID",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve external artifact filesystem") from exc
    fields = output.split()
    if len(fields) != 4:
        raise RuntimeError("external artifact filesystem identity is incomplete")
    return dict(zip(("source", "target", "filesystem", "uuid"), fields))


@dataclass(frozen=True)
class QualificationExternalArtifactRootV1:
    workspace_root: Path
    logical_package_root: Path
    external_root: Path
    allowed_top_levels: frozenset[str]
    binding_sha256: str
    binding: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        *,
        workspace_root: Path,
        logical_package_relative: Path,
        binding_path: Path,
    ) -> "QualificationExternalArtifactRootV1":
        workspace = workspace_root.resolve(strict=True)
        binding = json.loads(binding_path.read_text())
        payload = {
            key: item for key, item in binding.items()
            if key != "binding_sha256"
        }
        if binding.get("binding_sha256") != canonical_sha256(payload):
            raise RuntimeError("external artifact binding semantic hash drift")
        if binding.get("status") != "FROZEN_BEFORE_E263_EXECUTION":
            raise RuntimeError("external artifact binding is not active")
        if any(binding.get("authority", {}).values()):
            raise RuntimeError("storage binding grants forbidden authority")

        external = Path(str(binding["external_root"]))
        expected_realpath = Path(str(binding["external_root_realpath"]))
        if external.resolve(strict=True) != expected_realpath:
            raise RuntimeError("external artifact root realpath drift")
        if mounted_filesystem_identity(external) != binding.get("mount"):
            raise RuntimeError("external artifact filesystem identity drift")
        stat = external.stat()
        actual_owner = {
            "user": pwd.getpwuid(stat.st_uid).pw_name,
            "group": grp.getgrgid(stat.st_gid).gr_name,
        }
        if actual_owner != binding.get("owner"):
            raise RuntimeError("external artifact root owner drift")
        sentinel = external / str(binding["sentinel_name"])
        if json.loads(sentinel.read_text()) != binding:
            raise RuntimeError("external artifact sentinel drift")

        package = _lexical_absolute(workspace / logical_package_relative)
        if not package.is_relative_to(workspace):
            raise RuntimeError("logical artifact package escaped workspace")
        allowed = binding.get("allowed_top_level_outputs")
        if (
            not isinstance(allowed, list)
            or not allowed
            or len(set(allowed)) != len(allowed)
            or any(not isinstance(item, str) or not item for item in allowed)
        ):
            raise RuntimeError("external artifact allowlist is invalid")
        return cls(
            workspace_root=workspace,
            logical_package_root=package,
            external_root=expected_realpath,
            allowed_top_levels=frozenset(allowed),
            binding_sha256=str(binding["binding_sha256"]),
            binding=binding,
        )

    def _logical_parts(self, path: Path) -> tuple[Path, tuple[str, ...]]:
        logical = _lexical_absolute(path)
        if not logical.is_relative_to(self.logical_package_root):
            raise RuntimeError("artifact logical path escaped its package")
        relative = logical.relative_to(self.logical_package_root)
        if not relative.parts:
            raise RuntimeError("artifact path cannot be the package root")
        if relative.parts[0] not in self.allowed_top_levels:
            raise RuntimeError("artifact top-level namespace is not allowlisted")
        return logical, relative.parts

    def expected_physical_path(self, logical_path: Path) -> Path:
        _, parts = self._logical_parts(logical_path)
        expected = self.external_root.joinpath(*parts)
        resolved = expected.resolve(strict=False)
        if not resolved.is_relative_to(self.external_root):
            raise RuntimeError("external artifact target escaped frozen root")
        return expected

    def resolve_for_output(self, logical_path: Path) -> Path:
        logical, parts = self._logical_parts(logical_path)
        expected = self.external_root.joinpath(*parts)
        expected_resolved = expected.resolve(strict=False)
        if not expected_resolved.is_relative_to(self.external_root):
            raise RuntimeError("external output target escaped frozen root")
        logical_resolved = logical.resolve(strict=False)
        if logical_resolved != expected_resolved:
            raise RuntimeError("logical output is not mapped to its exact external target")
        return expected_resolved

    def resolve_for_read(self, logical_path: Path) -> Path:
        logical, parts = self._logical_parts(logical_path)
        expected = self.external_root.joinpath(*parts).resolve(strict=True)
        if not expected.is_relative_to(self.external_root):
            raise RuntimeError("external input escaped frozen root")
        actual = logical.resolve(strict=True)
        if actual != expected:
            raise RuntimeError("logical input mapping drifted from external target")
        return actual

    def workspace_relative_logical_path(self, logical_path: Path) -> str:
        logical, _ = self._logical_parts(logical_path)
        if not logical.is_relative_to(self.workspace_root):
            raise RuntimeError("logical record escaped workspace")
        return str(logical.relative_to(self.workspace_root))


__all__ = [
    "QualificationExternalArtifactRootV1",
    "canonical_sha256",
    "mounted_filesystem_identity",
]
