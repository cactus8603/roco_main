"""Strict, production-neutral contracts for action-bank descriptors.

The original eight-field action catalogs were useful inventories, but their
nested values do not share a type: for example, ``strength`` is a float in the
optical catalog and a policy mapping in the weather and learned catalogs.  A
selector adapter must not guess how those values map to Selector-v7 fields.

This module defines a homogeneous v2 *template* contract.  It deliberately
keeps an operator's physical parameter (sigma, iterations, quality, and so on)
separate from the normalized ``input_strength`` and ``output_beta`` that a
selector consumes.  Each normalized scalar has an explicit BOUND/UNBOUND
status, so a catalog row cannot use a guessed zero or one to hide missing
control semantics.  It also models aliases as metadata on one canonical arm so
that byte-equivalent names cannot silently become extra selector arms.

The records here do not bind a descriptor to a case and never grant production
authority.  ``CaseBindingPrerequisitesV2`` only validates that the immutable
inputs needed by a later adapter are present; it does not create a
``CandidateArmV7``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from typing import Mapping, Sequence


ACTION_DESCRIPTOR_SCHEMA_V2 = "stablebridge-action-descriptor/v2"
NATIVE_ACTION_ID_V2 = "native"
NATIVE_OPERATOR_ID_V2 = "identity"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_LEGACY_EIGHT_FIELDS = frozenset({
    "action_id",
    "operator",
    "strength",
    "endpoint",
    "support",
    "cost",
    "availability",
    "receipt",
})
_DESCRIPTOR_FIELDS = frozenset({
    "schema",
    "action_id",
    "aliases",
    "operator",
    "input_strength_status",
    "input_strength",
    "output_beta_status",
    "output_beta",
    "endpoint",
    "support",
    "cost",
    "availability",
    "receipt",
    "descriptor_hash",
})


class ActionLifecycleV2(str, Enum):
    """Evidence lifecycle of a descriptor; every state remains test-only."""

    BLOCKED = "BLOCKED"
    CATALOG_ONLY = "CATALOG_ONLY"
    NATIVE_REFERENCE_AVAILABLE_TEST_ONLY = "NATIVE_REFERENCE_AVAILABLE_TEST_ONLY"
    MATERIALIZABLE_TEST_ONLY = "MATERIALIZABLE_TEST_ONLY"
    CASE_BOUND_TEST_ONLY = "CASE_BOUND_TEST_ONLY"
    CHILD_CC_AVAILABLE_TEST_ONLY = "CHILD_CC_AVAILABLE_TEST_ONLY"
    CHILD_QUARTET_AVAILABLE_TEST_ONLY = "CHILD_QUARTET_AVAILABLE_TEST_ONLY"
    OUTCOME_AVAILABLE_TEST_ONLY = "OUTCOME_AVAILABLE_TEST_ONLY"
    SELECTOR_QUALIFIED_TEST_ONLY = "SELECTOR_QUALIFIED_TEST_ONLY"


class ActionEndpointV2(str, Enum):
    """Endpoint policy at template time.

    ``CASE_SELECTED_CORRUPTED`` is resolved to ``FIRST`` or ``SECOND`` by a
    case binding.  It is not an invitation for the operator to inspect both
    endpoints. ``CASE_SELECTED_SUPPORTED`` is resolved to ``FIRST``,
    ``SECOND``, or ``BOTH`` by a separately frozen, before-only physical
    applicability rule. ``CASE_SELECTED_SINGLE_SUPPORTED`` has the same
    before-only semantics but is restricted to exactly one of ``FIRST`` or
    ``SECOND``. The exact resolved endpoint and rule receipt remain mandatory
    in the case binding.
    """

    FIRST = "first"
    SECOND = "second"
    BOTH = "both"
    CASE_SELECTED_CORRUPTED = "case_selected_corrupted"
    CASE_SELECTED_SUPPORTED = "case_selected_supported"
    CASE_SELECTED_SINGLE_SUPPORTED = "case_selected_single_supported"
    NONE = "none"


class ActionCostStatusV2(str, Enum):
    BOUND = "BOUND"
    UNBOUND = "UNBOUND"


class ControlValueStatusV2(str, Enum):
    """Whether a normalized selector control has actually been frozen."""

    BOUND = "BOUND"
    UNBOUND = "UNBOUND"


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty canonical text")
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 binding")
    return value


def _require_fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite normalized number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return result


def _require_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _require_nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _require_mapping(
    value: object,
    expected_fields: frozenset[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    keys = frozenset(value)
    if keys != expected_fields:
        missing = sorted(expected_fields - keys)
        extra = sorted(keys - expected_fields)
        raise ValueError(
            f"{name} fields are not canonical; missing={missing}, extra={extra}"
        )
    return value


def _enum_from_value(value: object, enum_type: type[Enum], name: str) -> Enum:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a typed enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"unknown {name}: {value!r}") from exc


def _sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ActionAliasV2:
    """A non-arm name for a canonical action, backed by equivalence evidence."""

    alias_id: str
    equivalence_receipt_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.alias_id, "action alias id")
        _require_sha256(
            self.equivalence_receipt_sha256,
            "action alias equivalence receipt",
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "alias_id": self.alias_id,
            "equivalence_receipt_sha256": self.equivalence_receipt_sha256,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "ActionAliasV2":
        row = _require_mapping(
            value,
            frozenset({"alias_id", "equivalence_receipt_sha256"}),
            "action alias",
        )
        return cls(
            alias_id=row["alias_id"],  # type: ignore[arg-type]
            equivalence_receipt_sha256=row[  # type: ignore[arg-type]
                "equivalence_receipt_sha256"
            ],
        )


@dataclass(frozen=True)
class OperatorParameterV2:
    """One immutable physical/operator parameter, not selector strength."""

    name: str
    value: str | int | float | bool
    unit: str

    def __post_init__(self) -> None:
        _require_text(self.name, "operator parameter name")
        _require_text(self.unit, "operator parameter unit")
        value = self.value
        if isinstance(value, str):
            _require_text(value, f"operator parameter {self.name}")
        elif isinstance(value, bool):
            pass
        elif isinstance(value, int):
            pass
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"operator parameter {self.name} must be finite"
                )
        else:
            raise ValueError(
                "operator parameter values must be immutable JSON scalars"
            )

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value, "unit": self.unit}

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorParameterV2":
        row = _require_mapping(
            value,
            frozenset({"name", "value", "unit"}),
            "operator parameter",
        )
        return cls(
            name=row["name"],  # type: ignore[arg-type]
            value=row["value"],  # type: ignore[arg-type]
            unit=row["unit"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ActionOperatorV2:
    operator_id: str
    version: str
    parameters: tuple[OperatorParameterV2, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.operator_id, "operator id")
        _require_text(self.version, "operator version")
        if any(not isinstance(row, OperatorParameterV2) for row in self.parameters):
            raise ValueError("operator parameters must be typed")
        names = tuple(row.name for row in self.parameters)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError(
                "operator parameters must have sorted unique names"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.operator_id,
            "version": self.version,
            "parameters": [row.as_dict() for row in self.parameters],
        }

    @classmethod
    def from_mapping(cls, value: object) -> "ActionOperatorV2":
        row = _require_mapping(
            value,
            frozenset({"id", "version", "parameters"}),
            "action operator",
        )
        parameters = row["parameters"]
        if not isinstance(parameters, list):
            raise ValueError("action operator parameters must be a list")
        return cls(
            operator_id=row["id"],  # type: ignore[arg-type]
            version=row["version"],  # type: ignore[arg-type]
            parameters=tuple(
                OperatorParameterV2.from_mapping(item) for item in parameters
            ),
        )


@dataclass(frozen=True)
class ActionSupportV2:
    policy_id: str
    policy_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.policy_id, "support policy id")
        _require_sha256(self.policy_sha256, "support policy hash")

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "ActionSupportV2":
        row = _require_mapping(
            value,
            frozenset({"policy_id", "policy_sha256"}),
            "action support",
        )
        return cls(
            policy_id=row["policy_id"],  # type: ignore[arg-type]
            policy_sha256=row["policy_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ActionCostV2:
    """A prospective ceiling, kept distinct from observed runtime evidence."""

    status: ActionCostStatusV2
    runtime_wall_seconds_ceiling: float | None
    matcher_trajectories_ceiling: int | None
    peak_vram_bytes_ceiling: int | None
    source_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ActionCostStatusV2):
            raise ValueError("action cost status must be typed")
        _require_sha256(self.source_sha256, "action cost source")
        values = (
            self.runtime_wall_seconds_ceiling,
            self.matcher_trajectories_ceiling,
            self.peak_vram_bytes_ceiling,
        )
        if self.status is ActionCostStatusV2.UNBOUND:
            if any(value is not None for value in values):
                raise ValueError("an unbound action cost cannot carry ceilings")
            return
        if any(value is None for value in values):
            raise ValueError("a bound action cost needs every ceiling")
        wall = _require_nonnegative(
            self.runtime_wall_seconds_ceiling,
            "runtime wall-seconds ceiling",
        )
        trajectories = _require_nonnegative_integer(
            self.matcher_trajectories_ceiling,
            "matcher-trajectories ceiling",
        )
        vram = _require_nonnegative_integer(
            self.peak_vram_bytes_ceiling,
            "peak-VRAM ceiling",
        )
        object.__setattr__(self, "runtime_wall_seconds_ceiling", wall)
        object.__setattr__(self, "matcher_trajectories_ceiling", trajectories)
        object.__setattr__(self, "peak_vram_bytes_ceiling", vram)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "runtime_wall_seconds_ceiling": self.runtime_wall_seconds_ceiling,
            "matcher_trajectories_ceiling": self.matcher_trajectories_ceiling,
            "peak_vram_bytes_ceiling": self.peak_vram_bytes_ceiling,
            "source_sha256": self.source_sha256,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "ActionCostV2":
        row = _require_mapping(
            value,
            frozenset({
                "status",
                "runtime_wall_seconds_ceiling",
                "matcher_trajectories_ceiling",
                "peak_vram_bytes_ceiling",
                "source_sha256",
            }),
            "action cost",
        )
        return cls(
            status=_enum_from_value(  # type: ignore[arg-type]
                row["status"], ActionCostStatusV2, "action cost status"
            ),
            runtime_wall_seconds_ceiling=row[  # type: ignore[arg-type]
                "runtime_wall_seconds_ceiling"
            ],
            matcher_trajectories_ceiling=row[  # type: ignore[arg-type]
                "matcher_trajectories_ceiling"
            ],
            peak_vram_bytes_ceiling=row[  # type: ignore[arg-type]
                "peak_vram_bytes_ceiling"
            ],
            source_sha256=row["source_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ActionAvailabilityV2:
    lifecycle: ActionLifecycleV2
    execution_authorized: bool
    production_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.lifecycle, ActionLifecycleV2):
            raise ValueError("action lifecycle must be typed")
        execution = _require_bool(
            self.execution_authorized, "execution authorization"
        )
        production = _require_bool(
            self.production_authority, "production authority"
        )
        if production:
            raise ValueError(
                "action-bank descriptors cannot grant production authority"
            )
        if execution and self.lifecycle in {
            ActionLifecycleV2.BLOCKED,
            ActionLifecycleV2.CATALOG_ONLY,
        }:
            raise ValueError(
                "blocked or catalog-only actions cannot be execution-authorized"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "lifecycle": self.lifecycle.value,
            "execution_authorized": self.execution_authorized,
            "production_authority": self.production_authority,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "ActionAvailabilityV2":
        row = _require_mapping(
            value,
            frozenset({
                "lifecycle", "execution_authorized", "production_authority",
            }),
            "action availability",
        )
        return cls(
            lifecycle=_enum_from_value(  # type: ignore[arg-type]
                row["lifecycle"], ActionLifecycleV2, "action lifecycle"
            ),
            execution_authorized=row["execution_authorized"],  # type: ignore[arg-type]
            production_authority=row["production_authority"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ActionReceiptV2:
    receipt_id: str
    source_manifest_sha256: str
    operator_source_sha256: str
    evidence_sha256: str
    evidence_scope: str
    production_authority: bool = False

    def __post_init__(self) -> None:
        _require_text(self.receipt_id, "action receipt id")
        _require_text(self.evidence_scope, "action receipt evidence scope")
        _require_sha256(
            self.source_manifest_sha256, "action source-manifest hash"
        )
        _require_sha256(
            self.operator_source_sha256, "action operator-source hash"
        )
        _require_sha256(self.evidence_sha256, "action evidence hash")
        production = _require_bool(
            self.production_authority, "receipt production authority"
        )
        if production:
            raise ValueError(
                "action-bank receipts cannot grant production authority"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "operator_source_sha256": self.operator_source_sha256,
            "evidence_sha256": self.evidence_sha256,
            "evidence_scope": self.evidence_scope,
            "production_authority": self.production_authority,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "ActionReceiptV2":
        row = _require_mapping(
            value,
            frozenset({
                "receipt_id",
                "source_manifest_sha256",
                "operator_source_sha256",
                "evidence_sha256",
                "evidence_scope",
                "production_authority",
            }),
            "action receipt",
        )
        return cls(
            receipt_id=row["receipt_id"],  # type: ignore[arg-type]
            source_manifest_sha256=row[  # type: ignore[arg-type]
                "source_manifest_sha256"
            ],
            operator_source_sha256=row[  # type: ignore[arg-type]
                "operator_source_sha256"
            ],
            evidence_sha256=row["evidence_sha256"],  # type: ignore[arg-type]
            evidence_scope=row["evidence_scope"],  # type: ignore[arg-type]
            production_authority=row["production_authority"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ActionDescriptorV2:
    """One canonical, homogeneous, case-unbound action descriptor."""

    action_id: str
    aliases: tuple[ActionAliasV2, ...]
    operator: ActionOperatorV2
    input_strength_status: ControlValueStatusV2
    input_strength: float | None
    output_beta_status: ControlValueStatusV2
    output_beta: float | None
    endpoint: ActionEndpointV2
    support: ActionSupportV2
    cost: ActionCostV2
    availability: ActionAvailabilityV2
    receipt: ActionReceiptV2
    descriptor_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        _require_text(self.action_id, "canonical action id")
        if any(not isinstance(row, ActionAliasV2) for row in self.aliases):
            raise ValueError("action aliases must be typed")
        alias_ids = tuple(row.alias_id for row in self.aliases)
        if alias_ids != tuple(sorted(alias_ids)) or len(alias_ids) != len(set(alias_ids)):
            raise ValueError("action aliases must have sorted unique ids")
        if self.action_id in alias_ids:
            raise ValueError("a canonical action id cannot alias itself")
        if not isinstance(self.operator, ActionOperatorV2):
            raise ValueError("action operator must be typed")
        if not isinstance(self.input_strength_status, ControlValueStatusV2):
            raise ValueError("input strength status must be typed")
        if not isinstance(self.output_beta_status, ControlValueStatusV2):
            raise ValueError("output beta status must be typed")

        def normalized_control_value(
            status: ControlValueStatusV2,
            value: float | None,
            name: str,
        ) -> float | None:
            if status is ControlValueStatusV2.UNBOUND:
                if value is not None:
                    raise ValueError(f"unbound {name} cannot carry a value")
                return None
            if value is None:
                raise ValueError(f"bound {name} needs a value")
            return _require_fraction(value, name)

        strength = normalized_control_value(
            self.input_strength_status, self.input_strength, "input strength"
        )
        beta = normalized_control_value(
            self.output_beta_status, self.output_beta, "output beta"
        )
        if not isinstance(self.endpoint, ActionEndpointV2):
            raise ValueError("action endpoint must be typed")
        if not isinstance(self.support, ActionSupportV2):
            raise ValueError("action support must be typed")
        if not isinstance(self.cost, ActionCostV2):
            raise ValueError("action cost must be typed")
        if not isinstance(self.availability, ActionAvailabilityV2):
            raise ValueError("action availability must be typed")
        if not isinstance(self.receipt, ActionReceiptV2):
            raise ValueError("action receipt must be typed")
        if self.availability.production_authority != self.receipt.production_authority:
            raise ValueError("action and receipt production authority disagree")
        native = self.action_id == NATIVE_ACTION_ID_V2
        if (
            self.availability.lifecycle
            is ActionLifecycleV2.NATIVE_REFERENCE_AVAILABLE_TEST_ONLY
            and not native
        ):
            raise ValueError(
                "native-reference lifecycle is reserved for native action zero"
            )
        if native and self.availability.lifecycle in {
            ActionLifecycleV2.CHILD_CC_AVAILABLE_TEST_ONLY,
            ActionLifecycleV2.CHILD_QUARTET_AVAILABLE_TEST_ONLY,
        }:
            raise ValueError(
                "native action zero cannot use an intervention-child lifecycle"
            )
        advanced = self.availability.lifecycle not in {
            ActionLifecycleV2.BLOCKED,
            ActionLifecycleV2.CATALOG_ONLY,
        }
        if advanced and (
            self.input_strength_status is not ControlValueStatusV2.BOUND
            or self.output_beta_status is not ControlValueStatusV2.BOUND
        ):
            raise ValueError(
                "an action with materialized evidence needs bound normalized controls"
            )
        if (
            self.availability.execution_authorized
            and self.cost.status is not ActionCostStatusV2.BOUND
        ):
            raise ValueError(
                "an execution-authorized action needs frozen cost ceilings"
            )
        if native:
            if (
                self.operator.operator_id != NATIVE_OPERATOR_ID_V2
                or self.operator.parameters
            ):
                raise ValueError(
                    "the native action needs a parameter-free identity operator"
                )
            if self.endpoint is not ActionEndpointV2.NONE:
                raise ValueError("the native action endpoint must be none")
            if (
                self.input_strength_status is not ControlValueStatusV2.BOUND
                or self.output_beta_status is not ControlValueStatusV2.BOUND
                or strength != 0.0
                or beta != 0.0
            ):
                raise ValueError("the native action strength and beta must be zero")
            if (
                self.cost.status is not ActionCostStatusV2.BOUND
                or self.cost.runtime_wall_seconds_ceiling != 0.0
                or self.cost.matcher_trajectories_ceiling != 0
                or self.cost.peak_vram_bytes_ceiling != 0
            ):
                raise ValueError("the native action needs exact zero cost ceilings")
            if not self.availability.execution_authorized or not advanced:
                raise ValueError(
                    "the native action must be available as the action-zero fallback"
                )
        elif self.endpoint is ActionEndpointV2.NONE:
            raise ValueError("only the native action may use endpoint none")
        elif strength is not None and strength <= 0.0:
            raise ValueError(
                "a bound nonnative input strength must be positive"
            )
        elif beta is not None and beta <= 0.0:
            raise ValueError("a bound nonnative output beta must be positive")
        object.__setattr__(self, "input_strength", strength)
        object.__setattr__(self, "output_beta", beta)
        expected = _sha256(self.as_dict(include_hash=False))
        if self.descriptor_hash:
            _require_sha256(self.descriptor_hash, "action descriptor hash")
            if self.descriptor_hash != expected:
                raise ValueError("action descriptor hash drift")
        object.__setattr__(self, "descriptor_hash", expected)

    @property
    def canonical_action_id(self) -> str:
        return self.action_id

    def as_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": ACTION_DESCRIPTOR_SCHEMA_V2,
            "action_id": self.action_id,
            "aliases": [row.as_dict() for row in self.aliases],
            "operator": self.operator.as_dict(),
            "input_strength_status": self.input_strength_status.value,
            "input_strength": self.input_strength,
            "output_beta_status": self.output_beta_status.value,
            "output_beta": self.output_beta,
            "endpoint": self.endpoint.value,
            "support": self.support.as_dict(),
            "cost": self.cost.as_dict(),
            "availability": self.availability.as_dict(),
            "receipt": self.receipt.as_dict(),
        }
        if include_hash:
            result["descriptor_hash"] = self.descriptor_hash
        return result

    @classmethod
    def from_mapping(cls, value: object) -> "ActionDescriptorV2":
        if not isinstance(value, Mapping):
            raise ValueError("action descriptor v2 must be a mapping")
        keys = frozenset(value)
        if _LEGACY_EIGHT_FIELDS <= keys and keys != _DESCRIPTOR_FIELDS:
            raise ValueError(
                "legacy heterogeneous eight-field action row is not an "
                "action descriptor v2; normalize it explicitly"
            )
        row = _require_mapping(value, _DESCRIPTOR_FIELDS, "action descriptor v2")
        if row["schema"] != ACTION_DESCRIPTOR_SCHEMA_V2:
            raise ValueError("unknown action descriptor schema")
        aliases = row["aliases"]
        if not isinstance(aliases, list):
            raise ValueError("action aliases must be a list")
        return cls(
            action_id=row["action_id"],  # type: ignore[arg-type]
            aliases=tuple(ActionAliasV2.from_mapping(item) for item in aliases),
            operator=ActionOperatorV2.from_mapping(row["operator"]),
            input_strength_status=_enum_from_value(  # type: ignore[arg-type]
                row["input_strength_status"],
                ControlValueStatusV2,
                "input strength status",
            ),
            input_strength=row["input_strength"],  # type: ignore[arg-type]
            output_beta_status=_enum_from_value(  # type: ignore[arg-type]
                row["output_beta_status"],
                ControlValueStatusV2,
                "output beta status",
            ),
            output_beta=row["output_beta"],  # type: ignore[arg-type]
            endpoint=_enum_from_value(  # type: ignore[arg-type]
                row["endpoint"], ActionEndpointV2, "action endpoint"
            ),
            support=ActionSupportV2.from_mapping(row["support"]),
            cost=ActionCostV2.from_mapping(row["cost"]),
            availability=ActionAvailabilityV2.from_mapping(row["availability"]),
            receipt=ActionReceiptV2.from_mapping(row["receipt"]),
            descriptor_hash=row["descriptor_hash"],  # type: ignore[arg-type]
        )


def parse_action_descriptor_v2(value: object) -> ActionDescriptorV2:
    """Parse a sealed v2 descriptor without legacy shape coercion."""

    return ActionDescriptorV2.from_mapping(value)


def validate_action_bank_v2(
    descriptors: Sequence[ActionDescriptorV2],
) -> dict[str, str]:
    """Validate canonical and alias uniqueness, returning id -> canonical id."""

    if isinstance(descriptors, (str, bytes)) or not isinstance(descriptors, Sequence):
        raise ValueError("action bank must be a sequence of typed descriptors")
    if not descriptors:
        raise ValueError("action bank must contain at least one descriptor")
    if any(not isinstance(row, ActionDescriptorV2) for row in descriptors):
        raise ValueError("action bank entries must be typed descriptors")
    canonical_ids = tuple(row.action_id for row in descriptors)
    if canonical_ids != tuple(sorted(canonical_ids)):
        raise ValueError("canonical action ids must be sorted")
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("canonical action ids must be unique")
    if NATIVE_ACTION_ID_V2 not in canonical_ids:
        raise ValueError("a canonical action bank must contain native action zero")
    result = {action_id: action_id for action_id in canonical_ids}
    for descriptor in descriptors:
        for alias in descriptor.aliases:
            if alias.alias_id in result:
                raise ValueError(
                    f"action alias collision: {alias.alias_id}"
                )
            result[alias.alias_id] = descriptor.action_id
    return result


def canonical_action_id_v2(
    action_id: str,
    descriptors: Sequence[ActionDescriptorV2],
) -> str:
    """Resolve a frozen alias without turning it into another action arm."""

    _require_text(action_id, "action id to resolve")
    identities = validate_action_bank_v2(descriptors)
    try:
        return identities[action_id]
    except KeyError as exc:
        raise ValueError(f"unknown action id or alias: {action_id}") from exc


@dataclass(frozen=True)
class CaseBindingPrerequisitesV2:
    """Hashes required before a later adapter may build a case-bound arm."""

    case_id: str
    exact_endpoint: ActionEndpointV2
    source_input_sha256s: tuple[str, ...]
    endpoint_sha256: str
    support_realization_sha256: str
    rollback_sha256: str
    materialization_recipe_sha256: str
    cost_ceiling_receipt_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.case_id, "case id")
        if self.exact_endpoint not in {
            ActionEndpointV2.FIRST,
            ActionEndpointV2.SECOND,
            ActionEndpointV2.BOTH,
        }:
            raise ValueError("case binding needs an exact endpoint")
        if not self.source_input_sha256s:
            raise ValueError("case binding needs source-input hashes")
        if len(self.source_input_sha256s) != len(set(self.source_input_sha256s)):
            raise ValueError("case binding source-input hashes must be unique")
        for index, value in enumerate(self.source_input_sha256s):
            _require_sha256(value, f"source-input hash {index}")
        for value, name in (
            (self.endpoint_sha256, "endpoint hash"),
            (self.support_realization_sha256, "support realization hash"),
            (self.rollback_sha256, "rollback hash"),
            (self.materialization_recipe_sha256, "materialization recipe hash"),
            (self.cost_ceiling_receipt_sha256, "cost-ceiling receipt hash"),
        ):
            _require_sha256(value, name)


def validate_case_binding_prerequisites_v2(
    descriptor: ActionDescriptorV2,
    binding: CaseBindingPrerequisitesV2,
) -> None:
    """Fail closed before any separate CandidateArmV7 adapter is invoked."""

    if not isinstance(descriptor, ActionDescriptorV2):
        raise ValueError("case binding needs a typed action descriptor")
    if not isinstance(binding, CaseBindingPrerequisitesV2):
        raise ValueError("case binding prerequisites must be typed")
    if not descriptor.availability.execution_authorized:
        raise ValueError("action descriptor is not execution-authorized")
    if descriptor.cost.status is not ActionCostStatusV2.BOUND:
        raise ValueError("case binding needs frozen prospective cost ceilings")
    if descriptor.action_id == NATIVE_ACTION_ID_V2:
        if descriptor.endpoint is not ActionEndpointV2.NONE:
            raise ValueError("native descriptor endpoint drift")
    elif descriptor.endpoint is ActionEndpointV2.CASE_SELECTED_CORRUPTED:
        if binding.exact_endpoint is ActionEndpointV2.BOTH:
            raise ValueError(
                "a case-selected corrupted endpoint must resolve to one endpoint"
            )
    elif descriptor.endpoint is ActionEndpointV2.CASE_SELECTED_SUPPORTED:
        # CaseBindingPrerequisitesV2 already restricts the resolved endpoint
        # to FIRST/SECOND/BOTH.  The endpoint hash binds the before-only rule
        # receipt and its exact case result; no runtime outcome may choose it.
        pass
    elif descriptor.endpoint is ActionEndpointV2.CASE_SELECTED_SINGLE_SUPPORTED:
        if binding.exact_endpoint is ActionEndpointV2.BOTH:
            raise ValueError(
                "a case-selected single-supported endpoint must resolve to one endpoint"
            )
    elif binding.exact_endpoint is not descriptor.endpoint:
        raise ValueError("case binding endpoint disagrees with action descriptor")


__all__ = [
    "ACTION_DESCRIPTOR_SCHEMA_V2",
    "NATIVE_ACTION_ID_V2",
    "NATIVE_OPERATOR_ID_V2",
    "ActionAliasV2",
    "ActionAvailabilityV2",
    "ActionCostStatusV2",
    "ActionCostV2",
    "ActionDescriptorV2",
    "ActionEndpointV2",
    "ActionLifecycleV2",
    "ActionOperatorV2",
    "ActionReceiptV2",
    "ActionSupportV2",
    "CaseBindingPrerequisitesV2",
    "ControlValueStatusV2",
    "OperatorParameterV2",
    "canonical_action_id_v2",
    "parse_action_descriptor_v2",
    "validate_action_bank_v2",
    "validate_case_binding_prerequisites_v2",
]
