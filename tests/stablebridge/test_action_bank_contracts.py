from dataclasses import replace
import json
from pathlib import Path

import pytest

from stablebridge.physical_repair.action_bank_contracts import (
    ACTION_DESCRIPTOR_SCHEMA_V2,
    NATIVE_ACTION_ID_V2,
    NATIVE_OPERATOR_ID_V2,
    ActionAliasV2,
    ActionAvailabilityV2,
    ActionCostStatusV2,
    ActionCostV2,
    ActionDescriptorV2,
    ActionEndpointV2,
    ActionLifecycleV2,
    ActionOperatorV2,
    ActionReceiptV2,
    ActionSupportV2,
    CaseBindingPrerequisitesV2,
    ControlValueStatusV2,
    OperatorParameterV2,
    canonical_action_id_v2,
    parse_action_descriptor_v2,
    validate_action_bank_v2,
    validate_case_binding_prerequisites_v2,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def h(digit: str) -> str:
    return digit * 64


def descriptor(
    *,
    action_id: str = "action/canonical",
    aliases: tuple[ActionAliasV2, ...] | None = None,
    endpoint: ActionEndpointV2 = ActionEndpointV2.BOTH,
    execution_authorized: bool = True,
) -> ActionDescriptorV2:
    if aliases is None:
        aliases = (ActionAliasV2("action/legacy", h("1")),)
    return ActionDescriptorV2(
        action_id=action_id,
        aliases=aliases,
        operator=ActionOperatorV2(
            operator_id="matcher_iteration_override",
            version="v1",
            parameters=(
                OperatorParameterV2(
                    name="iterations",
                    value=8,
                    unit="iterations",
                ),
            ),
        ),
        input_strength_status=ControlValueStatusV2.BOUND,
        input_strength=0.5,
        output_beta_status=ControlValueStatusV2.BOUND,
        output_beta=1.0,
        endpoint=endpoint,
        support=ActionSupportV2(
            policy_id="whole_pair_finite_float32_v1",
            policy_sha256=h("2"),
        ),
        cost=ActionCostV2(
            status=ActionCostStatusV2.BOUND,
            runtime_wall_seconds_ceiling=2.0,
            matcher_trajectories_ceiling=1,
            peak_vram_bytes_ceiling=8_000_000_000,
            source_sha256=h("3"),
        ),
        availability=ActionAvailabilityV2(
            lifecycle=ActionLifecycleV2.MATERIALIZABLE_TEST_ONLY,
            execution_authorized=execution_authorized,
            production_authority=False,
        ),
        receipt=ActionReceiptV2(
            receipt_id=f"receipt:{action_id}",
            source_manifest_sha256=h("4"),
            operator_source_sha256=h("5"),
            evidence_sha256=h("6"),
            evidence_scope="development-only",
            production_authority=False,
        ),
    )


def native_descriptor() -> ActionDescriptorV2:
    return ActionDescriptorV2(
        action_id=NATIVE_ACTION_ID_V2,
        aliases=(),
        operator=ActionOperatorV2(
            operator_id=NATIVE_OPERATOR_ID_V2,
            version="v1",
            parameters=(),
        ),
        input_strength_status=ControlValueStatusV2.BOUND,
        input_strength=0.0,
        output_beta_status=ControlValueStatusV2.BOUND,
        output_beta=0.0,
        endpoint=ActionEndpointV2.NONE,
        support=ActionSupportV2(
            policy_id="whole_pair_finite_float32_v1",
            policy_sha256=h("2"),
        ),
        cost=ActionCostV2(
            status=ActionCostStatusV2.BOUND,
            runtime_wall_seconds_ceiling=0.0,
            matcher_trajectories_ceiling=0,
            peak_vram_bytes_ceiling=0,
            source_sha256=h("3"),
        ),
        availability=ActionAvailabilityV2(
            lifecycle=ActionLifecycleV2.MATERIALIZABLE_TEST_ONLY,
            execution_authorized=True,
            production_authority=False,
        ),
        receipt=ActionReceiptV2(
            receipt_id="receipt:native",
            source_manifest_sha256=h("4"),
            operator_source_sha256=h("5"),
            evidence_sha256=h("6"),
            evidence_scope="identity fallback",
            production_authority=False,
        ),
    )


def binding(
    endpoint: ActionEndpointV2 = ActionEndpointV2.BOTH,
) -> CaseBindingPrerequisitesV2:
    return CaseBindingPrerequisitesV2(
        case_id="case-1",
        exact_endpoint=endpoint,
        source_input_sha256s=(h("7"), h("8")),
        endpoint_sha256=h("9"),
        support_realization_sha256=h("a"),
        rollback_sha256=h("b"),
        materialization_recipe_sha256=h("c"),
        cost_ceiling_receipt_sha256=h("d"),
    )


def test_v2_round_trip_is_homogeneous_and_separates_parameter_from_strength() -> None:
    arm = descriptor()
    payload = arm.as_dict()

    assert payload["schema"] == ACTION_DESCRIPTOR_SCHEMA_V2
    assert "strength" not in payload
    assert payload["input_strength_status"] == "BOUND"
    assert payload["input_strength"] == 0.5
    assert payload["output_beta_status"] == "BOUND"
    assert payload["output_beta"] == 1.0
    assert payload["operator"]["parameters"] == [
        {"name": "iterations", "value": 8, "unit": "iterations"}
    ]
    assert len(payload["descriptor_hash"]) == 64
    assert parse_action_descriptor_v2(payload) == arm


def test_native_is_exact_action_zero_contract_and_resolves_case_endpoint() -> None:
    native = native_descriptor()
    assert native.action_id == "native"
    assert native.endpoint is ActionEndpointV2.NONE
    assert native.input_strength == native.output_beta == 0.0
    assert native.cost.runtime_wall_seconds_ceiling == 0.0
    validate_case_binding_prerequisites_v2(native, binding())

    with pytest.raises(ValueError, match="strength and beta must be zero"):
        replace(native, input_strength=0.1, descriptor_hash="")
    with pytest.raises(ValueError, match="parameter-free identity"):
        replace(native, operator=descriptor().operator, descriptor_hash="")
    with pytest.raises(ValueError, match="exact zero cost"):
        replace(native, cost=descriptor().cost, descriptor_hash="")
    with pytest.raises(ValueError, match="only the native action"):
        replace(descriptor(), endpoint=ActionEndpointV2.NONE, descriptor_hash="")
    with pytest.raises(ValueError, match="nonnative input strength must be positive"):
        replace(descriptor(), input_strength=0.0, descriptor_hash="")


@pytest.mark.parametrize(
    ("relative_path", "collection"),
    (
        (
            "experiments/E240_weather_restoration_action_pilot_v1/"
            "ACTION_MANIFEST.json",
            "actions",
        ),
        (
            "experiments/E242_recommended_restorer_route_v1/"
            "ACTION_MANIFEST.json",
            "routes",
        ),
    ),
)
def test_legacy_e240_e242_heterogeneous_rows_are_rejected_explicitly(
    relative_path: str,
    collection: str,
) -> None:
    manifest = json.loads((REPO_ROOT / relative_path).read_text())
    row = manifest[collection][0]

    with pytest.raises(ValueError, match="legacy heterogeneous eight-field"):
        parse_action_descriptor_v2(row)


def test_descriptor_hash_drift_and_noncanonical_hashes_fail_closed() -> None:
    arm = descriptor()
    with pytest.raises(ValueError, match="descriptor hash drift"):
        replace(arm, descriptor_hash=h("0"))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(arm.support, policy_sha256="A" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(arm.receipt, evidence_sha256="short")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_normalized_strength_is_strict_and_finite(bad: object) -> None:
    with pytest.raises(ValueError, match="input strength"):
        replace(descriptor(), input_strength=bad, descriptor_hash="")


def test_operator_parameters_and_cost_ceilings_are_finite() -> None:
    with pytest.raises(ValueError, match="operator parameter sigma must be finite"):
        OperatorParameterV2("sigma", float("nan"), "pixels")
    with pytest.raises(ValueError, match="runtime wall-seconds ceiling"):
        replace(
            descriptor().cost,
            runtime_wall_seconds_ceiling=float("inf"),
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        replace(descriptor().cost, matcher_trajectories_ceiling=True)


def test_authority_cannot_be_laundered_through_availability_or_receipt() -> None:
    with pytest.raises(ValueError, match="cannot grant production authority"):
        ActionAvailabilityV2(
            lifecycle=ActionLifecycleV2.MATERIALIZABLE_TEST_ONLY,
            execution_authorized=True,
            production_authority=True,
        )
    with pytest.raises(ValueError, match="cannot grant production authority"):
        replace(descriptor().receipt, production_authority=True)
    with pytest.raises(ValueError, match="cannot be execution-authorized"):
        ActionAvailabilityV2(
            lifecycle=ActionLifecycleV2.BLOCKED,
            execution_authorized=True,
        )


def test_unbound_cost_is_orthogonal_to_historical_evidence_but_blocks_execution() -> None:
    unbound = ActionCostV2(
        status=ActionCostStatusV2.UNBOUND,
        runtime_wall_seconds_ceiling=None,
        matcher_trajectories_ceiling=None,
        peak_vram_bytes_ceiling=None,
        source_sha256=h("e"),
    )
    blocked = replace(
        descriptor(),
        cost=unbound,
        availability=ActionAvailabilityV2(
            lifecycle=ActionLifecycleV2.BLOCKED,
            execution_authorized=False,
        ),
        descriptor_hash="",
    )
    assert blocked.cost.status is ActionCostStatusV2.UNBOUND

    historical = replace(
        blocked,
        input_strength_status=ControlValueStatusV2.BOUND,
        input_strength=1.0,
        output_beta_status=ControlValueStatusV2.BOUND,
        output_beta=1.0,
        availability=ActionAvailabilityV2(
            lifecycle=ActionLifecycleV2.CHILD_QUARTET_AVAILABLE_TEST_ONLY,
            execution_authorized=False,
        ),
        descriptor_hash="",
    )
    assert historical.cost.status is ActionCostStatusV2.UNBOUND
    assert historical.availability.lifecycle is (
        ActionLifecycleV2.CHILD_QUARTET_AVAILABLE_TEST_ONLY
    )

    with pytest.raises(ValueError, match="execution-authorized action needs"):
        replace(
            historical,
            availability=ActionAvailabilityV2(
                lifecycle=ActionLifecycleV2.CHILD_QUARTET_AVAILABLE_TEST_ONLY,
                execution_authorized=True,
            ),
            descriptor_hash="",
        )
    with pytest.raises(ValueError, match="cannot carry ceilings"):
        replace(unbound, runtime_wall_seconds_ceiling=1.0)


def test_unbound_normalized_controls_are_typed_and_cannot_reach_materialization() -> None:
    blocked = replace(
        descriptor(),
        input_strength_status=ControlValueStatusV2.UNBOUND,
        input_strength=None,
        output_beta_status=ControlValueStatusV2.UNBOUND,
        output_beta=None,
        cost=ActionCostV2(
            status=ActionCostStatusV2.UNBOUND,
            runtime_wall_seconds_ceiling=None,
            matcher_trajectories_ceiling=None,
            peak_vram_bytes_ceiling=None,
            source_sha256=h("e"),
        ),
        availability=ActionAvailabilityV2(
            lifecycle=ActionLifecycleV2.BLOCKED,
            execution_authorized=False,
        ),
        descriptor_hash="",
    )
    payload = blocked.as_dict()
    assert payload["input_strength_status"] == "UNBOUND"
    assert payload["input_strength"] is None
    assert payload["output_beta_status"] == "UNBOUND"
    assert payload["output_beta"] is None
    assert parse_action_descriptor_v2(payload) == blocked

    with pytest.raises(ValueError, match="unbound input strength cannot carry"):
        replace(blocked, input_strength=1.0, descriptor_hash="")
    with pytest.raises(ValueError, match="bound input strength needs a value"):
        replace(
            blocked,
            input_strength_status=ControlValueStatusV2.BOUND,
            descriptor_hash="",
        )
    with pytest.raises(ValueError, match="needs bound normalized controls"):
        replace(
            blocked,
            cost=descriptor().cost,
            availability=ActionAvailabilityV2(
                lifecycle=ActionLifecycleV2.MATERIALIZABLE_TEST_ONLY,
                execution_authorized=True,
            ),
            descriptor_hash="",
        )


def test_aliases_resolve_to_one_canonical_arm_and_collisions_fail() -> None:
    first = descriptor()
    second = descriptor(
        action_id="action/second",
        aliases=(ActionAliasV2("action/second-legacy", h("f")),),
    )
    native = native_descriptor()
    bank = (first, second, native)

    identities = validate_action_bank_v2(bank)
    assert identities["action/canonical"] == "action/canonical"
    assert identities["action/legacy"] == "action/canonical"
    assert identities["native"] == "native"
    assert canonical_action_id_v2("action/second-legacy", bank) == "action/second"

    with pytest.raises(ValueError, match="must be sorted"):
        validate_action_bank_v2(tuple(reversed(bank)))
    collision = descriptor(
        action_id="action/second",
        aliases=(ActionAliasV2("action/legacy", h("f")),),
    )
    with pytest.raises(ValueError, match="alias collision"):
        validate_action_bank_v2((first, collision, native))
    with pytest.raises(ValueError, match="must contain native"):
        validate_action_bank_v2((first, second))
    with pytest.raises(ValueError, match="cannot alias itself"):
        descriptor(
            aliases=(ActionAliasV2("action/canonical", h("f")),),
        )


def test_historical_native_id_is_a_receipt_backed_alias_not_a_second_arm() -> None:
    historical_id = "CSB/OF/SEA-RAFT/action/P0"
    native = replace(
        native_descriptor(),
        aliases=(ActionAliasV2(historical_id, h("f")),),
        descriptor_hash="",
    )
    action = descriptor(aliases=())
    bank = tuple(sorted((native, action), key=lambda row: row.action_id))

    identities = validate_action_bank_v2(bank)
    assert identities[historical_id] == NATIVE_ACTION_ID_V2
    assert canonical_action_id_v2(historical_id, bank) == NATIVE_ACTION_ID_V2
    assert len(bank) == 2


def test_native_reference_lifecycle_is_typed_and_native_only() -> None:
    native = replace(
        native_descriptor(),
        availability=ActionAvailabilityV2(
            lifecycle=ActionLifecycleV2.NATIVE_REFERENCE_AVAILABLE_TEST_ONLY,
            execution_authorized=True,
        ),
        descriptor_hash="",
    )
    assert (
        native.availability.lifecycle
        is ActionLifecycleV2.NATIVE_REFERENCE_AVAILABLE_TEST_ONLY
    )
    assert parse_action_descriptor_v2(native.as_dict()) == native

    with pytest.raises(ValueError, match="reserved for native action zero"):
        replace(
            descriptor(),
            availability=ActionAvailabilityV2(
                lifecycle=ActionLifecycleV2.NATIVE_REFERENCE_AVAILABLE_TEST_ONLY,
                execution_authorized=False,
            ),
            descriptor_hash="",
        )

    with pytest.raises(ValueError, match="cannot use an intervention-child"):
        replace(
            native,
            availability=ActionAvailabilityV2(
                lifecycle=ActionLifecycleV2.CHILD_QUARTET_AVAILABLE_TEST_ONLY,
                execution_authorized=True,
            ),
            descriptor_hash="",
        )


def test_case_binding_prerequisites_validate_without_creating_selector_arm() -> None:
    arm = descriptor()
    validate_case_binding_prerequisites_v2(arm, binding())

    disabled = replace(
        arm,
        availability=replace(
            arm.availability,
            execution_authorized=False,
        ),
        descriptor_hash="",
    )
    with pytest.raises(ValueError, match="not execution-authorized"):
        validate_case_binding_prerequisites_v2(disabled, binding())
    with pytest.raises(ValueError, match="endpoint disagrees"):
        validate_case_binding_prerequisites_v2(
            arm,
            binding(ActionEndpointV2.FIRST),
        )


def test_case_selected_corrupted_endpoint_resolves_to_exactly_one_endpoint() -> None:
    arm = descriptor(endpoint=ActionEndpointV2.CASE_SELECTED_CORRUPTED)
    validate_case_binding_prerequisites_v2(
        arm,
        binding(ActionEndpointV2.FIRST),
    )
    with pytest.raises(ValueError, match="must resolve to one endpoint"):
        validate_case_binding_prerequisites_v2(arm, binding())


@pytest.mark.parametrize(
    "exact_endpoint",
    [ActionEndpointV2.FIRST, ActionEndpointV2.SECOND, ActionEndpointV2.BOTH],
)
def test_case_selected_supported_endpoint_accepts_exact_before_only_result(
    exact_endpoint: ActionEndpointV2,
) -> None:
    arm = descriptor(endpoint=ActionEndpointV2.CASE_SELECTED_SUPPORTED)
    validate_case_binding_prerequisites_v2(arm, binding(exact_endpoint))


@pytest.mark.parametrize(
    "exact_endpoint",
    [ActionEndpointV2.FIRST, ActionEndpointV2.SECOND],
)
def test_case_selected_single_supported_accepts_exact_single_endpoint(
    exact_endpoint: ActionEndpointV2,
) -> None:
    arm = descriptor(endpoint=ActionEndpointV2.CASE_SELECTED_SINGLE_SUPPORTED)
    validate_case_binding_prerequisites_v2(arm, binding(exact_endpoint))


def test_case_selected_single_supported_rejects_both_endpoints() -> None:
    arm = descriptor(endpoint=ActionEndpointV2.CASE_SELECTED_SINGLE_SUPPORTED)
    with pytest.raises(ValueError, match="single-supported endpoint must resolve to one"):
        validate_case_binding_prerequisites_v2(
            arm, binding(ActionEndpointV2.BOTH)
        )
