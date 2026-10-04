from dataclasses import replace

import pytest

from stablebridge.physical_repair.action_bank_adapter import (
    FrozenCaseArmBindingV2,
    bind_action_descriptor_v2_to_candidate_arm_v7,
    case_binding_prerequisites_hash_v2,
)
from stablebridge.physical_repair.action_bank_contracts import (
    NATIVE_ACTION_ID_V2,
    NATIVE_OPERATOR_ID_V2,
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
)
from stablebridge.physical_repair.selector_v7 import (
    CandidateArmV7,
    CandidateScopeV7,
    CostCountersV7,
)


def h(digit: str) -> str:
    return digit * 64


def counters(
    *,
    wall_seconds: float = 2.0,
    trajectories: int = 1,
    peak_reserved_bytes: int = 8_000_000_000,
) -> CostCountersV7:
    return CostCountersV7(
        natural_forwards=0,
        reused_forwards=0,
        candidate_forwards=1,
        reverse_forwards=0,
        cpu_seconds=0.5,
        gpu_seconds=1.5,
        wall_seconds=wall_seconds,
        bytes_moved=1024,
        peak_allocated_bytes=min(1_000_000_000, peak_reserved_bytes),
        peak_reserved_bytes=peak_reserved_bytes,
        matcher_trajectories=trajectories,
        cache_state="cold",
    )


def descriptor(
    *,
    iterations: int = 8,
    input_strength: float = 0.5,
) -> ActionDescriptorV2:
    return ActionDescriptorV2(
        action_id=f"CSB/OF/action/iters{iterations}",
        aliases=(),
        operator=ActionOperatorV2(
            operator_id="matcher_iteration_override",
            version="v1",
            parameters=(
                OperatorParameterV2(
                    name="iterations",
                    value=iterations,
                    unit="iterations",
                ),
            ),
        ),
        input_strength_status=ControlValueStatusV2.BOUND,
        input_strength=input_strength,
        output_beta_status=ControlValueStatusV2.BOUND,
        output_beta=1.0,
        endpoint=ActionEndpointV2.BOTH,
        support=ActionSupportV2("whole_pair_v1", h("1")),
        cost=ActionCostV2(
            status=ActionCostStatusV2.BOUND,
            runtime_wall_seconds_ceiling=2.0,
            matcher_trajectories_ceiling=1,
            peak_vram_bytes_ceiling=8_000_000_000,
            source_sha256=h("2"),
        ),
        availability=ActionAvailabilityV2(
            lifecycle=ActionLifecycleV2.CASE_BOUND_TEST_ONLY,
            execution_authorized=True,
            production_authority=False,
        ),
        receipt=ActionReceiptV2(
            receipt_id=f"receipt:iters{iterations}",
            source_manifest_sha256=h("3"),
            operator_source_sha256=h("4"),
            evidence_sha256=h("5"),
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
        support=ActionSupportV2("whole_pair_v1", h("1")),
        cost=ActionCostV2(
            status=ActionCostStatusV2.BOUND,
            runtime_wall_seconds_ceiling=0.0,
            matcher_trajectories_ceiling=0,
            peak_vram_bytes_ceiling=0,
            source_sha256=h("2"),
        ),
        availability=ActionAvailabilityV2(
            lifecycle=ActionLifecycleV2.CASE_BOUND_TEST_ONLY,
            execution_authorized=True,
            production_authority=False,
        ),
        receipt=ActionReceiptV2(
            receipt_id="receipt:native",
            source_manifest_sha256=h("3"),
            operator_source_sha256=h("4"),
            evidence_sha256=h("5"),
            evidence_scope="identity fallback",
            production_authority=False,
        ),
    )


def prerequisites(
    endpoint: ActionEndpointV2 = ActionEndpointV2.BOTH,
) -> CaseBindingPrerequisitesV2:
    return CaseBindingPrerequisitesV2(
        case_id="case-1",
        exact_endpoint=endpoint,
        source_input_sha256s=(h("6"), h("7")),
        endpoint_sha256=h("8"),
        support_realization_sha256=h("9"),
        rollback_sha256=h("a"),
        materialization_recipe_sha256=h("b"),
        cost_ceiling_receipt_sha256=h("c"),
    )


def frozen_binding(
    arm: ActionDescriptorV2,
    case: CaseBindingPrerequisitesV2,
    *,
    action_index: int = 4,
    exact_control_id: str = "iters8",
    runtime_counters: CostCountersV7 | None = None,
    offline_cost: float = 2.0,
    offline_counters: CostCountersV7 | None = None,
) -> FrozenCaseArmBindingV2:
    return FrozenCaseArmBindingV2(
        descriptor_sha256=arm.descriptor_hash,
        prerequisites_sha256=case_binding_prerequisites_hash_v2(case),
        candidate_id=f"candidate:{arm.action_id}",
        action_index=action_index,
        exact_control_id=exact_control_id,
        control_sha256=h("d"),
        materialization_recipe_id="materialize-pair-v1",
        materialization_recipe_version="v1",
        source_provenance_sha256=h("e"),
        offline_bank_acquisition_cost_ceiling=offline_cost,
        offline_bank_acquisition_cost_counter_ceiling=(
            offline_counters if offline_counters is not None else counters()
        ),
        prospective_runtime_cost_counter_ceiling=(
            runtime_counters if runtime_counters is not None else counters()
        ),
        production_authority=False,
    )


def test_bridge_builds_exact_case_atomic_candidate_without_parameter_leakage() -> None:
    source = descriptor(iterations=8, input_strength=0.5)
    case = prerequisites()
    seal = frozen_binding(source, case)

    candidate = bind_action_descriptor_v2_to_candidate_arm_v7(
        source, case, seal
    )

    assert isinstance(candidate, CandidateArmV7)
    assert candidate.scope is CandidateScopeV7.CASE_ATOMIC
    assert candidate.unit_ids == ("case-1",)
    assert candidate.action_identity == source.action_id
    assert candidate.mechanism_id == "matcher_iteration_override"
    assert candidate.input_strength == 0.5
    assert candidate.output_beta == 1.0
    assert source.operator.parameters[0].value == 8
    assert candidate.action_hash == source.descriptor_hash
    assert candidate.support_policy_hash == source.support.policy_sha256
    assert candidate.support_hash == case.support_realization_sha256
    assert candidate.rollback_hash == case.rollback_sha256
    assert candidate.provenance_hash == seal.binding_hash
    assert candidate.hard_legal is True


def test_p4_r7_iterations_change_action_hash_not_normalized_strength() -> None:
    case = prerequisites()
    p4 = descriptor(iterations=8, input_strength=0.5)
    r7 = descriptor(iterations=12, input_strength=0.5)

    p4_arm = bind_action_descriptor_v2_to_candidate_arm_v7(
        p4, case, frozen_binding(p4, case, exact_control_id="iters8")
    )
    r7_arm = bind_action_descriptor_v2_to_candidate_arm_v7(
        r7,
        case,
        frozen_binding(r7, case, action_index=5, exact_control_id="iters12"),
    )

    assert p4_arm.input_strength == r7_arm.input_strength == 0.5
    assert p4_arm.action_hash != r7_arm.action_hash
    assert p4.operator.parameters[0].value == 8
    assert r7.operator.parameters[0].value == 12


def test_native_descriptor_becomes_exact_zero_cost_action_zero() -> None:
    source = native_descriptor()
    case = prerequisites(ActionEndpointV2.SECOND)
    zero = CostCountersV7.zero(cache_state="cold")
    seal = frozen_binding(
        source,
        case,
        action_index=0,
        exact_control_id="native",
        runtime_counters=zero,
        offline_cost=0.0,
        offline_counters=zero,
    )

    candidate = bind_action_descriptor_v2_to_candidate_arm_v7(
        source, case, seal
    )

    assert candidate.action_index == 0
    assert candidate.action_identity == "native"
    assert candidate.exact_control_id == "native"
    assert candidate.scope is CandidateScopeV7.NATIVE
    assert candidate.endpoint_id == "second"
    assert candidate.input_strength == candidate.output_beta == 0.0
    assert candidate.prospective_runtime_cost_ceiling == 0.0


def test_native_and_nonnative_action_indices_cannot_be_swapped() -> None:
    case = prerequisites()
    nonnative = descriptor()
    with pytest.raises(ValueError, match="only the native descriptor"):
        bind_action_descriptor_v2_to_candidate_arm_v7(
            nonnative,
            case,
            frozen_binding(nonnative, case, action_index=0),
        )

    native = native_descriptor()
    zero = CostCountersV7.zero(cache_state="cold")
    native_binding = frozen_binding(
        native,
        case,
        action_index=1,
        exact_control_id="native",
        runtime_counters=zero,
        offline_cost=0.0,
        offline_counters=zero,
    )
    with pytest.raises(ValueError, match="action index zero"):
        bind_action_descriptor_v2_to_candidate_arm_v7(
            native, case, native_binding
        )


def test_binding_is_sealed_to_descriptor_and_case_prerequisites() -> None:
    source = descriptor()
    case = prerequisites()
    seal = frozen_binding(source, case)

    with pytest.raises(ValueError, match="binding hash drift"):
        replace(seal, binding_hash=h("0"))

    wrong_descriptor = replace(
        seal,
        descriptor_sha256=h("f"),
        binding_hash="",
    )
    with pytest.raises(ValueError, match="descriptor hash mismatch"):
        bind_action_descriptor_v2_to_candidate_arm_v7(
            source, case, wrong_descriptor
        )

    wrong_case = replace(
        seal,
        prerequisites_sha256=h("f"),
        binding_hash="",
    )
    with pytest.raises(ValueError, match="prerequisites hash mismatch"):
        bind_action_descriptor_v2_to_candidate_arm_v7(
            source, case, wrong_case
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("source_input_sha256s", (), "source-input hashes"),
        ("support_realization_sha256", "", "support realization hash"),
        ("rollback_sha256", "", "rollback hash"),
        ("materialization_recipe_sha256", "", "materialization recipe hash"),
        ("cost_ceiling_receipt_sha256", "", "cost-ceiling receipt hash"),
    ),
)
def test_missing_case_receipts_fail_before_adaptation(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(prerequisites(), **{field: value})


def test_missing_recipe_binding_and_production_authority_fail_closed() -> None:
    source = descriptor()
    case = prerequisites()
    seal = frozen_binding(source, case)
    with pytest.raises(ValueError, match="materialization recipe id"):
        replace(seal, materialization_recipe_id="", binding_hash="")
    with pytest.raises(ValueError, match="cannot grant production authority"):
        replace(seal, production_authority=True, binding_hash="")


@pytest.mark.parametrize(
    ("bad_counters", "message"),
    (
        (counters(wall_seconds=3.0), "wall-seconds"),
        (counters(trajectories=2), "matcher trajectories"),
        (counters(peak_reserved_bytes=7_000_000_000), "reserved VRAM"),
    ),
)
def test_runtime_counter_ceiling_must_match_descriptor_receipt(
    bad_counters: CostCountersV7,
    message: str,
) -> None:
    source = descriptor()
    case = prerequisites()
    seal = frozen_binding(source, case, runtime_counters=bad_counters)
    with pytest.raises(ValueError, match=message):
        bind_action_descriptor_v2_to_candidate_arm_v7(source, case, seal)


def test_disabled_or_endpoint_mismatched_descriptor_cannot_be_bridged() -> None:
    source = descriptor()
    disabled = replace(
        source,
        availability=replace(
            source.availability,
            execution_authorized=False,
        ),
        descriptor_hash="",
    )
    case = prerequisites()
    with pytest.raises(ValueError, match="not execution-authorized"):
        bind_action_descriptor_v2_to_candidate_arm_v7(
            disabled, case, frozen_binding(disabled, case)
        )

    first = prerequisites(ActionEndpointV2.FIRST)
    with pytest.raises(ValueError, match="endpoint disagrees"):
        bind_action_descriptor_v2_to_candidate_arm_v7(
            source, first, frozen_binding(source, first)
        )


def test_case_selected_supported_descriptor_binds_exact_both_endpoint() -> None:
    source = replace(
        descriptor(),
        endpoint=ActionEndpointV2.CASE_SELECTED_SUPPORTED,
        descriptor_hash="",
    )
    case = prerequisites(ActionEndpointV2.BOTH)
    candidate = bind_action_descriptor_v2_to_candidate_arm_v7(
        source, case, frozen_binding(source, case)
    )
    assert candidate.endpoint_id == "both"
    assert candidate.action_identity == source.action_id


def test_case_selected_single_supported_descriptor_rejects_both_endpoint() -> None:
    source = replace(
        descriptor(),
        endpoint=ActionEndpointV2.CASE_SELECTED_SINGLE_SUPPORTED,
        descriptor_hash="",
    )
    case = prerequisites(ActionEndpointV2.BOTH)
    with pytest.raises(ValueError, match="single-supported endpoint must resolve to one"):
        bind_action_descriptor_v2_to_candidate_arm_v7(
            source, case, frozen_binding(source, case)
        )
