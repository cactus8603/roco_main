from __future__ import annotations

from dataclasses import replace

import pytest

from stablebridge.physical_repair.family_action_competition import (
    ActionFamilyManifestV1,
    ExactControlFamilyBindingV1,
    FrozenActionFamilyDefinitionV1,
    propose_family_competition_v1,
    validate_action_family_manifest_v1,
)
from stablebridge.physical_repair.selector_v7 import (
    ArmAliasReceiptV7,
    ArmRealizationReceiptV7,
    ArmRealizationStatusV7,
    CandidateArmV7,
    CandidateManifestV7,
    CandidateScopeV7,
    CostAccountV7,
    CostCountersV7,
    FoldRoleV7,
    HardRejectReasonV7,
    PlannerCandidateV7,
    PredictionKindV7,
    SELECTOR_V7_SCHEMA_VERSION,
    make_native_arm_v7,
)


def _counters(
    count: int = 1,
    *,
    wall: float = 1.0,
    cache_state: str = "cold",
) -> CostCountersV7:
    return CostCountersV7(
        natural_forwards=0,
        reused_forwards=0,
        candidate_forwards=count,
        reverse_forwards=0,
        cpu_seconds=0.1 * count,
        gpu_seconds=0.9 * count,
        wall_seconds=wall,
        bytes_moved=1024 * count,
        peak_allocated_bytes=1024 if count else 0,
        peak_reserved_bytes=2048 if count else 0,
        matcher_trajectories=count,
        cache_state=cache_state,
    )


def _verification_counters(
    *,
    wall: float = 0.1,
    cache_state: str = "cold",
) -> CostCountersV7:
    return CostCountersV7(
        natural_forwards=0,
        reused_forwards=0,
        candidate_forwards=0,
        reverse_forwards=0,
        cpu_seconds=wall,
        gpu_seconds=0.0,
        wall_seconds=wall,
        bytes_moved=64,
        peak_allocated_bytes=64,
        peak_reserved_bytes=64,
        matcher_trajectories=0,
        cache_state=cache_state,
    )


def _arm(
    index: int,
    candidate_id: str,
    *,
    strength: float,
    cost: float = 1.0,
    action_hash: str | None = None,
    hard_legal: bool = True,
    runtime_counters: CostCountersV7 | None = None,
) -> CandidateArmV7:
    return CandidateArmV7(
        candidate_id=candidate_id,
        action_index=index,
        action_identity=candidate_id.split(":", 1)[0],
        mechanism_id=f"mechanism:{candidate_id}",
        operator_version="v1",
        exact_control_id=f"control:{candidate_id}",
        endpoint_id="second",
        input_strength=strength,
        output_beta=1.0,
        action_hash=action_hash or f"action-hash:{candidate_id}",
        control_hash=f"control-hash:{candidate_id}",
        endpoint_hash="endpoint-hash",
        support_hash=f"support-hash:{candidate_id}",
        support_policy_hash="support-policy",
        rollback_hash=f"rollback-hash:{candidate_id}",
        source_input_hashes=("source-hash",),
        materialization_recipe_id=f"recipe:{candidate_id}",
        materialization_recipe_version="v1",
        materialization_recipe_hash=f"recipe-hash:{candidate_id}",
        provenance_hash=f"provenance:{candidate_id}",
        scope=CandidateScopeV7.CASE_ATOMIC,
        unit_ids=("case-1",),
        interaction_receipts=(),
        offline_bank_acquisition_cost_ceiling=cost,
        offline_bank_acquisition_cost_counter_ceiling=_counters(wall=cost),
        prospective_runtime_cost_ceiling=cost,
        prospective_runtime_cost_counter_ceiling=(
            runtime_counters or _counters(wall=cost)
        ),
        hard_legal=hard_legal,
        hard_reject_reasons=(
            ()
            if hard_legal
            else (HardRejectReasonV7.ILLEGAL_LINEAGE,)
        ),
    )


def _manifest(*arms: CandidateArmV7) -> CandidateManifestV7:
    native = make_native_arm_v7(
        execution_unit_id="case-1",
        endpoint_id="second",
        support_hash="native-support",
        support_policy_hash="support-policy",
        source_hash="source-hash",
        provenance_hash="native-provenance",
        source_materialization_hash="native-bytes",
    )
    return CandidateManifestV7(
        case_id="case-1",
        execution_unit_id="case-1",
        fold_role=FoldRoleV7.EVAL,
        source_hash="source-hash",
        support_policy_hash="support-policy",
        planner_allowlist_hash="planner-allowlist",
        assessor_allowlist_hash="assessor-allowlist",
        schema_version=SELECTOR_V7_SCHEMA_VERSION,
        provenance_hash="manifest-provenance",
        frozen_before_outcome=True,
        candidates=(native, *arms),
    )


def _binding(
    arm: CandidateArmV7,
    family_id: str,
) -> ExactControlFamilyBindingV1:
    return ExactControlFamilyBindingV1(
        candidate_id=arm.candidate_id,
        planned_arm_id=arm.planned_arm_id,
        action_identity=arm.action_identity,
        exact_control_id=arm.exact_control_id,
        action_family_id=family_id,
        strength_id=f"strength-{arm.input_strength:g}",
        variant_id="operator-default",
        beta_id=f"beta-{arm.output_beta:g}",
        input_strength=arm.input_strength,
        output_beta=arm.output_beta,
    )


def _family_manifest(
    bank: CandidateManifestV7,
    family_by_candidate: dict[str, str],
) -> ActionFamilyManifestV1:
    family_ids = tuple(sorted(set(family_by_candidate.values())))
    return ActionFamilyManifestV1(
        selector_manifest_hash=bank.manifest_hash,
        selector_candidate_family_hash=bank.candidate_family_hash,
        frozen_before_outcome=True,
        definitions=tuple(
            FrozenActionFamilyDefinitionV1(
                action_family_id=family_id,
                member_action_identities=(family_id,),
                registry_source_hash="test-family-registry",
            )
            for family_id in family_ids
        ),
        bindings=tuple(
            _binding(arm, family_by_candidate[arm.candidate_id])
            for arm in bank.candidates
            if arm.action_index != 0
        ),
    )


def _score(arm: CandidateArmV7, gain: float) -> PlannerCandidateV7:
    return PlannerCandidateV7(
        candidate_id=arm.candidate_id,
        planned_arm_id=arm.planned_arm_id,
        cost_hash=arm.cost_hash,
        planner_allowlist_hash="planner-allowlist",
        before_feature_hash=f"before:{arm.candidate_id}",
        prediction_kind=(
            PredictionKindV7.PLANNER_MODEL_PREDICTION_PREDECISION
        ),
        planner_predicted_gain_raw_px=gain,
        before_features_complete=True,
    )


def _realization(
    bank: CandidateManifestV7,
    arm: CandidateArmV7,
    *,
    shared_bytes_id: str | None = None,
) -> ArmRealizationReceiptV7:
    bytes_id = shared_bytes_id or arm.candidate_id
    return ArmRealizationReceiptV7(
        manifest_hash=bank.manifest_hash,
        planned_arm_id=arm.planned_arm_id,
        source_hash=bank.source_hash,
        status=ArmRealizationStatusV7.COMPLETE,
        status_reason="complete",
        realized_input_hash=f"input:{bytes_id}",
        realized_output_hash=f"output:{bytes_id}",
        realized_support_hash=f"support:{bytes_id}",
        realized_operator_hash=arm.action_hash,
        tensor_shape=(2, 3, 2),
        tensor_dtype="float32",
        realized_bytes=48,
        offline_bank_acquisition_cost=(
            arm.offline_bank_acquisition_cost_ceiling
        ),
        offline_bank_acquisition_cost_counters=(
            arm.offline_bank_acquisition_cost_counter_ceiling
        ),
        cost_account=CostAccountV7.OFFLINE_BANK_ACQUISITION,
        provenance_hash=f"realization:{arm.candidate_id}",
    )


def _failed_realization(
    bank: CandidateManifestV7,
    arm: CandidateArmV7,
) -> ArmRealizationReceiptV7:
    return ArmRealizationReceiptV7(
        manifest_hash=bank.manifest_hash,
        planned_arm_id=arm.planned_arm_id,
        source_hash=bank.source_hash,
        status=ArmRealizationStatusV7.FAILED,
        status_reason="materialization failed",
        realized_input_hash=None,
        realized_output_hash=None,
        realized_support_hash=None,
        realized_operator_hash=None,
        tensor_shape=(),
        tensor_dtype=None,
        realized_bytes=None,
        offline_bank_acquisition_cost=None,
        offline_bank_acquisition_cost_counters=None,
        cost_account=CostAccountV7.OFFLINE_BANK_ACQUISITION,
        provenance_hash=f"failed-realization:{arm.candidate_id}",
    )


def _alias(
    bank: CandidateManifestV7,
    first: tuple[CandidateArmV7, ArmRealizationReceiptV7],
    second: tuple[CandidateArmV7, ArmRealizationReceiptV7],
    *,
    verification_cost: float = 0.1,
    verification_counters: CostCountersV7 | None = None,
) -> tuple[
    CandidateArmV7,
    CandidateArmV7,
    ArmAliasReceiptV7,
]:
    representative, alias_arm = sorted(
        (first[0], second[0]),
        key=lambda arm: arm.planned_arm_id,
    )
    receipts = {first[0].candidate_id: first[1], second[0].candidate_id: second[1]}
    representative_receipt = receipts[representative.candidate_id]
    alias_receipt = receipts[alias_arm.candidate_id]
    assert representative_receipt.executable_bytes_hash == alias_receipt.executable_bytes_hash
    return representative, alias_arm, ArmAliasReceiptV7(
        manifest_hash=bank.manifest_hash,
        representative_planned_arm_id=representative.planned_arm_id,
        alias_planned_arm_id=alias_arm.planned_arm_id,
        representative_realization_receipt_hash=representative_receipt.receipt_hash,
        alias_realization_receipt_hash=alias_receipt.receipt_hash,
        executable_bytes_hash=str(representative_receipt.executable_bytes_hash),
        realized_operator_hash=str(representative_receipt.realized_operator_hash),
        prospective_runtime_verification_cost=verification_cost,
        prospective_runtime_verification_cost_counters=(
            verification_counters or _verification_counters()
        ),
        cost_account=CostAccountV7.PROSPECTIVE_RUNTIME,
        provenance_hash="alias-provenance",
    )


def _propose(
    bank: CandidateManifestV7,
    families: ActionFamilyManifestV1,
    scores: tuple[PlannerCandidateV7, ...],
    *,
    realizations: tuple[ArmRealizationReceiptV7, ...] | None = None,
    aliases: tuple[ArmAliasReceiptV7, ...] = (),
    top_k: int = 2,
    budget: float = 10.0,
    maximum_counters: CostCountersV7 | None = None,
    cost_penalty: float = 0.0,
):
    if realizations is None:
        realizations = tuple(
            _realization(bank, arm)
            for arm in bank.candidates
            if arm.action_index != 0
        )
    return propose_family_competition_v1(
        bank,
        families,
        scores,
        realizations,
        aliases,
        top_k=top_k,
        maximum_prospective_runtime_cost=budget,
        maximum_prospective_runtime_cost_counters=(
            maximum_counters or _counters(10, wall=budget)
        ),
        cost_penalty_raw_px_per_cost_unit=cost_penalty,
        planner_policy_hash="base-planner-policy",
        planner_provenance_hash="planner-provenance",
    )


def test_byte_alias_cannot_displace_its_canonical_representative() -> None:
    first = _arm(
        1,
        "blur:first",
        strength=0.25,
        action_hash="shared-realized-operator",
    )
    second = _arm(
        2,
        "jpeg:second",
        strength=0.5,
        action_hash="shared-realized-operator",
    )
    bank = _manifest(first, second)
    families = _family_manifest(
        bank,
        {first.candidate_id: "blur", second.candidate_id: "jpeg"},
    )
    first_receipt = _realization(bank, first, shared_bytes_id="same")
    second_receipt = _realization(bank, second, shared_bytes_id="same")
    representative, alias_arm, alias_receipt = _alias(
        bank,
        (first, first_receipt),
        (second, second_receipt),
    )

    result = _propose(
        bank,
        families,
        (
            _score(representative, 1.0),
            _score(alias_arm, 100.0),
        ),
        realizations=(first_receipt, second_receipt),
        aliases=(alias_receipt,),
    )

    assert [row.candidate_id for row in result.proposal.candidates] == [
        representative.candidate_id
    ]
    assert [row.candidate_id for row in result.family_winners] == [
        representative.candidate_id
    ]
    assert alias_arm.candidate_id in result.suppressed_candidate_ids


def test_duplicate_realization_receipts_are_rejected_before_ranking() -> None:
    candidate = _arm(1, "blur:weak", strength=0.25)
    bank = _manifest(candidate)
    families = _family_manifest(bank, {candidate.candidate_id: "blur"})
    receipt = _realization(bank, candidate)

    with pytest.raises(ValueError, match="realizations contain duplicates"):
        _propose(
            bank,
            families,
            (_score(candidate, 1.0),),
            realizations=(receipt, receipt),
        )


def test_foreign_realization_binding_fails_closed() -> None:
    candidate = _arm(1, "blur:weak", strength=0.25)
    bank = _manifest(candidate)
    families = _family_manifest(bank, {candidate.candidate_id: "blur"})
    foreign = replace(
        _realization(bank, candidate),
        manifest_hash="foreign-manifest",
        receipt_hash="",
    )

    with pytest.raises(ValueError, match="receipt and manifest bindings differ"):
        _propose(
            bank,
            families,
            (_score(candidate, 1.0),),
            realizations=(foreign,),
        )


def test_alias_with_forged_byte_identity_is_rejected() -> None:
    first = _arm(
        1, "blur:a", strength=0.25, action_hash="shared-realized-operator"
    )
    second = _arm(
        2, "blur:b", strength=0.5, action_hash="shared-realized-operator"
    )
    bank = _manifest(first, second)
    families = _family_manifest(
        bank,
        {first.candidate_id: "blur", second.candidate_id: "blur"},
    )
    first_receipt = _realization(bank, first, shared_bytes_id="same")
    second_receipt = _realization(bank, second, shared_bytes_id="same")
    _, _, valid_alias = _alias(
        bank,
        (first, first_receipt),
        (second, second_receipt),
    )
    forged_alias = replace(
        valid_alias,
        executable_bytes_hash="forged-executable-bytes",
        receipt_hash="",
    )

    with pytest.raises(ValueError, match="does not prove byte identity"):
        _propose(
            bank,
            families,
            (_score(first, 2.0), _score(second, 1.0)),
            realizations=(first_receipt, second_receipt),
            aliases=(forged_alias,),
        )


def test_duplicate_and_missing_canonical_alias_receipts_are_rejected() -> None:
    first = _arm(
        1, "blur:a", strength=0.25, action_hash="shared-realized-operator"
    )
    second = _arm(
        2, "blur:b", strength=0.5, action_hash="shared-realized-operator"
    )
    bank = _manifest(first, second)
    families = _family_manifest(
        bank,
        {first.candidate_id: "blur", second.candidate_id: "blur"},
    )
    first_receipt = _realization(bank, first, shared_bytes_id="same")
    second_receipt = _realization(bank, second, shared_bytes_id="same")
    _, _, alias_receipt = _alias(
        bank,
        (first, first_receipt),
        (second, second_receipt),
    )
    scores = (_score(first, 2.0), _score(second, 1.0))

    with pytest.raises(ValueError, match="multiple alias receipts"):
        _propose(
            bank,
            families,
            scores,
            realizations=(first_receipt, second_receipt),
            aliases=(alias_receipt, alias_receipt),
        )
    with pytest.raises(ValueError, match="need exact canonical aliases"):
        _propose(
            bank,
            families,
            scores,
            realizations=(first_receipt, second_receipt),
            aliases=(),
        )


def test_family_tie_break_prefers_lower_cost_then_planned_arm_id() -> None:
    expensive = _arm(1, "blur:expensive", strength=0.75, cost=2.0)
    cheap = _arm(2, "blur:cheap", strength=0.25, cost=1.0)
    bank = _manifest(expensive, cheap)
    families = _family_manifest(
        bank,
        {expensive.candidate_id: "blur", cheap.candidate_id: "blur"},
    )

    cheaper_result = _propose(
        bank,
        families,
        (_score(expensive, 4.0), _score(cheap, 4.0)),
    )
    assert cheaper_result.family_winners[0].candidate_id == cheap.candidate_id

    left = _arm(1, "blur:left", strength=0.25)
    right = _arm(2, "blur:right", strength=0.75)
    equal_bank = _manifest(left, right)
    equal_families = _family_manifest(
        equal_bank,
        {left.candidate_id: "blur", right.candidate_id: "blur"},
    )
    expected = min((left, right), key=lambda arm: arm.planned_arm_id)
    forward = _propose(
        equal_bank,
        equal_families,
        (_score(left, 4.0), _score(right, 4.0)),
    )
    reverse = _propose(
        equal_bank,
        equal_families,
        (_score(right, 4.0), _score(left, 4.0)),
    )

    assert forward.family_winners[0].candidate_id == expected.candidate_id
    assert reverse.family_winners == forward.family_winners
    assert reverse.proposal.proposal_hash == forward.proposal.proposal_hash


def test_hard_illegal_arm_is_bound_but_neither_scored_nor_reopened() -> None:
    legal = _arm(1, "blur:legal", strength=0.25)
    illegal = _arm(
        2,
        "jpeg:illegal",
        strength=0.5,
        hard_legal=False,
    )
    bank = _manifest(legal, illegal)
    families = _family_manifest(
        bank,
        {legal.candidate_id: "blur", illegal.candidate_id: "jpeg"},
    )

    result = _propose(bank, families, (_score(legal, 1.0),))
    assert [row.candidate_id for row in result.proposal.candidates] == [
        legal.candidate_id
    ]
    assert families.binding(illegal.candidate_id) is not None

    with pytest.raises(ValueError, match="exactly one score per hard-legal arm"):
        _propose(
            bank,
            families,
            (_score(legal, 1.0), _score(illegal, 100.0)),
        )


def test_binding_and_family_manifest_seals_reject_drift() -> None:
    candidate = _arm(1, "blur:weak", strength=0.25)
    bank = _manifest(candidate)
    row = _binding(candidate, "blur")
    families = _family_manifest(bank, {candidate.candidate_id: "blur"})

    with pytest.raises(ValueError, match="family binding hash drift"):
        replace(row, action_family_id="jpeg")
    with pytest.raises(ValueError, match="family manifest hash drift"):
        replace(families, family_manifest_hash="forged-family-manifest-hash")


def test_family_manifest_rejects_selector_bank_and_planned_arm_drift() -> None:
    candidate = _arm(1, "blur:weak", strength=0.25)
    bank = _manifest(candidate)
    families = _family_manifest(bank, {candidate.candidate_id: "blur"})
    another_bank = replace(
        bank,
        provenance_hash="another-manifest-provenance",
        manifest_hash="",
    )

    with pytest.raises(ValueError, match="bound to another Selector-v7 bank"):
        validate_action_family_manifest_v1(another_bank, families)

    drifted_binding = replace(
        families.bindings[0],
        planned_arm_id="unplanned-arm",
        binding_hash="",
    )
    drifted_families = replace(
        families,
        bindings=(drifted_binding,),
        family_manifest_hash="",
    )
    with pytest.raises(ValueError, match="exact-arm identity drifted"):
        validate_action_family_manifest_v1(bank, drifted_families)


def test_scalar_budget_is_applied_cumulatively_after_family_winners() -> None:
    blur = _arm(1, "blur:weak", strength=0.25, cost=1.0)
    jpeg = _arm(2, "jpeg:medium", strength=0.5, cost=1.0)
    bank = _manifest(blur, jpeg)
    families = _family_manifest(
        bank,
        {blur.candidate_id: "blur", jpeg.candidate_id: "jpeg"},
    )

    result = _propose(
        bank,
        families,
        (_score(blur, 2.0), _score(jpeg, 1.0)),
        budget=1.5,
        maximum_counters=_counters(2, wall=2.0),
    )

    assert len(result.family_winners) == 1
    assert [row.candidate_id for row in result.proposal.candidates] == [
        blur.candidate_id
    ]
    assert result.proposal.reserved_prospective_runtime_cost == pytest.approx(1.0)


def test_counter_budget_is_applied_cumulatively_even_with_scalar_headroom() -> None:
    blur = _arm(1, "blur:weak", strength=0.25, cost=1.0)
    jpeg = _arm(2, "jpeg:medium", strength=0.5, cost=1.0)
    bank = _manifest(blur, jpeg)
    families = _family_manifest(
        bank,
        {blur.candidate_id: "blur", jpeg.candidate_id: "jpeg"},
    )
    one_forward_only = replace(
        _counters(10, wall=10.0),
        candidate_forwards=1,
        matcher_trajectories=1,
    )

    result = _propose(
        bank,
        families,
        (_score(blur, 2.0), _score(jpeg, 1.0)),
        budget=10.0,
        maximum_counters=one_forward_only,
    )

    assert len(result.family_winners) == 1
    assert [row.candidate_id for row in result.proposal.candidates] == [
        blur.candidate_id
    ]
    assert result.proposal.reserved_prospective_runtime_cost_counters.candidate_forwards == 1


def test_counter_cache_state_mismatch_suppresses_candidate() -> None:
    candidate = _arm(1, "blur:weak", strength=0.25)
    bank = _manifest(candidate)
    families = _family_manifest(bank, {candidate.candidate_id: "blur"})
    warm_ceiling = replace(_counters(10, wall=10.0), cache_state="warm")

    result = _propose(
        bank,
        families,
        (_score(candidate, 10.0),),
        maximum_counters=warm_ceiling,
    )

    assert result.family_winners == ()
    assert result.proposal.candidates == ()
    assert result.suppressed_candidate_ids == (candidate.candidate_id,)


def test_alias_verification_cost_is_charged_to_representative_budget() -> None:
    first = _arm(
        1,
        "blur:first",
        strength=0.25,
        cost=0.4,
        action_hash="shared-realized-operator",
    )
    second = _arm(
        2,
        "blur:second",
        strength=0.5,
        cost=0.4,
        action_hash="shared-realized-operator",
    )
    bank = _manifest(first, second)
    families = _family_manifest(
        bank,
        {first.candidate_id: "blur", second.candidate_id: "blur"},
    )
    first_receipt = _realization(bank, first, shared_bytes_id="same")
    second_receipt = _realization(bank, second, shared_bytes_id="same")
    representative, alias_arm, alias_receipt = _alias(
        bank,
        (first, first_receipt),
        (second, second_receipt),
        verification_cost=0.7,
    )

    result = _propose(
        bank,
        families,
        (_score(representative, 1.0), _score(alias_arm, 100.0)),
        realizations=(first_receipt, second_receipt),
        aliases=(alias_receipt,),
        budget=1.0,
        maximum_counters=_counters(10, wall=10.0),
    )

    assert result.family_winners == ()
    assert result.proposal.candidates == ()
    assert set(result.suppressed_candidate_ids) == {
        representative.candidate_id,
        alias_arm.candidate_id,
    }


def test_manifest_canonicalizes_mutable_registry_and_binding_inputs() -> None:
    candidate = _arm(1, "blur:weak", strength=0.25)
    bank = _manifest(candidate)
    member_input = ["blur"]
    definition = FrozenActionFamilyDefinitionV1(
        action_family_id="blur",
        member_action_identities=member_input,  # type: ignore[arg-type]
        registry_source_hash="test-family-registry",
    )
    definition_input = [definition]
    binding_input = [_binding(candidate, "blur")]
    families = ActionFamilyManifestV1(
        selector_manifest_hash=bank.manifest_hash,
        selector_candidate_family_hash=bank.candidate_family_hash,
        frozen_before_outcome=True,
        definitions=definition_input,  # type: ignore[arg-type]
        bindings=binding_input,  # type: ignore[arg-type]
    )
    sealed_hash = families.family_manifest_hash

    member_input[0] = "jpeg"
    definition_input.clear()
    binding_input.clear()

    assert definition.member_action_identities == ("blur",)
    assert isinstance(families.definitions, tuple)
    assert isinstance(families.bindings, tuple)
    assert families.family_manifest_hash == sealed_hash
    validate_action_family_manifest_v1(bank, families)


def test_registry_rejects_family_split_merge_and_wrong_owner() -> None:
    with pytest.raises(ValueError, match="exactly one action identity"):
        FrozenActionFamilyDefinitionV1(
            action_family_id="restoration",
            member_action_identities=("blur", "jpeg"),
            registry_source_hash="test-family-registry",
        )
    with pytest.raises(ValueError, match="must equal its frozen action identity"):
        FrozenActionFamilyDefinitionV1(
            action_family_id="blur-strength-weak",
            member_action_identities=("blur",),
            registry_source_hash="test-family-registry",
        )

    blur = _arm(1, "blur:weak", strength=0.25)
    jpeg = _arm(2, "jpeg:medium", strength=0.5)
    bank = _manifest(blur, jpeg)
    families = _family_manifest(
        bank,
        {blur.candidate_id: "blur", jpeg.candidate_id: "jpeg"},
    )
    wrong_owner = replace(
        families.bindings[0],
        action_family_id="jpeg",
        binding_hash="",
    )
    wrong_families = replace(
        families,
        bindings=(wrong_owner, families.bindings[1]),
        family_manifest_hash="",
    )
    with pytest.raises(ValueError, match="wrong owner"):
        validate_action_family_manifest_v1(bank, wrong_families)


def test_binding_action_identity_drift_is_rejected_after_resealing() -> None:
    blur = _arm(1, "blur:weak", strength=0.25)
    jpeg = _arm(2, "jpeg:medium", strength=0.5)
    bank = _manifest(blur, jpeg)
    families = _family_manifest(
        bank,
        {blur.candidate_id: "blur", jpeg.candidate_id: "jpeg"},
    )
    blur_binding = next(
        row for row in families.bindings if row.candidate_id == blur.candidate_id
    )
    jpeg_binding = next(
        row for row in families.bindings if row.candidate_id == jpeg.candidate_id
    )
    drifted = replace(
        blur_binding,
        action_identity="jpeg",
        binding_hash="",
    )
    drifted_families = replace(
        families,
        bindings=(drifted, jpeg_binding),
        family_manifest_hash="",
    )

    with pytest.raises(ValueError, match="exact-arm identity drifted"):
        validate_action_family_manifest_v1(bank, drifted_families)


def test_joint_enumeration_uses_cheaper_strength_to_admit_second_family() -> None:
    expensive_blur = _arm(1, "blur:expensive", strength=0.75, cost=2.0)
    cheap_blur = _arm(2, "blur:cheap", strength=0.25, cost=1.0)
    jpeg = _arm(3, "jpeg:medium", strength=0.5, cost=1.0)
    bank = _manifest(expensive_blur, cheap_blur, jpeg)
    families = _family_manifest(
        bank,
        {
            expensive_blur.candidate_id: "blur",
            cheap_blur.candidate_id: "blur",
            jpeg.candidate_id: "jpeg",
        },
    )

    result = _propose(
        bank,
        families,
        (
            _score(expensive_blur, 10.0),
            _score(cheap_blur, 8.0),
            _score(jpeg, 8.0),
        ),
        budget=2.0,
        maximum_counters=_counters(2, wall=2.0),
    )

    assert {row.candidate_id for row in result.family_winners} == {
        cheap_blur.candidate_id,
        jpeg.candidate_id,
    }
    assert expensive_blur.candidate_id in result.suppressed_candidate_ids
    assert result.proposal.reserved_prospective_runtime_cost == pytest.approx(2.0)


def test_hard_illegal_and_incomplete_realization_both_fail_closed() -> None:
    complete = _arm(1, "blur:complete", strength=0.25)
    incomplete = _arm(2, "jpeg:failed", strength=0.5)
    illegal = _arm(
        3,
        "noise:illegal",
        strength=0.5,
        hard_legal=False,
    )
    bank = _manifest(complete, incomplete, illegal)
    families = _family_manifest(
        bank,
        {
            complete.candidate_id: "blur",
            incomplete.candidate_id: "jpeg",
            illegal.candidate_id: "noise",
        },
    )

    result = _propose(
        bank,
        families,
        (_score(complete, 1.0), _score(incomplete, 100.0)),
        realizations=(
            _realization(bank, complete),
            _failed_realization(bank, incomplete),
            _realization(bank, illegal),
        ),
    )

    assert [row.candidate_id for row in result.proposal.candidates] == [
        complete.candidate_id
    ]
    assert set(result.suppressed_candidate_ids) == {
        incomplete.candidate_id,
        illegal.candidate_id,
    }


def test_more_than_k_families_selects_best_sealed_distinct_portfolio() -> None:
    blur = _arm(1, "blur:weak", strength=0.25)
    jpeg = _arm(2, "jpeg:medium", strength=0.5)
    noise = _arm(3, "noise:light", strength=0.25)
    bank = _manifest(blur, jpeg, noise)
    families = _family_manifest(
        bank,
        {
            blur.candidate_id: "blur",
            jpeg.candidate_id: "jpeg",
            noise.candidate_id: "noise",
        },
    )

    result = _propose(
        bank,
        families,
        (_score(noise, 1.0), _score(jpeg, 2.0), _score(blur, 3.0)),
        top_k=2,
    )

    assert [row.candidate_id for row in result.proposal.candidates] == [
        blur.candidate_id,
        jpeg.candidate_id,
    ]
    assert result.suppressed_candidate_ids == (noise.candidate_id,)
    assert result.selection_receipt_hash


def test_cost_penalty_is_part_of_exact_control_competition() -> None:
    expensive = _arm(1, "blur:expensive", strength=0.75, cost=2.0)
    cheap = _arm(2, "blur:cheap", strength=0.25, cost=0.5)
    bank = _manifest(expensive, cheap)
    families = _family_manifest(
        bank,
        {expensive.candidate_id: "blur", cheap.candidate_id: "blur"},
    )

    result = _propose(
        bank,
        families,
        (_score(expensive, 3.0), _score(cheap, 2.0)),
        top_k=1,
        cost_penalty=1.0,
    )

    assert [row.candidate_id for row in result.proposal.candidates] == [
        cheap.candidate_id
    ]
    assert result.family_winners[0].conservative_objective == pytest.approx(1.5)
