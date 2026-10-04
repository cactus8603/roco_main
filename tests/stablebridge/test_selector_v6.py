from dataclasses import replace

import numpy as np
import pytest

from stablebridge.physical_repair.action_path_geometry import FIXED_PATH
from stablebridge.physical_repair.action_profiles import ACTION_PHYSICS_PROFILES
from stablebridge.physical_repair.contracts import ActionSpec
from stablebridge.physical_repair.epe_verifier_task_bounds import (
    EPEVerifierCalibration,
    independent_verifier_epe_path_bound,
)
from stablebridge.physical_repair.selector_v6 import (
    ActionHypothesisV6,
    EPETailRiskPoint,
    ExecutedControlV6,
    SelectionAwareEPETailRiskBound,
    SelectorV6Config,
    _required_laws,
    enumerate_certified_control_options_v6,
    physical_receipt_set_rejection_reasons_v6,
    select_action_paths_v6,
)
from stablebridge.physical_repair.action_plan import (
    ActionCandidateFailureV1,
    ActionCandidateResultV1,
    ActionPlanCandidateV1,
    ActionPlanConfigV1,
    action_plan_receipt_v1,
    advance_action_plan_v1,
    initialize_action_plan_v1,
    make_native_checkpoint_v1,
    record_action_candidate_v1,
    record_action_failure_v1,
    typed_law_receipt_set_sha256,
    validate_action_plan_state_v1,
)
from stablebridge.physical_repair.selector_v6_evidence import (
    ActionSupportDecomposition,
    CandidateResponseReceipt,
    RecoverabilitySupportReceipt,
    SequentialProbeReceipt,
    TypedPhysicalLawReceipt,
)
from stablebridge.physical_repair.verifier_task_bounds import support_mask_sha256


ACTION_IDENTITY = "common_gaussian@second"
SOURCE_HASH = "native-source"
POLICY_HASH = "response-risk-visibility-v1"
LATENT_HASH = "gaussian-theta-set"
LATENT_KEYS = ("sigma_interval", "endpoint")
ADMISSION_HASH = "admission-support"


def action() -> ActionSpec:
    return ActionSpec(
        operator_id="common_gaussian",
        operator_version="v6-test",
        domain="image",
        hypothesized_degraded_endpoint="second",
        modified_endpoint="second",
        coordinate_frame="second_native",
    )


def receipt(law_id: str, role: str, *, margin: float = 0.2
            ) -> TypedPhysicalLawReceipt:
    return TypedPhysicalLawReceipt(
        action_identity=ACTION_IDENTITY,
        law_id=law_id,
        role=role,
        null_id=f"null:{law_id}",
        statistic_id=f"stat:{law_id}",
        unit="dimensionless_signed_margin",
        direction="greater_is_better",
        margin_lower=margin,
        certificate_support_hash=ADMISSION_HASH,
        parameter_fit_support_hash=None,
        latent_set_hash=LATENT_HASH,
        evaluated_latent_set_hash=LATENT_HASH,
        latent_parameter_keys=LATENT_KEYS,
        worst_case_member_key="sigma_upper@endpoint_second",
        optimization_receipt_hash=f"interval-proof:{law_id}",
        deterministic=True,
        effective_n=0.0,
        hypotheses_tested=1,
        correction_method="fixed_single",
        familywise_error_rate=0.05,
        assumption_veto_margins={"law_assumption": 0.1},
    )


def all_receipts(*, omit: str | None = None,
                 failed_confound: str | None = None
                 ) -> tuple[TypedPhysicalLawReceipt, ...]:
    profile = ACTION_PHYSICS_PROFILES["common_gaussian"]
    required = set(profile.decision_witnesses) | {
        name for name in profile.closure_witnesses
        if name != "task_signed_utility"
    }
    rows = [
        receipt(name, "closure" if name in profile.closure_witnesses else "invariant")
        for name in sorted(required) if name != omit
    ]
    rows.extend(
        receipt(name, "confound_veto", margin=(-0.1 if name == failed_confound else 0.2))
        for name in sorted(profile.dangerous_confounds) if name != omit
    )
    return tuple(rows)


def support(*, uncertainty: float = 0.5, utility: float = 0.2,
            active_mask: np.ndarray | None = None,
            visibility_mask: np.ndarray | None = None,
            ) -> ActionSupportDecomposition:
    shape = (2, 3)
    full = np.ones(shape, dtype=bool)
    zeros = np.zeros(shape, dtype=bool)
    physical = full if active_mask is None else np.asarray(active_mask, dtype=bool)
    visibility = (
        full if visibility_mask is None
        else np.asarray(visibility_mask, dtype=bool)
    )
    utility_map = np.full(shape, utility, dtype=np.float32)
    eligible = physical & visibility
    delivery = eligible & (utility_map > 0.0)
    return ActionSupportDecomposition(
        source_input_hash=SOURCE_HASH,
        support_policy_hash=POLICY_HASH,
        coordinate_frame="flow_native",
        transport_receipt_hash="transport-proof",
        uncertainty=np.full(shape, uncertainty, dtype=np.float32),
        physical_applicability=physical,
        observable_evidence=full,
        visibility=visibility,
        candidate_response=full,
        candidate_risk_nonincrease=full,
        collision=zeros,
        local_fold=zeros,
        uncovered=zeros,
        influence_support=full,
        candidate_eligible_support=eligible,
        signed_utility_lower=utility_map,
        certified_delivery_support=delivery,
    )


def calibration(radius: float) -> EPEVerifierCalibration:
    return EPEVerifierCalibration(
        verifier_id="native-cycle-v1",
        representation_id="flow-px",
        calibration_version="dense-same-support-v1",
        calibration_data_hash="dense-calibration",
        support_policy_hash=POLICY_HASH,
        mean_epe_radius_upper=radius,
        effective_n=100.0,
        confidence_level=0.95,
        selection_aware=True,
        action_blind=True,
        frozen_before_selection=True,
    )


def control(*, hypothesis_key: str, key: str = "half-input",
            input_strength: float = 0.5, radius: float = 0.05,
            candidate_value: float = 0.5,
            uncertainty: float = 0.5,
            tail_points: tuple[EPETailRiskPoint, ...] | None = None,
            disjoint: tuple[str, ...] = (), overlap: tuple[str, ...] = (),
            interactions: dict[str, float] | None = None,
            expected_compute_seconds: float = 1.0,
            extra_matcher_trajectories: int = 1,
            active_mask: np.ndarray | None = None,
            visibility_mask: np.ndarray | None = None,
            covered_path_keys: tuple[str, ...] | None = None) -> ExecutedControlV6:
    spatial = support(
        uncertainty=uncertainty,
        active_mask=active_mask,
        visibility_mask=visibility_mask,
    )
    native = np.zeros((2, 3, 2), dtype=np.float32)
    verifier = np.ones_like(native)
    candidate = np.full_like(native, candidate_value)
    bound = independent_verifier_epe_path_bound(
        native,
        candidate,
        verifier,
        spatial.certified_delivery_support,
        action_key=hypothesis_key,
        control_key=key,
        control_selection_receipt_hash=f"selection:{key}",
        source_input_hash=SOURCE_HASH,
        verifier_input_hash=SOURCE_HASH,
        task_support_hash=spatial.delivery_support_hash,
        task_support_policy_hash=POLICY_HASH,
        calibration=calibration(radius),
    )
    if tail_points is None:
        tail_points = (
            EPETailRiskPoint(0.5, 0.02, 0.10, 0.01),
            EPETailRiskPoint(1.0, 0.02, 0.10, 0.01),
        )
    own_path = f"{hypothesis_key}/{key}"
    if covered_path_keys is None:
        covered_path_keys = (own_path,)
    tail = SelectionAwareEPETailRiskBound(
        action_identity=ACTION_IDENTITY,
        hypothesis_key=hypothesis_key,
        control_key=key,
        support_policy_hash=POLICY_HASH,
        support_content_hash=spatial.delivery_support_hash,
        calibration_version="scene-max-v1",
        calibration_data_hash="tail-calibration",
        calibration_policy_hash="action-control-support-selected-v1",
        selection_family_hash="selection-family:" + "|".join(sorted(covered_path_keys)),
        covered_path_keys=covered_path_keys,
        selection_method=(
            "fixed_single" if len(covered_path_keys) == 1 else "simultaneous_bound"
        ),
        familywise_error_rate=0.05,
        e_value=1.0,
        effective_n=100.0,
        confidence_level=0.95,
        severe_threshold_px=0.25,
        cvar_level=0.95,
        frozen_before_selection=True,
        points=tail_points,
    )
    sequential = SequentialProbeReceipt(
        policy_hash="fixed-control-policy",
        trace_hash=f"trace:{key}",
        support_policy_hash=POLICY_HASH,
        support_content_hash=spatial.candidate_support_hash,
        method="fixed_once",
        attempted_control_keys=(key,),
        selected_control_key=key,
        maximum_attempts=1,
        alpha_budget=0.05,
        alpha_spent=0.0,
        e_value=1.0,
        simultaneous_family_size=1,
        stopping_reason="fixed_control",
        frozen_before_probes=True,
    )
    return ExecutedControlV6(
        key=key,
        input_strength=input_strength,
        input_path_id=FIXED_PATH,
        control_parameter_hash=f"parameters:{key}",
        repaired_input_hash=f"input:{key}",
        native_output_hash=bound.native_output_hash,
        candidate_output_hash=bound.candidate_output_hash,
        physical_bound_support_hash=ADMISSION_HASH,
        latent_set_hash=LATENT_HASH,
        input_directional_gain_lower=0.5,
        input_curvature_upper=0.1,
        information_retention_lower=0.8,
        support=spatial,
        task_bound=bound,
        tail_risk=tail,
        sequential=sequential,
        expected_compute_seconds=expected_compute_seconds,
        extra_matcher_trajectories=extra_matcher_trajectories,
        disjoint_path_keys=disjoint,
        overlap_path_keys=overlap,
        interaction_epe_upper_px={} if interactions is None else interactions,
    )


def hypothesis(*, suffix: str = "region1",
               receipts: tuple[TypedPhysicalLawReceipt, ...] | None = None,
               controls: tuple[ExecutedControlV6, ...] | None = None,
               **control_kwargs) -> ActionHypothesisV6:
    key = f"{ACTION_IDENTITY}#{suffix}"
    if controls is None:
        controls = (control(hypothesis_key=key, **control_kwargs),)
    return ActionHypothesisV6(
        key=key,
        action=action(),
        region_key=suffix,
        source_input_hash=SOURCE_HASH,
        admission_support_hash=ADMISSION_HASH,
        latent_set_hash=LATENT_HASH,
        latent_parameter_keys=LATENT_KEYS,
        physical_status="supported",
        law_receipts=all_receipts() if receipts is None else receipts,
        controls=controls,
    )


def test_v6_selects_exact_control_and_metric_aligned_beta():
    row = hypothesis()
    decision = select_action_paths_v6((row,))
    assert decision.state == "selected"
    assert decision.selected_controls[row.key] == "half-input"
    assert decision.input_strengths[row.key] == pytest.approx(0.5)
    assert decision.output_trusts[row.key] == pytest.approx(1.0)
    assert decision.certified_epe_gain_lowers[row.key] > 0.0


def test_uncertainty_does_not_choose_or_reject_an_action():
    low = hypothesis(suffix="low", uncertainty=0.0)
    high = hypothesis(suffix="high", uncertainty=1.0)
    low_decision = select_action_paths_v6((low,))
    high_decision = select_action_paths_v6((high,))
    assert low_decision.state == high_decision.state == "selected"
    assert low_decision.output_trusts[low.key] == high_decision.output_trusts[high.key]


def test_missing_or_failed_required_physical_law_fails_closed():
    missing = hypothesis(receipts=all_receipts(omit="heldout_spatial_reblur"))
    failed_receipts = tuple(
        replace(row, margin_lower=-0.1)
        if row.law_id == "heldout_spatial_reblur" else row
        for row in all_receipts()
    )
    failed = hypothesis(suffix="failed", receipts=failed_receipts)
    first = select_action_paths_v6((missing,))
    second = select_action_paths_v6((failed,))
    assert first.state == second.state == "native"
    assert any("missing_typed_law" in item for item in first.rejected_hypotheses[missing.key])
    assert any("typed_law_failed" in item for item in second.rejected_hypotheses[failed.key])


def test_pairwise_endpoint_dominance_is_optional_not_an_admission_gate():
    receipts = tuple(
        row for row in all_receipts()
        if row.law_id != "same_parameter_endpoint"
    )
    row = hypothesis(receipts=receipts)
    decision = select_action_paths_v6((row,))
    assert decision.state == "selected"
    assert row.key not in decision.rejected_hypotheses


def test_motion_requires_own_null_otf_not_family_or_endpoint_dominance():
    laws, confounds = _required_laws("common_motion")
    assert "otf_magnitude" in laws
    assert "directional_otf_competitor" not in laws
    assert "same_parameter_endpoint" not in laws
    assert confounds == {"edge_orientation"}


def test_rival_blur_families_are_not_invalidating_confounds():
    _, disk = _required_laws("common_disk")
    _, gaussian = _required_laws("common_gaussian")
    assert disk == set()
    assert gaussian == set()


def test_warp_error_is_deferred_to_support_localization_not_ignored():
    no_warp_receipts = tuple(
        row for row in all_receipts() if row.law_id != "warp_error"
    )
    observable = hypothesis(receipts=no_warp_receipts)
    assert select_action_paths_v6((observable,)).state == "selected"

    hidden = np.zeros((2, 3), dtype=bool)
    with pytest.raises(ValueError, match="task support is empty"):
        hypothesis(
            suffix="unobservable-warp",
            receipts=no_warp_receipts,
            visibility_mask=hidden,
        )


def test_recoverability_and_unsigned_response_are_separate_exact_intersections():
    physical = np.array([[1, 1, 1], [1, 1, 0]], dtype=bool)
    observable = np.ones((2, 3), dtype=bool)
    visible = np.array([[1, 1, 0], [1, 1, 1]], dtype=bool)
    transport = np.ones((2, 3), dtype=bool)
    ambiguous = np.array([[0, 1, 0], [0, 0, 0]], dtype=bool)
    fold = np.array([[0, 0, 0], [1, 0, 0]], dtype=bool)
    influence = np.ones((2, 3), dtype=bool)
    expected = physical & observable & visible & transport & ~ambiguous & ~fold
    support = RecoverabilitySupportReceipt(
        source_input_hash=SOURCE_HASH,
        action_identity=ACTION_IDENTITY,
        support_policy_hash=POLICY_HASH,
        coordinate_frame="native_flow_lattice",
        transport_receipt_hash="transport",
        visibility_calibration_hash="visibility-calibration",
        influence_policy_id="full_lattice_conservative",
        physical_applicability=physical,
        observable_evidence=observable,
        visibility=visible,
        transport_complete=transport,
        transport_ambiguous=ambiguous,
        local_fold=fold,
        influence_support=influence,
        recoverability_support=expected,
    )
    response = np.array([[1, 0, 1], [1, 1, 1]], dtype=bool)
    receipt = CandidateResponseReceipt(
        source_input_hash=SOURCE_HASH,
        control_key="control",
        native_output_hash="native-output",
        candidate_output_hash="candidate-output",
        response_policy_hash="response-policy",
        recoverability_support_hash=support.recoverability_support_hash,
        recoverability_support=support.recoverability_support,
        candidate_response=response,
        response_on_recoverability=expected & response,
    )
    assert np.array_equal(support.recoverability_support, expected)
    assert np.array_equal(receipt.response_on_recoverability, expected & response)

    with pytest.raises(ValueError, match="expanded or changed"):
        CandidateResponseReceipt(
            source_input_hash=SOURCE_HASH,
            control_key="control",
            native_output_hash="native-output",
            candidate_output_hash="candidate-output",
            response_policy_hash="response-policy",
            recoverability_support_hash=support.recoverability_support_hash,
            recoverability_support=support.recoverability_support,
            candidate_response=response,
            response_on_recoverability=response,
        )


def test_public_physical_prefix_validator_reuses_selector_admission_gates():
    rows = all_receipts()
    assert physical_receipt_set_rejection_reasons_v6(
        operator_id="common_gaussian",
        action_identity=ACTION_IDENTITY,
        admission_support_hash=ADMISSION_HASH,
        latent_set_hash=LATENT_HASH,
        latent_parameter_keys=LATENT_KEYS,
        physical_status="supported",
        law_receipts=rows,
    ) == ()
    incomplete = tuple(row for row in rows if row.law_id != "warp_error")
    assert physical_receipt_set_rejection_reasons_v6(
        operator_id="common_gaussian",
        action_identity=ACTION_IDENTITY,
        admission_support_hash=ADMISSION_HASH,
        latent_set_hash=LATENT_HASH,
        latent_parameter_keys=LATENT_KEYS,
        physical_status="supported",
        law_receipts=incomplete,
    ) == ()


def test_public_physical_prefix_validator_rejects_cross_bound_receipts():
    rows = all_receipts()
    with pytest.raises(ValueError, match="admission supports differ"):
        physical_receipt_set_rejection_reasons_v6(
            operator_id="common_gaussian",
            action_identity=ACTION_IDENTITY,
            admission_support_hash="another-support",
            latent_set_hash=LATENT_HASH,
            latent_parameter_keys=LATENT_KEYS,
            physical_status="supported",
            law_receipts=rows,
        )


def test_small_harm_is_allowed_but_mean_harm_budget_is_enforced():
    safe = hypothesis(tail_points=(EPETailRiskPoint(1.0, 0.04, 0.10, 0.01),))
    unsafe = hypothesis(
        suffix="unsafe",
        tail_points=(EPETailRiskPoint(1.0, 0.051, 0.10, 0.01),),
    )
    assert select_action_paths_v6((safe,)).state == "selected"
    rejected = select_action_paths_v6((unsafe,))
    assert rejected.state == "native"
    assert any(
        "mean_harm_budget_exceeded" in item
        for item in rejected.rejected_controls[f"{unsafe.key}/half-input"]
    )


def test_severe_probability_and_cvar_are_independent_hard_vetoes():
    severe = hypothesis(
        tail_points=(EPETailRiskPoint(1.0, 0.01, 0.10, 0.051),),
    )
    cvar = hypothesis(
        suffix="cvar", tail_points=(EPETailRiskPoint(1.0, 0.01, 0.251, 0.01),),
    )
    severe_decision = select_action_paths_v6((severe,))
    cvar_decision = select_action_paths_v6((cvar,))
    assert severe_decision.state == cvar_decision.state == "native"
    assert any("severe_probability" in value for value in
               severe_decision.rejected_controls[f"{severe.key}/half-input"])
    assert any("cvar_harm" in value for value in
               cvar_decision.rejected_controls[f"{cvar.key}/half-input"])


def test_large_same_support_verifier_radius_naturally_returns_native():
    row = hypothesis(radius=1.0)
    decision = select_action_paths_v6((row,))
    assert decision.state == "native"
    assert any(
        "nonpositive_epe_utility" in item
        for item in decision.rejected_controls[f"{row.key}/half-input"]
    )


def test_tail_support_or_control_identity_cannot_be_reused():
    key = f"{ACTION_IDENTITY}#region1"
    good = control(hypothesis_key=key)
    with pytest.raises(ValueError, match="tail calibration and delivery support differ"):
        replace(good, tail_risk=replace(good.tail_risk, support_content_hash="other"))
    other_tail = replace(
        good.tail_risk,
        control_key="other",
        covered_path_keys=(f"{key}/other",),
        selection_family_hash="selection-family:other",
    )
    with pytest.raises(ValueError, match="another exact control"):
        replace(good, tail_risk=other_tail)


def test_only_exact_precalibrated_beta_points_are_considered():
    row = hypothesis(
        tail_points=(EPETailRiskPoint(0.7, 0.01, 0.10, 0.01),),
    )
    decision = select_action_paths_v6((row,))
    assert decision.state == "selected"
    assert decision.output_trusts[row.key] == pytest.approx(0.7)


def test_unknown_cross_region_relation_cannot_be_silently_composed():
    left_key = f"{ACTION_IDENTITY}#left"
    right_key = f"{ACTION_IDENTITY}#right"
    family = (f"{left_key}/half-input", f"{right_key}/half-input")
    left = hypothesis(
        suffix="left",
        controls=(control(hypothesis_key=left_key, covered_path_keys=family),),
    )
    right = hypothesis(
        suffix="right",
        controls=(control(hypothesis_key=right_key, covered_path_keys=family),),
    )
    decision = select_action_paths_v6((left, right))
    assert decision.state == "selected"
    assert len(decision.selected_hypotheses) == 1


def test_tail_calibration_must_cover_the_full_predeclared_choice_family():
    left = hypothesis(suffix="left")
    right = hypothesis(suffix="right")
    decision = select_action_paths_v6((left, right))
    assert decision.state == "native"
    for row in (left, right):
        assert "tail_calibration_selection_family_incomplete" in (
            decision.rejected_controls[f"{row.key}/half-input"]
        )


def test_tail_threshold_and_cvar_estimands_cannot_drift_from_protocol():
    key = f"{ACTION_IDENTITY}#threshold"
    base = control(hypothesis_key=key)
    changed_tail = replace(base.tail_risk, severe_threshold_px=0.5)
    changed = replace(base, tail_risk=changed_tail)
    row = hypothesis(suffix="threshold", controls=(changed,))
    decision = select_action_paths_v6((row,))
    assert decision.state == "native"
    assert "tail_severe_threshold_mismatch" in (
        decision.rejected_controls[f"{row.key}/half-input"]
    )


def test_symmetric_disjoint_proof_allows_two_regions_to_compose():
    left_key = f"{ACTION_IDENTITY}#left"
    right_key = f"{ACTION_IDENTITY}#right"
    left_path = f"{left_key}/half-input"
    right_path = f"{right_key}/half-input"
    family = (left_path, right_path)
    left_mask = np.zeros((2, 3), dtype=bool); left_mask[:, :1] = True
    right_mask = np.zeros((2, 3), dtype=bool); right_mask[:, 1:] = True
    left_control = control(
        hypothesis_key=left_key, disjoint=(right_path,), active_mask=left_mask,
        covered_path_keys=family,
    )
    right_control = control(
        hypothesis_key=right_key, disjoint=(left_path,), active_mask=right_mask,
        covered_path_keys=family,
    )
    left = hypothesis(suffix="left", controls=(left_control,))
    right = hypothesis(suffix="right", controls=(right_control,))
    decision = select_action_paths_v6(
        (left, right),
        config=SelectorV6Config(
            maximum_total_delivery_fraction=1.0,
            maximum_matcher_trajectories=2,
            maximum_compute_seconds=10.0,
        ),
    )
    assert len(decision.selected_hypotheses) == 2


def test_symmetric_measured_overlap_allows_one_pair_to_compose():
    left_key = f"{ACTION_IDENTITY}#left-overlap"
    right_key = f"{ACTION_IDENTITY}#right-overlap"
    left_path = f"{left_key}/half-input"
    right_path = f"{right_key}/half-input"
    family = (left_path, right_path)
    shared = np.zeros((2, 3), dtype=bool); shared[0, 0] = True
    left_control = control(
        hypothesis_key=left_key, overlap=(right_path,),
        interactions={right_path: 0.001}, active_mask=shared,
        covered_path_keys=family,
    )
    right_control = control(
        hypothesis_key=right_key, overlap=(left_path,),
        interactions={left_path: 0.001}, active_mask=shared,
        covered_path_keys=family,
    )
    left = hypothesis(suffix="left-overlap", controls=(left_control,))
    right = hypothesis(suffix="right-overlap", controls=(right_control,))
    decision = select_action_paths_v6(
        (left, right),
        config=SelectorV6Config(
            maximum_total_delivery_fraction=1.0,
            maximum_matcher_trajectories=2,
            maximum_compute_seconds=10.0,
        ),
    )
    assert len(decision.selected_hypotheses) == 2
    assert decision.total_interaction_upper_px == pytest.approx(0.001)


def test_pairwise_bounds_cannot_authorize_three_way_overlap():
    suffixes = ("overlap-a", "overlap-b", "overlap-c")
    keys = tuple(f"{ACTION_IDENTITY}#{suffix}" for suffix in suffixes)
    paths = tuple(f"{key}/half-input" for key in keys)
    shared = np.zeros((2, 3), dtype=bool); shared[0, 0] = True
    rows = []
    for suffix, key, own_path in zip(suffixes, keys, paths):
        other_paths = tuple(path for path in paths if path != own_path)
        exact = control(
            hypothesis_key=key, overlap=other_paths,
            interactions={path: 0.001 for path in other_paths},
            active_mask=shared, covered_path_keys=paths,
        )
        rows.append(hypothesis(suffix=suffix, controls=(exact,)))
    decision = select_action_paths_v6(
        tuple(rows),
        config=SelectorV6Config(
            maximum_total_delivery_fraction=1.0,
            maximum_matcher_trajectories=3,
            maximum_compute_seconds=10.0,
        ),
    )
    assert decision.state == "selected"
    assert len(decision.selected_hypotheses) == 2


def test_cost_budget_can_force_native_without_changing_physical_evidence():
    row = hypothesis(expected_compute_seconds=2.0)
    decision = select_action_paths_v6(
        (row,), config=SelectorV6Config(maximum_compute_seconds=1.0),
    )
    assert decision.state == "native"
    assert decision.stop_reason == "no_positive_budget_feasible_composition"


def _planned_candidate(root, row: ActionHypothesisV6, *, mandatory: bool,
                       rank: int, information: float = 0.1,
                       max_compute: float = 1.0,
                       max_trajectories: int = 1) -> ActionPlanCandidateV1:
    exact = row.controls[0]
    endpoint = row.action.hypothesized_degraded_endpoint
    return ActionPlanCandidateV1(
        hypothesis_key=row.key,
        control_key=exact.key,
        action_identity=f"{row.action.operator_id}@{endpoint}",
        region_key=row.region_key,
        source_checkpoint_id=root.checkpoint_id,
        source_input_hash=row.source_input_hash,
        admission_support_hash=row.admission_support_hash,
        latent_set_hash=row.latent_set_hash,
        law_receipt_set_sha256=typed_law_receipt_set_sha256(row.law_receipts),
        support_policy_hash=exact.support.support_policy_hash,
        input_path_id=exact.input_path_id,
        control_parameter_hash=exact.control_parameter_hash,
        expected_information_gain_lower=information,
        maximum_compute_seconds=max_compute,
        maximum_matcher_trajectories=max_trajectories,
        probe_rank=rank,
        mandatory=mandatory,
    )


def _plan_rows(*, values: tuple[float, ...], mandatory_count: int,
               complete_tail_family: bool = True,
               tail_points: tuple[EPETailRiskPoint, ...] | None = None):
    key = f"{ACTION_IDENTITY}#plan-region"
    control_keys = tuple(f"probe-{index}" for index in range(len(values)))
    paths = tuple(f"{key}/{control_key}" for control_key in control_keys)
    controls = []
    rows = []
    for index, (control_key, value) in enumerate(zip(control_keys, values)):
        covered = paths if complete_tail_family else (paths[index],)
        exact = control(
            hypothesis_key=key,
            key=control_key,
            candidate_value=value,
            covered_path_keys=covered,
            tail_points=tail_points,
        )
        controls.append(exact)
        rows.append(hypothesis(suffix="plan-region", controls=(exact,)))
    root = make_native_checkpoint_v1(
        source_input_hash=SOURCE_HASH,
        native_output_hash=controls[0].native_output_hash,
    )
    proposals = tuple(
        _planned_candidate(
            root, row,
            mandatory=index < mandatory_count,
            rank=index,
            information=1.0 / (index + 1),
        )
        for index, row in enumerate(rows)
    )
    return root, tuple(rows), proposals


def _dispatch_and_record(state, row):
    state, step = advance_action_plan_v1(state)
    assert step.kind == "repair_candidate"
    assert step.proposal_path_key == f"{row.key}/{row.controls[0].key}"
    state = record_action_candidate_v1(
        state,
        ActionCandidateResultV1(
            proposal_path_key=step.proposal_path_key,
            source_checkpoint_id=step.source_checkpoint_id,
            hypothesis=row,
        ),
    )
    return state


def test_public_v6_enumeration_checks_larger_predeclared_family():
    row = hypothesis(suffix="enumeration")
    own = f"{row.key}/half-input"
    declared = frozenset({own, f"{row.key}/unexecuted-midpoint"})
    result = enumerate_certified_control_options_v6(
        (row,), predeclared_path_keys=declared,
    )
    assert result.options == ()
    assert "tail_calibration_selection_family_incomplete" in (
        result.rejected_controls[own]
    )


def test_action_plan_executes_mandatory_controls_rolls_back_and_commits_best():
    root, rows, proposals = _plan_rows(
        values=(0.3, 0.9), mandatory_count=2,
    )
    state = initialize_action_plan_v1(root, proposals)
    state = _dispatch_and_record(state, rows[0])
    assert state.current_checkpoint == root
    state = _dispatch_and_record(state, rows[1])
    assert state.current_checkpoint == root
    state, step = advance_action_plan_v1(state)
    assert step.kind == "stop_prefix"
    assert state.state == "stop_prefix"
    assert state.selected_path_key == proposals[1].path_key
    assert state.current_checkpoint.parent_checkpoint_id == root.checkpoint_id
    assert sum(event.kind == "rollback" for event in state.history) == 2
    assert sum(event.kind == "commit" for event in state.history) == 1


def test_action_plan_replans_to_optional_midpoint_only_when_intervals_overlap():
    root, rows, proposals = _plan_rows(
        values=(0.50, 0.52, 0.90), mandatory_count=2,
    )
    state = initialize_action_plan_v1(root, proposals)
    state = _dispatch_and_record(state, rows[0])
    state = _dispatch_and_record(state, rows[1])
    state, step = advance_action_plan_v1(state)
    assert step.kind == "repair_candidate"
    assert step.proposal_path_key == proposals[2].path_key
    assert any(event.kind == "replan" for event in state.history)
    state = record_action_candidate_v1(
        state,
        ActionCandidateResultV1(
            proposal_path_key=step.proposal_path_key,
            source_checkpoint_id=step.source_checkpoint_id,
            hypothesis=rows[2],
        ),
    )
    state, step = advance_action_plan_v1(state)
    assert step.kind == "stop_prefix"
    assert state.selected_path_key == proposals[2].path_key
    receipt_value = action_plan_receipt_v1(state)
    assert receipt_value["rollback_count"] == 3
    assert receipt_value["replan_count"] == 1
    assert receipt_value["commit_count"] == 1
    assert receipt_value["GT_read"] is receipt_value["H2_read"] is False


def test_action_plan_failed_candidate_rolls_back_then_replans_safe_alternative():
    root, rows, proposals = _plan_rows(
        values=(0.4, 0.9), mandatory_count=1,
    )
    state = initialize_action_plan_v1(root, proposals)
    state, first = advance_action_plan_v1(state)
    state = record_action_failure_v1(
        state,
        ActionCandidateFailureV1(
            proposal_path_key=first.proposal_path_key,
            source_checkpoint_id=first.source_checkpoint_id,
            reason="postcheck_failed",
            spent_compute_seconds=1.0,
            spent_matcher_trajectories=1,
        ),
    )
    state, second = advance_action_plan_v1(state)
    assert second.proposal_path_key == proposals[1].path_key
    state = record_action_candidate_v1(
        state,
        ActionCandidateResultV1(
            proposal_path_key=second.proposal_path_key,
            source_checkpoint_id=second.source_checkpoint_id,
            hypothesis=rows[1],
        ),
    )
    state, final = advance_action_plan_v1(state)
    assert final.kind == "stop_prefix"
    assert state.selected_path_key == proposals[1].path_key


def test_action_plan_incomplete_choice_family_calibration_falls_back_native():
    root, rows, proposals = _plan_rows(
        values=(0.5, 0.9), mandatory_count=2,
        complete_tail_family=False,
    )
    state = initialize_action_plan_v1(root, proposals)
    state = _dispatch_and_record(state, rows[0])
    state = _dispatch_and_record(state, rows[1])
    state, step = advance_action_plan_v1(state)
    assert step.kind == "stop_native"
    assert state.current_checkpoint == root
    assert state.terminal_reason == "no_candidate_passed_complete_certificate_stack"


def test_action_plan_tail_harm_veto_rolls_back_to_bit_exact_native():
    unsafe_tail = (EPETailRiskPoint(1.0, 0.051, 0.10, 0.01),)
    root, rows, proposals = _plan_rows(
        values=(0.9,), mandatory_count=1, tail_points=unsafe_tail,
    )
    state = initialize_action_plan_v1(root, proposals)
    state = _dispatch_and_record(state, rows[0])
    state, step = advance_action_plan_v1(state)
    assert step.kind == "stop_native"
    assert state.current_checkpoint.checkpoint_id == root.checkpoint_id
    assert action_plan_receipt_v1(state)["selected_path_key"] is None


def test_action_plan_checkpoint_binds_exact_output_trust_not_raw_candidate_only():
    half_tail = (EPETailRiskPoint(0.5, 0.02, 0.10, 0.01),)
    root, rows, proposals = _plan_rows(
        values=(0.9,), mandatory_count=1, tail_points=half_tail,
    )
    state = initialize_action_plan_v1(root, proposals)
    state = _dispatch_and_record(state, rows[0])
    state, step = advance_action_plan_v1(state)
    assert step.kind == "stop_prefix"
    assert state.current_checkpoint.output_trust == pytest.approx(0.5)
    assert state.current_checkpoint.current_output_hash == rows[0].controls[
        0].candidate_output_hash
    assert state.current_checkpoint.delivery_commitment_sha256 != (
        state.current_checkpoint.current_output_hash
    )
    receipt_value = action_plan_receipt_v1(state)
    assert receipt_value["final_output_trust"] == pytest.approx(0.5)
    assert receipt_value["final_delivery_support_hash"] == rows[0].controls[
        0].support.delivery_support_hash


def test_action_plan_budget_blocks_optional_probe_instead_of_overrunning():
    root, rows, proposals = _plan_rows(
        values=(0.50, 0.52, 0.90), mandatory_count=2,
    )
    config = ActionPlanConfigV1(maximum_matcher_trajectories=2)
    state = initialize_action_plan_v1(root, proposals, config=config)
    state = _dispatch_and_record(state, rows[0])
    state = _dispatch_and_record(state, rows[1])
    state, step = advance_action_plan_v1(state)
    assert step.kind == "stop_native"
    assert state.terminal_reason == "optional_probe_exceeds_remaining_budget"
    assert state.spent_matcher_trajectories == 2


def test_action_plan_rejects_wrong_checkpoint_result_without_state_mutation():
    root, rows, proposals = _plan_rows(values=(0.9,), mandatory_count=1)
    state = initialize_action_plan_v1(root, proposals)
    dispatched, step = advance_action_plan_v1(state)
    with pytest.raises(ValueError, match="frozen proposal"):
        record_action_candidate_v1(
            dispatched,
            ActionCandidateResultV1(
                proposal_path_key=step.proposal_path_key,
                source_checkpoint_id="wrong-checkpoint",
                hypothesis=rows[0],
            ),
        )
    assert state.current_checkpoint == root
    assert state.results == ()


def test_action_plan_content_hash_and_history_are_fail_closed():
    root, _, proposals = _plan_rows(values=(0.9,), mandatory_count=1)
    state = initialize_action_plan_v1(root, proposals)
    with pytest.raises(ValueError, match="identity drift"):
        validate_action_plan_state_v1(replace(state, plan_id="fabricated"))
    bad_event = replace(state.history[0], ordinal=9)
    with pytest.raises(ValueError, match="ordinal drift"):
        validate_action_plan_state_v1(replace(state, history=(bad_event,)))
