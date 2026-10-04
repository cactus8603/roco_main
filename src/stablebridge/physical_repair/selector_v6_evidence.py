"""Selector-v6 evidence contracts for physical laws and regional delivery.

These objects close three gaps left by the exact-control v5 selector:

* a positive scalar with a witness name is not a physical proof;
* adaptive ``r1/r2/mid`` search needs an auditable stopping receipt; and
* uncertainty, physical applicability, observable evidence, candidate
  eligibility, and certified signed utility are different spatial objects.

The module deliberately does not select an action.  It makes the evidence
that a future selector is allowed to consume explicit and fail-closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Mapping

import numpy as np

from .action_path_geometry import OPERATOR_ALLOWED_PATHS
from .action_profiles import ACTION_PHYSICS_PROFILES
from .verifier_task_bounds import support_mask_sha256


LAW_ROLES = frozenset({"invariant", "closure", "confound_veto"})
LAW_DIRECTIONS = frozenset({"greater_is_better", "less_is_better", "inside_set"})
CORRECTION_METHODS = frozenset({
    "fixed_single", "bonferroni", "holm", "simultaneous_bound", "e_process",
})
SEQUENTIAL_METHODS = frozenset({"fixed_once", "simultaneous_bound", "e_process"})
ALGEBRAIC_STRUCTURES = frozenset({
    "order_statistic",
    "linear_spectral_contraction",
    "convolution_semigroup",
    "normalized_convolution_nonsemigroup",
    "anisotropic_convolution_nonsemigroup",
    "codec_feasible_projection",
    "polyphase_sampling",
    "monotone_order_projection",
})
INFORMATION_ORDERS = frozenset({
    "locally_destructive", "contractive", "attenuating", "projective",
    "irreversibly_aliased", "conditionally_invariant",
})


def _finite(value: float, name: str) -> float:
    item = float(value)
    if not np.isfinite(item):
        raise ValueError(f"{name} must be finite")
    return item


def _nonnegative(value: float, name: str) -> float:
    item = _finite(value, name)
    if item < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return item


def _fraction(value: float, name: str) -> float:
    item = _finite(value, name)
    if not 0.0 <= item <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return item


def _float_map(value: np.ndarray, name: str) -> np.ndarray:
    item = np.ascontiguousarray(value, dtype=np.float32)
    if item.ndim != 2 or not np.isfinite(item).all():
        raise ValueError(f"{name} must be a finite HxW map")
    item.setflags(write=False)
    return item


def _bool_map(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    item = np.asarray(value)
    if item.dtype != np.bool_ or item.shape != shape:
        raise ValueError(f"{name} must be an aligned boolean map")
    item = np.ascontiguousarray(item)
    item.setflags(write=False)
    return item


def _mapping_hash(value: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for key, item in sorted(value.items()):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class ActionOperatorSignature:
    """Static operator algebra; descriptive and never delivery authority."""

    operator_id: str
    forward_model_id: str
    parameter_coordinate: str
    algebraic_structure: str
    information_order: str
    discrepancy_law_id: str
    influence_geometry: str
    allowed_path_ids: frozenset[str]

    def __post_init__(self) -> None:
        values = (
            self.operator_id, self.forward_model_id, self.parameter_coordinate,
            self.discrepancy_law_id, self.influence_geometry,
        )
        if any(not value for value in values):
            raise ValueError("operator signature must be fully named")
        if self.algebraic_structure not in ALGEBRAIC_STRUCTURES:
            raise ValueError("unknown action algebra")
        if self.information_order not in INFORMATION_ORDERS:
            raise ValueError("unknown information order")
        if not self.allowed_path_ids:
            raise ValueError("operator signature needs an executable path")


ACTION_OPERATOR_SIGNATURES: Mapping[str, ActionOperatorSignature] = {
    "impulse_exact_median3": ActionOperatorSignature(
        "impulse_exact_median3", "sparse_impulse_observation_v1", "fixed",
        "order_statistic", "locally_destructive", "leave_one_out_median_risk",
        "sparse_isotropic_local", OPERATOR_ALLOWED_PATHS["impulse_exact_median3"],
    ),
    "impulse_median3": ActionOperatorSignature(
        "impulse_median3", "sparse_impulse_observation_v1", "fixed",
        "order_statistic", "locally_destructive", "leave_one_out_median_risk",
        "sparse_isotropic_local", OPERATOR_ALLOWED_PATHS["impulse_median3"],
    ),
    "wiener3": ActionOperatorSignature(
        "wiener3", "additive_noise_psd_v1", "noise_psd",
        "linear_spectral_contraction", "contractive", "j_invariant_or_sure_risk",
        "regional_isotropic_local", OPERATOR_ALLOWED_PATHS["wiener3"],
    ),
    "common_gaussian": ActionOperatorSignature(
        "common_gaussian", "gaussian_psf_v1", "variance_sigma_squared",
        "convolution_semigroup", "attenuating", "repair_reblur_closure",
        "isotropic_radius", OPERATOR_ALLOWED_PATHS["common_gaussian"],
    ),
    "common_disk": ActionOperatorSignature(
        "common_disk", "disk_psf_v1", "disk_radius",
        "normalized_convolution_nonsemigroup", "attenuating",
        "repair_reblur_closure", "isotropic_radius_with_ringing",
        OPERATOR_ALLOWED_PATHS["common_disk"],
    ),
    "common_motion": ActionOperatorSignature(
        "common_motion", "line_motion_psf_v1", "joint_length_angle",
        "anisotropic_convolution_nonsemigroup", "attenuating",
        "repair_reblur_closure", "anisotropic_oriented_segment",
        OPERATOR_ALLOWED_PATHS["common_motion"],
    ),
    "jpeg_deblock": ActionOperatorSignature(
        "jpeg_deblock", "jpeg_quantization_codec_v1", "quality_phase_strength",
        "codec_feasible_projection", "projective", "codec_reencode_fixed_point",
        "block_grid_and_chroma_phase", OPERATOR_ALLOWED_PATHS["jpeg_deblock"],
    ),
    "pixelate": ActionOperatorSignature(
        "pixelate", "polyphase_sampling_v1", "factor_phase",
        "polyphase_sampling", "irreversibly_aliased", "sampling_fixed_point",
        "sampling_lattice", OPERATOR_ALLOWED_PATHS["pixelate"],
    ),
    "rank3_pair": ActionOperatorSignature(
        "rank3_pair", "positive_monotone_radiometry_v1", "monotone_map",
        "monotone_order_projection", "conditionally_invariant",
        "rank_radiometry_warp_commutator", "local_rank_neighbourhood",
        OPERATOR_ALLOWED_PATHS["rank3_pair"],
    ),
}


def validate_action_operator_signatures() -> None:
    profile_keys = set(ACTION_PHYSICS_PROFILES)
    signature_keys = set(ACTION_OPERATOR_SIGNATURES)
    if profile_keys != signature_keys:
        raise RuntimeError(
            "operator-signature coverage drift: "
            f"missing={sorted(profile_keys - signature_keys)} "
            f"extra={sorted(signature_keys - profile_keys)}"
        )
    for key, signature in ACTION_OPERATOR_SIGNATURES.items():
        if key != signature.operator_id:
            raise RuntimeError(f"operator-signature identity drift: {key}")
        if signature.allowed_path_ids != OPERATOR_ALLOWED_PATHS[key]:
            raise RuntimeError(f"operator path coverage drift: {key}")


validate_action_operator_signatures()


@dataclass(frozen=True)
class TypedPhysicalLawReceipt:
    """One signed law with units, null, split, multiplicity, and theta coverage."""

    action_identity: str
    law_id: str
    role: str
    null_id: str
    statistic_id: str
    unit: str
    direction: str
    margin_lower: float
    certificate_support_hash: str
    parameter_fit_support_hash: str | None
    latent_set_hash: str
    evaluated_latent_set_hash: str
    latent_parameter_keys: tuple[str, ...]
    worst_case_member_key: str
    optimization_receipt_hash: str
    deterministic: bool
    effective_n: float
    hypotheses_tested: int
    correction_method: str
    familywise_error_rate: float
    assumption_veto_margins: Mapping[str, float] = field(default_factory=dict)
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        values = (
            self.action_identity, self.law_id, self.null_id, self.statistic_id,
            self.unit, self.certificate_support_hash, self.latent_set_hash,
            self.evaluated_latent_set_hash, self.worst_case_member_key,
            self.optimization_receipt_hash,
        )
        if any(not value for value in values):
            raise ValueError("typed physical law identity is incomplete")
        if self.role not in LAW_ROLES:
            raise ValueError("unknown physical law role")
        if self.direction not in LAW_DIRECTIONS:
            raise ValueError("unknown physical law direction")
        _finite(self.margin_lower, "physical law margin")
        _nonnegative(self.effective_n, "physical law effective n")
        if (not self.latent_parameter_keys
                or len(set(self.latent_parameter_keys))
                != len(self.latent_parameter_keys)):
            raise ValueError("physical law needs unique latent parameter keys")
        if self.latent_set_hash != self.evaluated_latent_set_hash:
            raise ValueError("physical law did not evaluate the complete latent set")
        if self.hypotheses_tested < 1:
            raise ValueError("physical law needs a hypothesis family size")
        if self.correction_method not in CORRECTION_METHODS:
            raise ValueError("unknown multiplicity correction")
        alpha = _fraction(self.familywise_error_rate, "familywise error rate")
        if alpha <= 0.0:
            raise ValueError("familywise error rate must be positive")
        if self.hypotheses_tested > 1 and self.correction_method == "fixed_single":
            raise ValueError("multiple hypotheses lack simultaneous correction")
        if self.deterministic:
            if self.parameter_fit_support_hash is not None:
                raise ValueError("deterministic law cannot claim a statistical fit fold")
        else:
            if not self.parameter_fit_support_hash:
                raise ValueError("statistical law needs a parameter-fit support")
            if self.parameter_fit_support_hash == self.certificate_support_hash:
                raise ValueError("fit and certificate supports must differ")
            if self.effective_n <= 0.0:
                raise ValueError("statistical law needs positive effective n")
        if any(not key for key in self.assumption_veto_margins):
            raise ValueError("assumption-veto names must be nonempty")
        for margin in self.assumption_veto_margins.values():
            _finite(margin, "assumption-veto margin")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("physical law cannot read labels, truth, or outcomes")

    @property
    def assumption_receipt_hash(self) -> str:
        return _mapping_hash({
            key: np.float64(value).hex()
            for key, value in self.assumption_veto_margins.items()
        })

    @property
    def passed(self) -> bool:
        return bool(
            self.margin_lower > 0.0
            and all(value > 0.0 for value in self.assumption_veto_margins.values())
        )


@dataclass(frozen=True)
class SequentialProbeReceipt:
    """Proof that an adaptive exact-control search respected its frozen rule."""

    policy_hash: str
    trace_hash: str
    support_policy_hash: str
    support_content_hash: str
    method: str
    attempted_control_keys: tuple[str, ...]
    selected_control_key: str
    maximum_attempts: int
    alpha_budget: float
    alpha_spent: float
    e_value: float
    simultaneous_family_size: int
    stopping_reason: str
    frozen_before_probes: bool
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        identities = (
            self.policy_hash, self.trace_hash, self.support_policy_hash,
            self.support_content_hash, self.selected_control_key,
            self.stopping_reason,
        )
        if any(not value for value in identities):
            raise ValueError("sequential probe provenance is incomplete")
        if self.method not in SEQUENTIAL_METHODS:
            raise ValueError("unknown sequential evidence method")
        if (not self.attempted_control_keys
                or len(set(self.attempted_control_keys))
                != len(self.attempted_control_keys)):
            raise ValueError("attempted controls must be nonempty and unique")
        if self.selected_control_key not in self.attempted_control_keys:
            raise ValueError("selected control was not actually attempted")
        attempts = len(self.attempted_control_keys)
        if self.maximum_attempts < attempts or self.maximum_attempts < 1:
            raise ValueError("attempt count exceeds the frozen probe budget")
        alpha_budget = _fraction(self.alpha_budget, "alpha budget")
        alpha_spent = _fraction(self.alpha_spent, "alpha spent")
        _nonnegative(self.e_value, "e-value")
        if not 0.0 < alpha_budget < 1.0:
            raise ValueError("alpha budget must lie in (0,1)")
        if alpha_spent > alpha_budget:
            raise ValueError("adaptive probes overspent the alpha budget")
        if not self.frozen_before_probes:
            raise ValueError("probe policy was not frozen before observing responses")
        if self.method == "fixed_once":
            if attempts != 1 or self.maximum_attempts != 1 or alpha_spent != 0.0:
                raise ValueError("fixed-once receipt cannot contain adaptive probes")
        elif self.method == "simultaneous_bound":
            if self.simultaneous_family_size < attempts:
                raise ValueError("simultaneous bound does not cover all attempted controls")
        elif self.e_value < 1.0 / alpha_budget:
            raise ValueError("e-process did not cross its frozen evidence threshold")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("probe selection cannot read labels, truth, or outcomes")


@dataclass(frozen=True)
class RecoverabilitySupportReceipt:
    """Physical/geometry support before observing candidate task utility.

    This stage deliberately stops before candidate response, matcher risk, and
    signed improvement.  ``transport_ambiguous`` is a conservative boundary
    veto, not a claim that splat density is physical occlusion.  A global
    matcher may use a full-frame ``influence_support``; that is honest but does
    not manufacture localization.
    """

    source_input_hash: str
    action_identity: str
    support_policy_hash: str
    coordinate_frame: str
    transport_receipt_hash: str
    visibility_calibration_hash: str
    influence_policy_id: str
    physical_applicability: np.ndarray
    observable_evidence: np.ndarray
    visibility: np.ndarray
    transport_complete: np.ndarray
    transport_ambiguous: np.ndarray
    local_fold: np.ndarray
    influence_support: np.ndarray
    recoverability_support: np.ndarray

    def __post_init__(self) -> None:
        identities = (
            self.source_input_hash, self.action_identity,
            self.support_policy_hash, self.coordinate_frame,
            self.transport_receipt_hash, self.visibility_calibration_hash,
            self.influence_policy_id,
        )
        if any(not value for value in identities):
            raise ValueError("recoverability support provenance is incomplete")
        physical = np.asarray(self.physical_applicability)
        if physical.dtype != np.bool_ or physical.ndim != 2:
            raise ValueError("physical applicability must be an HxW boolean map")
        shape = physical.shape
        names = (
            "physical_applicability", "observable_evidence", "visibility",
            "transport_complete", "transport_ambiguous", "local_fold",
            "influence_support", "recoverability_support",
        )
        masks = {
            name: _bool_map(getattr(self, name), shape, name)
            for name in names
        }
        expected = (
            masks["physical_applicability"]
            & masks["observable_evidence"]
            & masks["visibility"]
            & masks["transport_complete"]
            & ~masks["transport_ambiguous"]
            & ~masks["local_fold"]
            & masks["influence_support"]
        )
        if not np.array_equal(masks["recoverability_support"], expected):
            raise ValueError("recoverability is not the exact support-stage intersection")
        for name, value in masks.items():
            object.__setattr__(self, name, value)

    @property
    def recoverability_support_hash(self) -> str:
        return support_mask_sha256(self.recoverability_support)

    @property
    def recoverability_fraction(self) -> float:
        return float(self.recoverability_support.mean())


@dataclass(frozen=True)
class CandidateResponseReceipt:
    """Unsigned realized response on an already certified support.

    A changed matcher output can justify that the intervention had an effect;
    it cannot establish that the effect improved EPE.  The exact intersection
    is stored separately so later utility calibration cannot expand support.
    """

    source_input_hash: str
    control_key: str
    native_output_hash: str
    candidate_output_hash: str
    response_policy_hash: str
    recoverability_support_hash: str
    recoverability_support: np.ndarray
    candidate_response: np.ndarray
    response_on_recoverability: np.ndarray

    def __post_init__(self) -> None:
        identities = (
            self.source_input_hash, self.control_key, self.native_output_hash,
            self.candidate_output_hash, self.response_policy_hash,
            self.recoverability_support_hash,
        )
        if any(not value for value in identities):
            raise ValueError("candidate response provenance is incomplete")
        support = np.asarray(self.recoverability_support)
        if support.dtype != np.bool_ or support.ndim != 2:
            raise ValueError("recoverability support must be an HxW boolean map")
        masks = {
            name: _bool_map(getattr(self, name), support.shape, name)
            for name in (
                "recoverability_support", "candidate_response",
                "response_on_recoverability",
            )
        }
        if self.recoverability_support_hash != support_mask_sha256(
            masks["recoverability_support"],
        ):
            raise ValueError("recoverability support content hash changed")
        expected = masks["recoverability_support"] & masks["candidate_response"]
        if not np.array_equal(masks["response_on_recoverability"], expected):
            raise ValueError("candidate response expanded or changed recoverability support")
        for name, value in masks.items():
            object.__setattr__(self, name, value)

    @property
    def response_support_hash(self) -> str:
        return support_mask_sha256(self.response_on_recoverability)

    @property
    def response_fraction(self) -> float:
        return float(self.response_on_recoverability.mean())


@dataclass(frozen=True)
class ActionSupportDecomposition:
    """Concrete maps whose algebra prevents uncertainty/support/utility collapse."""

    source_input_hash: str
    support_policy_hash: str
    coordinate_frame: str
    transport_receipt_hash: str
    uncertainty: np.ndarray
    physical_applicability: np.ndarray
    observable_evidence: np.ndarray
    visibility: np.ndarray
    candidate_response: np.ndarray
    candidate_risk_nonincrease: np.ndarray
    collision: np.ndarray
    local_fold: np.ndarray
    uncovered: np.ndarray
    influence_support: np.ndarray
    candidate_eligible_support: np.ndarray
    signed_utility_lower: np.ndarray
    certified_delivery_support: np.ndarray

    def __post_init__(self) -> None:
        identities = (
            self.source_input_hash, self.support_policy_hash,
            self.coordinate_frame, self.transport_receipt_hash,
        )
        if any(not value for value in identities):
            raise ValueError("support decomposition provenance is incomplete")
        uncertainty = _float_map(self.uncertainty, "uncertainty")
        if np.any(uncertainty < 0.0) or np.any(uncertainty > 1.0):
            raise ValueError("uncertainty must lie in [0,1]")
        shape = uncertainty.shape
        names = (
            "physical_applicability", "observable_evidence", "visibility",
            "candidate_response", "candidate_risk_nonincrease", "collision",
            "local_fold", "uncovered", "influence_support",
            "candidate_eligible_support", "certified_delivery_support",
        )
        masks = {
            name: _bool_map(getattr(self, name), shape, name)
            for name in names
        }
        utility = _float_map(self.signed_utility_lower, "signed utility lower")
        if utility.shape != shape:
            raise ValueError("signed utility lower must align with support maps")
        expected_eligible = (
            masks["physical_applicability"]
            & masks["observable_evidence"]
            & masks["visibility"]
            & masks["candidate_response"]
            & masks["candidate_risk_nonincrease"]
            & ~masks["collision"]
            & ~masks["local_fold"]
            & ~masks["uncovered"]
            & masks["influence_support"]
        )
        if not np.array_equal(masks["candidate_eligible_support"], expected_eligible):
            raise ValueError("candidate eligibility is not the exact evidence intersection")
        expected_delivery = expected_eligible & (utility > 0.0)
        if not np.array_equal(
            masks["certified_delivery_support"], expected_delivery,
        ):
            raise ValueError("delivery support is not eligibility intersected with utility")
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "signed_utility_lower", utility)
        for name, value in masks.items():
            object.__setattr__(self, name, value)

    @property
    def uncertainty_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.uncertainty.tobytes())
        return digest.hexdigest()

    @property
    def candidate_support_hash(self) -> str:
        return support_mask_sha256(self.candidate_eligible_support)

    @property
    def physical_support_hash(self) -> str:
        return support_mask_sha256(self.physical_applicability)

    @property
    def visibility_support_hash(self) -> str:
        return support_mask_sha256(self.visibility)

    @property
    def influence_support_hash(self) -> str:
        return support_mask_sha256(self.influence_support)

    @property
    def delivery_support_hash(self) -> str:
        return support_mask_sha256(self.certified_delivery_support)

    @property
    def candidate_fraction(self) -> float:
        return float(self.candidate_eligible_support.mean())

    @property
    def delivery_fraction(self) -> float:
        return float(self.certified_delivery_support.mean())
