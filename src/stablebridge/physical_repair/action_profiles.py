"""Auditable action traits for the Selector-v3 design.

This registry does not select an action and never grants delivery authority.
It makes a crucial distinction that the older explanatory registry could not:
features already used by a physical decision, diagnostics that are recorded but
do not gate it, and evidence that is still missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .selector_v2 import ACTION_MECHANISMS


PARAMETER_GEOMETRIES = frozenset({"fixed", "scalar", "vector", "mixed"})
SPATIAL_SCOPES = frozenset({"point", "local", "regional", "whole_frame"})
INFORMATION_EFFECTS = frozenset({"preserve", "attenuate", "project", "discard"})
REVERSIBILITY_STATES = frozenset({"approximate", "conditional", "irreversible"})


@dataclass(frozen=True)
class ActionPhysicsProfile:
    operator_id: str
    family: str
    parameter_geometry: str
    spatial_scope: str
    information_effect: str
    reversibility: str
    decision_witnesses: tuple[str, ...]
    diagnostic_witnesses: tuple[str, ...]
    missing_witnesses: tuple[str, ...]
    closure_witnesses: tuple[str, ...]
    dangerous_confounds: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.operator_id or not self.family:
            raise ValueError("action profile needs an operator and family")
        if self.parameter_geometry not in PARAMETER_GEOMETRIES:
            raise ValueError("unknown parameter geometry")
        if self.spatial_scope not in SPATIAL_SCOPES:
            raise ValueError("unknown spatial scope")
        if self.information_effect not in INFORMATION_EFFECTS:
            raise ValueError("unknown information effect")
        if self.reversibility not in REVERSIBILITY_STATES:
            raise ValueError("unknown reversibility state")
        groups = (
            self.decision_witnesses,
            self.diagnostic_witnesses,
            self.missing_witnesses,
        )
        if any(not value for group in groups for value in group):
            raise ValueError("witness names must be nonempty")
        if any(len(set(group)) != len(group) for group in groups):
            raise ValueError("witness names must be unique within each stage")
        if set(self.decision_witnesses) & set(self.missing_witnesses):
            raise ValueError("a witness cannot be both used and missing")
        if not self.closure_witnesses or not self.dangerous_confounds:
            raise ValueError("closure tests and confounds are required")

    @property
    def physical_gap_count(self) -> int:
        return len(self.missing_witnesses)


_IMPULSE = ActionPhysicsProfile(
    operator_id="impulse_exact_median3", family="impulse",
    parameter_geometry="fixed", spatial_scope="point",
    information_effect="project", reversibility="irreversible",
    decision_witnesses=(
        "sparse_local_extremum", "paired_correspondence_disagreement",
        "spatial_distribution",
    ),
    diagnostic_witnesses=("raw_saturation_fraction",),
    missing_witnesses=(
        "j_invariant_median_risk", "natural_saturation_likelihood",
        "edge_contamination",
    ),
    closure_witnesses=(
        "heldout_center_prediction", "residual_impulse_rate", "task_signed_utility",
    ),
    dangerous_confounds=("specular_highlight", "traffic_light", "thin_structure"),
)


ACTION_PHYSICS_PROFILES: Mapping[str, ActionPhysicsProfile] = {
    "impulse_exact_median3": _IMPULSE,
    "impulse_median3": ActionPhysicsProfile(
        **{**_IMPULSE.__dict__, "operator_id": "impulse_median3"}
    ),
    "wiener3": ActionPhysicsProfile(
        operator_id="wiener3", family="additive_noise",
        parameter_geometry="scalar", spatial_scope="regional",
        information_effect="attenuate", reversibility="irreversible",
        decision_witnesses=("robust_tile_laplacian_excess", "spatial_coherence"),
        diagnostic_witnesses=("within_image_noise_baseline",),
        missing_witnesses=(
            "paired_auto_cross_psd", "heteroscedastic_noise_law",
            "j_invariant_or_sure_risk", "postrepair_residual_whiteness",
        ),
        closure_witnesses=(
            "j_invariant_risk", "residual_whiteness", "task_signed_utility",
        ),
        dangerous_confounds=("demosaic_correlation", "jpeg_ringing", "fine_texture"),
    ),
    "common_disk": ActionPhysicsProfile(
        operator_id="common_disk", family="defocus_disk",
        parameter_geometry="scalar", spatial_scope="regional",
        information_effect="attenuate", reversibility="irreversible",
        decision_witnesses=(
            "heldout_spatial_reblur", "otf_magnitude", "same_parameter_endpoint",
            "common_passband_information", "translation_nuisance",
            "observable_ab_fold_support",
        ),
        diagnostic_witnesses=("spectral_own_null",),
        missing_witnesses=(
            "bessel_zero_pattern", "complex_cross_spectrum_phase",
            "parameter_set_worst_case", "spatially_varying_psf",
        ),
        closure_witnesses=("repair_reblur_closure", "task_signed_utility"),
        dangerous_confounds=("gaussian_blur", "motion_blur", "warp_error"),
    ),
    "common_gaussian": ActionPhysicsProfile(
        operator_id="common_gaussian", family="gaussian_blur",
        parameter_geometry="scalar", spatial_scope="regional",
        information_effect="attenuate", reversibility="irreversible",
        decision_witnesses=(
            "heldout_spatial_reblur", "otf_magnitude", "same_parameter_endpoint",
            "common_passband_information", "translation_nuisance",
            "observable_ab_fold_support",
        ),
        diagnostic_witnesses=("spectral_own_null",),
        missing_witnesses=(
            "log_otf_quadratic", "complex_cross_spectrum_phase",
            "parameter_set_worst_case", "spatially_varying_psf",
        ),
        closure_witnesses=("repair_reblur_closure", "task_signed_utility"),
        dangerous_confounds=("additive_noise", "defocus_disk", "warp_error"),
    ),
    "common_motion": ActionPhysicsProfile(
        operator_id="common_motion", family="motion_blur",
        parameter_geometry="vector", spatial_scope="regional",
        information_effect="attenuate", reversibility="irreversible",
        decision_witnesses=(
            "heldout_spatial_reblur", "directional_otf_competitor",
            "same_parameter_endpoint", "common_passband_information",
            "translation_nuisance", "observable_ab_fold_support",
        ),
        diagnostic_witnesses=("spectral_own_null",),
        missing_witnesses=(
            "phase_only_autocorrelation", "cepstral_periodicity",
            "kernel_flow_direction_consistency", "parameter_set_worst_case",
            "nonuniform_motion_field",
        ),
        closure_witnesses=("repair_reblur_closure", "task_signed_utility"),
        dangerous_confounds=("edge_orientation", "warp_error", "defocus_disk"),
    ),
    "jpeg_deblock": ActionPhysicsProfile(
        operator_id="jpeg_deblock", family="jpeg",
        parameter_geometry="mixed", spatial_scope="regional",
        information_effect="project", reversibility="conditional",
        decision_witnesses=(
            "dct_quantization_lattice", "grid_phase", "endpoint_localization",
            "heldout_boundary_excess_reduction",
            "codec_reencode_noninferiority", "dct_interval_consistency",
            "rgb_gamut_realizability", "declared_footprint_exact",
        ),
        diagnostic_witnesses=("estimated_ijg_quality",),
        missing_witnesses=(
            "texture_retention", "chroma_subsampling",
            "codec_backend_robustness",
        ),
        closure_witnesses=(
            "codec_reencode_noninferiority", "task_signed_utility",
        ),
        dangerous_confounds=("natural_grid_texture", "pixelation", "resampling"),
    ),
    "pixelate": ActionPhysicsProfile(
        operator_id="pixelate", family="pixelate",
        parameter_geometry="scalar", spatial_scope="whole_frame",
        information_effect="discard", reversibility="irreversible",
        decision_witnesses=(),
        diagnostic_witnesses=("sampling_lattice_candidate",),
        missing_witnesses=(
            "sampling_factor_identifiability", "spectral_replica_phase",
            "alias_free_band", "heldout_action_efficacy",
        ),
        closure_witnesses=("downsample_upsample_closure", "task_signed_utility"),
        dangerous_confounds=("jpeg_grid", "regular_texture", "true_low_resolution"),
    ),
    "rank3_pair": ActionPhysicsProfile(
        operator_id="rank3_pair", family="radiometry",
        parameter_geometry="fixed", spatial_scope="local",
        information_effect="project", reversibility="conditional",
        decision_witnesses=(
            "heldout_affine_forward_gain", "monotone_derivative_lower",
            "invertibility", "clipping_censored_support",
            "rank_action_commutator",
        ),
        diagnostic_witnesses=(
            "rank_residual", "gain_channel_span", "rank_warp_commutator",
            "rank_total_commutator", "candidate_response",
        ),
        missing_witnesses=(
            "nonlinear_monotone_response", "rank_action_efficacy",
        ),
        closure_witnesses=("heldout_inverse_closure", "task_signed_utility"),
        dangerous_confounds=("clipping", "quantization", "fine_texture"),
    ),
}


def validate_action_profiles() -> None:
    """Fail if the explanatory and auditable registries drift apart."""
    expected, observed = set(ACTION_MECHANISMS), set(ACTION_PHYSICS_PROFILES)
    if expected != observed:
        raise RuntimeError(
            f"action profile coverage drift: missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )
    for key, profile in ACTION_PHYSICS_PROFILES.items():
        if key != profile.operator_id:
            raise RuntimeError(f"action profile identity drift: {key}")


validate_action_profiles()
