from stablebridge.physical_repair.action_profiles import (
    ACTION_PHYSICS_PROFILES,
    validate_action_profiles,
)
from stablebridge.physical_repair.selector_v2 import ACTION_MECHANISMS


def test_every_selector_action_has_an_auditable_profile():
    validate_action_profiles()
    assert set(ACTION_PHYSICS_PROFILES) == set(ACTION_MECHANISMS)


def test_declared_but_unimplemented_signals_are_not_decision_witnesses():
    motion = ACTION_PHYSICS_PROFILES["common_motion"]
    assert "phase_only_autocorrelation" in motion.missing_witnesses
    assert "phase_only_autocorrelation" not in motion.decision_witnesses
    noise = ACTION_PHYSICS_PROFILES["wiener3"]
    assert "paired_auto_cross_psd" in noise.missing_witnesses
    jpeg = ACTION_PHYSICS_PROFILES["jpeg_deblock"]
    assert "dct_interval_consistency" in jpeg.decision_witnesses
    assert "rgb_gamut_realizability" in jpeg.decision_witnesses
    assert "codec_reencode_noninferiority" in jpeg.decision_witnesses
    assert "texture_retention" in jpeg.missing_witnesses


def test_information_destroying_actions_require_closure_and_task_utility():
    for profile in ACTION_PHYSICS_PROFILES.values():
        assert "task_signed_utility" in profile.closure_witnesses
        if profile.information_effect in {"attenuate", "discard"}:
            assert profile.reversibility == "irreversible"
