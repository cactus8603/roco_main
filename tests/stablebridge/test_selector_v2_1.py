import pytest

from stablebridge.physical_repair.selector_v2 import PairwiseDominance
from stablebridge.physical_repair.selector_v2_1 import select_action_set_v2_1

from test_selector_v2 import candidate


def rows():
    return (
        candidate(prefix="noise"),
        candidate("common_motion@first", "common_motion", "first", "motion"),
    )


def test_reciprocal_pairwise_conflict_is_order_independent_and_prunes_nothing():
    evidence = (
        PairwiseDominance(
            "wiener3@first", "common_motion@first", 0.1, "pair-support",
        ),
        PairwiseDominance(
            "common_motion@first", "wiener3@first", 0.1, "pair-support",
        ),
    )
    forward = select_action_set_v2_1(rows(), evidence)
    reverse = select_action_set_v2_1(rows(), tuple(reversed(evidence)))
    assert forward.pruned_actions == reverse.pruned_actions == {}
    assert forward.supported_actions == reverse.supported_actions
    assert set(forward.probe_actions) == set(forward.supported_actions)


def test_clear_undominated_pairwise_witness_prunes_loser():
    result = select_action_set_v2_1(rows(), (
        PairwiseDominance(
            "wiener3@first", "common_motion@first", 0.1, "pair-support",
        ),
    ))
    assert result.supported_actions == ("wiener3@first",)
    assert result.pruned_actions == {"common_motion@first": "wiener3@first"}


def test_unknown_pairwise_action_fails_closed():
    with pytest.raises(ValueError, match="unknown action"):
        select_action_set_v2_1(rows(), (
            PairwiseDominance("wiener3@first", "unknown@first", 0.1, "support"),
        ))


def test_candidate_key_must_match_certificate_identity():
    bad = candidate("common_motion@first", "wiener3", "first", "noise")
    with pytest.raises(ValueError, match="action identity"):
        select_action_set_v2_1((bad,))
