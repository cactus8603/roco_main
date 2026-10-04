import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs/specs/best_rejected_candidate_ledger_20261002.json"


def load_ledger():
    return json.loads(LEDGER.read_text())


def candidate(ledger, experiment):
    return next(
        item for item in ledger["candidates"]
        if item["experiment"] == experiment
    )


def test_relaxation_is_narrow_provisional_and_effect_preserving():
    policy = load_ledger()["policy"]
    assert policy["minimum_independent_prefrozen_successor_attempts"] == 3
    assert policy["maximum_relaxable_gate_count"] == 1
    assert policy["maximum_absolute_rate_gap_percentage_points"] == 2.0
    assert policy["minimum_safe_effect_retention_fraction"] == 0.95
    assert "PROVISIONAL_SHADOW_INCUMBENT" in \
        policy["provisional_substitution_semantics"]
    assert "effect lower bound is non-positive" in policy["materially_worse_veto"]


def test_tail_safety_and_integrity_can_never_be_relaxed():
    never = set(load_ledger()["policy"]["never_relax"])
    assert {"severe_harm_budget", "worst_harm_budget"} <= never
    assert {"data_or_outcome_leakage", "hash_or_lineage_integrity"} <= never
    assert {"sealed_external_confirmation", "h2_seal"} <= never


def test_e169_remains_best_recorded_reject_but_is_not_relaxed_now():
    ledger = load_ledger()
    e169 = candidate(ledger, "E169_noise_law_receipt_audit")
    assert e169["rank"] == 1
    assert e169["formal_status"].startswith("REJECT_")
    assert e169["absolute_gap_percentage_points"] < 2.0
    assert e169["successor_attempts_after_candidate"] == 4
    assert e169["qualifying_same_bottleneck_successor_attempts"] == 0
    assert e169["relaxation_eligible_now"] is False
    assert e169["provisional_substitution_status"] == \
        "NOT_NEEDED_AND_NOT_ELIGIBLE"


def test_same_panel_e175_success_does_not_count_as_independent_relaxation():
    ledger = load_ledger()
    e169 = candidate(ledger, "E169_noise_law_receipt_audit")
    e175 = next(
        item for item in e169["same_panel_mechanism_history"]
        if item["experiment"] == "E175_balanced_impulse_law_successor"
    )
    assert e175["result"].startswith("PASS_")
    assert e175["counts_toward_relaxation_prerequisite"] is False
    assert "23/23" in e175["reason"]


def test_small_numeric_gap_cannot_override_e177_signed_severe_harm():
    ledger = load_ledger()
    e177 = candidate(ledger, "E177_cross_source_full_impulse_conjunction")
    assert e177["absolute_gap_percentage_points"] < 2.0
    assert e177["other_key_results"][
        "e178_non_impulse_severe_harm_endpoints"
    ] == 8
    assert e177["other_key_results"][
        "e178_non_impulse_worst_clean_action_efficiency"
    ] == -1.0
    assert e177["relaxation_eligible_now"] is False
    assert e177["provisional_substitution_status"].startswith("PROHIBITED_")
