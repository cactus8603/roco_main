"""Contracts for evidence-dependent reliability, action value, and revision."""
import unittest

from stablebridge.contracts import (
    ActionAssessment,
    ActionBudget,
    QueryIdentity,
    QueryState,
    ResourceCost,
    RiskEstimate,
    advance_query_state,
)


def query() -> QueryIdentity:
    return QueryIdentity("scene", "left", 4, "left", 5, native_xy=(10.0, 20.0))


def risk(probability: float, calibration_id: str = "risk-v1") -> RiskEstimate:
    return RiskEstimate(probability, 1.0, calibration_id)


def state(status: str = "unresolved") -> QueryState:
    return QueryState(
        query=query(),
        revision=0,
        evidence_version="e0",
        prediction_id="u0",
        risk=risk(0.8),
        budget=ActionBudget(10.0, 1000, 8, 3, 2.0),
        status=status,
        ambiguity=1.2,
        evidence_ids=("pair",),
        admissible_action_ids=("a1", "retain"),
        information_cutoff=5,
        decision_event=5.5,
        deadline_event=7,
    )


def action(kind: str = "read_memory", **kwargs) -> ActionAssessment:
    values = dict(
        action_id="a1",
        action_kind=kind,
        based_on_revision=0,
        evidence_version="e0",
        success_probability=0.7,
        expected_gain=1.2,
        harm_probability=0.1,
        expected_tail_harm=0.2,
        cost=ResourceCost(compute=2.0, read_bytes=100, candidates=2,
                          revisions=1, delay_events=1.0),
        calibration_id="action-v1",
        candidate_ids=("c1",),
    )
    values.update(kwargs)
    return ActionAssessment(**values)


class AdaptiveQueryContractsTest(unittest.TestCase):
    def test_unresolved_is_a_mutable_state_after_new_evidence(self):
        updated, transition = advance_query_state(
            state(), action(), outcome="accept", next_status="trusted",
            next_risk=risk(0.15), next_prediction_id="u1",
            next_ambiguity=0.2,
            next_evidence_version="e1", added_evidence_ids=("history-1",),
            next_admissible_action_ids=("retain", "rematch-2"),
            accepted_candidate_id="c1", reason="new evidence resolved ambiguity",
        )
        self.assertEqual(updated.status, "trusted")
        self.assertEqual(updated.revision, 1)
        self.assertEqual(updated.predecessor_revision, 0)
        self.assertEqual(updated.prediction_id, "u1")
        self.assertEqual(updated.risk.error_probability, 0.15)
        self.assertEqual(updated.ambiguity, 0.2)
        self.assertEqual(updated.admissible_action_ids, ("retain", "rematch-2"))
        self.assertEqual(updated.evidence_ids, ("pair", "history-1"))
        self.assertEqual(updated.budget.compute, 8.0)
        self.assertEqual(transition.outcome, "accept")

    def test_acceptance_does_not_imply_support_eligibility(self):
        _, transition = advance_query_state(
            state(), action(), outcome="accept", next_status="actionable",
            next_prediction_id="u1", accepted_candidate_id="c1",
            support_eligible=False,
        )
        self.assertFalse(transition.support_eligible)

    def test_retain_is_an_explicit_identity_fallback(self):
        keep = action(
            kind="retain", action_id="retain", success_probability=0.2,
            expected_gain=0.0, harm_probability=0.0,
            expected_tail_harm=0.0, cost=ResourceCost(), candidate_ids=(),
        )
        updated, transition = advance_query_state(
            state(), keep, outcome="retain", next_status="unresolved",
            reason="no admissible action passed the harm constraint",
        )
        self.assertEqual(updated.prediction_id, "u0")
        self.assertEqual(updated.budget, state().budget)
        self.assertEqual(transition.outcome, "retain")

    def test_stale_action_and_hidden_new_evidence_are_rejected(self):
        with self.assertRaises(ValueError):
            advance_query_state(
                state(), action(based_on_revision=1), outcome="retain",
                next_status="unresolved",
            )
        with self.assertRaises(ValueError):
            advance_query_state(
                state(), action(), outcome="retain", next_status="waiting",
                added_evidence_ids=("future",),
            )
        with self.assertRaises(ValueError):
            advance_query_state(
                state(), action(action_id="not-admissible"), outcome="retain",
                next_status="unresolved",
            )

    def test_budget_is_enforced_before_transition(self):
        expensive = action(cost=ResourceCost(compute=11.0, revisions=1))
        with self.assertRaises(ValueError):
            advance_query_state(
                state(), expensive, outcome="retain", next_status="abstained",
            )

    def test_action_probabilities_and_risk_require_calibratable_ranges(self):
        with self.assertRaises(ValueError):
            risk(1.1)
        with self.assertRaises(ValueError):
            action(harm_probability=-0.1)


if __name__ == "__main__":
    unittest.main()
