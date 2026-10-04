"""Order-independent validation wrapper for the frozen Selector-v2 core.

E139 binds ``selector_v2.py`` by hash, so compatibility fixes live in this
explicit successor rather than rewriting the source used by that artifact.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .selector_v2 import (
    ActionCandidate,
    PairwiseDominance,
    SelectorV2Config,
    SelectorV2Decision,
    select_action_set,
)


def select_action_set_v2_1(
    candidates: Sequence[ActionCandidate],
    pairwise: Sequence[PairwiseDominance] = (),
    *,
    config: SelectorV2Config = SelectorV2Config(),
) -> SelectorV2Decision:
    """Validate identities and prune only by an undominated unique witness.

    Reciprocal or longer positive-dominance cycles preserve every involved
    hypothesis for the independent task-verifier stage.  This makes the result
    invariant to the order in which pairwise records are serialized.
    """
    rows = tuple(candidates)
    witnesses = tuple(pairwise)
    keys = {row.key for row in rows}
    if len(keys) != len(rows):
        raise ValueError("candidate keys must be unique")
    for row in rows:
        expected = (
            f"{row.certificate.action.operator_id}@"
            f"{row.certificate.action.hypothesized_degraded_endpoint}"
        )
        if row.key != expected:
            raise ValueError("candidate key does not match certificate action identity")
    for witness in witnesses:
        if witness.winner_key not in keys or witness.loser_key not in keys:
            raise ValueError("pairwise witness references an unknown action")

    # Ask the frozen core only for the physical supported set.  Its task state
    # is irrelevant here; the final call below recomputes the complete result.
    physical = set(select_action_set(rows, (), config=config).supported_actions)
    positive = tuple(
        witness for witness in witnesses
        if witness.margin_lower > 0.0
        and witness.winner_key in physical and witness.loser_key in physical
    )
    incoming: dict[str, set[str]] = defaultdict(set)
    for witness in positive:
        incoming[witness.loser_key].add(witness.winner_key)

    proposed: dict[str, set[str]] = defaultdict(set)
    by_edge: dict[tuple[str, str], PairwiseDominance] = {}
    for witness in positive:
        if not incoming[witness.winner_key]:
            proposed[witness.loser_key].add(witness.winner_key)
            edge = (witness.winner_key, witness.loser_key)
            incumbent = by_edge.get(edge)
            if incumbent is None or witness.margin_lower > incumbent.margin_lower:
                by_edge[edge] = witness

    safe = []
    for loser, winners in sorted(proposed.items()):
        if len(winners) != 1:
            continue
        winner = next(iter(winners))
        safe.append(by_edge[(winner, loser)])
    return select_action_set(rows, tuple(safe), config=config)
