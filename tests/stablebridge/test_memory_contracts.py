"""Behavioral checks for information rights, provenance, and bounded state."""
from dataclasses import replace
import pickle
import unittest

import numpy as np

from stablebridge.contracts import FrameEvidence, QueryIdentity, ReadBudget, TimeSnapshot
from stablebridge.memory import EvidenceStore


def frame(index, *, view="left", **kwargs):
    return FrameEvidence("scene", index, view, np.full((2, 3, 4), index, np.float32),
                         image_hw=(24, 32), **kwargs)


def flow(t=4, **kwargs):
    return QueryIdentity("scene", "left", t, "left", t + 1, **kwargs)


class MemoryContractsTest(unittest.TestCase):
    def test_future_views_and_computation_are_not_leaked(self):
        store = EvidenceStore()
        past = store.add(frame(2))
        store.add(frame(6))
        store.add(frame(3, view="right"))
        store.add(frame(4, availability=8))
        result = store.retrieve(flow(), TimeSnapshot(5, available=5))
        self.assertEqual([r.evidence_id for r in result], [past.evidence_id])
        self.assertEqual(result.costs["read_items"], 1)
        with self.assertRaises(ValueError):
            store.retrieve(flow(), TimeSnapshot(6))
        with self.assertRaises(ValueError):
            store.retrieve(flow(), TimeSnapshot(5, allowed_views=("left", "right")))

    def test_derived_evidence_inherits_information_and_view_restrictions(self):
        store = EvidenceStore()
        future = store.add(frame(9))
        child = store.add(frame(3, parent_versions=(future.evidence_id,)))
        self.assertEqual(child.information_time, 9)
        self.assertFalse(store.check_support(child.evidence_id, flow()))
        right = store.add(frame(2, view="right"))
        cross_view = store.add(frame(3, parent_versions=(right.evidence_id,)))
        self.assertFalse(store.check_support(cross_view.evidence_id, flow()))
        stereo = QueryIdentity("scene", "left", 4, "right", 4)
        self.assertTrue(store.check_support(cross_view.evidence_id, stereo))

    def test_pair_mode_cannot_use_past_or_derived_past(self):
        store = EvidenceStore(mode="M0")
        old = store.add(frame(2))
        current = store.add(frame(4))
        store.add(frame(5, parent_versions=(old.evidence_id,)))
        result = store.retrieve(flow())
        self.assertEqual([r.evidence_id for r in result], [current.evidence_id])

    def test_duplicate_root_does_not_inflate_observation_count_or_read_budget(self):
        store = EvidenceStore()
        original = store.add(frame(2))
        store.add(frame(2, parent_versions=(original.evidence_id,), quality=5))
        store.add(frame(2, parent_versions=(original.evidence_id,), quality=9))
        independent = store.add(frame(3))
        result = store.retrieve(flow(), budget=ReadBudget(max_reads=4), policy="quality")
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result.independent_groups), 2)
        self.assertIn(independent.evidence_id, [r.evidence_id for r in result])
        self.assertEqual(result.costs["read_items"], 2)

    def test_invalidation_stales_descendants_but_keeps_other_support(self):
        store = EvidenceStore()
        bad = store.add(frame(1))
        good = store.add(frame(2))
        child = store.add(frame(3, parent_versions=(bad.evidence_id, good.evidence_id)))
        grandchild = store.add(frame(4, parent_versions=(child.evidence_id,)))
        affected = store.invalidate(bad.evidence_id)
        self.assertEqual(set(affected), {bad.evidence_id, child.evidence_id, grandchild.evidence_id})
        self.assertFalse(store.check_support(child.evidence_id))
        self.assertTrue(store.check_support(good.evidence_id))
        self.assertEqual(set(store.revision_queue), {child.evidence_id, grandchild.evidence_id})
        new = store.complete_revision(child.evidence_id, frame(3), support_ids=(good.evidence_id,))
        self.assertTrue(store.check_support(new.evidence_id))
        self.assertFalse(store.check_support(child.evidence_id))
        self.assertEqual(new.observation_ids, good.observation_ids)
        self.assertNotIn(child.evidence_id, store.revision_queue)
        self.assertIn(grandchild.evidence_id, store.revision_queue)

    def test_group_revocation_reaches_copies_and_preserves_revision_semantics(self):
        store = EvidenceStore()
        source = store.add(frame(1, source_groups=("shared-source",)))
        child = store.add(frame(2, parent_versions=(source.evidence_id,)))
        store.invalidate(source_group="shared-source")
        self.assertFalse(store.check_support(source.evidence_id))
        self.assertFalse(store.check_support(child.evidence_id))
        clone = store.add(frame(3, source_groups=("shared-source",)))
        self.assertFalse(store.check_support(clone.evidence_id))

    def test_evicted_bad_observation_cannot_reenter_as_a_fresh_version(self):
        store = EvidenceStore(capacity=1, tombstone_capacity=1)
        bad = store.add(frame(1))
        store.invalidate(bad.evidence_id)
        store.add(frame(2))
        store.add(frame(3))  # Both bad payload and tombstone have now expired.
        clone = store.add(frame(1))
        self.assertFalse(store.check_support(clone.evidence_id))
        state = store.state_dict()
        restored = EvidenceStore(capacity=1, tombstone_capacity=1)
        restored.load_state_dict(state)
        self.assertFalse(restored.check_support(restored.add(frame(1)).evidence_id))

    def test_bad_derived_estimate_does_not_invalidate_original_observation(self):
        store = EvidenceStore()
        source = store.add(frame(1))
        estimate = store.add(frame(2, parent_versions=(source.evidence_id,)))
        store.invalidate(estimate.evidence_id)
        self.assertTrue(store.check_support(source.evidence_id))
        revised = store.complete_revision(estimate.evidence_id, frame(2), support_ids=(source.evidence_id,))
        self.assertTrue(store.check_support(revised.evidence_id))

    def test_copy_cannot_claim_extra_independent_groups(self):
        store = EvidenceStore()
        source = store.add(frame(1))
        with self.assertRaises(ValueError):
            store.add(frame(2, parent_versions=(source.evidence_id,), source_groups=("invented-independent",)))

    def test_revocation_overflow_is_bounded_and_fail_closed(self):
        store = EvidenceStore(capacity=2, tombstone_capacity=1, max_lineage=1)
        source = store.add(frame(1))
        store.invalidate(source_group="bad-a")
        store.invalidate(source_group="bad-b")
        self.assertFalse(store.check_support(source.evidence_id))
        self.assertEqual(len(store.state_dict()["revoked_groups"]), 1)
        self.assertFalse(store.check_support(store.add(frame(3)).evidence_id))
        store.reset()
        self.assertTrue(store.check_support(store.add(frame(4)).evidence_id))

    def test_eviction_tombstones_and_expiration_do_not_create_trust(self):
        store = EvidenceStore(capacity=2, tombstone_capacity=1)
        source = store.add(frame(1))
        child = store.add(frame(2, parent_versions=(source.evidence_id,)))
        store.add(frame(3))  # Source is a tombstone with known valid provenance.
        self.assertTrue(store.check_support(child.evidence_id))
        self.assertEqual(len(store), 2)
        self.assertEqual(store.tombstone_count, 1)
        store.invalidate(source.evidence_id)
        self.assertFalse(store.check_support(child.evidence_id))
        # A wider live buffer leaves a child alive when its source's tombstone expires.
        store = EvidenceStore(capacity=3, tombstone_capacity=1)
        source = store.add(frame(0))
        store.add(frame(1))
        child = store.add(frame(2, parent_versions=(source.evidence_id,)))
        store.add(frame(3))
        store.add(frame(4))
        self.assertFalse(store.check_support(child.evidence_id))
        self.assertIn(child.evidence_id, store.revision_queue)
        for i in range(5, 30):
            store.add(frame(i))
        self.assertEqual(len(store), 3)
        self.assertEqual(store.tombstone_count, 1)
        self.assertLessEqual(len(store.revision_queue), 3)

    def test_metadata_ranking_and_probes_pay_for_access_before_selection(self):
        store = EvidenceStore()
        store.add(frame(1, quality=100, metadata={"descriptor": [0, 1]}))
        compatible = store.add(frame(2, quality=1, metadata={"descriptor": [1, 0]}))
        query = flow(metadata={"descriptor": [1, 0]})
        result = store.retrieve(query, budget=ReadBudget(max_reads=1), policy="query_conditioned")
        self.assertEqual(result.records[0].evidence_id, compatible.evidence_id)
        self.assertEqual(result.costs["index_items"], 2)
        self.assertEqual(result.costs["probe_items"], 0)
        size = compatible.features.nbytes
        result = store.retrieve(query, budget=ReadBudget(max_reads=1, max_probes=2),
                                policy="quality", probe=lambda evidence, _: evidence.frame)
        self.assertEqual(result.records[0].evidence_id, compatible.evidence_id)
        self.assertEqual(result.costs["probe_items"], 2)
        self.assertEqual(result.costs["read_items"], 1)
        self.assertEqual(result.costs["read_bytes"], 3 * size)
        result = store.retrieve(query, budget=ReadBudget(max_reads=2, max_probes=1, max_read_bytes=size),
                                probe=lambda *_: 1)
        self.assertEqual(result.costs["read_bytes"], size)
        self.assertEqual(len(result), 0)  # The paid probe consumed the full byte budget.

    def test_resume_is_equivalent_and_reset_removes_sequence_and_cost_state(self):
        store = EvidenceStore(capacity=3)
        source = store.add(frame(1))
        store.add(frame(2, parent_versions=(source.evidence_id,)))
        store.retrieve(flow(), budget=ReadBudget(max_reads=1))
        restored = EvidenceStore(capacity=3)
        restored.load_state_dict(pickle.loads(pickle.dumps(store.state_dict())))
        a, b = store.retrieve(flow()), restored.retrieve(flow())
        self.assertEqual([r.evidence_id for r in a], [r.evidence_id for r in b])
        self.assertEqual(store.cost_summary(), restored.cost_summary())
        self.assertEqual(store.add(frame(3)).evidence_id, restored.add(frame(3)).evidence_id)
        with self.assertRaises(ValueError):
            EvidenceStore(capacity=4).load_state_dict(store.state_dict())
        with self.assertRaises(ValueError):
            restored.add(replace(frame(4), sequence="other"))
        restored.reset(sequence="other")
        self.assertEqual(len(restored), 0)
        self.assertFalse(any(restored.cost_summary().values()))
        restored.add(replace(frame(1), sequence="other"))

    def test_immutable_versions_isolate_mutable_payloads(self):
        store = EvidenceStore()
        value = frame(1, metadata={"descriptor": [1, 0]})
        added = store.add(value)
        value.features[:] = 50
        value.metadata["descriptor"][0] = 50
        added.features[:] = 70
        first = store.retrieve(flow()).records[0]
        np.testing.assert_array_equal(first.features, 1)
        self.assertEqual(first.payload.metadata["descriptor"], [1, 0])
        first.features[:] = 99
        np.testing.assert_array_equal(store.retrieve(flow()).records[0].features, 1)
        with self.assertRaises(ValueError):
            store.add(replace(frame(3), evidence_id=added.evidence_id))

    def test_unknown_parent_cannot_later_form_a_cycle(self):
        store = EvidenceStore()
        a = store.add(frame(1, evidence_id="a", parent_versions=("b",)))
        self.assertFalse(store.check_support(a.evidence_id))
        with self.assertRaises(ValueError):
            store.add(frame(2, evidence_id="b", parent_versions=("a",)))

    def test_bounded_lineage_rejects_silent_provenance_truncation(self):
        store = EvidenceStore(max_lineage=2)
        with self.assertRaises(ValueError):
            store.add(frame(1, source_groups=("a", "b", "c")))
        self.assertEqual(len(store), 0)

    def test_feature_byte_budget_evicts_even_with_free_record_slots(self):
        size = frame(1).features.nbytes
        store = EvidenceStore(capacity=10, max_feature_bytes=2 * size)
        old = store.add(frame(1))
        second = store.add(frame(2))
        third = store.add(frame(3))
        self.assertEqual(len(store), 2)
        self.assertEqual(store.feature_bytes, 2 * size)
        self.assertEqual([r.evidence_id for r in store.records], [second.evidence_id, third.evidence_id])
        self.assertTrue(store.check_support(old.evidence_id))  # Metadata tombstone remains known.
        with self.assertRaises(ValueError):
            store.add(replace(frame(4), features=np.zeros((2, 30, 40), np.float32)))
        self.assertEqual(store.feature_bytes, 2 * size)
        restored = EvidenceStore(capacity=10, max_feature_bytes=2 * size)
        restored.load_state_dict(store.state_dict())
        self.assertEqual(restored.feature_bytes, store.feature_bytes)
        with self.assertRaises(ValueError):
            EvidenceStore(capacity=10, max_feature_bytes=3 * size).load_state_dict(store.state_dict())
        with self.assertRaises(ValueError):
            replace(frame(4), features=[[[0] * 100000]])


if __name__ == "__main__":
    unittest.main()
