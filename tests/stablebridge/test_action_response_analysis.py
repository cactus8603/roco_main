"""Small numerical checks for the S01-v5 read-only evaluator."""
from pathlib import Path
import sys
import unittest

import numpy as np


REPORTS=(Path(__file__).resolve().parents[2]/'experiments/E01_evidence_mechanism/studies/'
         'S01_spatial_repair/reports')
sys.path.insert(0,str(REPORTS))
import analyze_action_response_gate_v5 as v5  # noqa: E402


class ActionResponseAnalysisTest(unittest.TestCase):
    def test_candidate_ranks_preserve_ties_and_invalid_slots(self):
        costs=np.array([[1.,2.],[0.,2.],[0.,9.]],np.float32)
        valid=np.array([[1,1],[1,1],[1,0]],bool)
        ranks=v5.candidate_ranks(costs,valid)
        np.testing.assert_allclose(ranks[:,0],[1.,.25,.25])
        np.testing.assert_allclose(ranks[:2,1],[.5,.5])
        self.assertTrue(np.isinf(ranks[2,1]))

    def test_missing_identity_evidence_fails_closed_to_retain(self):
        scores=np.array([[np.inf,0.],[0.,1.]],np.float32)
        errors=np.array([[3.,3.],[1.,5.]],np.float32)
        allowed=np.ones_like(scores,bool)
        result=v5.selector(scores,errors,allowed,1.,1.,5.)
        np.testing.assert_array_equal(result['best'],[0,0])
        self.assertEqual(result['changed_count'],0)

    def test_switch_accounting_distinguishes_help_and_avoided_harm(self):
        errors=np.array([[3.,3.,3.],[1.,6.,4.],[4.,2.,2.]],np.float32)
        combined={'best':np.array([0,1,1])}
        aggregate={'best':np.array([1,0,2])}
        result=v5.switch_counts(combined,aggregate,errors,1.)
        self.assertEqual(result['retain_to_action'],1)
        self.assertEqual(result['retain_to_action_helpful'],1)
        self.assertEqual(result['action_to_retain_avoided_harm'],1)
        self.assertEqual(result['action_to_action'],1)


if __name__=='__main__':unittest.main()
