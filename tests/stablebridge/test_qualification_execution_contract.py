from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from stablebridge.physical_repair import qualification_execution_contract as qec
from stablebridge.physical_repair.qualification_matrix import canonical_sha256


class QualificationExecutionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(17)
        self.first = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
        self.second = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
        self.flow = rng.normal(size=(32, 48, 2)).astype(np.float32)
        self.case_id = "qualification-case-001"
        self.receipt = qec.build_native_matcher_qualification_receipt_v1(
            case_id=self.case_id,
            matcher_id="frozen-native-matcher-v1",
            corrupted_first_rgb=self.first,
            corrupted_second_rgb=self.second,
            native_predicted_flow=self.flow,
            corruption_receipt_sha256="1" * 64,
            matcher_execution_receipt_sha256="2" * 64,
        )

    @staticmethod
    def _rehash(receipt: dict) -> dict:
        receipt["receipt_sha256"] = canonical_sha256({
            key: value for key, value in receipt.items()
            if key != "receipt_sha256"
        })
        return receipt

    def test_receipt_binds_same_corrupted_pair_and_native_prediction(self) -> None:
        observed = qec.validate_native_matcher_qualification_receipt_v1(
            self.receipt,
            case_id=self.case_id,
            corrupted_first_rgb=self.first,
            corrupted_second_rgb=self.second,
            native_predicted_flow=self.flow,
        )
        self.assertEqual(
            observed["outputs"]["flow"]["semantic_role"],
            qec.NATIVE_FLOW_ROLE_V1,
        )
        self.assertFalse(observed["chronology"]["task_ground_truth_decoded"])

    def test_official_gt_role_is_rejected_even_with_valid_shape_and_hash(self) -> None:
        bad = deepcopy(self.receipt)
        bad["outputs"]["flow"]["semantic_role"] = "OFFICIAL_TASK_GROUND_TRUTH"
        self._rehash(bad)
        with self.assertRaisesRegex(RuntimeError, "rejects non-native or GT"):
            qec.validate_native_matcher_qualification_receipt_v1(
                bad,
                case_id=self.case_id,
                corrupted_first_rgb=self.first,
                corrupted_second_rgb=self.second,
                native_predicted_flow=self.flow,
            )

    def test_clean_pair_flow_and_premature_gt_decode_are_rejected(self) -> None:
        for mutation in ("clean", "gt"):
            with self.subTest(mutation=mutation):
                bad = deepcopy(self.receipt)
                if mutation == "clean":
                    bad["clean_pair_flow_read"] = True
                else:
                    bad["chronology"]["task_ground_truth_decoded"] = True
                self._rehash(bad)
                with self.assertRaises(RuntimeError):
                    qec.validate_native_matcher_qualification_receipt_v1(
                        bad,
                        case_id=self.case_id,
                        corrupted_first_rgb=self.first,
                        corrupted_second_rgb=self.second,
                        native_predicted_flow=self.flow,
                    )

    def test_array_or_case_rebinding_is_rejected(self) -> None:
        changed = self.first.copy()
        changed[0, 0, 0] ^= np.uint8(1)
        with self.assertRaisesRegex(RuntimeError, "first input binding drift"):
            qec.validate_native_matcher_qualification_receipt_v1(
                self.receipt,
                case_id=self.case_id,
                corrupted_first_rgb=changed,
                corrupted_second_rgb=self.second,
                native_predicted_flow=self.flow,
            )
        with self.assertRaisesRegex(RuntimeError, "case binding drift"):
            qec.validate_native_matcher_qualification_receipt_v1(
                self.receipt,
                case_id="qualification-case-002",
                corrupted_first_rgb=self.first,
                corrupted_second_rgb=self.second,
                native_predicted_flow=self.flow,
            )

    def test_provenance_failure_occurs_before_matrix_call(self) -> None:
        bad = deepcopy(self.receipt)
        bad["ground_truth_fields_read"] = ["flow_gt"]
        self._rehash(bad)
        with mock.patch.object(qec, "build_uncapped_qualification_matrix") as matrix:
            with self.assertRaisesRegex(RuntimeError, "forbidden evidence"):
                qec.build_bound_uncapped_qualification_matrix_v1(
                    case_id=self.case_id,
                    corrupted_first_rgb=self.first,
                    corrupted_second_rgb=self.second,
                    native_predicted_flow=self.flow,
                    native_matcher_receipt=bad,
                )
            matrix.assert_not_called()

    def test_bound_wrapper_seals_native_receipt_to_matrix_receipt(self) -> None:
        matrix_receipt = {
            "native_reference": {
                "first_sha256": self.receipt["inputs"]["first"]["array_sha256"],
                "second_sha256": self.receipt["inputs"]["second"]["array_sha256"],
                "native_flow_sha256": self.receipt["outputs"]["flow"]["array_sha256"],
            },
            "receipt_sha256": "3" * 64,
        }
        fake = SimpleNamespace(receipt=matrix_receipt)
        with mock.patch.object(
            qec, "build_uncapped_qualification_matrix", return_value=fake,
        ) as matrix:
            result = qec.build_bound_uncapped_qualification_matrix_v1(
                case_id=self.case_id,
                corrupted_first_rgb=self.first,
                corrupted_second_rgb=self.second,
                native_predicted_flow=self.flow,
                native_matcher_receipt=self.receipt,
            )
        matrix.assert_called_once()
        self.assertIs(result.matrix, fake)
        binding = result.binding_receipt
        self.assertEqual(
            binding["native_matcher_receipt_sha256"],
            self.receipt["receipt_sha256"],
        )
        self.assertEqual(binding["qualification_matrix_receipt_sha256"], "3" * 64)
        self.assertFalse(binding["task_ground_truth_joined"])
        self.assertEqual(
            binding["receipt_sha256"],
            canonical_sha256({
                key: value for key, value in binding.items()
                if key != "receipt_sha256"
            }),
        )


if __name__ == "__main__":
    unittest.main()
