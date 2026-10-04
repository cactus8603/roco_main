"""Analytic path checks; image support is distinct from geometric correctness."""
import unittest
from types import SimpleNamespace

import numpy as np

from stablebridge.path_geometry import DenseEdge, compose_path, project_stereo_path
from stablebridge.pipeline import _predict_reverse
from stablebridge.quartet_experiment import restore_translation


def affine_field(matrix, offset, hw=(64, 64), origin=(0, 0)):
    """Displacement for endpoint mapping p -> matrix @ p + offset."""
    yy, xx = np.mgrid[:hw[0], :hw[1]]
    grid = np.stack((xx + origin[0], yy + origin[1]))
    return np.einsum("ij,jhw->ihw", np.asarray(matrix) - np.eye(2), grid) + np.asarray(offset)[:, None, None]


class PathGeometryTests(unittest.TestCase):
    def test_nonconstant_fields_sample_each_arrival_not_original_query(self):
        matrices = (np.diag([1.5, 1.25]), np.diag([0.9, 1.1]),
                    np.array([[1.0, 0.05], [-0.1, 1.0]]))
        offsets = ((1, 0), (2, 1), (0, 0.5))
        edges = [DenseEdge(affine_field(a, b), str(i), str(i + 1))
                 for i, (a, b) in enumerate(zip(matrices, offsets))]
        query = np.array([[5.25, 7.5], [10.5, 10.25]])
        expected = query.copy()
        for matrix, offset in zip(matrices, offsets):
            expected = expected @ matrix.T + offset
        result = compose_path(query, edges)
        np.testing.assert_allclose(result["raw_displacement"], expected - query, atol=1e-12)
        np.testing.assert_allclose(result["coordinates"][-1], expected, atol=1e-12)
        self.assertTrue(result["edge_computable"].all())
        wrong = sum(query @ (matrix - np.eye(2)).T + offset
                    for matrix, offset in zip(matrices, offsets))
        self.assertGreater(np.max(np.abs(wrong - result["raw_displacement"])), 0.1)

    def test_backward_field_is_inverse_mapping_not_negative_forward(self):
        matrix = np.diag([1.2, 0.9])
        offset = np.array([2.0, 1.0])
        inverse = np.linalg.inv(matrix)
        forward = DenseEdge(affine_field(matrix, offset), "A", "B")
        reverse = DenseEdge(affine_field(inverse, -inverse @ offset), "B", "A")
        query = np.array([[10.25, 15.5]])
        correct = compose_path(query, [forward, reverse])
        np.testing.assert_allclose(correct["raw_displacement"], 0.0, atol=1e-12)
        self.assertTrue(correct["computable"].all())
        wrong_reverse = DenseEdge(-forward.field, "B", "A")
        wrong = compose_path(query, [forward, wrong_reverse])
        self.assertGreater(np.linalg.norm(wrong["raw_displacement"]), 0.5)
        self.assertFalse(np.allclose(reverse.field, -forward.field))

    def test_reverse_stereo_adapter_swaps_flips_and_restores_native_endpoints(self):
        height, width = 5, 9
        image0 = np.arange(height * width * 3).reshape(height, width, 3)
        image1 = image0 + 1000
        origin = (768, 380)
        flipped_field = affine_field(np.diag([1.05, 1]), (-2, 0.25), (height, width))

        class RecordingBackbone:
            def predict(self, source, target, origin_xy):
                self.inputs = (source.copy(), target.copy(), origin_xy)
                return SimpleNamespace(displacement=flipped_field,
                                       metadata={"diagnostic": "fake_reverse"})

        backbone = RecordingBackbone()
        native_field, metadata = _predict_reverse(backbone, "stereo", image0, image1, origin)
        np.testing.assert_array_equal(backbone.inputs[0], image1[:, ::-1])
        np.testing.assert_array_equal(backbone.inputs[1], image0[:, ::-1])
        self.assertEqual(backbone.inputs[2], origin)
        self.assertEqual(metadata["diagnostic"], "fake_reverse")

        # Reflect both endpoints around the crop's global centre. A shared
        # nonzero crop origin cancels from dx; vertical displacement does not
        # change sign under a horizontal reflection.
        query_local = np.array([2.0, 1.0])
        query_global = query_local + origin
        flipped_query_local = np.array([width - 1 - query_local[0], query_local[1]])
        flipped_endpoint = flipped_query_local + flipped_field[:, 1, 6]
        expected_endpoint = np.array([width - 1 - flipped_endpoint[0], flipped_endpoint[1]]) + origin
        edge = DenseEdge(native_field, "R", "L", source_origin_xy=origin,
                         target_origin_xy=origin)
        composed = compose_path(query_global[None], [edge])
        self.assertTrue(composed["computable"][0])
        np.testing.assert_allclose(composed["coordinates"][-1, 0], expected_endpoint, atol=1e-6)

        # Flow reverse inference swaps the pair without any reflection.
        flow_field, _ = _predict_reverse(backbone, "flow", image0, image1, origin)
        np.testing.assert_array_equal(backbone.inputs[0], image1)
        np.testing.assert_array_equal(backbone.inputs[1], image0)
        np.testing.assert_allclose(flow_field, flipped_field)

    def test_crop_origins_do_not_change_global_displacement_units(self):
        first = DenseEdge(affine_field(np.eye(2), (10, 5), (4, 4)), "A", "B",
                          source_origin_xy=(100, 200), target_origin_xy=(110, 205),
                          target_hw=(4, 4))
        second = DenseEdge(affine_field(np.eye(2), (2, -1), (6, 6)), "B", "C",
                           source_origin_xy=(109, 204), target_origin_xy=(112, 204),
                           target_hw=(4, 4))
        result = compose_path([[100.5, 201.25]], [first, second])
        self.assertTrue(result["computable"][0])
        np.testing.assert_allclose(result["raw_displacement"], [[12, 4]])
        np.testing.assert_allclose(result["coordinates"][:, 0],
                                   [[100.5, 201.25], [110.5, 206.25], [112.5, 205.25]])

    def test_invalid_destination_remains_invalid_despite_later_large_roi(self):
        first = DenseEdge(affine_field(np.eye(2), (3, 0), (5, 5)), "A", "B")
        second = DenseEdge(affine_field(np.eye(2), (0, 0)), "B", "C")
        result = compose_path([[0, 1], [3, 1], [-0.01, 1], [np.nan, 1]], [first, second])
        np.testing.assert_array_equal(result["edge_computable"],
                                      [[True, False, False, False], [True, False, False, False]])
        np.testing.assert_allclose(result["raw_displacement"], [[3, 0], [0, 0], [0, 0], [0, 0]])
        self.assertTrue(np.isfinite(result["coordinates"]).all())
        np.testing.assert_array_equal(result["coordinates"][1:, 1], [[3, 1], [3, 1]])

    def test_nonzero_bilinear_corners_require_finite_observed_values(self):
        field = np.zeros((2, 5, 5))
        field[:, 1, 2] = np.nan
        observed = np.ones((5, 5), bool)
        observed[3, 2] = False
        edge = DenseEdge(field, "A", "B", observed_mask=observed)
        result = compose_path([[1, 1], [1.5, 1], [1, 3], [1.5, 3], [4, 4]], [edge])
        np.testing.assert_array_equal(result["computable"], [True, False, True, False, True])
        self.assertTrue(np.isfinite(result["raw_displacement"]).all())

    def test_source_crop_of_next_edge_must_cover_previous_destination(self):
        first = DenseEdge(np.zeros((2, 5, 5)), "A", "B")
        second = DenseEdge(np.zeros((2, 2, 2)), "B", "C", source_origin_xy=(2, 2),
                           target_origin_xy=(2, 2))
        result = compose_path([[1, 1], [2, 2]], [first, second])
        np.testing.assert_array_equal(result["edge_computable"], [[True, True], [False, True]])

    def test_roll_restoration_preserves_displacement_without_offset_or_scaling(self):
        native = affine_field(np.diag([1.03, 0.98]), (0.75, -0.25), (24, 32))
        yy, xx = np.mgrid[:24, :32]
        queries = np.stack((xx, yy), axis=-1)
        endpoints = queries + native.transpose(1, 2, 0)
        for shift in ((9, 0), (0, 9), (-9, 0), (0, -9), (9, -9)):
            with self.subTest(shift=shift):
                dx, dy = shift
                rolled = np.roll(native, (dy, dx), axis=(1, 2))
                restored, valid = restore_translation(rolled, shift)
                np.testing.assert_array_equal(restored, native)
                # Independent scalar inequalities include original endpoints
                # and transformed source/target positions, without clipping.
                expected = np.ones((24, 32), dtype=bool)
                for point in (endpoints, queries + shift, endpoints + shift):
                    expected &= ((point[..., 0] >= 0) & (point[..., 0] <= 31)
                                 & (point[..., 1] >= 0) & (point[..., 1] <= 23))
                np.testing.assert_array_equal(valid, expected)

    def test_roll_mask_rejects_source_wrap_endpoint_wrap_and_original_oob(self):
        native = np.zeros((2, 5, 16), dtype=np.float64)
        native[0, 1, 2] = 8       # Original endpoint 10 is inside; shifted 19 wraps.
        native[0, 1, 14] = -10   # Endpoint and shifted endpoint fit; source wraps.
        native[0, 1, 3] = -5     # Shifted endpoint fits; original endpoint is -2.
        native[0, 2, 1] = 4.99   # Shifted endpoint 14.99 fits continuously.
        native[0, 2, 2] = 4.99   # Shifted endpoint 15.99 exceeds last pixel centre.
        native[1, 1, 4] = np.nan
        rolled = np.roll(native, 9, axis=2)
        restored, valid = restore_translation(rolled, (9, 0))
        np.testing.assert_array_equal(restored, native)
        self.assertFalse(valid[1, 2])
        self.assertFalse(valid[1, 14])
        self.assertFalse(valid[1, 3])
        self.assertFalse(valid[1, 4])
        self.assertTrue(valid[2, 1])
        self.assertFalse(valid[2, 2])
        self.assertTrue(valid[0, 6])
        self.assertFalse(valid[0, 7])

    def test_stereo_projection_retains_signed_vertical_residual_and_rejects_it(self):
        raw = np.array([[-3, 0.2], [-4, 1.0], [-5, -1.01], [-6, 0]])
        result = project_stereo_path(raw, [True, True, True, False], tolerance=1.0)
        np.testing.assert_array_equal(result["eligible"], [True, True, False, False])
        np.testing.assert_array_equal(result["candidate"], [[-3, 0], [-4, 0], [-5, 0], [-6, 0]])
        np.testing.assert_array_equal(result["raw_displacement"], raw)
        np.testing.assert_array_equal(result["vertical_residual"], raw[:, 1])

    def test_large_disagreement_with_wrong_direct_prediction_is_not_a_veto(self):
        # True disparity is 12 px, while the old answer is 2 px. An independently
        # supported path must be allowed to offer its 10 px correction.
        path = DenseEdge(affine_field(np.eye(2), (-12, 0)), "L", "R")
        result = compose_path([[20, 10]], [path])
        stereo = project_stereo_path(result["raw_displacement"], result["computable"])
        self.assertTrue(stereo["eligible"][0])
        np.testing.assert_allclose(stereo["candidate"], [[-12, 0]])
        self.assertEqual(np.linalg.norm(stereo["candidate"][0] - [-2, 0]), 10)

    def test_consistent_wrong_fields_close_without_becoming_correct(self):
        # Actual scene is stationary. These two mutually inverse translations
        # close exactly while each correspondence is still wrong by 4 pixels.
        forward = DenseEdge(affine_field(np.eye(2), (4, 0)), "A", "B")
        backward = DenseEdge(affine_field(np.eye(2), (-4, 0)), "B", "A")
        query = [[10, 10]]
        closed = compose_path(query, [forward, backward])
        one_way = compose_path(query, [forward])
        np.testing.assert_allclose(closed["raw_displacement"], 0)
        self.assertTrue(closed["computable"][0])
        self.assertEqual(np.linalg.norm(one_way["raw_displacement"]), 4)

    def test_bad_contracts_rejected_and_empty_query_batch_supported(self):
        edge = DenseEdge(np.zeros((2, 3, 3)), "A", "B")
        with self.assertRaises(ValueError):
            compose_path([[0, 0]], [edge, edge])
        with self.assertRaises(ValueError):
            compose_path([[0, 0]], [])
        with self.assertRaises(ValueError):
            compose_path([[0, 0, 0]], [edge])
        with self.assertRaises(ValueError):
            DenseEdge(np.zeros((3, 3, 3)), "A", "B")
        with self.assertRaises(ValueError):
            DenseEdge(np.zeros((2, 3, 3)), "A", "B", target_hw=(3.5, 3))
        with self.assertRaises(ValueError):
            project_stereo_path([[0, 0]], [True], tolerance=-1)
        empty = compose_path(np.empty((0, 2)), [edge])
        self.assertEqual(empty["raw_displacement"].shape, (0, 2))
        self.assertEqual(empty["coordinates"].shape, (2, 0, 2))


if __name__ == "__main__":
    unittest.main()
