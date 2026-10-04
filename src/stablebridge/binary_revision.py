"""S05 observable descriptors and a binary candidate revision model.

The module consumes no GT, hard-region mask, case ID, or corruption label. It
does not build candidates, regress a replacement displacement, choose a query
set, fit descriptor statistics, or calibrate an acceptance threshold. The caller
hard-selects U0 or the already-fallback candidate using the returned score.

Frozen CroCo maps are [C,H/16,W/16], post-encoder-normalization. Features are
sampled at token centers 7.5 + 16*k. Samples outside that token-center rectangle
are explicitly invalid, even when the endpoint remains inside the RGB image.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


DESCRIPTOR_VERSION = "stereo_binary_revision_descriptors_v1"
GEOMETRY_FEATURE_NAMES = (
    "source_x_over_width", "source_y_over_height",
    "initial_dx_over_width", "initial_dy_over_height",
    "candidate_dx_over_width", "candidate_dy_over_height",
    "delta_dx_over_width", "delta_dy_over_height",
    "initial_norm_over_diagonal", "candidate_norm_over_diagonal", "delta_norm_over_diagonal",
    "absolute_delta_x_over_width", "absolute_delta_y_over_height",
    "initial_endpoint_x_over_width", "initial_endpoint_y_over_height",
    "candidate_endpoint_x_over_width", "candidate_endpoint_y_over_height",
    "source_horizontal_boundary_distance_over_width", "source_vertical_boundary_distance_over_height",
    "initial_horizontal_boundary_distance_over_width", "initial_vertical_boundary_distance_over_height",
    "candidate_horizontal_boundary_distance_over_width", "candidate_vertical_boundary_distance_over_height",
    "initial_endpoint_in_image", "candidate_endpoint_in_image", "candidate_valid",
    "initial_finite", "candidate_finite", "source_feature_geometrically_valid",
    "initial_feature_pair_geometrically_valid", "candidate_feature_pair_geometrically_valid",
    "candidate_changed",
)
_ENDPOINT_MATCH_NAMES = (
    "center_rgb_l1", "patch_rgb_l1_r", "patch_rgb_l1_g", "patch_rgb_l1_b",
    "patch_rgb_rmse", "patch_soft_census_l1",
    "encoder_cosine_cost", "encoder_unit_vector_l1",
    "center_rgb_valid", "patch_pair_valid", "encoder_pair_valid",
)
MATCHING_FEATURE_NAMES = (
    "source_patch_mean_r", "source_patch_mean_g", "source_patch_mean_b",
    "source_patch_rgb_std_mean", "source_patch_luma_std", "source_patch_valid",
) + tuple(f"{endpoint}_{name}" for endpoint in ("initial", "candidate") for name in _ENDPOINT_MATCH_NAMES) + (
    "candidate_minus_initial_center_rgb_l1", "candidate_minus_initial_patch_rgb_l1",
    "candidate_minus_initial_encoder_cosine_cost", "candidate_minus_initial_encoder_unit_vector_l1",
)
FEATURE_NAMES = GEOMETRY_FEATURE_NAMES + MATCHING_FEATURE_NAMES
GEOMETRY_DIM = len(GEOMETRY_FEATURE_NAMES)
MATCHING_DIM = len(MATCHING_FEATURE_NAMES)
DESCRIPTOR_DIM = len(FEATURE_NAMES)
assert (GEOMETRY_DIM, MATCHING_DIM, DESCRIPTOR_DIM) == (32, 32, 64)

REQUIRED_ARRAY_KEYS = frozenset({"image0", "image1", "source_features", "target_features",
                               "initial", "candidate", "candidate_valid"})
ALLOWED_ARRAY_KEYS = REQUIRED_ARRAY_KEYS | {"raw_uncertainty"}


def _validate_arrays(arrays):
    missing = REQUIRED_ARRAY_KEYS - set(arrays)
    extra = set(arrays) - ALLOWED_ARRAY_KEYS
    if missing or extra:
        raise ValueError(f"Descriptor input must use the inference allowlist; missing={sorted(missing)}, unexpected={sorted(extra)}")
    image0, image1 = np.asarray(arrays["image0"]), np.asarray(arrays["image1"])
    if image0.ndim != 3 or image0.shape[-1] != 3 or image0.dtype != np.uint8:
        raise ValueError("image0 must be an RGB uint8 [H,W,3] array")
    if image1.shape != image0.shape or image1.dtype != np.uint8:
        raise ValueError("image1 must be RGB uint8 with the same dimensions as image0")
    h, w = image0.shape[:2]
    if min(h, w) < 16 or h % 16 or w % 16:
        raise ValueError("RGB dimensions must be positive multiples of the 16-pixel feature stride")
    source, target = np.asarray(arrays["source_features"]), np.asarray(arrays["target_features"])
    if (source.ndim != 3 or source.shape[0] < 1 or source.shape[1:] != (h // 16, w // 16)
            or target.shape != source.shape):
        raise ValueError("Frozen encoder maps must have identical [C,H/16,W/16] shapes")
    if source.dtype.kind != "f" or target.dtype.kind != "f" or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Frozen encoder features must be finite floating point arrays")
    initial, candidate = np.asarray(arrays["initial"], np.float64), np.asarray(arrays["candidate"], np.float64)
    if initial.shape != (2, h, w) or candidate.shape != initial.shape:
        raise ValueError("initial and candidate must have native displacement shape [2,H,W]")
    valid = np.asarray(arrays["candidate_valid"])
    if valid.shape != (h, w) or not np.all(np.isin(valid, [False, True])):
        raise ValueError("candidate_valid must contain a binary [H,W] mask")
    # raw_uncertainty is deliberately not even opened or shape-checked. It is
    # neither geometry nor matching evidence in this locked information ablation.
    return image0, image1, source, target, initial, candidate, valid.astype(bool)


def _inside(points, width, height):
    return ((points[..., 0] >= 0) & (points[..., 0] <= width - 1)
            & (points[..., 1] >= 0) & (points[..., 1] <= height - 1))


def _feature_inside(points, width, height):
    return ((points[..., 0] >= 7.5) & (points[..., 0] <= width - 8.5)
            & (points[..., 1] >= 7.5) & (points[..., 1] <= height - 8.5))


def _sample(values, points, *, stride=1.0, center=0.0):
    """Sample CHW at [...,2] coordinates; padding is never evidence validity."""
    channels, height, width = values.shape
    token = (points - center) / stride
    extent = points.new_tensor([width, height])
    grid = 2 * (token + .5) / extent - 1
    grid = torch.nan_to_num(grid, nan=2., posinf=2., neginf=-2.)
    sampled = F.grid_sample(values[None], grid.reshape(1, 1, -1, 2),
                            align_corners=False, mode="bilinear", padding_mode="zeros")
    return sampled[0, :, 0].T.reshape(*points.shape[:-1], channels)


def _geometry(initial, candidate, supplied_valid, height, width):
    yy, xx = np.indices((height, width), dtype=np.float64)
    xy = np.stack((xx, yy), axis=-1)
    finite0, finitea = np.isfinite(initial).all(0), np.isfinite(candidate).all(0)
    # Keep the finite flags, but replace an entire invalid vector to prevent NaN
    # components or synthetic zero-padded samples from contaminating descriptors.
    u0 = np.where(finite0[None], initial, 0).transpose(1, 2, 0)
    ua = np.where(finitea[None], candidate, 0).transpose(1, 2, 0)
    delta = ua - u0
    endpoint0, endpointa = xy + u0, xy + ua
    extent = np.array([width, height], dtype=np.float64)
    diagonal = np.hypot(width, height)
    image0_valid = finite0 & _inside(endpoint0, width, height)
    candidate_in_image = finitea & _inside(endpointa, width, height)
    imagea_valid = supplied_valid & candidate_in_image
    source_feature_valid = _feature_inside(xy, width, height)
    feature0_valid = source_feature_valid & finite0 & _feature_inside(endpoint0, width, height)
    candidate_feature_geometric_valid = source_feature_valid & finitea & _feature_inside(endpointa, width, height)
    featurea_valid = supplied_valid & candidate_feature_geometric_valid
    fields = [xy / extent, u0 / extent, ua / extent, delta / extent,
              (np.linalg.norm(u0, axis=-1) / diagonal)[..., None],
              (np.linalg.norm(ua, axis=-1) / diagonal)[..., None],
              (np.linalg.norm(delta, axis=-1) / diagonal)[..., None],
              np.abs(delta) / extent, endpoint0 / extent, endpointa / extent]
    for coordinates in (xy, endpoint0, endpointa):
        fields.append(np.stack((np.minimum(coordinates[..., 0], width - 1 - coordinates[..., 0]) / width,
                                np.minimum(coordinates[..., 1], height - 1 - coordinates[..., 1]) / height), axis=-1))
    for flag in (image0_valid, candidate_in_image, supplied_valid, finite0, finitea, source_feature_valid,
                 feature0_valid, candidate_feature_geometric_valid, finite0 & finitea & np.any(delta != 0, axis=-1)):
        fields.append(flag[..., None].astype(np.float64))
    result = np.concatenate(fields, axis=-1)
    if result.shape != (height, width, GEOMETRY_DIM) or not np.isfinite(result).all():
        raise FloatingPointError("Native geometry cannot be represented by finite descriptors")
    result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Native geometry exceeds float32 descriptor range")
    return result, xy, endpoint0, endpointa, image0_valid, imagea_valid, feature0_valid, featurea_valid


def build_descriptors(arrays: Mapping, device="cpu", chunk_size=2048):
    """Return finite float32 [H,W,64] descriptors and an exact feature contract.

    Inputs must be restricted to ALLOWED_ARRAY_KEYS. Geometry uses only original
    image extent, displacement fields and supplied candidate validity. Matching
    uses fixed RGB descriptors and unit-normalized frozen encoder samples. No
    sample statistics are fitted, no feature projection is learned, and gradients
    never flow into images, candidates or the frozen encoder.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    image0, image1, source, target, initial, candidate, valid = _validate_arrays(arrays)
    height, width = image0.shape[:2]
    geometry, xy, endpoint0, endpointa, rgb0_valid, rgba_valid, feature0_valid, featurea_valid = _geometry(
        initial, candidate, valid, height, width)
    device = torch.device(device)
    def tensor(value):
        return torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=torch.float32)
    rgb0 = tensor(image0.transpose(2, 0, 1)) / 255.
    rgb1 = tensor(image1.transpose(2, 0, 1)) / 255.
    sf, tf = tensor(source), tensor(target)
    flat = [np.ascontiguousarray(value.reshape(-1, 2)) for value in (xy, endpoint0, endpointa)]
    masks = [value.reshape(-1) for value in (rgb0_valid, rgba_valid, feature0_valid, featurea_valid)]
    offsets = tensor(np.array([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)], np.float32))
    luma_weights = tensor(np.array([.299, .587, .114], np.float32))
    matching = np.empty((height * width, MATCHING_DIM), dtype=np.float32)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, height * width, chunk_size):
            stop = min(start + chunk_size, height * width)
            points, endpoints0, endpointsa = [tensor(value[start:stop]) for value in flat]
            endpoints = torch.stack((endpoints0, endpointsa), dim=0)
            center_ok = torch.stack([torch.as_tensor(mask[start:stop], device=device) for mask in masks[:2]])
            feature_ok = torch.stack([torch.as_tensor(mask[start:stop], device=device) for mask in masks[2:]])
            source_patch_xy = points[:, None] + offsets[None]
            target_patch_xy = endpoints[:, :, None] + offsets[None, None]
            source_patch_ok = _inside(source_patch_xy, width, height).all(-1)
            target_patch_ok = _inside(target_patch_xy, width, height).all(-1)
            pair_patch_ok = source_patch_ok[None] & target_patch_ok & center_ok
            source_patch = _sample(rgb0, source_patch_xy)
            target_patch = _sample(rgb1, target_patch_xy)
            source_center, target_center = source_patch[:, 4], target_patch[:, :, 4]
            source_stats = torch.cat((source_patch.mean(dim=1),
                                      source_patch.std(dim=1, unbiased=False).mean(-1, keepdim=True),
                                      (source_patch @ luma_weights).std(dim=1, unbiased=False)[:, None]), dim=-1)
            source_stats = torch.where(source_patch_ok[:, None], source_stats, torch.zeros_like(source_stats))
            columns = [source_stats, source_patch_ok[:, None].to(torch.float32)]
            patch_abs = (source_patch[None] - target_patch).abs()
            center_l1 = (source_center[None] - target_center).abs().mean(-1)
            patch_l1 = patch_abs.mean(dim=-2)
            patch_rmse = patch_abs.square().mean(dim=(-2, -1)).sqrt()
            source_change = source_patch - source_center[:, None]
            target_change = target_patch - target_center[:, :, None]
            source_census = source_change / (source_change.square() + .04 ** 2).sqrt()
            target_census = target_change / (target_change.square() + .04 ** 2).sqrt()
            census_l1 = (source_census[None] - target_census).abs().mean(dim=(-2, -1))
            source_encoder = _sample(sf, points, stride=16., center=7.5)
            target_encoder = _sample(tf, endpoints, stride=16., center=7.5)
            source_norm = source_encoder.norm(dim=-1)
            target_norm = target_encoder.norm(dim=-1)
            feature_ok &= (source_norm[None] > 1e-8) & (target_norm > 1e-8)
            source_encoder = F.normalize(source_encoder, dim=-1, eps=1e-8)
            target_encoder = F.normalize(target_encoder, dim=-1, eps=1e-8)
            cosine_cost = (1 - (source_encoder[None] * target_encoder).sum(-1)).clamp(0, 2)
            encoder_l1 = (source_encoder[None] - target_encoder).abs().mean(-1)
            # Mask every numeric cost, rather than relying on grid_sample's
            # zero padding or a validity flag that a downstream model may ignore.
            center_l1 = torch.where(center_ok, center_l1, 0.)
            patch_l1 = torch.where(pair_patch_ok[:, :, None], patch_l1, 0.)
            patch_rmse = torch.where(pair_patch_ok, patch_rmse, 0.)
            census_l1 = torch.where(pair_patch_ok, census_l1, 0.)
            cosine_cost = torch.where(feature_ok, cosine_cost, 0.)
            encoder_l1 = torch.where(feature_ok, encoder_l1, 0.)
            for index in range(2):
                columns.append(torch.cat((center_l1[index, :, None], patch_l1[index],
                                          patch_rmse[index, :, None], census_l1[index, :, None],
                                          cosine_cost[index, :, None], encoder_l1[index, :, None],
                                          center_ok[index, :, None].float(), pair_patch_ok[index, :, None].float(),
                                          feature_ok[index, :, None].float()), dim=-1))
            center_difference = torch.where(center_ok.all(0), center_l1[1] - center_l1[0], 0.)
            patch_difference = torch.where(pair_patch_ok.all(0), patch_l1[1].mean(-1) - patch_l1[0].mean(-1), 0.)
            cosine_difference = torch.where(feature_ok.all(0), cosine_cost[1] - cosine_cost[0], 0.)
            encoder_difference = torch.where(feature_ok.all(0), encoder_l1[1] - encoder_l1[0], 0.)
            columns.append(torch.stack((center_difference, patch_difference, cosine_difference, encoder_difference), dim=-1))
            result = torch.cat(columns, dim=-1)
            if result.shape != (stop - start, MATCHING_DIM) or not bool(torch.isfinite(result).all()):
                raise FloatingPointError("Matching descriptors are not finite [N,32]")
            matching[start:stop] = result.cpu().numpy()
    descriptors = np.concatenate((geometry, matching.reshape(height, width, MATCHING_DIM)), axis=-1)
    metadata = {
        "version": DESCRIPTOR_VERSION, "feature_names": list(FEATURE_NAMES),
        "geometry_count": GEOMETRY_DIM, "matching_count": MATCHING_DIM, "descriptor_count": DESCRIPTOR_DIM,
        "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES), "matching_feature_names": list(MATCHING_FEATURE_NAMES),
        "shape": list(descriptors.shape), "feature_channels": int(source.shape[0]),
        "feature_stride": 16, "feature_center_offset": 7.5,
        "coordinate_frame": "shared_crop_local_native_pixel_centers",
        "geometry_normalization": "x_by_width_y_by_height_norm_by_hypot_width_height; signed_nearest_boundary_distances",
        "invalid_vectors": "whole_vector_zero_for_finite_computation_with_explicit_finite_flag",
        "rgb_patch": "3x3_offsets_minus1_to_plus1_bilinear_sampling_uint8_div255",
        "patch_validity": "all_9_source_and_target_samples_inside_native_RGB_bounds",
        "feature_validity": "source_and_target_inside_token_center_rectangle_and_nonzero_sampled_feature_norm",
        "invalid_matching_costs": "zero_numeric_cost_and_zero_validity; differences_only_when_both_endpoints_valid",
        "candidate_matching_validity": "also_requires_supplied_candidate_valid; geometric_flags_remain_physical_coordinate_properties",
        "encoder_normalization": "L2_unit_vector_after_bilinear_sampling; no_learned_projection",
        "soft_census_epsilon": .04, "luma_weights": [.299, .587, .114],
        "learned_feature_extraction": False, "fit_statistics": False,
        "uncertainty_used": False, "gt_used": False, "query_mask_used": False,
        "input_allowlist": sorted(ALLOWED_ARRAY_KEYS), "device": str(device), "chunk_size": chunk_size,
    }
    return descriptors.astype(np.float32, copy=False), metadata


class BinaryRevisionMLP(nn.Module):
    """Equal input/trunk capacity for information and target ablations.

    Geometry mode zeros matching channels before the same input layer. Outputs
    are unconstrained real values: one gain, or [initial_error, candidate_error].
    Neither objective enforces positive errors or produces a new displacement.
    """
    def __init__(self, input_dim, geometry_dim, information="geometry", objective="gain", hidden_dim=128):
        super().__init__()
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (input_dim, geometry_dim, hidden_dim)) or geometry_dim > input_dim:
            raise ValueError("Require positive dimensions with geometry_dim <= input_dim")
        if information not in ("geometry", "full") or objective not in ("gain", "errors"):
            raise ValueError("information must be geometry/full and objective gain/errors")
        self.input_dim, self.geometry_dim, self.hidden_dim = input_dim, geometry_dim, hidden_dim
        self.information, self.objective = information, objective
        mask = torch.ones(input_dim, dtype=torch.bool)
        if information == "geometry":
            mask[geometry_dim:] = False
        # The ablation is a constructor contract, not learned checkpoint state:
        # copying identical weights between full/geometry must not copy the mask.
        self.register_buffer("information_mask", mask, persistent=False)
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.head = nn.Linear(hidden_dim, 1 if objective == "gain" else 2)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.input_dim or not features.is_floating_point():
            raise ValueError(f"Expected floating point [N,{self.input_dim}] descriptors")
        visible = torch.where(self.information_mask[None], features, torch.zeros_like(features))
        if not bool(torch.isfinite(visible).all()):
            raise ValueError("Visible model inputs must be finite")
        return self.head(self.trunk(visible))

    def selection_scores(self, outputs):
        expected = 1 if self.objective == "gain" else 2
        if outputs.ndim != 2 or outputs.shape[1] != expected:
            raise ValueError(f"Expected [N,{expected}] model outputs")
        return outputs[:, 0] if self.objective == "gain" else outputs[:, 0] - outputs[:, 1]

    def score(self, features):
        return self.selection_scores(self(features))

    def contract(self):
        trunk_parameters = sum(p.numel() for p in self.trunk.parameters())
        head_parameters = sum(p.numel() for p in self.head.parameters())
        return {"version": "binary_revision_mlp_v1", "input_dim": self.input_dim,
                "geometry_dim": self.geometry_dim, "information": self.information, "objective": self.objective,
                "hidden_dim": self.hidden_dim, "hidden_layers": 2,
                "output_dim": self.head.out_features, "output_positivity_constraint": False,
                "target_scale": "caller_owned_fixed_scale_not_applied_in_this_module",
                "score": "gain_output" if self.objective == "gain" else "initial_error_minus_candidate_error",
                "decision": "caller_hard_selects_existing_initial_or_candidate; no_displacement_regression",
                "parameters": trunk_parameters + head_parameters,
                "trunk_parameters": trunk_parameters, "head_parameters": head_parameters,
                "effective_visible_input_dim": self.geometry_dim if self.information == "geometry" else self.input_dim}
