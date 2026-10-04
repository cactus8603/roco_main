"""S04: a small learned interface for support-conditioned native-pixel matching.

This module never accepts ground truth or an oracle query mask. Frozen CroCo
encoder maps use stride 16 and first center 7.5; RGB is optional float [0, 1].
Coordinates are LOCAL to a shared source/target crop. Displacements are source
to target (negative horizontal displacement for ordinary left stereo).

Each query always has 30 evaluated candidate slots: identity, nine independent
local searches, and four support hypotheses with five local offsets each. An
absent support is replaced by an independently specified search anchor. Thus a
wrong or absent support cannot remove identity or the independent search. The
output is a learned soft combination of bounded candidate refinements, not a
calibrated accept/reject policy. Identity itself is never refined.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def sample_native_map(values: Tensor, xy: Tensor, *, stride: float = 1.0,
                      center_offset: float = 0.0) -> Tensor:
    """Sample [B,C,H,W] at arbitrary [B,...,2] native xy; output [B,...,C].

    Native image bounds, rather than token-center bounds, govern candidate
    validity in the caller. Border padding extends the outer half feature patch.
    """
    if values.ndim != 4 or xy.ndim < 3 or xy.shape[-1] != 2 or values.shape[0] != xy.shape[0]:
        raise ValueError("Expected values [B,C,H,W] and xy [B,...,2]")
    h, w = values.shape[-2:]
    token_xy = (xy - center_offset) / stride
    # align_corners=False also correctly handles a one-token map.
    extent = values.new_tensor([w, h])
    grid = (2.0 * (token_xy + 0.5) / extent - 1.0).reshape(values.shape[0], -1, 1, 2)
    result = F.grid_sample(values, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return result.squeeze(-1).transpose(1, 2).reshape(*xy.shape[:-1], values.shape[1])


class SupportConditionedRematcher(nn.Module):
    """Shared stereo/flow architecture; instantiate separate weights per task.

    ``support_mode='none'`` ignores every supplied support value and provenance.
    ``rematch=False`` is the direct-propagation ablation: it has the exact same
    parameters/candidates but cannot read target descriptors or target RGB.
    Caller must enforce split isolation and omit GT support locations from the
    scoring query set. Every support within the declared query exclusion radius
    is additionally excluded before nearest-neighbor selection or descriptors.
    """

    candidate_count = 30
    nearest_supports = 4

    def __init__(self, feature_channels: int = 1024, hidden_dim: int = 96,
                 projected_dim: int = 32, query_chunk_size: int = 1024,
                 num_sources: int = 16, residual_bound_px: float = 2.0,
                 min_support_distance_px: float = 8.0, nearest_supports: int = 4):
        super().__init__()
        if min(feature_channels, hidden_dim, projected_dim, query_chunk_size, num_sources) <= 0:
            raise ValueError("Model dimensions must be positive")
        if residual_bound_px <= 0:
            raise ValueError("residual_bound_px must be positive")
        if min_support_distance_px < 0:
            raise ValueError("min_support_distance_px must be nonnegative")
        if nearest_supports != 4:
            raise ValueError("This locked 30-candidate interface requires nearest_supports=4")
        self.feature_channels = feature_channels
        self.query_chunk_size = query_chunk_size
        self.num_sources = num_sources
        self.residual_bound_px = float(residual_bound_px)
        self.min_support_distance_px = float(min_support_distance_px)
        self.feature_projection = nn.Conv2d(feature_channels, projected_dim, 1)
        self.source_embedding = nn.Embedding(num_sources, 8)
        self.kind_embedding = nn.Embedding(4, 4)
        self.scorer = nn.Sequential(nn.Linear(4 * projected_dim + 108 + 10 + 2 + 8 + 4, hidden_dim),
                                    nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.score_head = nn.Linear(hidden_dim, 1)
        self.residual_head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.identity_bias = nn.Parameter(torch.tensor(2.0))

    def contract(self) -> dict:
        return {"version": "support_conditioned_rematcher_v1", "candidate_count": 30,
                "identity_slots": 1, "independent_slots": 9, "support_slots": 20,
                "nearest_supports": 4, "local_candidates_per_support": 5,
                "feature_stride": 16, "feature_center_offset": 7.5,
                "residual_bound_px": self.residual_bound_px,
                "output": "soft_weighted_bounded_candidate_refinements",
                "identity_refinement": False, "input_gradients": False,
                "support_exclusion_radius_px": self.min_support_distance_px,
                "same_location_supports": "excluded", "parameters": sum(p.numel() for p in self.parameters())}

    @staticmethod
    def _offsets(task: str, ref: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if task == "stereo":
            independent = [(x, 0) for x in (-8, -6, -4, -2, 0, 2, 4, 6, 8)]
            local = [(x, 0) for x in (-2, -1, 0, 1, 2)]
            anchors = [(-16, 0), (-8, 0), (8, 0), (16, 0)]
        else:
            independent = [(x, y) for y in (-4, 0, 4) for x in (-4, 0, 4)]
            local = [(0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)]
            anchors = [(-16, 0), (16, 0), (0, -16), (0, 16)]
        return tuple(ref.new_tensor(v) for v in (independent, local, anchors))

    def _candidates(self, initial: Tensor, query: Tensor, supports_xy: Tensor,
                    supports_uv: Tensor, support_valid: Tensor, support_source: Tensor,
                    task: str, support_mode: str, h: int, w: int) -> dict[str, Tensor]:
        b, n = query.shape[:2]
        independent, local, anchors = self._offsets(task, query)
        centers = initial[:, :, None] + anchors[None, None]
        neighbor_xy = query[:, :, None].expand(b, n, 4, 2).clone()
        neighbor_index = torch.full((b, n, 4), -1, dtype=torch.long, device=query.device)
        available = torch.zeros((b, n, 4), dtype=torch.bool, device=query.device)
        source = torch.zeros((b, n, 4), dtype=torch.long, device=query.device)
        s = supports_xy.shape[1]
        if support_mode == "provided" and s:
            finite = torch.isfinite(supports_xy).all(-1) & torch.isfinite(supports_uv).all(-1)
            inside = ((supports_xy >= 0) & (supports_xy <= query.new_tensor([w - 1, h - 1]))).all(-1)
            usable = support_valid & finite & inside
            safe_xy, safe_uv = torch.nan_to_num(supports_xy), torch.nan_to_num(supports_uv)
            squared_distance = (query[:, :, None] - safe_xy[:, None]).square().sum(-1)
            # Even privileged support must not expose the exact query answer.
            eligible = usable[:, None] & (squared_distance > max(self.min_support_distance_px ** 2, 1e-8))
            distance = squared_distance.masked_fill(~eligible, torch.inf)
            k = min(s, 4)
            nearest_distance, nearest = torch.topk(distance, k, dim=-1, largest=False, sorted=True)
            ok = torch.isfinite(nearest_distance)
            batch = torch.arange(b, device=query.device)[:, None, None]
            selected_uv, selected_xy = safe_uv[batch, nearest], safe_xy[batch, nearest]
            centers[:, :, :k] = torch.where(ok[..., None], selected_uv, centers[:, :, :k])
            neighbor_xy[:, :, :k] = torch.where(ok[..., None], selected_xy, neighbor_xy[:, :, :k])
            neighbor_index[:, :, :k] = torch.where(ok, nearest, -1)
            available[:, :, :k] = ok
            source[:, :, :k] = torch.where(ok, support_source[batch, nearest], 0)
        support_candidates = centers[:, :, :, None] + local[None, None, None]
        candidate_uv = torch.cat((initial[:, :, None], initial[:, :, None] + independent[None, None],
                                  support_candidates.flatten(2, 3)), dim=2)
        if task == "stereo":
            candidate_uv = candidate_uv * candidate_uv.new_tensor([1, 0])
        candidate_xy = query[:, :, None] + candidate_uv
        endpoint_valid = ((candidate_xy >= 0) & (candidate_xy <= query.new_tensor([w - 1, h - 1]))).all(-1)
        candidate_valid = endpoint_valid.clone()
        candidate_valid[:, :, 0] = True  # Identity remains available even outside target crop.
        prefix_sources = torch.zeros((b, n, 10), dtype=torch.long, device=query.device)
        candidate_source = torch.cat((prefix_sources, source.repeat_interleave(5, dim=-1)), dim=-1)
        kinds = torch.ones((b, n, 30), dtype=torch.long, device=query.device)
        kinds[:, :, 0] = 0
        kinds[:, :, 10:] = torch.where(available, 2, 3).repeat_interleave(5, dim=-1)
        support_delta = torch.zeros((b, n, 30, 2), dtype=query.dtype, device=query.device)
        support_delta[:, :, 10:] = (neighbor_xy - query[:, :, None]).repeat_interleave(5, dim=2)
        candidate_has_support = torch.cat((torch.zeros((b, n, 10), dtype=torch.bool, device=query.device),
                                           available.repeat_interleave(5, dim=-1)), dim=-1)
        return {"candidate_uv": candidate_uv, "candidate_xy": candidate_xy,
                "candidate_valid": candidate_valid, "endpoint_valid": endpoint_valid,
                "candidate_source": candidate_source, "candidate_kind": kinds,
                "support_delta": support_delta, "candidate_has_support": candidate_has_support,
                "support_neighbor_indices": neighbor_index}

    @staticmethod
    def _rgb_patches(rgb: Tensor | None, xy: Tensor) -> Tensor:
        if rgb is None:
            return xy.new_zeros((*xy.shape[:-1], 27))
        offsets = xy.new_tensor([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)])
        return sample_native_map(rgb, xy[..., None, :] + offsets).flatten(-2)

    def forward(self, source_features: Tensor, target_features: Tensor, initial_uv: Tensor,
                query_xy: Tensor, supports_xy: Tensor, supports_uv: Tensor,
                support_valid: Tensor, support_source: Tensor, *, task: Literal["stereo", "flow"],
                source_rgb: Tensor | None = None, target_rgb: Tensor | None = None,
                support_mode: Literal["provided", "none"] = "provided", rematch: bool = True) -> dict[str, Tensor]:
        if task not in ("stereo", "flow") or support_mode not in ("provided", "none"):
            raise ValueError("Invalid task or support_mode")
        if source_features.ndim != 4 or source_features.shape != target_features.shape:
            raise ValueError("Feature maps must have matching [B,C,H/16,W/16] shapes")
        b, c, fh, fw = source_features.shape
        if c != self.feature_channels or initial_uv.shape != (b, 2, fh * 16, fw * 16):
            raise ValueError("Feature stride must be 16 and initial_uv must be [B,2,H,W]")
        h, w = initial_uv.shape[-2:]
        if query_xy.ndim != 3 or query_xy.shape[0] != b or query_xy.shape[-1] != 2 or not query_xy.shape[1]:
            raise ValueError("query_xy must be nonempty [B,N,2]")
        if supports_xy.ndim != 3 or supports_xy.shape[0] != b or supports_xy.shape[-1] != 2 or supports_uv.shape != supports_xy.shape:
            raise ValueError("supports_xy and supports_uv must be [B,S,2]")
        if support_valid.shape != supports_xy.shape[:2] or support_source.shape != support_valid.shape:
            raise ValueError("support_valid and support_source must be [B,S]")
        if support_valid.dtype != torch.bool or support_source.dtype != torch.long:
            raise ValueError("support_valid must be bool and support_source must be int64")
        if bool(((support_source < 0) | (support_source >= self.num_sources)).any()):
            raise ValueError("Support source ids are outside the declared vocabulary")
        if not bool(torch.isfinite(query_xy).all()) or bool(((query_xy < 0) | (query_xy > query_xy.new_tensor([w - 1, h - 1]))).any()):
            raise ValueError("query_xy must be finite and inside the source crop")
        if not bool(torch.isfinite(initial_uv).all()):
            raise ValueError("initial_uv must be finite")
        if (source_rgb is None) != (target_rgb is None):
            raise ValueError("Provide both RGB images or neither")
        for rgb in (source_rgb, target_rgb):
            if rgb is not None and (rgb.shape != (b, 3, h, w) or not rgb.is_floating_point()):
                raise ValueError("RGB must be floating point [B,3,H,W] in [0,1]")
        # No gradient may reach a frozen backbone, preprocessed support, or RGB.
        initial_uv, query_xy = initial_uv.detach(), query_xy.detach()
        supports_xy, supports_uv = supports_xy.detach(), supports_uv.detach()
        source_rgb = None if source_rgb is None else source_rgb.detach()
        target_rgb = None if target_rgb is None else target_rgb.detach()
        projected_source = self.feature_projection(source_features.detach())
        # The propagation control must not even evaluate the target projection.
        projected_target = self.feature_projection(target_features.detach()) if rematch else None
        parts: dict[str, list[Tensor]] = {}
        for start in range(0, query_xy.shape[1], self.query_chunk_size):
            query = query_xy[:, start:start + self.query_chunk_size]
            initial = sample_native_map(initial_uv, query)
            if task == "stereo":
                initial = initial * initial.new_tensor([1, 0])
            bank = self._candidates(initial, query, supports_xy, supports_uv, support_valid,
                                    support_source, task, support_mode, h, w)
            candidate_uv, endpoint = bank["candidate_uv"], bank["candidate_xy"]
            source = sample_native_map(projected_source, query, stride=16, center_offset=7.5)
            source = source[:, :, None].expand(-1, -1, 30, -1)
            target = (sample_native_map(projected_target, endpoint, stride=16, center_offset=7.5)
                      if rematch else torch.zeros_like(source))
            visual = torch.cat((source, target, (source - target).abs() if rematch else torch.zeros_like(source),
                                source * target), dim=-1)
            source_patch = self._rgb_patches(source_rgb, query)[:, :, None].expand(-1, -1, 30, -1)
            target_patch = self._rgb_patches(target_rgb, endpoint) if rematch else torch.zeros_like(source_patch)
            rgb = torch.cat((source_patch, target_patch,
                             (source_patch - target_patch).abs() if rematch else torch.zeros_like(source_patch),
                             source_patch * target_patch), dim=-1)
            extent = initial.new_tensor([w, h])
            init = initial[:, :, None].expand(-1, -1, 30, -1)
            geometry = torch.cat((candidate_uv / extent, init / extent, (candidate_uv - init) / 16,
                                  bank["support_delta"] / extent,
                                  bank["candidate_has_support"][..., None].to(initial.dtype),
                                  (bank["endpoint_valid"] & rematch)[..., None].to(initial.dtype)), dim=-1)
            task_bit = initial.new_tensor([task == "stereo", task == "flow"])
            task_bits = task_bit.expand(*candidate_uv.shape[:-1], 2)
            description = torch.cat((visual, rgb, geometry, task_bits,
                                     self.source_embedding(bank["candidate_source"]),
                                     self.kind_embedding(bank["candidate_kind"])), dim=-1)
            hidden = self.scorer(description)
            logits = self.score_head(hidden).squeeze(-1)
            identity_mask = torch.zeros_like(logits)
            identity_mask[:, :, 0] = 1
            logits = logits + identity_mask * self.identity_bias
            logits = logits.masked_fill(~bank["candidate_valid"], -torch.inf)
            probabilities = torch.softmax(logits, dim=-1)
            residual = torch.tanh(self.residual_head(hidden)) * self.residual_bound_px
            residual = residual * (1 - identity_mask[..., None])
            if task == "stereo":
                residual = residual * residual.new_tensor([1, 0])
            refined = candidate_uv + residual
            output = (probabilities[..., None] * refined).sum(dim=-2)
            selected = logits.argmax(-1)
            selected_uv = refined.gather(2, selected[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
            current = {"output_uv": output, "candidate_uv": candidate_uv,
                       "refined_candidate_uv": refined, "candidate_valid": bank["candidate_valid"],
                       "logits": logits, "probabilities": probabilities,
                       "selected_index": selected, "selected_uv": selected_uv,
                       "support_neighbor_indices": bank["support_neighbor_indices"],
                       "candidate_has_support": bank["candidate_has_support"]}
            for key, value in current.items():
                parts.setdefault(key, []).append(value)
        return {key: torch.cat(values, dim=1) for key, values in parts.items()}
