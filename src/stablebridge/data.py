"""Read-only Spring provider and deterministic, geometry-preserving corruptions.

Inference reads ``read_pair``; labels can only be obtained through ``read_gt``
or the explicitly evaluation-only ``read_pair_with_gt`` convenience method.
Corruptions operate on full-resolution images before cropping. Nothing writes
to a dataset root. These are controlled probes, not RobustSpring renderers.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .geometry import PixelTransform
from .robust_proxy import apply_observation as apply_robust20_observation
from .robust_proxy import observation_maps, parse_profile, transform_vector_ground_truth


CORRUPTION_VERSION = "stablebridge_controlled_rgb_v1"
GT_OFFSETS = ((0, 0), (1, 0), (0, 1), (1, 1))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identifier(value):
    number = int(value)
    if number < 0:
        raise ValueError("Scene and frame identifiers must be nonnegative integers")
    return f"{number:04d}"


def _derived_seed(seed, scene, frame, view, profile, temporal_correlation, view_correlation):
    if temporal_correlation not in ("independent", "persistent"):
        raise ValueError("temporal_correlation must be independent or persistent")
    if view_correlation not in ("independent", "shared"):
        raise ValueError("view_correlation must be independent or shared")
    identity = {"seed": int(seed), "scene": _identifier(scene), "profile": profile,
                "frame": int(frame) if temporal_correlation == "independent" else None,
                "view": view if view_correlation == "independent" else None,
                "version": CORRUPTION_VERSION}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little"), identity


def apply_corruption(image, profile="clean", *, seed=0, scene="0000", frame=0,
                     view="left", options=None):
    """Return a new uint8 RGB image and its fully specified generation record.

    Gaussian noise defaults to sigma=.04 in [0,1], independent across time and
    views. persistent_overlay uses an opaque image-plane rectangle with fixed
    location/colour over time; views are independent unless explicitly shared.
    recovery requires an absolute recovery_frame; noise defaults to sigma=.16
    before that event and exactly clean thereafter. All profiles preserve GT.
    """
    options = dict(options or {})
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Corruption input must be uint8 HxWx3")
    if view not in ("left", "right"):
        raise ValueError("view must be left or right")
    allowed = {"clean", "gaussian_noise", "persistent_overlay", "recovery"}
    if profile not in allowed:
        raise ValueError(f"Unknown controlled profile: {profile}")
    accepted_options = {"temporal_correlation", "view_correlation", "sigma", "recovery_frame",
                        "heavy_sigma", "rectangle_xywh", "rectangle_fraction", "colour_rgb"}
    unknown = options.keys() - accepted_options
    if unknown:
        raise ValueError(f"Unknown corruption options: {sorted(unknown)}")
    temporal = options.get("temporal_correlation", "persistent" if profile == "persistent_overlay" else "independent")
    if profile == "persistent_overlay" and temporal != "persistent":
        raise ValueError("persistent_overlay requires persistent temporal correlation")
    across_views = options.get("view_correlation", "independent")
    rng_seed, identity = _derived_seed(seed, scene, frame, view, profile, temporal, across_views)
    rng = np.random.default_rng(rng_seed)
    output = image.copy()
    metadata = {"version": CORRUPTION_VERSION, "profile": profile, "seed_identity": identity,
                "derived_seed": rng_seed, "temporal_correlation": temporal,
                "view_correlation": across_views, "applied_before_crop": True,
                "geometry_preserved": True, "shape_hw": list(image.shape[:2])}
    if profile in ("gaussian_noise", "recovery"):
        if profile == "recovery":
            if "recovery_frame" not in options:
                raise ValueError("recovery requires a predeclared absolute recovery_frame")
            event = int(options["recovery_frame"])
            sigma = float(options.get("heavy_sigma", .16)) if int(frame) < event else 0.0
            metadata["recovery_frame"] = event
        else:
            sigma = float(options.get("sigma", .04))
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("Noise sigma must be finite and nonnegative")
        if sigma:
            output = np.rint(np.clip(image.astype(np.float32)/255 +
                                    rng.normal(0, sigma, image.shape), 0, 1)*255).astype(np.uint8)
        metadata["sigma_01"] = sigma
    elif profile == "persistent_overlay":
        h, w = image.shape[:2]
        fraction = float(options.get("rectangle_fraction", .25))
        if not 0 < fraction <= 1:
            raise ValueError("rectangle_fraction must lie in (0,1]")
        if "rectangle_xywh" in options:
            x, y, rw, rh = map(int, options["rectangle_xywh"])
        else:
            rw, rh = max(1, round(w*fraction)), max(1, round(h*fraction))
            x, y = int(rng.integers(0, w-rw+1)), int(rng.integers(0, h-rh+1))
        if min(x, y) < 0 or min(rw, rh) <= 0 or x+rw > w or y+rh > h:
            raise ValueError("Overlay rectangle must lie within the full-resolution image")
        colour = np.asarray(options.get("colour_rgb", rng.integers(0, 256, 3)), dtype=np.float64)
        if colour.shape != (3,) or not np.isfinite(colour).all() or (colour < 0).any() or (colour > 255).any():
            raise ValueError("colour_rgb must have three values in [0,255]")
        colour = colour.astype(np.uint8)
        output[y:y+rh, x:x+rw] = colour
        metadata.update({"rectangle_xywh": [x, y, rw, rh], "colour_rgb": colour.tolist(),
                         "attachment": "image_plane_not_scene_surface"})
    metadata["full_corrupted_rgb_sha256"] = hashlib.sha256(output.tobytes()).hexdigest()
    return output, metadata


@dataclass(frozen=True)
class SpringProvider:
    rgb_root: Path
    flow_root: Path
    disp_root: Path
    dataset_version: str = "spring_local_version_unverified"

    @classmethod
    def from_registry(cls, path):
        registry = json.loads(Path(path).read_text())
        config = registry["datasets"]["spring"]
        return cls(*(Path(config[key]) for key in ("rgb_root", "flow_root", "disp_root")),
                   dataset_version=config.get("dataset_version", "spring_local_version_unverified"))

    def image_path(self, scene, frame, view="left", split="train"):
        self._validate(view, split)
        modality = f"frame_{view}"
        return Path(self.rgb_root)/split/_identifier(scene)/modality/f"{modality}_{_identifier(frame)}.png"

    def gt_path(self, task, scene, frame, view="left", split="train"):
        self._validate(view, split)
        if split != "train":
            raise ValueError("Local Spring test split has no ground truth")
        if task == "stereo":
            root, modality, suffix = self.disp_root, f"disp1_{view}", ".dsp5"
        elif task == "flow":
            root, modality, suffix = self.flow_root, f"flow_FW_{view}", ".flo5"
        else:
            raise ValueError("task must be stereo or flow")
        return Path(root)/split/_identifier(scene)/modality/f"{modality}_{_identifier(frame)}{suffix}"

    @staticmethod
    def _validate(view, split):
        if view not in ("left", "right") or split not in ("train", "test"):
            raise ValueError("Expected left/right view and train/test split")

    def read_image(self, scene, frame, view="left", split="train"):
        with Image.open(self.image_path(scene, frame, view, split)) as image:
            return np.array(image.convert("RGB"), copy=True)

    def read_gt(self, task, scene, frame, view="left", split="train", roi_xyhw=None):
        """Read four subpixel GT samples [4,2,H,W] in NATIVE image-pixel units.

        ROI order is (x,y,height,width). GT has twice the spatial resolution;
        its displacement values are NOT divided by two. Stereo uses (-d,0)
        for left-to-right and (+d,0) for right-to-left.
        """
        path = self.gt_path(task, scene, frame, view, split)
        key = "disparity" if task == "stereo" else "flow"
        with h5py.File(path, "r") as file:
            field = file[key]
            expected_ndim = 2 if task == "stereo" else 3
            if field.ndim != expected_ndim or (task == "flow" and field.shape[2] != 2):
                raise ValueError(f"Unexpected Spring {key} dimensions: {field.shape}")
            gh, gw = field.shape[:2]
            if gh % 2 or gw % 2:
                raise ValueError("Spring GT dimensions must be exactly even")
            h, w = gh//2, gw//2
            with Image.open(self.image_path(scene, frame, view, split)) as image:
                if image.size != (w,h):
                    raise ValueError("Spring GT must be exactly twice the RGB spatial resolution")
            x, y, rh, rw = (0, 0, h, w) if roi_xyhw is None else tuple(map(int, roi_xyhw))
            self._validate_roi((x, y, rh, rw), (h, w))
            raw = np.asarray(field[2*y:2*(y+rh), 2*x:2*(x+rw)], dtype=np.float32)
        branches = []
        for dy, dx in GT_OFFSETS:
            value = raw[dy::2, dx::2]
            if task == "stereo":
                signed = -value if view == "left" else value
                branches.append(np.stack((signed, np.zeros_like(value))))
            else:
                branches.append(value.transpose(2, 0, 1))
        return np.stack(branches).astype(np.float32)

    @staticmethod
    def _validate_roi(roi, hw):
        x, y, h, w = roi
        if min(x, y) < 0 or min(h, w) < 1 or y+h > hw[0] or x+w > hw[1]:
            raise ValueError(f"ROI {roi} is outside image dimensions {hw}")

    def read_pair(self, task, scene, frame, context_hw, *, view="left", split="train",
                  profile="clean", seed=0, corruption_options=None, origin_xy=None):
        """Read inference inputs and provenance only; this method never opens GT.

        Both endpoints use the same spatial ROI. Flow's target is frame+1;
        stereo's target is the opposite view at the same frame. Full images
        are independently corrupted according to explicit correlation options.
        """
        if task not in ("stereo", "flow"):
            raise ValueError("task must be stereo or flow")
        target_frame = int(frame)+(task == "flow")
        target_view = ("right" if view == "left" else "left") if task == "stereo" else view
        images, records = [], []
        for current_frame, current_view in ((int(frame), view), (target_frame, target_view)):
            raw = self.read_image(scene, current_frame, current_view, split)
            if parse_profile(profile) is not None:
                if corruption_options:
                    raise ValueError("Robust20 profiles encode severity and accept no ad-hoc options")
                corrupted, record = apply_robust20_observation(
                    raw, profile=profile, seed=seed, scene=scene,
                    frame=current_frame, view=current_view)
            else:
                corrupted, record = apply_corruption(raw, profile, seed=seed, scene=scene,
                                                      frame=current_frame, view=current_view,
                                                      options=corruption_options)
            path = self.image_path(scene, current_frame, current_view, split)
            record.update({"path": str(path), "file_sha256": file_sha256(path),
                           "frame": current_frame, "view": current_view})
            images.append(corrupted)
            records.append(record)
        if images[0].shape != images[1].shape:
            raise ValueError("Pair endpoints must have equal full image dimensions")
        full_h, full_w = images[0].shape[:2]
        h, w = map(int, context_hw)
        x, y = ((full_w-w)//2, (full_h-h)//2) if origin_xy is None else tuple(map(int, origin_xy))
        roi = (x, y, h, w)
        self._validate_roi(roi, (full_h, full_w))
        crops = [np.ascontiguousarray(image[y:y+h, x:x+w]) for image in images]
        transform = PixelTransform(origin_xy=(x,y), native_hw=(h,w)).as_dict()
        metadata = {"dataset_version": self.dataset_version, "split": split,
                    "task": task, "scene": _identifier(scene), "source_frame": int(frame),
                    "target_frame": target_frame, "source_view": view, "target_view": target_view,
                    "information_cutoff_event": target_frame, "full_hw": [full_h, full_w],
                    "roi_xyhw": list(roi), "source_transform": transform,
                    "target_transform": dict(transform), "inputs": records,
                    "gt_read": False, "time_basis": "frame_index"}
        return {"image0": crops[0], "image1": crops[1], "origin_xy": (x,y),
                "roi_xyhw": roi, "metadata": metadata}

    def read_corrupted_gt(self, task, scene, frame, *, view="left", split="train",
                          roi_xyhw=None, profile="clean", seed=0):
        """Read task GT in the coordinate system of a possibly elastic pair."""
        parsed = parse_profile(profile)
        if parsed is None or parsed[0] != "elastic_transform":
            return self.read_gt(task, scene, frame, view=view, split=split, roi_xyhw=roi_xyhw)
        full = self.read_gt(task, scene, frame, view=view, split=split, roi_xyhw=None)
        h, w = full.shape[-2:]
        target_frame = int(frame) + (task == "flow")
        target_view = ("right" if view == "left" else "left") if task == "stereo" else view
        source_maps = observation_maps((h, w), profile=profile, seed=seed, scene=scene,
                                       frame=int(frame), view=view)
        target_maps = observation_maps((h, w), profile=profile, seed=seed, scene=scene,
                                       frame=target_frame, view=target_view)
        if source_maps is None or target_maps is None:
            raise RuntimeError("Elastic profile failed to produce geometry maps")
        transformed = transform_vector_ground_truth(full, source_maps, target_maps)
        if roi_xyhw is None:
            return transformed
        x, y, rh, rw = tuple(map(int, roi_xyhw))
        self._validate_roi((x, y, rh, rw), (h, w))
        return np.ascontiguousarray(transformed[:, :, y:y+rh, x:x+rw])

    def read_pair_with_gt(self, task, scene, frame, context_hw, **kwargs):
        """Evaluation convenience only; DO NOT pass this dict to a controller."""
        pair = self.read_pair(task, scene, frame, context_hw, **kwargs)
        pair["gt"] = self.read_corrupted_gt(
            task, scene, frame, view=kwargs.get("view", "left"),
            split=kwargs.get("split", "train"), roi_xyhw=pair["roi_xyhw"],
            profile=kwargs.get("profile", "clean"), seed=kwargs.get("seed", 0))
        pair["metadata"]["gt_read"] = True
        gt_path = self.gt_path(task, scene, frame, view=kwargs.get("view", "left"),
                               split=kwargs.get("split", "train"))
        pair["metadata"]["evaluation_gt"] = {"path": str(gt_path), "file_sha256": file_sha256(gt_path),
                                               "layout": "four_subpixels_native_vector_units"}
        return pair
