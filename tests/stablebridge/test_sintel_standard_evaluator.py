from pathlib import Path
import struct

import numpy as np
from PIL import Image

from scripts.evaluate_sintel_standard import (
    _read_flo,
    discover_sintel_training_pairs,
)


def _write_flow(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = value.shape
    assert channels == 2
    with path.open("wb") as stream:
        stream.write(b"PIEH")
        stream.write(struct.pack("<ii", width, height))
        stream.write(value.astype("<f4").tobytes())


def test_standard_sintel_inventory_keeps_clean_final_identity(tmp_path):
    flow = np.zeros((3, 4, 2), dtype=np.float32)
    flow[..., 0] = 1.25
    for render_pass in ("clean", "final"):
        scene = tmp_path / "training" / render_pass / "alley"
        scene.mkdir(parents=True)
        for index in (1, 2, 3):
            Image.new("RGB", (4, 3), color=(index, 2, 3)).save(
                scene / f"frame_{index:04d}.png"
            )
    for index in (1, 2):
        _write_flow(
            tmp_path / "training" / "flow" / "alley" / f"frame_{index:04d}.flo",
            flow,
        )

    inventory = discover_sintel_training_pairs(tmp_path)

    assert tuple(inventory) == ("clean", "final")
    assert len(inventory["clean"]) == len(inventory["final"]) == 2
    assert inventory["clean"][0].scene == "alley"
    assert np.array_equal(_read_flo(inventory["clean"][0].flow), flow)


def test_standard_sintel_inventory_rejects_render_pass_drift(tmp_path):
    for render_pass, count in (("clean", 3), ("final", 2)):
        scene = tmp_path / "training" / render_pass / "alley"
        scene.mkdir(parents=True)
        for index in range(1, count + 1):
            Image.new("RGB", (2, 2)).save(scene / f"frame_{index:04d}.png")
    flow = np.zeros((2, 2, 2), dtype=np.float32)
    for index in (1, 2):
        _write_flow(
            tmp_path / "training" / "flow" / "alley" / f"frame_{index:04d}.flo",
            flow,
        )

    try:
        discover_sintel_training_pairs(tmp_path)
    except ValueError as error:
        assert "identities differ" in str(error)
    else:
        raise AssertionError("render-pass drift was accepted")
