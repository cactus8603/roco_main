"""CPU integration coverage for a real repair model and portable feature data."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from waft_repair.checkpoint import cpu_tree, load_model, read_checkpoint, state_digest
from waft_repair.data import (CHEAP_METHOD, compute_sigma, load_manifest, load_packet,
                             import_dense_cache, load_sample, make_demo, prepare_manifest, sha256)
from waft_repair.training import epoch_order, feature_mse, infer_features, train


@pytest.fixture(autouse=True)
def small_cpu_thread_count():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_normalization_is_original_native_rms_and_loss_covers_padding():
    original = torch.arange(1, 1 + 2 * 64 * 16, dtype=torch.float32).reshape(1, 2, 64, 4, 4)
    sigma, mask = compute_sigma(original, [6, 6], [1, 1, 1, 1])
    # Independent indexing of native centers (indices 1..3 on each axis).
    expected = original[0, :, :, 1:, 1:].double().square().mean(dim=(0, 2, 3)).sqrt()
    expected = expected.clamp_min(0.01 * expected.sort().values[31] + 1e-8).float()
    assert torch.equal(sigma, expected)
    target = original.clone()
    target[:, :, :, 0, 0] += sigma[None, None, :]
    loss = feature_mse(original, target, sigma, mask)
    assert float(loss) == pytest.approx(1 / 16, rel=1e-4)
    assert not mask[0, 0, 0, 0, 0]


def test_npz_schema_and_bound_file_change_are_checked(tmp_path):
    manifest_path = make_demo(tmp_path / "data")
    spec_path = manifest_path.parent / "samples.json"
    spec = json.loads(spec_path.read_text())
    packet = load_packet(manifest_path.parent / "example-1-input.pt")
    tensor = packet.pop("features")
    np.savez(manifest_path.parent / "input.npz", features=tensor.numpy(), metadata=json.dumps(packet))
    spec["samples"][0]["input_file"] = "input.npz"
    spec_path.write_text(json.dumps(spec))
    prepared = tmp_path / "bound.json"
    prepare_manifest(spec_path, prepared)
    value = load_sample(prepared, 0)
    assert torch.equal(value["original"], tensor)
    path = manifest_path.parent / "input.npz"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_sample(prepared, 0)


def test_pair_checkpoint_kind_and_second_frame_contract(tmp_path):
    manifest_path = make_demo(tmp_path / "data")
    spec_path = manifest_path.parent / "samples.json"
    spec = json.loads(spec_path.read_text())
    target_path = manifest_path.parent / "example-1-target.pt"
    packet = cpu_tree(load_packet(target_path))
    packet["pair_id"] = "different-pair"
    torch.save(packet, target_path)
    with pytest.raises(ValueError, match="identity"):
        prepare_manifest(spec_path, tmp_path / "wrong-pair.json")
    packet["pair_id"] = "example-1"
    packet["waft_checkpoint_sha256"] = "1" * 64
    torch.save(packet, target_path)
    with pytest.raises(ValueError, match="checkpoint"):
        prepare_manifest(spec_path, tmp_path / "wrong-checkpoint.json")
    packet["waft_checkpoint_sha256"] = "0" * 64
    packet["target_kind"] = "expensive"
    torch.save(packet, target_path)
    with pytest.raises(ValueError, match="kind"):
        prepare_manifest(spec_path, tmp_path / "wrong-kind.json")
    packet["target_kind"] = "cheap"
    packet["teacher_features"][:, 1, :, 0, 0] += 1
    torch.save(packet, target_path)
    with pytest.raises(ValueError, match="second-frame"):
        prepare_manifest(spec_path, tmp_path / "wrong-second-frame.json")


def test_final_cheap_target_requires_complete_paired_clean_provenance(tmp_path):
    manifest_path = make_demo(tmp_path / "data")
    spec_path = manifest_path.parent / "samples.json"
    spec = json.loads(spec_path.read_text())
    row = spec["samples"][0]
    spec["samples"] = [row]
    row["domain"] = "sintel-final"
    input_path = manifest_path.parent / row["input_file"]
    original = cpu_tree(load_packet(input_path))
    original["domain"] = "sintel-final"
    torch.save(original, input_path)
    target_path = manifest_path.parent / row["target_file"]
    target = cpu_tree(load_packet(target_path))
    target["domain"] = "sintel-final"
    target["target_source_pair_id"] = "clean-pair"
    target["teacher_domain"] = "sintel_clean"
    target["teacher_features"][:, 1] += 0.3  # complete paired Clean F2 can differ from Final F2.
    torch.save(target, target_path)
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="target_source_pair_id"):
        prepare_manifest(spec_path, tmp_path / "missing-source.json")
    row["target_source_pair_id"] = "clean-pair"
    spec_path.write_text(json.dumps(spec))
    prepared = tmp_path / "paired.json"
    prepare_manifest(spec_path, prepared)
    assert not torch.equal(load_sample(prepared, 0)["original"][:, 1], load_sample(prepared, 0)["target"][:, 1])


def test_real_model_train_resume_reload_and_target_free_prediction(tmp_path):
    manifest_path = make_demo(tmp_path / "data")
    uninterrupted = train(manifest_path, tmp_path / "uninterrupted", epochs=2, seed=13)
    paused = train(manifest_path, tmp_path / "resumed", epochs=2, seed=13, max_updates=3)
    assert paused["status"] == "paused_at_update_limit"
    resumed = train(manifest_path, tmp_path / "resumed", epochs=2, seed=13,
                    resume=tmp_path / "resumed/checkpoint.pt")
    full_checkpoint = read_checkpoint(uninterrupted["checkpoint"])
    resumed_checkpoint = read_checkpoint(resumed["checkpoint"])
    assert uninterrupted["updates_completed"] == resumed["updates_completed"] == 4
    assert full_checkpoint["history"] == resumed_checkpoint["history"]
    assert full_checkpoint["model_state_sha256"] == resumed_checkpoint["model_state_sha256"]
    for key, value in full_checkpoint["model_state"].items():
        assert torch.equal(value.view(torch.uint8), resumed_checkpoint["model_state"][key].view(torch.uint8))
    model, _ = load_model(resumed["checkpoint"], backend="reference")
    assert sum(p.numel() for p in model.parameters()) == 3_923_200
    assert len(model.state_dict()) == 90
    sample = load_sample(manifest_path, 0)
    with torch.no_grad():
        expected = model(sample["original"], sample["sigma"], sample["native_mask"])
    assert not torch.equal(expected, sample["original"])
    # Target files are absent during inference; only the original field and metadata remain.
    for path in manifest_path.parent.glob("*-target.pt"):
        path.unlink()
    output = tmp_path / "prediction.pt"
    result = infer_features(resumed["checkpoint"], manifest_path.parent / "example-1-input.pt", output,
                            backend="reference")
    assert result["targets_or_gt_used"] is False
    assert torch.equal(load_packet(output)["features"], expected)
    with pytest.raises(ValueError, match="already exists"):
        infer_features(resumed["checkpoint"], manifest_path.parent / "example-1-input.pt", output)
    # A corrupted checkpoint must fail before prediction.
    corrupted = dict(resumed_checkpoint)
    corrupted["model_state"]["readout.bias"] = corrupted["model_state"]["readout.bias"].clone() + 1
    torch.save(corrupted, tmp_path / "corrupted.pt")
    with pytest.raises(ValueError, match="weights changed"):
        read_checkpoint(tmp_path / "corrupted.pt")


def test_natural_epoch_order_is_reproducible_and_complete():
    order = epoch_order(160, 0, 17)
    assert sorted(order) == list(range(160))
    assert order == epoch_order(160, 0, 17)
    assert order != epoch_order(160, 1, 17)


def test_expensive_adapter_uses_original_and_target_keys_in_one_accepted_packet(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    entries = tmp_path / "entries"
    entries.mkdir()
    original = torch.randn(1, 2, 64, 4, 4, generator=torch.Generator().manual_seed(2))
    sigma, mask = compute_sigma(original, [6, 6], [1, 1, 1, 1])
    method = dict(maximum_updates=512, fixed_total_updates=512, normalized_delta_bound=0.25,
                  learning_rate=0.001, betas=[0.9, 0.999], epsilon=1e-8, weight_decay=0.0,
                  clip_norm=1.0, clip_precision="FP64", warmup_updates=0, stop_geometry=False,
                  objective="FP32_mean_final_fifth_flow_vector_EPE_on_fixed_finite_valid_GT",
                  projection="after_AdamW_hardclamp_delta_to_plusminus0.25_then_zero_padding",
                  selection="first_strict_minimum_search_EPE_including_original_F0")
    packet = dict(case_id="accepted-pair", domain="synthetic", original=original,
                  teacher_features=original + 0.1, sigma=sigma, native_mask=mask,
                  native_hw=[6, 6], padding=[1, 1, 1, 1], quality={"passed": True, "selected_step": 23},
                  provenance={"checkpoint": {"sha256": "1" * 64}, "fixed_method": method})
    packet_path = data_dir / "features.pt"
    torch.save(packet, packet_path)
    audit_path = data_dir / "audit.json"
    audit_path.write_text('{"independently_verified":true}')
    def descriptor(path):
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    (entries / "accepted").mkdir()
    entry = {"case_id": "accepted-pair", "domain": "synthetic", "qualified": True,
             "teacher_updates": 512, "selected_step": 23, "packet": descriptor(packet_path), "audit": descriptor(audit_path)}
    (entries / "accepted/entry.json").write_text(json.dumps(entry))
    (entries / "unfinished").mkdir()
    (entries / "unfinished/entry.json").write_text('{"qualified":false}')
    output = tmp_path / "expensive.json"
    imported = import_dense_cache(entries, output)
    assert len(imported["samples"]) == 1
    row = imported["samples"][0]
    assert row["input_file"] == row["target_file"]
    assert row["input_key"] == "original"
    sample = load_sample(output, 0)
    assert torch.equal(sample["original"], original)
    assert torch.equal(sample["target"], original + 0.1)
    assert imported["target_kind"] == "expensive"
