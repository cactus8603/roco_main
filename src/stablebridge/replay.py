"""Frozen-candidate utility replay: query-level diagnostics, no new CroCo passes.

Decisions consume saved observable candidate features only. Saved GT errors are
read by a separate scoring step after the decision. This does not re-run memory,
generate candidates under the learned policy, or measure whole-image accuracy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .artifacts import sha256_file
from .learning import FEATURE_SCHEMA, load_acceptor, select_queries
from .util import save_json


_SPLITS = {"fit": "fit", "calibration": "calibration", "calib": "calibration",
           "test": "evaluation", "eval": "evaluation", "dev_eval": "evaluation",
           "development_evaluation": "evaluation", "evaluation": "evaluation"}


def decision_view(feature_records):
    """Build the selection view with no labels; query-gain field is a zero dummy."""
    features = np.asarray(feature_records["features"], np.float32)
    if features.ndim != 2 or features.shape[1] != 12 or not len(features) or not np.isfinite(features).all():
        raise ValueError("features must be nonempty finite [M,12]")
    size = len(features)
    data = {"features": features, "gain": np.zeros(size, np.float32)}
    for name in ("query_id", "scene", "split"):
        array = np.asarray(feature_records[name])
        if array.shape != (size,) or array.dtype.kind not in "USiu":
            raise ValueError(f"{name} must be a string/integer vector aligned with features")
        data[name] = array.astype(str)
    unknown = set(data["split"]) - set(_SPLITS)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    data["split"] = np.asarray([_SPLITS[s] for s in data["split"]])
    indices = np.asarray(feature_records["candidate_index"])
    if indices.shape != (size,) or indices.dtype.kind not in "iu" or np.any(indices < 0):
        raise ValueError("Replay requires explicit nonnegative candidate_index [M]")
    data["candidate_index"] = indices.astype(np.int64)
    queries, inverse = np.unique(data["query_id"], return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1)
    identities = []
    for name, rows in zip(queries, groups):
        if len(np.unique(data["scene"][rows])) != 1 or len(np.unique(data["split"][rows])) != 1:
            raise ValueError(f"Query crosses scene/split boundaries: {name}")
        if len(np.unique(indices[rows])) != len(rows):
            raise ValueError(f"Duplicate candidate indices: {name}")
        roots = rows[indices[rows] == 0]
        if len(roots) != 1 or abs(float(features[roots[0], 5])) > 1e-6:
            raise ValueError(f"Query requires exactly one zero-update identity: {name}")
        if np.any(features[rows, 9] != features[roots[0], 8]):
            raise ValueError("Baseline-support flags disagree with identity")
        if np.any(features[rows, 10] != features[roots[0], 10]):
            raise ValueError("Candidates disagree on task")
        identities.append(roots[0])
    for scene in np.unique(data["scene"]):
        if len(np.unique(data["split"][data["scene"] == scene])) != 1:
            raise ValueError(f"Scene crosses split boundaries: {scene}")
    for column in (8, 9, 10, 11):
        if not np.all(np.isin(features[:, column], [0, 1])):
            raise ValueError("Candidate support/task/history flags must be binary")
    if np.any(features[:, 5] < 0):
        raise ValueError("Candidate update lengths cannot be negative")
    data["groups_rows"], data["identity_rows"] = groups, np.asarray(identities, np.int64)
    return data


def decide_candidates(feature_records, acceptor):
    """Return one actual decision per query, without receiving error labels.

    Pass a mapping containing features, candidate_index, query_id, scene, split.
    Any additional keys (including GT errors) are ignored, never inspected.
    """
    data = decision_view(feature_records)
    gain, harm = acceptor.predict(data["features"])
    decisions = []
    for split in ("fit", "calibration", "evaluation"):
        selected = select_queries(data, gain, harm, acceptor.config, split=split)
        accepted = (selected["eligible_nonidentity"]
                    & (selected["predicted_gain"] > acceptor.gain_threshold)
                    & (selected["predicted_harm"] <= acceptor.harm_threshold))
        for index in range(len(accepted)):
            winner, identity = int(selected["row"][index]), int(selected["identity_row"][index])
            decisions.append({"query_id": str(selected["query_id"][index]),
                              "scene": str(selected["scene"][index]), "split": split,
                              "selected_row": winner, "identity_row": identity,
                              "output_row": winner if accepted[index] else identity,
                              "selected_candidate_index": int(data["candidate_index"][winner]),
                              "output_candidate_index": int(data["candidate_index"][winner]) if accepted[index] else 0,
                              "accepted": bool(accepted[index]),
                              "predicted_gain_px": float(selected["predicted_gain"][index]),
                              "predicted_harm_probability": float(selected["predicted_harm"][index]),
                              "eligible_nonidentity": bool(selected["eligible_nonidentity"][index])})
    return decisions


def _identity(query_id, scene):
    parts = query_id.split("/")
    if len(parts) != 4:
        raise ValueError(f"Expected case_id/arm/x/y query identity: {query_id}")
    case_id, arm, x, y = parts
    case_parts = case_id.split("_", 3)
    if len(case_parts) != 4 or case_parts[0] not in ("stereo", "flow") or case_parts[1] != scene:
        raise ValueError(f"Malformed/mismatched task-scene case identity: {query_id}")
    task, _, frame, profile = case_parts
    try:
        return {"case_id": case_id, "task": task, "arm": arm, "frame": int(frame),
                "profile": profile, "x": int(x), "y": int(y)}
    except ValueError as error:
        raise ValueError(f"Query frame/pixel coordinates must be integers: {query_id}") from error


def score_decisions(decisions, baseline_error, candidate_error):
    """Attach metrics after selection; invalid GT queries remain in decision logs."""
    baseline = np.asarray(baseline_error, np.float64)
    candidate = np.asarray(candidate_error, np.float64)
    if baseline.ndim != 1 or candidate.shape != baseline.shape:
        raise ValueError("Saved baseline and candidate errors must be aligned vectors")
    result = []
    for decision in decisions:
        root, output, chosen = decision["identity_row"], decision["output_row"], decision["selected_row"]
        if max(root, output, chosen) >= len(baseline):
            raise ValueError("Decision row exceeds the saved label array")
        before, after, identity_error = baseline[root], candidate[output], candidate[root]
        valid = bool(np.isfinite(before) and np.isfinite(after) and np.isfinite(identity_error)
                     and before >= 0 and after >= 0 and identity_error >= 0)
        if np.isfinite(before) and np.isfinite(identity_error) and abs(before - identity_error) > 1e-6:
            raise ValueError("Saved identity error differs from baseline; refusing misaligned replay")
        gain = float(before - after) if valid else None
        evaluation = {"valid_gt": valid, "baseline_error_px": float(before) if valid else None,
                      "output_error_px": float(after) if valid else None, "signed_gain_px": gain,
                      "benefit_px": max(gain, 0) if valid else None,
                      "harm_px": max(-gain, 0) if valid else None,
                      "severe_harm_over_1px": bool(gain < -1) if valid else None}
        result.append({**decision, **_identity(decision["query_id"], decision["scene"]), "evaluation": evaluation})
    return result


def _metrics(rows):
    valid = [row for row in rows if row["evaluation"]["valid_gt"]]
    accepted_valid = [row for row in valid if row["accepted"]]
    result = {"queries": len(rows), "valid_gt_queries": len(valid), "invalid_gt_queries": len(rows) - len(valid),
              "accepted_queries": sum(row["accepted"] for row in rows),
              "accepted_valid_gt_queries": len(accepted_valid),
              "acceptance_pct": 100 * sum(row["accepted"] for row in rows) / len(rows) if rows else None}
    for name in ("baseline_error_px", "output_error_px", "signed_gain_px", "benefit_px", "harm_px"):
        result[name] = float(np.mean([row["evaluation"][name] for row in valid])) if valid else None
    result["severe_harm_pct_valid_queries"] = (100 * float(np.mean([row["evaluation"]["severe_harm_over_1px"] for row in valid]))
                                               if valid else None)
    result["severe_harm_pct_accepted_valid_queries"] = (100 * float(np.mean([row["evaluation"]["severe_harm_over_1px"] for row in accepted_valid]))
                                                        if accepted_valid else None)
    return result


def summarize(decisions):
    groups = defaultdict(list)
    for row in decisions:
        groups[(row["split"], row["task"], row["arm"])].append(row)
    summary = []
    mean_names = ("baseline_error_px", "output_error_px", "signed_gain_px", "benefit_px", "harm_px",
                  "acceptance_pct", "severe_harm_pct_valid_queries", "severe_harm_pct_accepted_valid_queries")
    for (split, task, arm), rows in sorted(groups.items()):
        scenes = {scene: _metrics([row for row in rows if row["scene"] == scene])
                  for scene in sorted({row["scene"] for row in rows})}
        macro = {}
        for name in mean_names:
            values = [scene[name] for scene in scenes.values() if scene[name] is not None]
            macro[name] = float(np.mean(values)) if values else None
        summary.append({"split": split, "task": task, "arm": arm,
                        "scene_count": len(scenes), "scene_macro": macro, "per_scene": scenes,
                        "query_pooled": _metrics(rows),
                        "aggregation": "equal_query_mean_within_scene_then_equal_scene_mean; groups never mix splits/tasks/arms"})
    return summary


def _markdown(summary, manifest):
    lines = ["# Learned acceptor：固定候選重播", "",
             "此結果只評估原先選出的 query，不是全圖成績。候選與記憶狀態固定，沒有重跑 CroCo，也沒有驗證 learned policy 改變後的閉環行為。",
             "", "決策只讀保存的可觀測特徵；GT errors 在選定輸出後才用來評分。fit／calibration 是開發診斷，evaluation 分開列出。",
             "", "| Split | Task | Arm | Scenes | Queries（有效 GT） | 原始誤差 | 輸出誤差 | Signed gain ↑ | Harm ↓ | 接受率 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def number(value):
        return "—" if value is None else f"{value:.5f}"
    for row in summary:
        macro, pooled = row["scene_macro"], row["query_pooled"]
        lines.append(f"| {row['split']} | {row['task']} | {row['arm']} | {row['scene_count']} | "
                     f"{pooled['queries']}（{pooled['valid_gt_queries']}） | {number(macro['baseline_error_px'])} | "
                     f"{number(macro['output_error_px'])} | {number(macro['signed_gain_px'])} | "
                     f"{number(macro['harm_px'])} | {number(macro['acceptance_pct'])}% |")
    lines.extend(["", "誤差／gain／harm 為原始像素單位，先在每個 scene 的有效 GT queries 平均，再等權平均 scenes。"
                  "接受率涵蓋所有 queries；沒有有效 GT 的 query 仍保留決策，但不納入誤差平均。不同任務不能直接合併誤差。",
                  "", "拒絕更新的 query 保留 baseline，仍在分母內。Harm = max(輸出誤差 − 原始誤差, 0)。"
                  "對照相同 policy 的全圖與時序閉環結果，必須另跑完整 pipeline。",
                  "", f"Model SHA256：`{manifest['acceptor']['sha256']}`。輸入與程式雜湊見 `manifest.json`；"
                  "每個 query 的候選選擇、接受判斷與評分見 `decisions.jsonl`。", ""])
    return "\n".join(lines)


def replay_run(run_dir, acceptor_path, output_dir):
    run_dir, acceptor_path, output_dir = map(lambda p: Path(p).resolve(), (run_dir, acceptor_path, output_dir))
    files = sorted((run_dir / "supervision").glob("*.npz"))
    if not files:
        raise ValueError("No saved candidate supervision found")
    input_manifest = run_dir / "manifest.json"
    if not input_manifest.is_file():
        raise ValueError("Source run lacks its immutable manifest")
    original = json.loads(input_manifest.read_text())
    acceptor = load_acceptor(acceptor_path)
    matching = original.get("matching", {})
    for name in ("max_update_px", "require_baseline_support"):
        if name not in matching or matching[name] != getattr(acceptor, name):
            raise ValueError(f"Replay eligibility differs from the calibrated acceptor: {name}")
    manifest = {"schema_version": 1, "kind": "frozen_candidate_query_replay", "feature_schema": FEATURE_SCHEMA,
                "input_run": str(run_dir), "input_manifest_sha256": sha256_file(input_manifest),
                "input_files": [{"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in files],
                "acceptor": {"path": str(acceptor_path), "sha256": sha256_file(acceptor_path),
                             "sidecar_sha256": sha256_file(acceptor_path.with_suffix(acceptor_path.suffix + ".json")),
                             "policy": {"gain_threshold": acceptor.gain_threshold if np.isfinite(acceptor.gain_threshold) else None,
                                        "harm_threshold": acceptor.harm_threshold,
                                        "reject_all": not np.isfinite(acceptor.gain_threshold)}},
                "eligibility": {name: matching[name] for name in ("max_update_px", "require_baseline_support")},
                "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in ("replay.py", "learning.py")},
                "prediction_inputs": ["features", "candidate_index", "query_id", "scene", "split"],
                "gt_use": "post-decision_scoring_only", "whole_image_result": False,
                "closed_loop_result": False, "backbone_forward_calls": 0,
                "development_exposure": original.get("exposure", "source_run_exposure_not_specified")}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Replay input/model/source changed; choose a new output directory")
    if not manifest_path.exists():
        save_json(manifest_path, manifest)
    started = time.perf_counter()
    all_decisions, seen_queries, scene_splits = [], set(), {}
    inference_seconds = 0.0
    for path, registered in zip(files, manifest["input_files"]):
        with np.load(path, allow_pickle=False) as archive:
            observable = {name: archive[name] for name in manifest["prediction_inputs"]}
            start = time.perf_counter()
            decisions = decide_candidates(observable, acceptor)
            inference_seconds += time.perf_counter() - start
            # Neither of these label arrays is loaded until decisions exist.
            scored = score_decisions(decisions, archive["baseline_error"], archive["candidate_error"])
        if sha256_file(path) != registered["sha256"]:
            raise ValueError("Candidate input changed during replay")
        for row in scored:
            if row["query_id"] in seen_queries:
                raise ValueError(f"Duplicate query across supervision files: {row['query_id']}")
            seen_queries.add(row["query_id"])
            previous = scene_splits.setdefault(row["scene"], row["split"])
            if previous != row["split"]:
                raise ValueError(f"Scene leakage across source files: {row['scene']}")
            row["source_file"] = str(path)
        all_decisions.extend(scored)
    if sha256_file(acceptor_path) != manifest["acceptor"]["sha256"] or sha256_file(input_manifest) != manifest["input_manifest_sha256"]:
        raise ValueError("Model or source manifest changed during replay")
    summary = summarize(all_decisions)
    temporary = output_dir / "decisions.jsonl.tmp"
    with temporary.open("w") as stream:
        for row in all_decisions:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output_dir / "decisions.jsonl")
    result = {"kind": manifest["kind"], "whole_image_result": False, "closed_loop_result": False,
              "query_count": len(all_decisions), "candidate_files": len(files),
              "utility_inference_and_selection_seconds": inference_seconds,
              "elapsed_seconds": time.perf_counter() - started,
              "denominator": "all_saved_queries_for_acceptance; valid_GT_queries_for_error; rejected_queries_kept",
              "groups": summary}
    save_json(output_dir / "results.json", result)
    temporary = output_dir / "results.md.tmp"
    temporary.write_text(_markdown(summary, manifest))
    temporary.replace(output_dir / "results.md")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--acceptor", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    result = replay_run(args.run_dir, args.acceptor, args.output_dir)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), "query_count": result["query_count"],
                      "whole_image_result": False, "closed_loop_result": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
