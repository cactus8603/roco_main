"""Rebuild auditable reports from recorded per-query metrics, without rerunning.

CLI: ``python -m stablebridge.reporting --run-dir RUN``. Input rows contain
task, scene, frame, profile, arm, split, metrics=evaluate_pair(...), and optional
costs/selected_pixels/accepted_pixels. Oracle arms are labelled diagnostics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import os

from .evaluation import aggregate_scene_macro


REPORT_VERSION = "stablebridge_recorded_scene_macro_v1"
METRICS = ("baseline_error_px", "error_px", "gain_px", "benefit_px", "harm_px",
           "over_1px_pct", "severe_harm_pct", "repaired_to_threshold_pct")
REQUIRED_REGIONS = ("all", "hard", "initial_good")


def _reject_constant(value):
    raise ValueError(f"Non-standard JSON constant {value}; failures must use null plus status")


def _read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=_reject_constant)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_text(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_rows(path):
    rows, identities = [], set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_constant)
            key = tuple(str(row[name]) for name in ("task", "scene", "frame", "profile", "arm", "split"))
            if key in identities:
                raise ValueError(f"Duplicate query-arm identity {key}; refusing double counting")
            if row["task"] not in ("stereo", "flow"):
                raise ValueError("Unknown task; flow and stereo must stay separate")
            if not isinstance(row["metrics"]["regions"], dict):
                raise ValueError("metrics must contain evaluate_pair regions")
            for region in row["metrics"]["regions"].values():
                if not isinstance(region["pixels"], int) or region["pixels"] < 0:
                    raise ValueError("Region pixels must be a nonnegative integer")
            identities.add(key)
            rows.append(row)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid metrics row {number}: {error}") from error
    return rows


def _scene_macro_scalar(rows, getter):
    scenes = defaultdict(list)
    for row in rows:
        scenes[str(row["scene"])].append(getter(row))
    results = {}
    for scene, values in sorted(scenes.items()):
        missing = sum(value is None or not math.isfinite(value) for value in values)
        results[scene] = {"value": None if missing else sum(values)/len(values),
                           "cases": len(values), "missing_cases": missing}
    complete = bool(results) and all(result["value"] is not None for result in results.values())
    return {"scene_macro": sum(result["value"] for result in results.values())/len(results) if complete else None,
            "scenes": results, "missing_cases": sum(result["missing_cases"] for result in results.values())}


def _coverage(row, name):
    all_pixels = row["metrics"]["regions"].get("all", {}).get("pixels", 0)
    if all_pixels == 0:
        return None
    pixels = row.get(name+"_pixels")
    if pixels is None:
        pixels = row["metrics"]["regions"].get(name, {}).get("pixels")
    if pixels is None:
        return None
    if not isinstance(pixels, (int,float)) or not 0 <= pixels <= all_pixels:
        raise ValueError(f"{name}_pixels must be a count within the valid GT denominator")
    return 100*float(pixels)/all_pixels


def _costs(rows):
    keys = sorted({key for row in rows for key,value in row.get("costs", {}).items()
                   if isinstance(value,(int,float)) and not isinstance(value,bool)})
    result = {}
    for key in keys:
        def value(row):
            item = row.get("costs", {}).get(key)
            return float(item) if isinstance(item,(int,float)) and math.isfinite(item) else None
        values = [value(row) for row in rows]
        missing = sum(item is None for item in values)
        result[key] = {"sum_recorded": sum(item for item in values if item is not None),
                       "missing_rows": missing, **_scene_macro_scalar(rows,value)}
    return result


def _summarize(rows):
    task = rows[0]["task"]
    output = {"rows":len(rows), "scenes":sorted({str(row["scene"]) for row in rows}),
              "regions":{}, "costs":_costs(rows),
              "selected_coverage_pct":_scene_macro_scalar(rows,lambda row:_coverage(row,"selected")),
              "accepted_coverage_pct":_scene_macro_scalar(rows,lambda row:_coverage(row,"accepted"))}
    regions = sorted(set(REQUIRED_REGIONS) | {name for row in rows for name in row["metrics"]["regions"]})
    for region in regions:
        missing = sum(region not in row["metrics"]["regions"] for row in rows)
        if missing:
            output["regions"][region] = {"status":"missing_region", "missing_rows":missing}
            continue
        values = [row["metrics"]["regions"][region] for row in rows]
        result = {"pixels_recorded":sum(value["pixels"] for value in values),
                  "empty_rows":sum(value["pixels"]==0 for value in values),
                  "failed_rows":sum(value.get("status")=="failed_prediction" for value in values),
                  "nonfinite_prediction_pixels":sum(value.get("nonfinite_prediction_pixels",0) for value in values),
                  "nonfinite_baseline_pixels":sum(value.get("nonfinite_baseline_pixels",0) for value in values),
                  "metrics":{}}
        for metric in METRICS:
            result["metrics"][metric] = aggregate_scene_macro(rows,metric=metric,region=region,value_key="metrics")[task]
        result["status"] = "failed_prediction" if result["failed_rows"] else ("empty" if not result["pixels_recorded"] else "ok")
        output["regions"][region] = result
    return output


def _classification(rows):
    scopes = sorted({str(row.get("oracle_scope")) for row in rows if row.get("oracle_scope")})
    is_oracle = bool(scopes) or any("oracle" in str(row["arm"]).lower() for row in rows)
    return {"role":"candidate_oracle_diagnostic" if is_oracle else "actual_output",
            "oracle_scopes":scopes,
            "oracle_scope_status":"recorded" if scopes else ("unspecified_diagnostic_scope" if is_oracle else "not_applicable")}


def build_report(rows, manifest, *, source_paths=None, run_status=None):
    """Pure aggregation helper. Never average tasks, arms, or data splits."""
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["split"]),str(row["task"]),str(row["arm"]))].append(row)
    results = {}
    for (split,task,arm), group in sorted(groups.items()):
        summary = _summarize(group)
        summary.update(_classification(group))
        summary["per_profile"] = {profile:_summarize([row for row in group if str(row["profile"])==profile])
                                  for profile in sorted({str(row["profile"]) for row in group})}
        results.setdefault(split,{}).setdefault(task,{})[arm] = summary
    expected = manifest.get("expected_metric_rows")
    completeness = "recorded_rows_only"
    if isinstance(expected,int):
        completeness = "complete_by_row_count" if expected==len(rows) else "row_count_mismatch"
    limits = ["Only recorded query outputs are scored; missing or not-yet-run queries are not inferred.",
              "Stereo and flow are separate metrics and are never averaged together.",
              "Nonfinite predictions stay in the GT denominator and invalidate the affected mean.",
              "Candidate oracle diagnostics are privileged selections from generated candidates, not deployable results or global upper bounds.",
              "Controlled RGB profiles are not official RobustSpring renderer results.",
              "Task fine-tuning and broader pretraining exposure are different; absence of a declared exposure audit is not evidence of unseen data."]
    declared = manifest.get("limitations",[])
    if isinstance(declared,str):
        limits.append(declared)
    elif isinstance(declared,list):
        limits.extend(str(value) for value in declared)
    provenance_keys = ("dataset", "dataset_version", "data", "model", "models", "model_role", "model_artifacts", "model_profile",
                       "exposure", "exposure_audit", "data_scope", "scientific_scope", "gt_status",
                       "source", "config", "effective_config", "study", "study_id", "run_id",
                       "config_sha256", "clip_manifest_sha256", "hard_mask", "runtime_query_mask",
                       "oracle", "corruption_claim", "gt_for_evaluation_only", "scorer_version")
    return {"report_version":REPORT_VERSION, "metric_rows":len(rows), "results":results,
            "aggregation":{"within_scene":"equal_registered_query_rows",
                           "between_scene":"equal_nonempty_scenes",
                           "empty_region":"explicitly_counted_excluded_only_when_zero_pixels",
                           "failure":"nonempty_failed_case_invalidates_mean",
                           "tasks_and_splits":"reported_separately"},
            "completeness":completeness,"expected_metric_rows":expected,"run_status":run_status,
            "source_paths":source_paths or {},
            "provenance":{key:manifest[key] for key in provenance_keys if key in manifest},
            "exposure_status":"declared_in_manifest" if any(key in manifest for key in ("exposure","exposure_audit")) else "not_explicitly_declared_at_manifest_top_level",
            "gt_status":manifest.get("gt_status","scoring_only_required_by_contract; inspect input provenance"),
            "cost_scope":"recorded per-output costs; shared or cached computation may appear in multiple arm rows; these sums are not automatically total physical runtime",
            "limitations":limits}


def _metric(summary, region, metric):
    return summary.get("regions",{}).get(region,{}).get("metrics",{}).get(metric,{}).get("scene_macro")


def _number(value):
    return "—" if value is None else f"{value:.5f}"


def markdown_report(report):
    lines = ["# StableBridge recorded-run report", "", f"Metric rows: {report['metric_rows']}. Completeness: `{report['completeness']}`.", "",
             "Scene macro: equal query rows within each scene, then equal scenes. Error and signed gain are in native pixels; positive gain means improvement.", "",
             "Empty regions are counted explicitly. A failed nonempty prediction does not disappear from the mean; `—` can indicate failure or missing data and must be read with the JSON status."]
    for split,tasks in report["results"].items():
        lines.extend(["",f"## Split: {split}"])
        for task,arms in tasks.items():
            lines.extend(["",f"### {task}","",
                          "| Arm | Role | All error ↓ | H* error ↓ | Initial-good error ↓ | All gain ↑ | All harm ↓ | Severe harm % ↓ | Accepted % | Failed rows |",
                          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
            for arm,summary in arms.items():
                values = [_metric(summary,"all","error_px"),_metric(summary,"hard","error_px"),
                          _metric(summary,"initial_good","error_px"),_metric(summary,"all","gain_px"),
                          _metric(summary,"all","harm_px"),_metric(summary,"all","severe_harm_pct"),
                          summary["accepted_coverage_pct"]["scene_macro"]]
                failed = summary["regions"].get("all",{}).get("failed_rows","unknown")
                lines.append(f"| {arm} | {summary['role']} | "+" | ".join(_number(value) for value in values)+f" | {failed} |")
            lines.extend(["","Per-profile values:","","| Arm | Profile | All error ↓ | H* error ↓ | All gain ↑ |","|---|---|---:|---:|---:|"])
            for arm,summary in arms.items():
                for profile,values in summary["per_profile"].items():
                    entries = [_metric(values,"all","error_px"),_metric(values,"hard","error_px"),_metric(values,"all","gain_px")]
                    lines.append(f"| {arm} | {profile} | "+" | ".join(_number(value) for value in entries)+" |")
    lines.extend(["","## Provenance and limits","",f"GT status: {report['gt_status']}","",f"Exposure status: {report['exposure_status']}","",
                  f"Cost scope: {report['cost_scope']}",""])
    lines.extend("- "+limit for limit in report["limitations"])
    lines.extend(["","Full scene, cohort, failure, cost, and source details: `report.json`.",""])
    return "\n".join(lines)


def write_report(run_dir):
    """Write report.json and REPORT.md and return the full structured report."""
    run_dir = Path(run_dir).resolve()
    manifest_path,metrics_path = run_dir/"manifest.json",run_dir/"metrics.jsonl"
    manifest,rows = _read_json(manifest_path),load_rows(metrics_path)
    status_path = run_dir/"status.json"
    report = build_report(rows,manifest,run_status=_read_json(status_path) if status_path.exists() else None,
                          source_paths={"manifest":str(manifest_path),"manifest_sha256":_digest(manifest_path),
                                        "metrics":str(metrics_path),"metrics_sha256":_digest(metrics_path)})
    _atomic_text(run_dir/"report.json",json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    _atomic_text(run_dir/"REPORT.md",markdown_report(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True,type=Path)
    args = parser.parse_args(argv)
    report = write_report(args.run_dir)
    print(json.dumps({"report":str(args.run_dir.resolve()/"report.json"),
                      "metric_rows":report["metric_rows"],"completeness":report["completeness"]}))


if __name__ == "__main__":
    main()
