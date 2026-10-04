#!/usr/bin/env python3
"""Posthoc fixed-rule correction on immutable cached banks; no RGB/GT reads.

The decision function consumes only candidates, observed costs, validity and the
original matcher configuration. Cached supervision is opened strictly afterward
for scoring. It cannot create candidates or simulate a new temporal rollout.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


REGIONS = ("all", "hard", "initial_good", "selected")


def select(bank, config, version):
    """Pure observable decision; version 1 reproduces the saved frozen rule."""
    candidates = bank["candidates"]
    delta = np.linalg.norm(candidates - candidates[0][None], axis=-1)
    penalized = bank["candidate_costs"] + config["movement_penalty"] * delta
    eligible = bank["candidate_valid"] & (delta <= config["max_update_px"])
    if config["require_baseline_support"]:
        eligible &= bank["candidate_valid"][0][None]
    costs = penalized.copy()
    if version == 2:
        costs = np.where(eligible, costs, np.inf)
        costs[0] = penalized[0]  # Identity survives as fallback, even unsupported.
    elif version != 1:
        raise ValueError(version)
    best = costs.argmin(axis=0)
    ids = np.arange(len(best))
    accepted = eligible[best, ids] & (best != 0)
    accepted &= penalized[best, ids] + config["min_improvement"] < bank["candidate_costs"][0]
    output = np.where(accepted[:, None], candidates[best, ids], candidates[0])
    return {"best": best, "accepted": accepted, "output": output, "delta": delta}


def load_npz(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def scene_macro(rows, get):
    groups = defaultdict(list)
    missing = 0
    for row in rows:
        value = get(row)
        if value is None:
            missing += 1
        groups[row["scene"]].append(value)
    scenes = {scene: None if any(v is None for v in values) else float(np.mean(values))
              for scene, values in sorted(groups.items())}
    return {"value": None if not scenes or missing else float(np.mean(list(scenes.values()))),
            "scenes": scenes, "unscored_cases": missing}


def lookup_errors(output, sources):
    """Reuse a GT error only when displacement and query position match exactly."""
    error = np.full(len(output), np.nan, np.float32)
    found = np.zeros(len(output), bool)
    for bank, labels in sources:
        same = (bank["candidates"] == output[None]).all(axis=-1)
        hit = same.any(axis=0)
        ids = np.flatnonzero(hit)
        picked = labels[same[:, ids].argmax(axis=0), ids]
        overlap = found[ids]
        assert np.array_equal(error[ids[overlap]], picked[overlap], equal_nan=True)
        error[ids] = picked
        found[ids] = True
    return error, found


def corrected_metrics(original, base, new, changed, scored, region_masks):
    """Exact aggregate replacement at formerly untouched pixels; fixed denominators.

    Unscored changed pixels invalidate a region, rather than being omitted.
    Original dense evaluation supplies the error of all unchanged pixels.
    """
    result = deepcopy(original)
    for region in REGIONS:
        item = result[region]
        changed_here = changed & region_masks[region]
        if (changed_here & ~scored).any():
            result[region] = {"pixels": item["pixels"], "status": "cached_label_unavailable",
                              "unscored_changed_pixels": int((changed_here & ~scored).sum())}
            continue
        n = item["pixels"]
        if not n:
            continue
        b, c = base[changed_here], new[changed_here]
        # Error means use sums of each original float32 error. Gain means use
        # float32 differences, matching evaluation.region_metrics exactly.
        gain = b-c
        item["error_px"] -= float((b.astype(np.float64)-c).sum()) / n
        item["gain_px"] += float(gain.sum(dtype=np.float64)) / n
        item["benefit_px"] += float(np.maximum(gain, 0).sum(dtype=np.float64)) / n
        item["harm_px"] += float(np.maximum(-gain, 0).sum(dtype=np.float64)) / n
        item["over_1px_pct"] += 100 * int((c > 1).sum()-(b > 1).sum()) / n
        item["severe_harm_pct"] += 100 * int((gain < -3).sum()) / n
        item["repaired_to_threshold_pct"] += 100 * int((c <= 1).sum()-(b <= 1).sum()) / n
    return result


def summarize(rows):
    output = {"cases": len(rows), "scenes": len({r["scene"] for r in rows}), "counts": {}, "regions": {}}
    for key in ("query_pixels", "valid_query_pixels", "invalid_gt_query_pixels", "changed_query_pixels",
                "changed_valid_query_pixels", "scored_changed_valid_pixels", "unscored_changed_valid_pixels",
                "v1_accepted_pixels", "v2_accepted_pixels", "v1_history_accepted_pixels", "v2_history_accepted_pixels"):
        output["counts"][key] = sum(r[key] for r in rows)
    for region in REGIONS:
        entry = {"fixed_denominator_pixels_pooled": sum(r["v1"][region]["pixels"] for r in rows),
                 "nonempty_cases": sum(r["v1"][region]["pixels"] > 0 for r in rows), "metrics": {}}
        active = [r for r in rows if r["v1"][region]["pixels"] > 0]
        for metric in ("error_px", "gain_px", "harm_px", "severe_harm_pct", "over_1px_pct"):
            entry["metrics"][metric] = {
                version: scene_macro(active, lambda r, v=version, m=metric: r[v][region].get(m))
                for version in ("v1", "v2")}
        entry["v1_minus_v2_error"] = scene_macro(active, lambda r: None if r["v2"][region].get("error_px") is None else
                                                r["v1"][region]["error_px"]-r["v2"][region]["error_px"])
        output["regions"][region] = entry
    return output


def comparisons(rows, left, right, version):
    pairs = defaultdict(dict)
    for row in rows:
        if row["arm"] in (left, right):
            pairs[row["case_id"]][row["arm"]] = row
    result = {}
    for region in REGIONS:
        differences = defaultdict(list)
        missing = 0
        for pair in pairs.values():
            if len(pair) != 2:
                raise ValueError("Unpaired case")
            a, b = pair[left], pair[right]
            assert a[version][region]["pixels"] == b[version][region]["pixels"]
            if not a[version][region]["pixels"]:
                continue
            ea, eb = a[version][region].get("error_px"), b[version][region].get("error_px")
            if ea is None or eb is None:
                missing += 1
            else:
                differences[a["scene"]].append(ea-eb)
        gains = {scene: float(np.mean(values)) for scene, values in sorted(differences.items())}
        vals = np.array(list(gains.values()))
        ci = None
        if len(vals) and len(vals) <= 6 and not missing:
            indices = np.array(list(itertools.product(range(len(vals)), repeat=len(vals))))
            ci = np.quantile(vals[indices].mean(axis=1), [.025, .975]).tolist()
        result[region] = {"gain_px": float(vals.mean()) if len(vals) and not missing else None,
                          "scene_gains": gains, "unscored_cases": missing,
                          "descriptive_paired_scene_bootstrap_95pct": ci}
    return result


def analyze(run):
    raw_rows = [json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
    by_case = defaultdict(dict)
    for row in raw_rows:
        by_case[row["case_id"]][row["arm"]] = row
    details, cancellation = [], []
    checked_banks = 0
    for cid, rows in sorted(by_case.items()):
        meta = json.loads((run/"decisions"/f"{cid}.json").read_text())
        banks, selections = {}, {}
        # All decisions are computed and v1 parity checked before labels open.
        for arm, info in meta["arms"].items():
            bank = load_npz(run/"decisions"/f"{cid}_{arm}.npz")
            cfg = info["diagnostics"]["config"]
            v1, v2 = select(bank, cfg, 1), select(bank, cfg, 2)
            assert np.array_equal(v1["best"], bank["best_indices"]), (cid, arm, "best parity")
            assert np.array_equal(v1["accepted"], bank["accepted"]), (cid, arm, "accept parity")
            changed = (v1["output"] != v2["output"]).any(axis=1)
            assert not v1["accepted"][changed].any(), "Order correction must preserve every v1 accepted update"
            banks[arm], selections[arm] = bank, (v1, v2)
            checked_banks += 1
        sources, base = [], None
        scored_v1_parity = []
        for path in sorted((run/"supervision").glob(cid+"_*.npz")):
            saved = load_npz(path)
            arm = str(saved["query_id"][0]).split("/")[1]
            bank = banks[arm]
            k, n = bank["candidates"].shape[:2]
            labels = saved["candidate_error"].reshape(k, n)
            candidate_index = saved["candidate_index"].reshape(k, n)
            assert np.array_equal(candidate_index[:, 0], np.arange(k))
            current_base = saved["baseline_error"].reshape(k, n)[0]
            v1 = selections[arm][0]
            ids = np.arange(n)
            finite = np.isfinite(current_base)
            old_error = np.where(v1["accepted"], labels[v1["best"], ids], current_base)
            selected_metrics = rows[arm]["metrics"]["regions"]["selected"]
            assert np.array_equal(labels[0, finite], current_base[finite]), (cid, arm, "identity error")
            assert selected_metrics["pixels"] == int(finite.sum())
            assert abs(float(old_error[finite].mean(dtype=np.float64))-selected_metrics["error_px"]) < 1e-9, (cid, arm, "cached error parity")
            scored_v1_parity.append(arm)
            if base is None:
                base = current_base
            else:
                assert np.array_equal(base, current_base)
            sources.append((bank, labels))
        assert sources and base is not None
        assert rows["U0"]["metrics"]["regions"]["all"]["nonfinite_baseline_pixels"] == 0
        valid = np.isfinite(base)  # Verified no baseline prediction failures.
        common_xy = sources[0][0]["query_xy"].astype(int)
        x, y = common_xy.T
        with np.load(run/"predictions"/f"{cid}.npz") as predictions:
            hard = predictions["hard_mask"][y, x]
        masks = {"all": valid, "selected": valid, "hard": valid & hard, "initial_good": valid & (base <= 1)}
        for arm, bank in banks.items():
            assert np.array_equal(bank["query_xy"], common_xy)
            v1, v2 = selections[arm]
            names = np.array(meta["arms"][arm]["candidate_names"])
            history = np.array([name.startswith("history_") for name in names])
            changed = (v1["output"] != v2["output"]).any(axis=1)
            new_error, found = lookup_errors(v2["output"], sources)
            scored = found & np.isfinite(new_error)
            original = {region: rows[arm]["metrics"]["regions"][region] for region in REGIONS}
            assert original["selected"]["pixels"] == int(valid.sum())
            corrected = corrected_metrics(original, base, new_error, changed, scored, masks)
            d = {key: rows[arm][key] for key in ("task", "scene", "frame", "profile", "split", "case_id", "arm")}
            d.update(cold_start=bool(rows[arm]["warmup"]), query_pixels=len(base), valid_query_pixels=int(valid.sum()),
                     v1_cached_selected_error_parity=arm in scored_v1_parity,
                     invalid_gt_query_pixels=int((~valid).sum()), changed_query_pixels=int(changed.sum()),
                     changed_valid_query_pixels=int((changed & valid).sum()),
                     scored_changed_valid_pixels=int((changed & valid & scored).sum()),
                     unscored_changed_valid_pixels=int((changed & valid & ~scored).sum()),
                     v1_accepted_pixels=int((v1["accepted"] & valid).sum()),
                     v2_accepted_pixels=int((v2["accepted"] & valid).sum()),
                     v1_history_accepted_pixels=int((history[v1["best"]] & v1["accepted"] & valid).sum()),
                     v2_history_accepted_pixels=int((history[v2["best"]] & v2["accepted"] & valid).sum()),
                     known_changed_signed_gain_sum=float((base-new_error)[changed & valid & scored].sum(dtype=np.float64)),
                     v1=original, v2=corrected)
            details.append(d)
        if "temporal" in selections:
            s1, s2 = selections["solved"]
            t1, t2 = selections["temporal"]
            h = np.array([name.startswith("history_") for name in meta["arms"]["temporal"]["candidate_names"]])
            ids = np.arange(len(base))
            cancelled1 = s1["accepted"] & ~t1["accepted"] & valid
            oversized = h[t1["best"]] & (t1["delta"][t1["best"], ids] > meta["arms"]["temporal"]["diagnostics"]["config"]["max_update_px"])
            cancelled2 = s2["accepted"] & ~t2["accepted"] & valid
            cancelled = {key: rows["U0"][key] for key in ("task", "scene", "frame", "profile", "split", "case_id")}
            cancelled.update(cold_start=bool(rows["U0"]["warmup"]), v1_cancelled=int(cancelled1.sum()),
                             v1_oversized_history_cancelled=int((cancelled1 & oversized).sum()),
                             v2_cancelled=int(cancelled2.sum()))
            if "temporal_duplicate" in selections:
                dup = selections["temporal_duplicate"][1]
                cancelled["v2_duplicate_identical"] = bool(np.array_equal(t2["output"], dup["output"]) and
                                                            np.array_equal(t2["accepted"], dup["accepted"]))
                assert cancelled["v2_duplicate_identical"]
            if cancelled["cold_start"]:
                assert np.array_equal(s2["output"], t2["output"])
            cancellation.append(cancelled)
    summary, paired = {}, {}
    for split in sorted({d["split"] for d in details}):
        summary[split], paired[split] = {}, {}
        for task in sorted({d["task"] for d in details}):
            task_rows = [d for d in details if d["split"] == split and d["task"] == task]
            summary[split][task], paired[split][task] = {}, {}
            scopes = {"all": task_rows, "cold_start": [d for d in task_rows if d["cold_start"]],
                      "with_history": [d for d in task_rows if not d["cold_start"]]}
            scopes.update({"profile:"+p: [d for d in task_rows if d["profile"] == p]
                           for p in sorted({d["profile"] for d in task_rows})})
            for scope, subset in scopes.items():
                if not subset:
                    continue
                arms = sorted({d["arm"] for d in subset})
                summary[split][task][scope] = {a: summarize([d for d in subset if d["arm"] == a]) for a in arms}
                paired[split][task][scope] = {}
                for a, b in (("solved", "temporal"), ("raw", "solved"), ("temporal", "temporal_wrong")):
                    if a in arms and b in arms:
                        paired[split][task][scope][a+"_to_"+b] = {v: comparisons(subset, a, b, v) for v in ("v1", "v2")}
    provenance = {name: hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ("manifest.json", "metrics.jsonl")}
    provenance["analysis_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"analysis_version": "eligibility_order_v2", "run": str(run.resolve()), "provenance": provenance,
            "scope": "Posthoc cached-bank engineering correction, not a fresh full rollout or preregistered result",
            "decision_inputs": ["cached candidate vectors", "observed costs", "observed candidate validity", "original config"],
            "scoring_inputs": ["cached post-inference candidate errors", "saved fixed hard mask", "original dense metrics"],
            "new_rgb_or_gt_reads": False, "training": False, "candidate_oracle_unchanged": True,
            "v1_run_mutated": False, "cases": len(by_case), "v1_parity_checked_banks": checked_banks,
            "v1_cached_selected_error_checked_banks": sum(d["v1_cached_selected_error_parity"] for d in details),
            "dense_aggregate_method": "At changed query pixels v1 equals U0; subtract cached (baseline error - v2 candidate error) sums from saved dense error sums; retain original full region denominator. No rerun.",
            "missing_label_policy": "Any unmatched changed valid pixel invalidates the region estimate; zero-change unknown labels do not affect its exact delta.",
            "summary": summary, "paired_comparisons": paired, "history_cancellation": cancellation, "cases_and_arms": details}


def number(value):
    return "未可完整評分" if value is None else f"{value:.9f}"


def markdown(result):
    lines = ["# Fixed-rule eligibility 順序修正：v2 離線分析", "", f"原始 run：`{result['run']}`", "",
             "這是 **posthoc cached-bank 工程修正**。它重用 v1 保存的候選、觀測 cost、validity 與配置，沒有重跑模型、重建候選、重新訓練或執行新的時序 rollout；不是預註冊新實驗。原始 v1 run 與主報告保留。", "",
             "## 唯一修正與評分契約", "",
             "v1 先選最低 cost，再檢查位移上限；不合格候選可能擠掉合格更新，最後退回 U0。v2 先以 observed validity、最大更新幅度與 baseline support 排除候選，再選最低 cost。identity 保留為 fallback，movement penalty 與 improvement 門檻不變。GT 不進入選擇。", "",
             "候選 GT error 只在選擇完成後評分。保存的 solved／temporal supervision 可提供同 query、**完全相同位移**的候選誤差；其他 arm 只在位移精確相同時借用此標籤。若任何改變的有效像素缺少標籤，該區域標為未可完整評分，不丟掉它後平均。", "",
             "修正不會更換已被 v1 接受的更新，只可能替原本保留 U0 的 query 加入更新，因此可將變動像素的 error 差加回原始全圖 sums。下面 all／H*／initial-good 仍使用各 case 原本固定的**整個區域**分母；selected 是原本有效 query 分母，不改成 accepted-only。它是精確的 cached-bank aggregate replacement，沒有新的 dense GT 評估或新 rollout。", "",
             f"v1 parity：{result['cases']} cases、{result['v1_parity_checked_banks']} 個 candidate bank 的 best indices／accepted flags 均與保存結果完全一致。沒有新讀 RGB 或原始 GT。candidate oracle 的 eligibility 原已先套用，因此 oracle 報告完全不變。", "",
             "## 按 split／task 的全部已登錄 case", "",
             "正 gain = v1 error − v2 error，正值代表修正後更好；兩個 task 各自 scene macro，不跨 task 平均。", "",
             "| Split | Task | Arm | Cases | 有效 query | 新增有效更新 | 未評分新增 | all v1 → v2 | H* v1 → v2 | initial-good v1 → v2 |", "|---|---|---|---:|---:|---:|---:|---|---|---|"]
    for split, tasks in result["summary"].items():
        for task, scopes in tasks.items():
            for arm, s in scopes["all"].items():
                vals = []
                for region in ("all", "hard", "initial_good"):
                    m = s["regions"][region]["metrics"]["error_px"]
                    vals.append(number(m["v1"]["value"])+" → "+number(m["v2"]["value"]))
                c = s["counts"]
                lines.append(f"| {split} | {task} | {arm} | {s['cases']} | {c['valid_query_pixels']} | {c['changed_valid_query_pixels']} | {c['unscored_changed_valid_pixels']} | "+" | ".join(vals)+" |")
    lines.extend(["", "## Temporal 對 solved：只比較相同 case", "",
                  "正 gain = solved error − temporal error。bootstrap 為 paired scene resampling、等權 scene macro，只有 2–4 個 development scenes 時僅供描述，不能視為外推證據。", "",
                  "| Split | Task | Scope | all gain v1 | all gain v2 | v2 descriptive 95% CI | H* gain v2 |", "|---|---|---|---:|---:|---|---:|"])
    for split, tasks in result["paired_comparisons"].items():
        for task, scopes in tasks.items():
            for scope in ("all", "cold_start", "with_history"):
                pair = scopes.get(scope, {}).get("solved_to_temporal")
                if pair:
                    ci = pair["v2"]["all"]["descriptive_paired_scene_bootstrap_95pct"]
                    lines.append(f"| {split} | {task} | {scope} | {number(pair['v1']['all']['gain_px'])} | {number(pair['v2']['all']['gain_px'])} | {ci} | {number(pair['v2']['hard']['gain_px'])} |")
    lines.extend(["", "## History 擠掉空間更新", "", "| Split | Task | Cases with history | v1 cancellation | 其中 oversized history | v2 cancellation |", "|---|---|---:|---:|---:|---:|"])
    groups = defaultdict(list)
    for row in result["history_cancellation"]:
        if not row["cold_start"]:
            groups[(row["split"], row["task"])].append(row)
    for (split, task), rows in sorted(groups.items()):
        lines.append(f"| {split} | {task} | {len(rows)} | {sum(r['v1_cancelled'] for r in rows)} | {sum(r['v1_oversized_history_cancelled'] for r in rows)} | {sum(r['v2_cancelled'] for r in rows)} |")
    lines.extend(["", "## 邊界與解讀", "",
                  "- JSON 保留每 case／arm 的固定分母、GT 無效 query、未評分像素、signed gain、harm、>3px harm tail、per-profile 與 cold-start／with-history subgroup；未評分不作零分或排除平均。",
                  "- 已觀察過的 development scenes 不是 unseen final test。fit／calibration 數字僅用於診斷。候選與 cost 在原 v1 流程下取得，修正後若真實 rollout 改變記憶體／後续候選，這份分析不能預測該效應。",
                  "- 現行記憶體保存 raw observed features；accepted output 不寫回 evidence，但仍應將正式 v2 runtime rerun 與這份離線比較分開命名。",
                  "- 修掉 cancellation 只證明選擇器的工程順序問題已處理，無法把會傷害預測的 history candidate 變為正確對應；要以修正後 temporal 對修正後 solved 的配對結果判斷新增證據。",
                  "- S02 僅 3 個 source queries（flow 共 4 個 RGB 時點），profiles 為 clean、每幀 noise、全段 overlay、noise→clean recovery；没有 past-clean/current-corrupted episode、真 cycle verification 或 learned VOI。負結果不能否定所有可信 history 修復假說。",
                  "- S03 temporal_duplicate 若存在，v2 的 output／accepted 仍逐像素完全相同；wrong history 的新位移若沒有 cached GT label，必須保持未評分。",
                  "", "完整逐 case 數字與 provenance hashes 見同名 JSON。", ""])
    return "\n".join(lines)


def self_test():
    bank = {"candidates": np.array([[[0., 0.]], [[1., 0.]], [[50., 0.]]], np.float32),
            "candidate_costs": np.array([[1.], [.4], [.1]], np.float32),
            "candidate_valid": np.ones((3, 1), bool)}
    cfg = {"movement_penalty": .001, "max_update_px": 32., "require_baseline_support": True, "min_improvement": .015}
    assert not select(bank, cfg, 1)["accepted"][0]
    assert select(bank, cfg, 2)["best"][0] == 1
    assert select(bank, cfg, 2)["accepted"][0]
    bank["candidate_valid"][0] = False
    assert not select(bank, cfg, 2)["accepted"].any()


def aggregate_report(results, output_dir):
    """Top-level entry point with links to the full per-study case accounting."""
    lines = ["# Eligibility-order v2：cached-bank 修正總結", "",
             "## 結論", "",
             "**修正消除了 history cancellation，但沒有扭轉目前研究的負結果，也不足以支持 ICCV main 的正向主張。** Stereo 困難區域有局部正向訊號；兩個 task 在有歷史的 development case 中，temporal 的全圖誤差仍比修正後 solved 更差。", "",
             "本報告是 **posthoc cached-bank 工程修正**，不是重新執行完整 rollout，也不是预註冊結果。GT 只用保存於 supervision 的後驗誤差作評分；選擇只讀候選、observed costs、validity、原配置。沒有訓練、沒有新讀 RGB／原始 GT，未改動 v1 runs。", "",
             "## 精確重建範圍", "",
             "v2 先排除超過 max-update 或沒有合法 support 的候選，再作 penalized-cost argmin，identity 永遠是 fallback。其他門檻不變。每個 bank 都重現 v1 best indices／accepted flags，並確認修正不會更換任何 v1 已接受的更新。", "",
             "變動 query 的舊值必為 U0，因此以 cached (baseline error − new candidate error) 加回原 dense sums，可精確更新 all／固定 H*／initial-good／selected 的 error、signed gain、harm 與 tail 指標；分母保留原本完整區域。這是 aggregate replacement，不是新 dense GT forward。cached solved／temporal 的 v1 selected-error 均另與原始評分逐 case 核對。", "",
             "只在同一 query、完全相同的位移上共用 cached label。所有 solved／temporal 的變動有效像素都能完整評分；generic／部分 raw 或 wrong-support 若有任何缺標的變動像素，對應 region 和 scene macro 為未可完整評分，不排除缺標後宣稱結果。S03 三個 temporal control 的變动像素都能由精確位移匹配完整評分。", "",
             "| Study | Cases | v1 決策 parity banks | cached selected-error parity banks | 改變有效 query（所有 arms 合計） | 其中缺少 cached label |", "|---|---:|---:|---:|---:|---:|"]
    for result in results:
        study = Path(result["run"]).parents[1].name
        rows = result["cases_and_arms"]
        lines.append(f"| {study} | {result['cases']} | {result['v1_parity_checked_banks']} | {result['v1_cached_selected_error_checked_banks']} | {sum(r['changed_valid_query_pixels'] for r in rows)} | {sum(r['unscored_changed_valid_pixels'] for r in rows)} |")
    lines.extend(["", "上述 counts 是 case × arm 中的 query，不是跨 arm 去重的唯一影像像素。各 case 固定分母、失敗／缺標與每個 split／profile 的明細均保存在 study JSON。", "",
                  "## Development evaluation：修正後主要對照", "",
                  "| Study / cohort | Task | 對照 | v1 gain px | v2 gain px | v2 H* gain px |", "|---|---|---|---:|---:|---:|"])
    for result in results:
        study = Path(result["run"]).parents[1].name
        for task in ("stereo", "flow"):
            if study.startswith("S01"):
                selected = [d for d in result["cases_and_arms"] if d["split"] == "development_evaluation" and d["task"] == task and d["arm"] == "solved"]
                get_gain = lambda version, region: scene_macro(selected, lambda r: r[version][region].get("gain_px"))["value"]
                lines.append(f"| S01 / all | {task} | U0 → solved | {number(get_gain('v1','all'))} | {number(get_gain('v2','all'))} | {number(get_gain('v2','hard'))} |")
            else:
                p = result["paired_comparisons"]["development_evaluation"][task]["with_history"]["solved_to_temporal"]
                lines.append(f"| {study[:3]} / with history | {task} | solved → temporal | {number(p['v1']['all']['gain_px'])} | {number(p['v2']['all']['gain_px'])} | {number(p['v2']['hard']['gain_px'])} |")
    lines.extend(["", "gain = 對照 error − 新方法 error，正值為改善。各 task 各自先 within-scene 平均，再 scene macro；不同 study 的 cohort 不同，不能拿其總平均直接互比。S02 為 4 個 development scenes，S03 為 2 個；bootstrap 只屬描述性分析。", "",
                  "## Cancellation 與機制結論", ""])
    for result in results:
        study = Path(result["run"]).parents[1].name
        for task in ("stereo", "flow"):
            rows = [d for d in result["history_cancellation"] if d["split"] == "development_evaluation" and d["task"] == task and not d["cold_start"]]
            if rows:
                lines.append(f"- {study[:3]} {task}：原本取消 solved 更新 {sum(r['v1_cancelled'] for r in rows)} 個有效 query，其中 oversized history {sum(r['v1_oversized_history_cancelled'] for r in rows)}；v2 cancellation = {sum(r['v2_cancelled'] for r in rows)}。")
    lines.extend(["", "S02 stereo 的 whole-H* 指標轉成小幅正 gain，但 all／initial-good 仍未同時改善；flow 的 all／H* 仍負向。修正改善了選擇器順序，沒有提供『新增 history 足以修復不可相信區域』的證據。S03 duplicate 的 v2 output／accepted flags 逐像素一致；錯來源控制仍是有限合成壓力測試，不能宣稱 confidence calibration 或遞迴錯誤防護。", "",
                  "candidate oracle 未改變（原 oracle 已先套 eligibility）；它只表達既有候選可達到的上限，不是 runtime 的真實增益。後續研究仍需提升候選／association 品質及可信度，並用獨立 unseen 場景與完整新版本 rollout 驗證。", "",
                  "S02 僅 3 個 source queries（flow 共 4 個 RGB 時點），沒有 past-clean/current-corrupted episode，也沒有真 cycle verification 或 learned VOI；本結果不能否定所有可信 history 修復方案。development scenes 已曝光，不是 final test。", "",
                  "## 詳細報告", ""])
    entries = []
    for result in results:
        run = Path(result["run"])
        study = run.parents[1].name
        report_dir = run.parents[1]/"reports"
        lines.append(f"- [{study}]({report_dir/'eligibility_order_v2.md'})；[逐 case JSON]({report_dir/'eligibility_order_v2.json'})。")
        entries.append({"study": study, "run": str(run), "report_json": str(report_dir/"eligibility_order_v2.json"),
                        "provenance": result["provenance"], "cases": result["cases"],
                        "v1_parity_checked_banks": result["v1_parity_checked_banks"],
                        "v1_cached_selected_error_checked_banks": result["v1_cached_selected_error_checked_banks"],
                        "summary": result["summary"], "paired_comparisons": result["paired_comparisons"]})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/"eligibility_order_v2.md").write_text("\n".join(lines)+"\n")
    (output_dir/"eligibility_order_v2.json").write_text(json.dumps({"scope": results[0]["scope"],
        "full_region_reconstruction": results[0]["dense_aggregate_method"],
        "missing_label_policy": results[0]["missing_label_policy"], "studies": entries}, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    self_test()
    results = []
    for run in args.runs:
        result = analyze(run.resolve())
        results.append(result)
        directory = run.parent.parent/"reports"
        directory.mkdir(parents=True, exist_ok=True)
        (directory/"eligibility_order_v2.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        (directory/"eligibility_order_v2.md").write_text(markdown(result))
        print(json.dumps({"run": str(run), "cases": result["cases"], "banks": result["v1_parity_checked_banks"],
                          "changed_valid_pixels": sum(r["changed_valid_query_pixels"] for r in result["cases_and_arms"]),
                          "unscored_changed_valid_pixels": sum(r["unscored_changed_valid_pixels"] for r in result["cases_and_arms"])}), flush=True)
    roots = {run.resolve().parents[3] for run in args.runs}
    if len(roots) == 1 and len(results) == 3:
        aggregate_report(results, roots.pop()/"reports")


if __name__ == "__main__":
    main()
