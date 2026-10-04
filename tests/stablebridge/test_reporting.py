"""Reporting and launch construction tests; never submit a real service."""
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from stablebridge.evaluation import evaluate_pair
from stablebridge.reporting import build_report,load_rows,write_report


ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("stablebridge_detached_test",ROOT/"scripts/run_stablebridge_detached.py")
launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launcher)


def metric_row(scene,frame,error,*,task="flow",arm="actual",profile="clean",split="development",pixels=2):
    gt = np.zeros((4,2,1,pixels),np.float32)
    base = np.zeros((2,1,pixels),np.float32);base[0] = 5
    current = np.zeros_like(base);current[0] = error
    # Keep initial-good nonempty on one pixel while retaining a true hard region.
    base[0,0,0] = .5
    hard = np.ones((1,pixels),bool);hard[0,0] = False
    selected = hard.copy()
    return {"task":task,"scene":scene,"frame":frame,"profile":profile,"arm":arm,"split":split,
            "metrics":evaluate_pair(base,current,gt,hard,selected_mask=selected),
            "accepted_pixels":int(hard.sum()),"costs":{"matching_candidates":12,"elapsed_ms":10}}


class ReportingTests(unittest.TestCase):
    def test_scene_macro_not_case_weighted_and_tasks_and_oracles_stay_separate(self):
        rows = [metric_row("a",1,2),metric_row("a",2,4),metric_row("b",1,9),
                metric_row("a",1,100,task="stereo"),
                metric_row("a",1,0,arm="decision_oracle")]
        report = build_report(rows,{"expected_metric_rows":5,"exposure":"development only"})
        flow = report["results"]["development"]["flow"]
        self.assertEqual(flow["actual"]["regions"]["all"]["metrics"]["error_px"]["scene_macro"],6)
        stereo = report["results"]["development"]["stereo"]["actual"]
        self.assertEqual(stereo["regions"]["all"]["metrics"]["error_px"]["scene_macro"],100)
        self.assertEqual(flow["decision_oracle"]["role"],"candidate_oracle_diagnostic")
        self.assertEqual(flow["actual"]["role"],"actual_output")
        self.assertEqual(report["completeness"],"complete_by_row_count")
        self.assertEqual(flow["actual"]["selected_coverage_pct"]["scene_macro"],50)

    def test_failed_prediction_does_not_disappear_and_profiles_separate(self):
        rows = [metric_row("a",1,2),metric_row("a",2,np.nan,profile="gaussian_noise")]
        report = build_report(rows,{})
        result = report["results"]["development"]["flow"]["actual"]
        self.assertIsNone(result["regions"]["all"]["metrics"]["error_px"]["scene_macro"])
        self.assertEqual(result["regions"]["all"]["failed_rows"],1)
        self.assertEqual(result["regions"]["all"]["nonfinite_prediction_pixels"],2)
        self.assertEqual(result["per_profile"]["clean"]["regions"]["all"]["metrics"]["error_px"]["scene_macro"],2)
        self.assertIsNone(result["per_profile"]["gaussian_noise"]["regions"]["all"]["metrics"]["error_px"]["scene_macro"])
        json.dumps(report,allow_nan=False)

    def test_write_report_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path/"manifest.json").write_text(json.dumps({"expected_metric_rows":1,"gt_status":"evaluation only"}))
            row = metric_row("a",1,2)
            (path/"metrics.jsonl").write_text(json.dumps(row)+"\n")
            report = write_report(path)
            self.assertEqual(report["metric_rows"],1)
            self.assertTrue((path/"REPORT.md").exists())
            self.assertEqual(json.loads((path/"report.json").read_text())["gt_status"],"evaluation only")
            self.assertEqual(len(report["source_paths"]["manifest_sha256"]),64)
            (path/"metrics.jsonl").write_text((json.dumps(row)+"\n")*2)
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                load_rows(path/"metrics.jsonl")

    def test_missing_region_never_looks_like_success(self):
        row = metric_row("a",1,2)
        del row["metrics"]["regions"]["hard"]
        report = build_report([row],{})
        hard = report["results"]["development"]["flow"]["actual"]["regions"]["hard"]
        self.assertEqual(hard["status"],"missing_region")
        self.assertEqual(hard["missing_rows"],1)


class DetachedLauncherTests(unittest.TestCase):
    def test_encoded_command_preserves_literal_shell_and_systemd_characters(self):
        with tempfile.TemporaryDirectory() as temporary:
            weird = ["a b","$(touch should-not-exist)","`echo bad`","$HOME","%n","a\nb",'quote"value']
            command = [sys.executable,"-c","import json,sys; print(json.dumps(sys.argv[1:]))",*weird]
            record = launcher.build_launch("stablebridge-test",Path(temporary)/"log % dir",command,cwd=ROOT)
            encoded = record["systemd_argv"][-1]
            payload = json.loads(base64.b64decode(encoded))
            self.assertEqual(payload["command"],command)
            self.assertFalse(record["shell"])
            self.assertIn("--remain-after-exit",record["systemd_argv"])
            self.assertNotIn("--wait",record["systemd_argv"])
            self.assertTrue(any("log %% dir" in arg for arg in record["systemd_argv"]))
            # Run only the fixed exec trampoline, with no systemd invocation.
            result = subprocess.run([sys.executable,"-c",launcher._TRAMPOLINE,encoded],check=True,
                                    capture_output=True,text=True)
            self.assertEqual(json.loads(result.stdout),weird)
            self.assertFalse((ROOT/"should-not-exist").exists())

    def test_dry_run_records_plan_without_calling_systemd(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = launcher.build_launch("stablebridge-test",temporary,[sys.executable,"-V"],cwd=ROOT)
            with patch.object(launcher.subprocess,"run") as run:
                result = launcher.launch(record,dry_run=True)
                run.assert_not_called()
            self.assertEqual(result["status"],"planned")
            saved = json.loads((Path(temporary)/"launch.json").read_text())
            self.assertEqual(saved["command"],[sys.executable,"-V"])
            # An identical planned command may subsequently be submitted.
            fake = subprocess.CompletedProcess(record["systemd_argv"],0,"accepted","")
            with patch.object(launcher.subprocess,"run",return_value=fake) as run:
                result = launcher.launch(record)
                self.assertEqual(run.call_args.args[0],record["systemd_argv"])
            self.assertEqual(result["status"],"submitted")
            with self.assertRaises(FileExistsError):
                launcher.launch(record,dry_run=True)

    def test_unit_and_nul_arguments_rejected(self):
        with self.assertRaises(ValueError):
            launcher.build_launch("x;touch danger","/tmp/output",["echo"],cwd=ROOT)
        with self.assertRaises(ValueError):
            launcher.build_launch("safe","/tmp/output",["echo","a\0b"],cwd=ROOT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
