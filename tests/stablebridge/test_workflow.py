"""Workflow tests use isolated files and fake workers; no model/GPU is loaded."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from stablebridge import workflow


class FakeWorkers:
    def __init__(self, plan, *, fail_stage=None, missing_artifact=False):
        self.by_command = {tuple(stage["command"]): stage for stage in plan["stages"]}
        self.fail_stage = fail_stage
        self.missing_artifact = missing_artifact
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append(tuple(command))
        stage = self.by_command[tuple(command)]
        code = 7 if stage["stage_id"] == self.fail_stage else 0
        kwargs["stdout"].write(f"fake worker {stage['stage_id']} exit={code}\n")
        if code == 0:
            for index, name in enumerate(stage["required_files"]):
                if self.missing_artifact and index == 1:
                    continue
                path = Path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.name == "status.json":
                    path.write_text(json.dumps({"state": "completed"}))
                elif path.name == "results.json":
                    path.write_text(json.dumps({"kind": "frozen_candidate_query_replay", "whole_image_result": False,
                                                "closed_loop_result": False, "query_count": 10}))
                else:
                    path.write_text("{}\n")
        return mock.Mock(pid=3000+len(self.calls), wait=mock.Mock(return_value=code), poll=mock.Mock(return_value=code))


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="workflow tests ")
        self.root = Path(self.temporary.name)
        (self.root / "src/stablebridge").mkdir(parents=True)
        (self.root / "src/stablebridge/__init__.py").write_text("SOURCE_VERSION = 1\n")
        configs = self.root / "configs/stablebridge"
        configs.mkdir(parents=True)
        (configs / "data_models_v1.json").write_text("{}\n")
        clips = self.root / "experiments/E01_evidence_mechanism/manifests/clips.json"
        clips.parent.mkdir(parents=True)
        clips.write_text('{"clips": []}\n')
        registration = {"clips_manifest": str(clips.relative_to(self.root)),
                        "matching": {"max_update_px": 32., "require_baseline_support": True}}
        for name in ("e01_smoke_v1", "e01_s01_spatial_v1", "e01_s02_temporal_pilot_v1",
                     "e01_s03_source_control_v1", "e01_s02_temporal_full_v1"):
            (configs / f"{name}.json").write_text(json.dumps(registration))
        self.output = self.root / "operations/e01_workflow_test"
        self.plan = workflow.build_plan(self.output, root=self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def execute(self, workers, plan=None):
        with mock.patch.object(workflow.subprocess, "Popen", side_effect=workers), redirect_stdout(io.StringIO()):
            workflow.execute_plan(self.plan if plan is None else plan)

    def test_plan_keeps_experiment_hierarchy_and_excludes_stress_rows_from_training(self):
        self.assertFalse(self.output.exists())
        stages = self.plan["stages"]
        self.assertEqual([s["stage_id"] for s in stages],
                         ["smoke", "s01_spatial", "s02_temporal_pilot", "s03_source_control",
                          "collect_supervision", "train_utility", "replay_s01_spatial", "replay_s02_temporal_pilot"])
        self.assertIn("S01_spatial_repair/runs/e01_workflow_test_smoke", stages[0]["completion_status"])
        self.assertIn("S03_source_control/runs/", stages[3]["completion_status"])
        collection = stages[4]["command"]
        self.assertTrue(any("_s01_spatial" in part for part in collection))
        self.assertTrue(any("_s02_temporal_pilot" in part for part in collection))
        self.assertFalse(any("_smoke" in part or "_s03_source_control" in part for part in collection))
        self.assertEqual(stages[5]["command"][stages[5]["command"].index("--epochs")+1], "30")
        self.assertTrue(all(stage["command"][0] == sys.executable for stage in stages))
        self.assertFalse(self.plan["scientific_success_claim"])

    def test_full_temporal_is_explicit_opt_in(self):
        enabled = workflow.build_plan(self.output, root=self.root, include_full_temporal=True)
        self.assertEqual(len(enabled["stages"]), 9)
        self.assertIn("s02_temporal_full", [s["stage_id"] for s in enabled["stages"]])
        collection = next(s for s in enabled["stages"] if s["stage_id"] == "collect_supervision")
        self.assertFalse(any("_s02_temporal_full" in part for part in collection["command"]))

    def test_dry_run_only_emits_plan(self):
        with mock.patch.object(workflow, "build_plan", return_value=self.plan), \
                mock.patch.object(workflow, "execute_plan") as execution, redirect_stdout(io.StringIO()) as output:
            workflow.main(["--output-dir", str(self.output), "--dry-run"])
        execution.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["workflow_name"], "e01_workflow_test")
        self.assertFalse(self.output.exists())

    def test_smoke_failure_stops_all_later_stages_and_records_child_exit(self):
        workers = FakeWorkers(self.plan, fail_stage="smoke")
        with self.assertRaisesRegex(RuntimeError, "exited with code 7"):
            self.execute(workers)
        self.assertEqual(len(workers.calls), 1)
        status = json.loads((self.output / "status.json").read_text())
        stages = json.loads((self.output / "stages.json").read_text())
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["stage"], "smoke")
        self.assertEqual(stages["smoke"]["returncode"], 7)
        self.assertEqual(stages["smoke"]["child_pid"], 3001)
        self.assertTrue(Path(stages["smoke"]["log"]).read_text())
        self.assertFalse(status["scientific_success_claim"])

    def test_successful_resume_checks_artifacts_and_launches_no_second_worker(self):
        workers = FakeWorkers(self.plan)
        self.execute(workers)
        self.assertEqual(len(workers.calls), 8)
        status = json.loads((self.output / "status.json").read_text())
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["scientific_conclusion"], "not_inferred_from_process_completion")
        resumed = FakeWorkers(self.plan)
        self.execute(resumed)
        self.assertEqual(resumed.calls, [])
        self.assertTrue((self.output / "snapshot/src/stablebridge/__init__.py").is_file())

    def test_failed_stage_retries_but_completed_prefix_is_not_reexecuted(self):
        with self.assertRaises(RuntimeError):
            self.execute(FakeWorkers(self.plan, fail_stage="s01_spatial"))
        retry = FakeWorkers(self.plan)
        self.execute(retry)
        self.assertEqual(len(retry.calls), 7)
        self.assertEqual(retry.by_command[retry.calls[0]]["stage_id"], "s01_spatial")
        stages = json.loads((self.output / "stages.json").read_text())
        self.assertEqual(stages["s01_spatial"]["attempt"], 2)
        self.assertTrue((self.output / "logs/s01_spatial.attempt001.log").is_file())
        self.assertTrue((self.output / "logs/s01_spatial.attempt002.log").is_file())

    def test_exit_zero_without_completed_artifacts_is_failure(self):
        workers = FakeWorkers(self.plan, missing_artifact=True)
        with self.assertRaisesRegex(RuntimeError, "missing its required artifact"):
            self.execute(workers)
        self.assertEqual(len(workers.calls), 1)
        self.assertEqual(json.loads((self.output / "status.json").read_text())["state"], "failed")

    def test_resume_rejects_changed_source_config_snapshot_and_completed_artifacts(self):
        self.execute(FakeWorkers(self.plan))
        original_source = self.root / "src/stablebridge/__init__.py"
        original_source.write_text("SOURCE_VERSION = 2\n")
        resumed = FakeWorkers(self.plan)
        with self.assertRaisesRegex(ValueError, "input changed"):
            self.execute(resumed)
        self.assertEqual(resumed.calls, [])
        original_source.write_text("SOURCE_VERSION = 1\n")
        frozen = self.output / "snapshot/src/stablebridge/__init__.py"
        frozen.write_text("SOURCE_VERSION = 3\n")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            self.execute(FakeWorkers(self.plan))
        frozen.write_text("SOURCE_VERSION = 1\n")
        artifact = Path(self.plan["stages"][0]["required_files"][0])
        artifact.write_text('{"changed":true}\n')
        with self.assertRaisesRegex(ValueError, "artifacts changed"):
            self.execute(FakeWorkers(self.plan))

    def test_resume_rejects_manifest_options_changed_under_same_workflow_name(self):
        self.execute(FakeWorkers(self.plan))
        different = workflow.build_plan(self.output, root=self.root, epochs=31)
        with self.assertRaisesRegex(ValueError, "manifest hashes/settings changed"):
            self.execute(FakeWorkers(different), different)

    def test_training_policy_is_pinned_to_both_source_experiments(self):
        config_path = self.root / "configs/stablebridge/e01_s02_temporal_pilot_v1.json"
        config = json.loads(config_path.read_text())
        config["matching"]["require_baseline_support"] = False
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "identical candidate eligibility"):
            workflow.build_plan(self.output, root=self.root)
        with self.assertRaises(ValueError):
            workflow.build_plan(self.output, root=self.root, epochs=0)

    def test_worker_directory_has_single_owner(self):
        self.output.mkdir(parents=True)
        with workflow._worker_lock(self.output):
            with self.assertRaisesRegex(RuntimeError, "Another worker"):
                with workflow._worker_lock(self.output):
                    pass

    def test_actual_cpu_child_records_pid_and_completion_without_a_shell(self):
        artifact = self.output / "cpu_worker.txt"
        command = (sys.executable, "-c", f"from pathlib import Path; Path({str(artifact)!r}).write_text('finished\\n')")
        plan = dict(self.plan)
        plan["stages"] = [{"stage_id": "cpu_fixture", "command": command,
                           "required_files": (str(artifact),), "completion_status": None}]
        with redirect_stdout(io.StringIO()):
            workflow.execute_plan(plan)
        self.assertEqual(artifact.read_text(), "finished\n")
        stage = json.loads((self.output / "stages.json").read_text())["cpu_fixture"]
        self.assertEqual(stage["returncode"], 0)
        self.assertGreater(stage["child_pid"], 0)


if __name__ == "__main__":
    unittest.main()
