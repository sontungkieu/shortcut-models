from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import submit_gmm_tide_fm_jobs as submit_jobs  # noqa: E402


class KjoResumePipelineTest(unittest.TestCase):
    def test_parent_gate_path_is_stable_per_exact_slug(self) -> None:
        root = Path("gates")
        self.assertEqual(
            submit_jobs.parent_resume_gate_path(root, "owner/parent-slug"),
            root / "owner__parent-slug.json",
        )

    def test_required_parent_gate_rejects_unconfigured_resume(self) -> None:
        with self.assertRaisesRegex(ValueError, "resume_parent_gate is required"):
            submit_jobs.evaluate_parent_resume_gate(
                config={"resume_kernel_ref": "owner/parent-slug"},
                gate_root=Path("gates"),
                require_cache_hit=True,
            )

    def test_stage_job_defaults_to_blueprint_dispatch(self) -> None:
        expected = (Path("stage"), "owner/run")
        with mock.patch.object(submit_jobs, "_stage_job_blueprint", return_value=expected) as blueprint:
            actual = submit_jobs.stage_job(
                owner="owner",
                config={"run_name": "run"},
                staging_root=Path("staging"),
                accelerator="cpu",
                wandb_api_key="",
            )
        self.assertEqual(actual, expected)
        blueprint.assert_called_once()

    def test_copy_submission_artifacts_keeps_blueprint_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage = root / "stage"
            run = root / "run"
            stage.mkdir()
            notebook = {
                "cells": [],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
            (stage / "submitted_notebook.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
            (stage / "kernel-metadata.json").write_text(
                json.dumps({"code_file": "submitted_notebook.ipynb"}),
                encoding="utf-8",
            )
            (stage / "gmm_tide_config.json").write_text("{}\n", encoding="utf-8")
            (stage / "stage_package_manifest.json").write_text("{}\n", encoding="utf-8")
            (stage / "staging_blueprint_result.json").write_text("{}\n", encoding="utf-8")
            (stage / "operation_timeline.jsonl").write_text("{}\n", encoding="utf-8")

            copied = submit_jobs.copy_submission_artifacts(stage, run)

            self.assertTrue(Path(copied["stage_package_manifest.json"]).is_file())
            self.assertTrue(Path(copied["staging_blueprint_result.json"]).is_file())
            self.assertTrue(Path(copied["operation_timeline.jsonl"]).is_file())

    def test_atomic_reservation_is_scoped_to_one_exact_owner(self) -> None:
        reservation_payload = {
            "ok": True,
            "reserved": [
                {
                    "owner": "owner-a",
                    "accelerator": "tpu",
                    "slot_id": 0,
                    "reservation_token": "token-a",
                }
            ],
        }
        with mock.patch.object(submit_jobs, "run_json_command", return_value=reservation_payload) as run:
            with mock.patch.object(submit_jobs, "kaggle_command", return_value=["/tmp/kaggle"]):
                result = submit_jobs.reserve_exact_owner(
                    owner="owner-a",
                    accelerator="TpuV5E8",
                    accounts_file=Path("/secret/accounts.json"),
                    project_root=Path("/project"),
                    run_id="resume-a",
                    task_id="task-a",
                    estimated_runtime_minutes=480,
                    ttl_minutes=30,
                )

        self.assertEqual(result["reservation_token"], "token-a")
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--owners") + 1], "owner-a")
        self.assertEqual(command[command.index("--preferred-owners") + 1], "owner-a")
        self.assertEqual(command[command.index("--count") + 1], "1")
        self.assertIn("--live", command)
        self.assertEqual(command[command.index("--registry-sync-mode") + 1], "db-only")

    def test_atomic_submit_consumes_reservation_and_records_embedded_key_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage = root / "stage"
            stage.mkdir()
            notebook = stage / "resume.ipynb"
            notebook.write_text("{}\n", encoding="utf-8")
            (stage / "kernel-metadata.json").write_text(
                json.dumps({"code_file": notebook.name, "id": "owner-a/resume-a"}) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(submit_jobs, "kaggle_command", return_value=["/tmp/kaggle"]):
                command = submit_jobs.build_atomic_submit_command(
                    run_dir=root / "run",
                    staging_dir=stage,
                    owner="owner-a",
                    accelerator="TpuV5E8",
                    reservation_token="token-a",
                    registry=Path("/project/.secrets/kaggle_notebooks.jsonl"),
                    project_root=Path("/project"),
                    run_id="resume-a",
                    task_id="task-a",
                    artifact_mode="has-artifacts",
                    retention_action="keep-while-artifacts-needed",
                    embedded_key_names=["WANDB_API_KEY", "KAGGLE_CREDENTIAL"],
                    kaggle_config_dir=root / "kaggle-config",
                    runtime_dataset_source="owner-a/kjo-runtime-0-10-0",
                )

        self.assertEqual(command[command.index("--reservation-token") + 1], "token-a")
        self.assertEqual(command[command.index("--owner") + 1], "owner-a")
        self.assertEqual(command[command.index("--secret-mode") + 1], "embedded")
        embedded_names = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--embedded-key-name"
        ]
        self.assertEqual(embedded_names, ["KAGGLE_CREDENTIAL", "WANDB_API_KEY"])
        self.assertIn("--require-notebook-logging-contract", command)
        self.assertIn("--require-accelerator-probe-contract", command)
        self.assertEqual(
            command[command.index("--required-dataset-source") + 1],
            "owner-a/kjo-runtime-0-10-0",
        )

    def test_unused_atomic_reservation_is_released_by_exact_token(self) -> None:
        payload = {"ok": True, "released": True}
        with mock.patch.object(submit_jobs, "run_json_command", return_value=payload) as run:
            result = submit_jobs.release_unused_reservation(
                owner="owner-a",
                accelerator="TpuV5E8",
                reservation={"slot_id": 0, "reservation_token": "token-a"},
            )

        self.assertEqual(result, payload)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--owner") + 1], "owner-a")
        self.assertEqual(command[command.index("--slot-id") + 1], "0")
        self.assertEqual(command[command.index("--reservation-token") + 1], "token-a")
        self.assertNotIn("--force-active", command)

    def test_atomic_evidence_copy_does_not_overwrite_kjo_operation_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage = root / "stage"
            run = root / "run"
            stage.mkdir()
            for name in (
                "gmm_tide_config.json",
                "stage_package_manifest.json",
                "staging_blueprint_result.json",
                "operation_timeline.jsonl",
            ):
                (stage / name).write_text(f"{name}\n", encoding="utf-8")
            (run / "operation_timeline.jsonl").parent.mkdir(parents=True)
            (run / "operation_timeline.jsonl").write_text("kjo timeline\n", encoding="utf-8")

            copied = submit_jobs.copy_atomic_submission_evidence(stage, run)

            self.assertEqual((run / "operation_timeline.jsonl").read_text(), "kjo timeline\n")
            self.assertTrue((run / "submit/stage_operation_timeline.jsonl").is_file())
            self.assertEqual(len(copied), 4)

    def test_atomic_main_keeps_accepted_submit_when_status_fails_and_skips_it_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "submit_report.json"
            job_root = root / "jobs"
            staging_root = root / "staging"
            registry = root / "registry.jsonl"
            job = {
                "candidate_family": "naive_gaussian",
                "expected_submit_owner": "owner-a",
                "gmm_num_modes": 1,
                "gmm_router_topk": 1,
                "grid_index": 0,
                "run_name": "naive-s1-resume400",
                "training_seed": 1,
            }

            def fake_stage_job(**kwargs):
                stage = staging_root / "naive-s1-resume400-owner-a"
                stage.mkdir(parents=True)
                notebook = stage / "submitted.ipynb"
                notebook.write_text(
                    json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
                    encoding="utf-8",
                )
                (stage / "kernel-metadata.json").write_text(
                    json.dumps({"code_file": notebook.name, "id": "owner-a/naive-s1-resume400-owner-a"}),
                    encoding="utf-8",
                )
                return stage, "owner-a/naive-s1-resume400-owner-a"

            atomic_payload = {"ok": True, "registry_result": {"ok": True}}
            argv = [
                "submit_gmm_tide_fm_jobs.py",
                "--grid-config",
                str(root / "grid.json"),
                "--accounts-file",
                str(root / "accounts.json"),
                "--env-file",
                str(root / "empty.env"),
                "--owners",
                "owner-a",
                "--exclude-owners",
                "",
                "--staging-root",
                str(staging_root),
                "--report-path",
                str(report_path),
                "--job-root",
                str(job_root),
                "--notebook-registry",
                str(registry),
                "--no-shared-context",
                "--kjo-atomic-submit",
            ]
            reservation = {
                "payload": {"ok": True},
                "reservation": {"owner": "owner-a", "slot_id": 0, "reservation_token": "token-a"},
                "reservation_token": "token-a",
            }
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(
                    submit_jobs,
                    "load_kaggle_accounts",
                    return_value={"owner-a": {"username": "owner-a", "key": "secret-not-logged"}},
                ):
                    with mock.patch.object(submit_jobs, "load_env_file", return_value={}):
                        with mock.patch.object(submit_jobs, "load_grid", return_value=[job]):
                            with mock.patch.object(submit_jobs, "ensure_submit_source_ready"):
                                with mock.patch.object(submit_jobs, "ensure_kaggle_cli_for_submit"):
                                    with mock.patch.object(
                                        submit_jobs.subprocess,
                                        "check_output",
                                        return_value="commit-a\n",
                                    ):
                                        with mock.patch.object(
                                            submit_jobs,
                                            "evaluate_parent_resume_gate",
                                            return_value=None,
                                        ):
                                            with mock.patch.object(submit_jobs, "stage_job", side_effect=fake_stage_job):
                                                with mock.patch.object(
                                                    submit_jobs,
                                                    "validate_staged_metadata",
                                                    return_value={"ok": True},
                                                ):
                                                    with mock.patch.object(
                                                        submit_jobs,
                                                        "record_injected_notebook",
                                                    ):
                                                        with mock.patch.object(
                                                            submit_jobs,
                                                            "reserve_exact_owner",
                                                            return_value=reservation,
                                                        ) as reserve:
                                                            with mock.patch.object(
                                                                submit_jobs,
                                                                "kaggle_command",
                                                                return_value=["/tmp/kaggle"],
                                                            ):
                                                                with mock.patch.object(
                                                                    submit_jobs,
                                                                    "run_json_command",
                                                                    side_effect=[
                                                                        atomic_payload,
                                                                        RuntimeError("status temporarily unavailable"),
                                                                    ],
                                                                ) as run_json:
                                                                    submit_jobs.main()
                                                                    submit_jobs.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["submit_mode"], "kjo-atomic")
            self.assertEqual(len(report["submitted"]), 1)
            self.assertFalse(report["failed"])
            self.assertEqual(report["submitted"][0]["kernel_status"], "UNKNOWN")
            self.assertIn("status temporarily unavailable", report["submitted"][0]["status_error"])
            self.assertNotIn("reservation_token", report["submitted"][0]["reservation"])
            self.assertTrue(report["submitted"][0]["reservation_consumed"])
            self.assertEqual(reserve.call_count, 1)
            self.assertEqual(run_json.call_count, 2)


if __name__ == "__main__":
    unittest.main()
