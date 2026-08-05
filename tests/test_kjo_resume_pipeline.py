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


if __name__ == "__main__":
    unittest.main()
