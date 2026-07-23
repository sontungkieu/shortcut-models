import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fid_repeat_utils import parse_eval_fid_seeds, summarize_fid_repeat_records
from scripts.analyze_gmm_tide_fid_repeats import (
    analyze_rows,
    audit_comparability,
    load_jobs,
    load_protocol,
)
from scripts.submit_gmm_tide_fm_jobs import (
    make_notebook,
    resume_download_credential_owner,
    resume_file_pattern,
    scrub_notebook_embedded_credentials,
)


def compile_notebook_cells(notebook):
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if source.strip():
            compile(source, f"cell_{index}.py", "exec")


class FidRepeatUtilsTest(unittest.TestCase):
    def test_parse_eval_fid_seeds_deduplicates_in_order(self):
        self.assertEqual(parse_eval_fid_seeds("101, 202,101,303"), [101, 202, 303])

    def test_parse_eval_fid_seeds_rejects_empty_or_negative(self):
        with self.assertRaises(ValueError):
            parse_eval_fid_seeds("")
        with self.assertRaises(ValueError):
            parse_eval_fid_seeds("1,-2")

    def test_summarize_fid_repeat_records_uses_sample_std(self):
        summary = summarize_fid_repeat_records(
            [
                {"metric_name": "fid/timesteps/128", "eval_seed": 1, "value": 7.0},
                {"metric_name": "fid/timesteps/128", "eval_seed": 2, "value": 7.2},
            ]
        )["metrics"]["fid/timesteps/128"]
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["mean"], 7.1)
        self.assertAlmostEqual(summary["sample_std"], 2 ** 0.5 / 10)


class FidRepeatAnalysisTest(unittest.TestCase):
    @staticmethod
    def _jobs():
        jobs = {}
        for family in ("C0", "C4"):
            for training_seed in (0, 1):
                run_name = f"{family.lower()}-s{training_seed}"
                jobs[run_name] = {
                    "run_name": run_name,
                    "candidate_family": family,
                    "training_seed": training_seed,
                    "resume_kernel_ref": f"owner/{run_name}",
                    "eval_fid_seeds": "1,2,3",
                    "eval_fid_generations": 50048,
                }
        return jobs

    @staticmethod
    def _rows(delta):
        rows = []
        for family in ("C0", "C4"):
            for training_seed in (0, 1):
                run_name = f"{family.lower()}-s{training_seed}"
                for eval_seed, offset in zip((1, 2, 3), (-0.01, 0.0, 0.01)):
                    value = 7.2 + 0.05 * training_seed + offset
                    if family == "C4":
                        value += delta
                    rows.append(
                        {
                            "run_name": run_name,
                            "eval_seed": eval_seed,
                            "value": value,
                            "step": 400000,
                            "eval_fid_generations": 50048,
                        }
                    )
        return rows

    def test_measurement_gate_passes_consistent_practical_gain(self):
        analysis = analyze_rows(self._jobs(), self._rows(delta=-0.2))
        self.assertTrue(analysis["all_training_seeds_favor_c4"])
        self.assertTrue(analysis["measurement_gate_passed"])
        self.assertAlmostEqual(analysis["paired_eval_delta_c4_minus_c0"]["mean"], -0.2)

    def test_measurement_gate_rejects_subthreshold_gain(self):
        analysis = analyze_rows(self._jobs(), self._rows(delta=-0.05))
        self.assertTrue(analysis["all_training_seeds_favor_c4"])
        self.assertFalse(analysis["measurement_gate_passed"])

    def test_secondary_diagnostics_do_not_change_primary_gate(self):
        rows = self._rows(delta=-0.2)
        for row in rows:
            family_offset = 0.01 if row["run_name"].startswith("c4-") else 0.0
            row["secondary_metrics"] = {
                "flow/curvature_proxy_mean": 0.02 + family_offset,
            }
        protocol = {
            "control_family": "C0",
            "candidate_family": "C4",
            "training_seeds": [0, 1],
            "evaluation_seeds": [1, 2, 3],
            "secondary_diagnostics": [
                "flow/curvature_proxy_mean",
                "training/fm/pred_variance",
            ],
            "decision_gate": {
                "minimum_absolute_fid_improvement": 0.1,
                "pooled_evaluation_sd_multiplier": 2.0,
                "require_candidate_to_win_every_training_seed": True,
                "require_complete_evaluation_seed_pairs": True,
            },
        }

        analysis = analyze_rows(self._jobs(), rows, protocol=protocol)

        self.assertTrue(analysis["measurement_gate_passed"])
        diagnostics = analysis["secondary_diagnostics"]
        self.assertEqual(diagnostics["available_metrics"], ["flow/curvature_proxy_mean"])
        self.assertEqual(diagnostics["unavailable_metrics"], ["training/fm/pred_variance"])
        for seed_group in diagnostics["paired_by_training_seed"]:
            self.assertAlmostEqual(
                seed_group["metrics"][0]["delta_candidate_minus_control"]["mean"],
                0.01,
            )

    def test_frozen_protocol_matches_grid_and_checks_loaded_step(self):
        repo_root = Path(__file__).resolve().parents[1]
        jobs = load_jobs(repo_root / "configs/gmm_tide_fid_repeat4_grid.json")
        protocol = load_protocol(
            repo_root / "configs/gmm_tide_fid_repeat_analysis_protocol.json"
        )
        rows = []
        for run_name in jobs:
            for eval_seed in protocol["evaluation_seeds"]:
                rows.append(
                    {
                        "run_name": run_name,
                        "eval_seed": eval_seed,
                        "value": 7.0,
                        "step": protocol["expected_checkpoint_step"],
                        "eval_fid_generations": protocol["generations_per_evaluation_seed"],
                    }
                )

        audit = audit_comparability(jobs, rows, protocol)

        self.assertEqual(audit["status"], "PASS")
        self.assertFalse(audit["errors"])

        rows[0]["step"] = 399999
        audit = audit_comparability(jobs, rows, protocol)
        self.assertEqual(audit["status"], "FAIL")
        self.assertTrue(any("steps=" in error for error in audit["errors"]))


class SubmittedNotebookScrubTest(unittest.TestCase):
    def test_scrub_removes_embedded_secret_values_from_local_notebook(self):
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "outputs": [],
                    "execution_count": None,
                    "source": [
                        'WANDB_API_KEY = "wandb-secret-value"\n',
                        'KAGGLE_CREDENTIAL = json.loads("{\\"username\\": \\"owner\\", \\"key\\": \\"kaggle-secret-value\\"}")\n',
                    ],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            notebook_path = Path(temp_dir) / "submitted.ipynb"
            notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

            result = scrub_notebook_embedded_credentials(notebook_path)

            scrubbed = notebook_path.read_text(encoding="utf-8")
            scrubbed_source = "".join(json.loads(scrubbed)["cells"][0]["source"])
            self.assertTrue(result["ok"])
            self.assertEqual(result["replacements"], 2)
            self.assertEqual(result["key_names"], ["KAGGLE_CREDENTIAL", "WANDB_API_KEY"])
            self.assertNotIn("wandb-secret-value", scrubbed)
            self.assertNotIn("kaggle-secret-value", scrubbed)
            self.assertIn('WANDB_API_KEY = ""', scrubbed_source)
            self.assertIn("KAGGLE_CREDENTIAL = {}", scrubbed_source)


class CrossAccountResumeNotebookTest(unittest.TestCase):
    def test_source_owner_credential_is_required(self):
        config = {
            "resume_kernel_ref": "source-owner/source-run",
            "resume_download_output": True,
        }
        accounts = {
            "source-owner": {"username": "source-owner", "key": "source-key"},
            "runtime-owner": {"username": "runtime-owner", "key": "runtime-key"},
        }
        self.assertEqual(
            resume_download_credential_owner(
                config=config,
                target_owner="runtime-owner",
                accounts=accounts,
            ),
            "source-owner",
        )
        with self.assertRaisesRegex(ValueError, "exact source-owner credential"):
            resume_download_credential_owner(
                config=config,
                target_owner="runtime-owner",
                accounts={"runtime-owner": accounts["runtime-owner"]},
            )

    def test_cross_account_cell_owns_download_and_parent_auth_is_untouched(self):
        notebook = make_notebook(
            {"run_name": "cross-account-test", "resume_output_preloaded": True},
            kaggle_credential={"username": "source-owner", "key": "source-key"},
            cross_account_output_source="KJO_CROSS_ACCOUNT_OUTPUT_RESULT = {'ok': True}\n",
        )
        source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("/tmp/.kaggle_source_owner", source)
        self.assertIn("KJO_CROSS_ACCOUNT_OUTPUT_RESULT", source)
        self.assertNotIn('os.environ["KAGGLE_USERNAME"]', source)
        self.assertNotIn('os.environ["KAGGLE_KEY"]', source)

    def test_notebook_pins_cross_account_compatible_kaggle_cli(self):
        notebook = make_notebook({"run_name": "kaggle-cli-contract"})
        source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn('"kaggle==2.2.3"', source)
        self.assertIn('"kagglesdk==0.1.31"', source)
        self.assertIn('if "--page-size" not in kaggle_help:', source)

    def test_resume_pattern_selects_only_required_artifacts(self):
        pattern = resume_file_pattern({"resume_checkpoint_step": 200000})
        self.assertIn("gmm_stats", pattern)
        self.assertIn("gmm_router", pattern)
        self.assertIn("ckpts/", pattern)
        self.assertIn("pkl", pattern)
        self.assertNotIn("200000", pattern)

    def test_audit_resume_pattern_does_not_request_checkpoint(self):
        pattern = resume_file_pattern(
            {
                "execution_mode": "router_geometry_audit",
                "resume_checkpoint_step": 400000,
            }
        )
        self.assertIn("gmm_stats", pattern)
        self.assertIn("gmm_router", pattern)
        self.assertNotIn("ckpts", pattern)
        self.assertNotIn("400000", pattern)

    def test_audit_notebook_cells_compile(self):
        notebook = make_notebook(
            {
                "run_name": "audit-test",
                "execution_mode": "router_geometry_audit",
                "resume_output_preloaded": True,
            },
            cross_account_output_source="KJO_CROSS_ACCOUNT_OUTPUT_RESULT = {'ok': True}\n",
            router_geometry_audit_script_source="print('audit')\n",
        )
        compile_notebook_cells(notebook)
        source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("router_geometry_audit", source)
        self.assertIn("/tmp/audit_gmm_tide_router_geometry.py", source)


if __name__ == "__main__":
    unittest.main()
