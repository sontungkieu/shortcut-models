from __future__ import annotations

import json
from pathlib import Path

from scripts.process_gmm_shift_scale_seed_parents import (
    _manifest_step_check,
    build_parent_items,
    gate_command,
    heavy_download_pattern,
)


def _fixture(tmp_path: Path) -> tuple[dict, dict]:
    parents = []
    resume_jobs = []
    for index in range(15):
        owner = f"owner{index}"
        run_name = f"parent{index}"
        kernel_id = f"{owner}/{run_name}-remote"
        run_dir = tmp_path / f"{owner}__{run_name}-remote"
        parents.append(
            {
                "kernel_id": kernel_id,
                "owner": owner,
                "run_name": run_name,
                "run_dir": str(run_dir),
            }
        )
        if index % 5 == 4:
            continue
        diagnostics = run_dir / f"output/gmm_tide_fm/{run_name}/diagnostics"
        gate = {
            "checkpoint": str(run_dir / f"output/ckpts/{run_name}.pkl"),
            "diagnostic_manifest": str(diagnostics / "train_metrics_summary.json"),
            "summary": str(run_dir / "reports/summary.json"),
            "audit": str(run_dir / "reports/audit_run_dir.json"),
        }
        if index >= 5:
            gate.update(
                {
                    "gmm_stats": str(run_dir / f"output/gmm_tide_fm/{run_name}/gmm_stats.npz"),
                    "router": str(run_dir / f"output/gmm_tide_fm/{run_name}/gmm_router.pkl"),
                }
            )
        resume_jobs.append(
            {
                "resume_kernel_ref": kernel_id,
                "resume_parent_gate": gate,
            }
        )
    return {"submitted": parents}, {
        "defaults": {"resume_expected_checkpoint_step": 200000},
        "jobs": resume_jobs,
    }


def test_build_parent_items_preserves_frozen_twelve_parent_allocation(tmp_path: Path) -> None:
    submit_report, resume_grid = _fixture(tmp_path)
    items = build_parent_items(
        submit_report,
        resume_grid,
        project_root=tmp_path,
        gate_root=tmp_path / "gates",
    )

    assert len(items) == 15
    assert sum(item.allocated_for_resume for item in items) == 12
    assert {index for index, item in enumerate(items) if not item.allocated_for_resume} == {4, 9, 14}
    assert all(item.expected_checkpoint_step == 200000 for item in items if item.allocated_for_resume)


def test_heavy_pattern_and_gate_command_include_exact_resume_artifacts(tmp_path: Path) -> None:
    submit_report, resume_grid = _fixture(tmp_path)
    item = build_parent_items(
        submit_report,
        resume_grid,
        project_root=tmp_path,
        gate_root=tmp_path / "gates",
    )[5]

    pattern = heavy_download_pattern(item)
    assert "ckpts/parent5\\.pkl" in pattern
    assert "gmm_stats\\.npz" in pattern
    assert "gmm_router\\.pkl" in pattern
    assert "gmm_tide_fm/parent5/diagnostics/.*" in pattern
    assert "kaggle_job_ops/.*" in pattern

    command = gate_command(item, Path("/tmp/kjo.py"), record=True)
    assert "--record" in command
    assert "--gmm-stats" in command
    assert "--router" in command
    assert command[command.index("--kernel-id") + 1] == "owner5/parent5-remote"


def test_manifest_step_gate_rejects_wrong_parent_step(tmp_path: Path) -> None:
    submit_report, resume_grid = _fixture(tmp_path)
    item = build_parent_items(
        submit_report,
        resume_grid,
        project_root=tmp_path,
        gate_root=tmp_path / "gates",
    )[0]
    manifest = item.gate_spec["diagnostic_manifest"]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"step": 199999}) + "\n", encoding="utf-8")

    assert _manifest_step_check(item) == {
        "checked": True,
        "ok": False,
        "expected": 200000,
        "actual": 199999,
    }

    manifest.write_text(json.dumps({"step": 200000}) + "\n", encoding="utf-8")
    assert _manifest_step_check(item)["ok"] is True
