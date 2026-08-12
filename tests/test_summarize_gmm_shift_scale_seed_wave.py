from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_gmm_shift_scale_seed_wave import build_wave_state


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_wave_state_requires_terminal_artifacts_audit_and_recorded_gate(tmp_path: Path) -> None:
    gate_root = tmp_path / "gates"
    parent_rows = []
    resume_jobs = []
    for index in range(15):
        owner = f"owner{index}"
        parent_name = f"parent{index}"
        kernel_id = f"{owner}/{parent_name}"
        run_dir = tmp_path / parent_name
        _write_json(
            run_dir / "status/status_result.json",
            {"record": {"normalized_status": "COMPLETE", "checked_at_utc": "2026-08-12T18:00:00Z"}},
        )
        checkpoint = run_dir / f"output/ckpts/{parent_name}.pkl"
        manifest = run_dir / f"output/gmm_tide_fm/{parent_name}/diagnostics/train_metrics_summary.json"
        for path in (checkpoint, manifest):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"evidence")
        summary = run_dir / "reports/summary.json"
        audit = run_dir / "reports/audit_run_dir.json"
        _write_json(summary, {"ok": True})
        _write_json(audit, {"ok": True})
        _write_json(
            gate_root / f"{owner}__{parent_name}.json",
            {"ok": True, "kernel_id": kernel_id},
        )
        parent_rows.append(
            {
                "candidate_family": "naive_gaussian",
                "kernel_id": kernel_id,
                "owner": owner,
                "run_dir": str(run_dir),
                "training_seed": index % 5 + 1,
            }
        )
        resume_jobs.append(
            {
                "resume_kernel_ref": kernel_id,
                "resume_parent_gate": {
                    "audit": str(audit),
                    "checkpoint": str(checkpoint),
                    "diagnostic_manifest": str(manifest),
                    "summary": str(summary),
                    "terminal_status": "COMPLETE",
                },
                "run_name": f"resume{index}",
            }
        )

    payload = build_wave_state({"submitted": parent_rows}, {"jobs": resume_jobs}, gate_root)

    assert payload["summary"]["status_counts"] == {"COMPLETE": 15}
    assert payload["summary"]["gate_inputs_ready"] == 15
    assert payload["summary"]["gate_recorded"] == 15
    assert payload["summary"]["resume_submit_ready"] == 15
