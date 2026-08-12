from __future__ import annotations

import pytest

from scripts.build_gmm_shift_scale_training_seed_fid_grid import build_fid_grid


def _resume_grid() -> dict:
    jobs = []
    for family, token, topk, scale in (
        ("naive_gaussian", "naive", 1, 0.0),
        ("top2_c01", "t2c01", 2, 0.1),
        ("top4_c02", "t4c02", 4, 0.2),
    ):
        for seed in range(1, 5):
            jobs.append(
                {
                    "candidate_family": family,
                    "expected_submit_owner": f"source-{family}-{seed}",
                    "gmm_router_temperature": 0.75 if topk == 4 else 1.0,
                    "gmm_router_topk": topk,
                    "gmm_source_center_scale": scale,
                    "model_train_type": "naive-gaussian" if topk == 1 else "gmm-tide",
                    "resume_kernel_ref": f"parent/{family}-s{seed}-200k",
                    "resume_run_name": f"{family}-s{seed}-parent200",
                    "run_name": f"{token}-s{seed}-resume400",
                    "training_seed": seed,
                }
            )
    return {
        "defaults": {
            "branch": "moe2",
            "eval_fid_timesteps": "1,4,32,128",
            "protocol_id": "gmm-shift-scale-training-seed-replication-v1",
            "train_max_steps": 200000,
            "train_resume_start_step": 200000,
            "train_target_step_abs": 400000,
        },
        "jobs": jobs,
    }


def _accepted(run_name: str, *, owner: str) -> dict:
    return {
        "kernel_id": f"{owner}/{run_name}-remote",
        "owner": owner,
        "run_dir": f"outputs/kaggle_jobs/resume/{owner}__{run_name}-remote",
        "run_name": run_name,
    }


def test_fid_grid_is_metric_blind_transform_of_accepted_children() -> None:
    report = {
        "submitted": [
            _accepted("naive-s1-resume400", owner="dest-naive"),
            _accepted("t2c01-s2-resume400", owner="dest-top2"),
            _accepted("t4c02-s3-resume400", owner="dest-top4"),
        ]
    }

    payload = build_fid_grid(_resume_grid(), report)

    assert payload["provenance"]["build_policy"].startswith("metric-blind")
    assert payload["defaults"]["execution_mode"] == "fid_repeats"
    assert payload["defaults"]["eval_fid_seeds"] == "101,202,303,404,505"
    assert payload["defaults"]["eval_fid_generations"] == 50048
    assert payload["defaults"]["eval_fid_timesteps"] == "128"
    assert payload["defaults"]["resume_expected_checkpoint_step"] == 400000
    assert payload["defaults"]["train_max_steps"] == 0

    assert [job["run_name"] for job in payload["jobs"]] == [
        "fidrep-naive-s1-400k",
        "fidrep-t2c01-s2-400k",
        "fidrep-t4c02-s3-400k",
    ]
    assert [job["expected_submit_owner"] for job in payload["jobs"]] == [
        "dest-naive",
        "dest-top2",
        "dest-top4",
    ]
    assert all(job["resume_attach_kernel_source"] is True for job in payload["jobs"])
    assert all(job["resume_download_output"] is False for job in payload["jobs"])

    naive_gate = payload["jobs"][0]["resume_parent_gate"]
    assert naive_gate["checkpoint"].endswith("ckpts/naive-s1-resume400.pkl")
    assert "gmm_stats" not in naive_gate
    assert "router" not in naive_gate
    top2_gate = payload["jobs"][1]["resume_parent_gate"]
    assert top2_gate["gmm_stats"].endswith("gmm_stats.npz")
    assert top2_gate["router"].endswith("gmm_router.pkl")


def test_fid_grid_rejects_accepted_child_outside_frozen_resume_grid() -> None:
    with pytest.raises(ValueError, match="absent from frozen resume grid"):
        build_fid_grid(
            _resume_grid(),
            {"submitted": [_accepted("unknown-s1-resume400", owner="dest")]},
        )


def test_fid_grid_rejects_duplicate_accepted_resume_identity() -> None:
    row = _accepted("naive-s1-resume400", owner="dest-naive")
    with pytest.raises(ValueError, match="duplicate accepted resume identity"):
        build_fid_grid(_resume_grid(), {"submitted": [row, dict(row)]})
