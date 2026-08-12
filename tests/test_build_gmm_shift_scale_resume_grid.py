from __future__ import annotations

from scripts.build_gmm_shift_scale_resume_grid import build_resume_grid


def _parent_grid() -> dict:
    jobs = []
    owners = [f"owner{i}" for i in range(15)]
    families = ["naive_gaussian"] * 5 + ["top2_c01"] * 5 + ["top4_c02"] * 5
    for index, (owner, family) in enumerate(zip(owners, families, strict=True)):
        seed = index % 5 + 1
        jobs.append(
            {
                "candidate_family": family,
                "expected_submit_owner": owner,
                "gmm_router_temperature": 0.75 if family == "top4_c02" else 1.0,
                "gmm_router_topk": {"naive_gaussian": 1, "top2_c01": 2, "top4_c02": 4}[family],
                "gmm_source_center_scale": {"naive_gaussian": 0.0, "top2_c01": 0.1, "top4_c02": 0.2}[family],
                "model_train_type": "naive-gaussian" if family == "naive_gaussian" else "gmm-tide",
                "run_name": f"{family}-s{seed}-parent200",
                "training_seed": seed,
            }
        )
    return {"defaults": {"train_max_steps": 200000}, "jobs": jobs}


def _submit_report(parent_grid: dict) -> dict:
    rows = []
    for index, job in enumerate(parent_grid["jobs"]):
        owner = job["expected_submit_owner"]
        run_name = job["run_name"]
        rows.append(
            {
                "candidate_family": job["candidate_family"],
                "grid_index": index,
                "kernel_id": f"{owner}/{run_name}-remote",
                "owner": owner,
                "run_dir": f"outputs/jobs/{owner}__{run_name}-remote",
                "run_name": run_name,
                "training_seed": job["training_seed"],
            }
        )
    return {"submitted": rows, "failed": [], "not_submitted": []}


def test_build_resume_grid_preserves_exact_parent_matrix_and_requires_gate() -> None:
    parent_grid = _parent_grid()
    payload = build_resume_grid(parent_grid, _submit_report(parent_grid))

    assert len(payload["jobs"]) == 15
    assert payload["defaults"]["train_resume_start_step"] == 200000
    assert payload["defaults"]["train_target_step_abs"] == 400000
    assert payload["defaults"]["resume_require_checkpoint"] is True
    assert payload["defaults"]["resume_attach_kernel_source"] is True
    assert payload["defaults"]["resume_download_output"] is False

    for job in payload["jobs"]:
        assert job["resume_kernel_ref"].startswith(f"{job['expected_submit_owner']}/")
        assert job["run_name"].endswith("-resume400")
        assert job["resume_run_name"].endswith("-parent200")
        assert job["resume_parent_gate"]["terminal_status"] == "COMPLETE"
        assert job["resume_parent_gate"]["checkpoint"].endswith(
            f"ckpts/{job['resume_run_name']}.pkl"
        )
        if job["candidate_family"] == "naive_gaussian":
            assert job["resume_reuse_gmm_router"] is False
            assert "gmm_stats" not in job["resume_parent_gate"]
            assert "router" not in job["resume_parent_gate"]
        else:
            assert job["resume_reuse_gmm_router"] is True
            assert job["resume_parent_gate"]["gmm_stats"].endswith("gmm_stats.npz")
            assert job["resume_parent_gate"]["router"].endswith("gmm_router.pkl")


def test_allocation_plan_maximizes_complete_seed_blocks_with_cross_account_overrides() -> None:
    parent_grid = _parent_grid()
    allocation = {
        "included_training_seeds": [1, 2, 3, 4],
        "destination_overrides": [
            {
                "candidate_family": "top2_c01",
                "destination_owner": "owner4",
                "training_seed": 2,
            },
            {
                "candidate_family": "naive_gaussian",
                "destination_owner": "owner9",
                "training_seed": 4,
            },
        ],
    }
    payload = build_resume_grid(parent_grid, _submit_report(parent_grid), allocation)

    assert len(payload["jobs"]) == 12
    assert {job["training_seed"] for job in payload["jobs"]} == {1, 2, 3, 4}
    assert len({job["expected_submit_owner"] for job in payload["jobs"]}) == 12
    cross_account = [job for job in payload["jobs"] if job["resume_download_output"]]
    assert {(job["candidate_family"], job["training_seed"]) for job in cross_account} == {
        ("top2_c01", 2),
        ("naive_gaussian", 4),
    }
    assert all(job["resume_attach_kernel_source"] is False for job in cross_account)
