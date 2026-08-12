from __future__ import annotations

from scripts.summarize_gmm_shift_scale_quota_forecast import build_quota_forecast


def test_forecast_projects_parent_residual_before_resume_and_fid() -> None:
    quota = {
        "generated_at_utc": "2026-08-12T00:00:00Z",
        "kjo_version": "0.12.0",
        "entries": [
            {
                "owner": "owner1",
                "weekly_remaining_s": 250.0,
                "accounting_confidence": "medium",
                "untracked_usage_possible": True,
                "usage": {
                    "records": [
                        {
                            "kernel_id": "owner1/a-parent200",
                            "seconds": 20.0,
                            "source": "running_status_estimate",
                        }
                    ]
                },
            },
            {
                "owner": "owner2",
                "weekly_remaining_s": 150.0,
                "accounting_confidence": "high",
                "untracked_usage_possible": False,
                "usage": {
                    "records": [
                        {
                            "kernel_id": "owner2/b-parent200",
                            "seconds": 50.0,
                            "source": "running_status_estimate",
                        }
                    ]
                },
            },
        ],
    }
    grid = {
        "jobs": [
            {"run_name": "a-resume400", "expected_submit_owner": "owner1"},
            {"run_name": "b-resume400", "expected_submit_owner": "owner2"},
        ]
    }
    allocation = {"runtime_estimate": {"historical_max_seconds": 100.0}}

    result = build_quota_forecast(quota, grid, allocation, fid_runtime_s=20.0)

    assert result["summary"]["resume_safe"] == 2
    assert result["summary"]["resume_and_fid_safe"] == 1
    assert result["summary"]["minimum_margin_after_resume_s"] == 0.0
    assert result["summary"]["minimum_margin_after_resume_and_fid_s"] == -20.0
    assert result["rows"][0]["projected_remaining_after_parent_s"] == 170.0
    assert result["rows"][0]["margin_after_resume_and_fid_s"] == 50.0
