from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.build_strict_fid_grid import build_fid_grid
from scripts.build_strict_shift_scale_grid import build_grid
from scripts.submit_gmm_tide_fm_jobs import load_grid, make_notebook
from strict_ablation import (
    build_repro_manifest,
    npz_content_sha256,
    pickle_content_sha256,
    validate_strict_jobs,
    write_repro_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_grid(path: Path, payload: dict) -> list[dict]:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_grid(path)


def _strict_training_grid() -> dict:
    return build_grid(
        base_grid=REPO_ROOT / "configs/gmm_tide_moe2_shift_scale_raw_200k_grid.json",
        artifact_ref="owner/canonical-artifact",
        artifact_run_name="canonical-run",
        artifact_block="gmm-seed-0",
        artifact_seed=0,
        training_seeds=[0, 1, 2],
        scales=[0.75, 0.875, 1.125, 1.25],
        dataset_seed=52000,
        vae_seed=62000,
        eval_fid_seeds="101,202,303,404,505",
    )


def _compile_notebook_cells(notebook: dict) -> None:
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if source.strip():
            compile(source, f"strict_notebook_cell_{index}.py", "exec")


def test_semantic_hashes_ignore_container_serialization_details(tmp_path: Path):
    first_npz = tmp_path / "first.npz"
    second_npz = tmp_path / "second.npz"
    np.savez(first_npz, b=np.asarray([3, 4]), a=np.asarray([1.0, 2.0]))
    np.savez(second_npz, a=np.asarray([1.0, 2.0]), b=np.asarray([3, 4]))
    assert npz_content_sha256(first_npz) == npz_content_sha256(second_npz)

    first_pkl = tmp_path / "first.pkl"
    second_pkl = tmp_path / "second.pkl"
    with first_pkl.open("wb") as handle:
        pickle.dump({"b": np.asarray([3, 4]), "a": {"x": 1}}, handle, protocol=4)
    with second_pkl.open("wb") as handle:
        pickle.dump({"a": {"x": 1}, "b": np.asarray([3, 4])}, handle, protocol=5)
    assert pickle_content_sha256(first_pkl) == pickle_content_sha256(second_pkl)


def test_reused_artifact_must_match_source_hash_and_block(tmp_path: Path):
    gmm_path = tmp_path / "gmm_stats.npz"
    router_path = tmp_path / "gmm_router.pkl"
    np.savez(gmm_path, pi=np.asarray([0.4, 0.6]), mu=np.asarray([[1.0], [2.0]]))
    with router_path.open("wb") as handle:
        pickle.dump({"params": {"w": np.asarray([1.0])}, "config": {}}, handle)

    source_config = {
        "strict_ablation": 1,
        "strict_artifact_block": "gmm-seed-0",
        "execution_mode": "artifact_prep",
        "run_name": "canonical",
        "dataset_seed": 1,
        "gmm_prep_seed": 2,
        "gmm_init_seed": 3,
        "gmm_mix_seed": 4,
        "router_seed": 5,
    }
    source_manifest_path = tmp_path / "source.json"
    source_manifest = write_repro_manifest(
        source_manifest_path,
        source_config,
        gmm_path,
        router_path,
    )

    reused = build_repro_manifest(
        {
            "strict_ablation": 1,
            "strict_artifact_block": "gmm-seed-0",
            "execution_mode": "train",
            "run_name": "treatment",
            "dataset_seed": 10,
            "vae_seed": 11,
            "training_seed": 12,
        },
        gmm_path,
        router_path,
        source_artifact_manifest_path=source_manifest_path,
    )
    assert reused["artifacts"]["gmm_content_sha256"] == source_manifest["artifacts"]["gmm_content_sha256"]
    assert reused["source_artifact_manifest"]["run_name"] == "canonical"

    with pytest.raises(ValueError, match="artifact block"):
        build_repro_manifest(
            {
                "strict_ablation": 1,
                "strict_artifact_block": "gmm-seed-1",
                "execution_mode": "train",
            },
            gmm_path,
            router_path,
            source_artifact_manifest_path=source_manifest_path,
        )


def test_canonical_artifact_grid_has_explicit_independent_seeds():
    jobs = load_grid(REPO_ROOT / "configs/gmm_tide_moe2_strict_artifacts_grid.json")
    assert len(jobs) == 3
    assert [job["strict_artifact_block"] for job in jobs] == [
        "strict-gmmseed0",
        "strict-gmmseed1",
        "strict-gmmseed2",
    ]
    assert {job["gmm_init_seed"] for job in jobs} == {0, 1, 2}
    assert {job["dataset_seed"] for job in jobs} == {42000}
    assert {job["gmm_prep_seed"] for job in jobs} == {1000}
    assert {job["router_seed"] for job in jobs} == {2000}
    for job in jobs:
        assert job["execution_mode"] == "artifact_prep"
        assert job["strict_deterministic_data"] == 1
        for key in ("dataset_seed", "gmm_prep_seed", "gmm_init_seed", "gmm_mix_seed", "router_seed"):
            assert key in job


def test_training_grid_is_paired_and_notebook_cells_compile(tmp_path: Path):
    jobs = _write_grid(tmp_path / "training.json", _strict_training_grid())
    assert len(jobs) == 18
    assert {job["training_seed"] for job in jobs} == {0, 1, 2}
    assert {job["resume_kernel_ref"] for job in jobs} == {"owner/canonical-artifact"}
    assert {job["dataset_seed"] for job in jobs} == {52000}
    assert {job["vae_seed"] for job in jobs} == {62000}
    treatments = {
        (job["gmm_source_shift_mean"], job["gmm_source_center_scale"])
        for job in jobs
        if job["training_seed"] == 0
    }
    assert treatments == {
        (0, 1.0),
        (1, 1.0),
        (1, 0.75),
        (1, 0.875),
        (1, 1.125),
        (1, 1.25),
    }
    _compile_notebook_cells(make_notebook(jobs[0]))


def test_validator_rejects_unpaired_data_seed_and_multiple_artifact_sources(tmp_path: Path):
    jobs = _write_grid(tmp_path / "training.json", _strict_training_grid())
    jobs[1]["dataset_seed"] += 1
    with pytest.raises(ValueError, match="outside the treatment allowlist"):
        validate_strict_jobs(jobs)

    jobs = _write_grid(tmp_path / "training-again.json", _strict_training_grid())
    seed_one_job = next(job for job in jobs if job["training_seed"] == 1)
    seed_one_job["resume_kernel_ref"] = "owner/different-artifact"
    with pytest.raises(ValueError, match="multiple GMM/router sources"):
        validate_strict_jobs(jobs)


def test_fid_grid_uses_repeated_eval_seeds_for_every_checkpoint(tmp_path: Path):
    training_path = tmp_path / "training.json"
    training_path.write_text(json.dumps(_strict_training_grid()), encoding="utf-8")
    training_jobs = load_grid(training_path)
    checkpoint_path = tmp_path / "checkpoints.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "run_name": job["run_name"],
                        "kernel_ref": f"owner/checkpoint-{index}",
                    }
                    for index, job in enumerate(training_jobs)
                ]
            }
        ),
        encoding="utf-8",
    )
    fid_payload = build_fid_grid(
        training_path,
        checkpoint_path,
        "101,202,303,404,505",
        50048,
    )
    fid_jobs = _write_grid(tmp_path / "fid.json", fid_payload)
    assert len(fid_jobs) == len(training_jobs)
    assert all(job["execution_mode"] == "fid_repeats" for job in fid_jobs)
    assert all(job["resume_require_checkpoint"] is True for job in fid_jobs)
    assert all(job["eval_fid_seeds"] == "101,202,303,404,505" for job in fid_jobs)
    assert all(job["eval_fid_generations"] == 50048 for job in fid_jobs)
    _compile_notebook_cells(make_notebook(fid_jobs[0]))
