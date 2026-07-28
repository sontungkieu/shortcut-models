from __future__ import annotations

import hashlib
import json
import pickle
import struct
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STRICT_TREATMENT_KEYS = {
    "candidate_family",
    "expected_submit_owner",
    "gmm_source_center_scale",
    "gmm_source_shift_mean",
    "grid_index",
    "run_name",
}

STRICT_REQUIRED_SEED_KEYS = {
    "dataset_seed",
    "training_seed",
    "vae_seed",
}

STRICT_ARTIFACT_REQUIRED_SEED_KEYS = {
    "dataset_seed",
    "gmm_init_seed",
    "gmm_mix_seed",
    "gmm_prep_seed",
    "router_seed",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_semantic_hash(digest, value: Any) -> None:
    if isinstance(value, np.ndarray):
        digest.update(b"array:")
        digest.update(value.dtype.str.encode("utf-8"))
        digest.update(json.dumps(value.shape).encode("utf-8"))
        if value.dtype.hasobject:
            _update_semantic_hash(digest, value.tolist())
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
        return
    if isinstance(value, np.generic):
        _update_semantic_hash(digest, value.item())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping:")
        for key in sorted(value, key=lambda item: str(item)):
            _update_semantic_hash(digest, str(key))
            _update_semantic_hash(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence:")
        for item in value:
            _update_semantic_hash(digest, item)
        return
    if value is None:
        digest.update(b"none")
    elif isinstance(value, bool):
        digest.update(b"bool:1" if value else b"bool:0")
    elif isinstance(value, int):
        digest.update(f"int:{value}".encode("utf-8"))
    elif isinstance(value, float):
        digest.update(b"float:")
        digest.update(struct.pack(">d", value))
    elif isinstance(value, bytes):
        digest.update(b"bytes:")
        digest.update(value)
    elif isinstance(value, str):
        digest.update(b"str:")
        digest.update(value.encode("utf-8"))
    else:
        raise TypeError(f"Unsupported value in semantic artifact hash: {type(value).__name__}")


def npz_content_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=True) as payload:
        for name in sorted(payload.files):
            _update_semantic_hash(digest, name)
            _update_semantic_hash(digest, np.asarray(payload[name]))
    return digest.hexdigest()


def pickle_content_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    digest = hashlib.sha256()
    _update_semantic_hash(digest, payload)
    return digest.hexdigest()


def artifact_fingerprints(
    gmm_stats_path: str | Path,
    router_path: str | Path,
) -> dict[str, str]:
    return {
        "gmm_file_sha256": sha256_file(gmm_stats_path),
        "gmm_content_sha256": npz_content_sha256(gmm_stats_path),
        "router_file_sha256": sha256_file(router_path),
        "router_content_sha256": pickle_content_sha256(router_path),
    }


def build_repro_manifest(
    config: Mapping[str, Any],
    gmm_stats_path: str | Path,
    router_path: str | Path,
    source_artifact_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    fingerprints = artifact_fingerprints(gmm_stats_path, router_path)
    expected = {
        "gmm_content_sha256": str(config.get("strict_expected_gmm_content_sha256", "")),
        "router_content_sha256": str(config.get("strict_expected_router_content_sha256", "")),
    }
    mismatches = {
        key: {"expected": expected[key], "actual": fingerprints[key]}
        for key in expected
        if expected[key] and expected[key] != fingerprints[key]
    }
    if mismatches:
        raise ValueError(f"Strict artifact fingerprint mismatch: {json.dumps(mismatches, sort_keys=True)}")
    source_artifact_manifest = None
    if source_artifact_manifest_path:
        source_path = Path(source_artifact_manifest_path)
        if source_path.exists():
            source_artifact_manifest = json.loads(source_path.read_text(encoding="utf-8"))
            source_fingerprints = source_artifact_manifest.get("artifacts", {})
            source_mismatches = {
                key: {"source": source_fingerprints.get(key), "actual": fingerprints[key]}
                for key in (
                    "gmm_file_sha256",
                    "gmm_content_sha256",
                    "router_file_sha256",
                    "router_content_sha256",
                )
                if source_fingerprints.get(key) != fingerprints[key]
            }
            if source_mismatches:
                raise ValueError(
                    "Copied artifact does not match its source manifest: "
                    + json.dumps(source_mismatches, sort_keys=True)
                )
            source_block = str(source_artifact_manifest.get("strict_artifact_block", ""))
            current_block = str(config.get("strict_artifact_block", ""))
            if source_block != current_block:
                raise ValueError(
                    "Copied artifact block does not match the current strict block: "
                    f"source={source_block!r}, current={current_block!r}"
                )
    execution_mode = str(config.get("execution_mode", "train"))
    artifact_build_seeds = None
    if execution_mode == "artifact_prep":
        artifact_build_seeds = {
            "dataset_seed": int(config["dataset_seed"]),
            "gmm_prep_seed": int(config["gmm_prep_seed"]),
            "gmm_init_seed": int(config["gmm_init_seed"]),
            "gmm_mix_seed": int(config["gmm_mix_seed"]),
            "router_seed": int(config["router_seed"]),
        }
    return {
        "schema_version": 1,
        "strict_ablation": bool(config.get("strict_ablation", False)),
        "strict_artifact_block": str(config.get("strict_artifact_block", "")),
        "repo_commit": str(config.get("repo_commit", "")),
        "run_name": str(config.get("run_name", "")),
        "execution_mode": execution_mode,
        "runtime_seeds": {
            "dataset_seed": int(config.get("dataset_seed", 42)),
            "vae_seed": int(config.get("vae_seed", 42)),
            "training_seed": (
                int(config["training_seed"])
                if config.get("training_seed") not in (None, "")
                else None
            ),
            "eval_fid_seeds": str(config.get("eval_fid_seeds", "42")),
        },
        "artifact_build_seeds": artifact_build_seeds,
        "treatment": {
            "gmm_source_shift_mean": int(config.get("gmm_source_shift_mean", 0)),
            "gmm_source_center_scale": float(config.get("gmm_source_center_scale", 1.0)),
        },
        "artifacts": fingerprints,
        "source_artifact_manifest": source_artifact_manifest,
    }


def write_repro_manifest(
    path: str | Path,
    config: Mapping[str, Any],
    gmm_stats_path: str | Path,
    router_path: str | Path,
    source_artifact_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = build_repro_manifest(
        config,
        gmm_stats_path,
        router_path,
        source_artifact_manifest_path=source_artifact_manifest_path,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _missing_keys(job: Mapping[str, Any], required: Iterable[str]) -> list[str]:
    return sorted(key for key in required if key not in job or job[key] in ("", None))


def validate_strict_jobs(jobs: list[dict[str, Any]]) -> None:
    strict_jobs = [job for job in jobs if bool(job.get("strict_ablation", False))]
    if not strict_jobs:
        return
    if len(strict_jobs) != len(jobs):
        raise ValueError("A strict-ablation grid cannot mix strict and legacy jobs")

    training_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    training_sources: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
    for job in strict_jobs:
        if not bool(job.get("strict_deterministic_data", False)):
            raise ValueError(
                f"{job.get('run_name', '<unnamed>')}: strict_ablation requires strict_deterministic_data=1"
            )
        if int(job.get("debug_overfit", 0) or 0):
            raise ValueError(f"{job.get('run_name', '<unnamed>')}: strict ablations forbid debug_overfit")
        block = str(job.get("strict_artifact_block", "")).strip()
        if not block:
            raise ValueError(f"{job.get('run_name', '<unnamed>')}: strict_artifact_block is required")

        execution_mode = str(job.get("execution_mode", "train")).strip().lower()
        if execution_mode == "artifact_prep":
            missing = _missing_keys(job, STRICT_ARTIFACT_REQUIRED_SEED_KEYS)
            if missing:
                raise ValueError(
                    f"{job.get('run_name', '<unnamed>')}: missing explicit artifact seeds {missing}"
                )
            if job.get("resume_kernel_ref"):
                raise ValueError("artifact_prep must create a new canonical pair, not resume another artifact")
            continue

        missing = _missing_keys(job, STRICT_REQUIRED_SEED_KEYS)
        if missing:
            raise ValueError(f"{job.get('run_name', '<unnamed>')}: missing explicit seeds {missing}")
        if not job.get("resume_kernel_ref"):
            raise ValueError(
                f"{job.get('run_name', '<unnamed>')}: strict jobs require resume_kernel_ref"
            )
        if execution_mode == "fid_repeats":
            if not bool(job.get("resume_require_checkpoint", True)):
                raise ValueError("strict fid_repeats jobs must require their source checkpoint")
            if not str(job.get("eval_fid_seeds", "")).strip():
                raise ValueError("strict fid_repeats jobs require explicit eval_fid_seeds")
            continue
        if execution_mode != "train":
            continue
        if bool(job.get("resume_require_checkpoint", True)):
            raise ValueError(
                f"{job.get('run_name', '<unnamed>')}: set resume_require_checkpoint=false for artifact-only reuse"
            )
        if not bool(job.get("resume_reuse_gmm_router", False)):
            raise ValueError(
                f"{job.get('run_name', '<unnamed>')}: strict treatment jobs must reuse GMM/router"
            )
        training_groups[(block, int(job["training_seed"]))].append(job)
        training_sources[block].add(
            (
                str(job.get("resume_kernel_ref", "")),
                str(job.get("resume_run_name", "")),
                str(job.get("strict_expected_gmm_content_sha256", "")),
                str(job.get("strict_expected_router_content_sha256", "")),
            )
        )

    for block, sources in training_sources.items():
        if len(sources) != 1:
            raise ValueError(
                f"Strict artifact block {block!r} references multiple GMM/router sources: "
                f"{sorted(sources)}"
            )

    for (block, training_seed), group in training_groups.items():
        reference = group[0]
        reference_base = {
            key: value
            for key, value in reference.items()
            if key not in STRICT_TREATMENT_KEYS
        }
        for candidate in group[1:]:
            candidate_base = {
                key: value
                for key, value in candidate.items()
                if key not in STRICT_TREATMENT_KEYS
            }
            if candidate_base != reference_base:
                differing = sorted(
                    key
                    for key in set(reference_base) | set(candidate_base)
                    if reference_base.get(key) != candidate_base.get(key)
                )
                raise ValueError(
                    "Strict paired jobs differ outside the treatment allowlist "
                    f"for block={block!r}, training_seed={training_seed}: {differing}"
                )
