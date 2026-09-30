"""Hugging Face Hub offline mode for disconnected clusters (S3 / PVC / git test_data_ref).

Eval Hub stages benchmark data under ``/test_data``. Inspect tasks (Open-Telco, inspect-evals)
load datasets via ``datasets`` / the Hub. Without offline env vars they call huggingface.co and
fail on air-gapped clusters.

Detection mirrors ``lm-evaluation-harness`` (tokenizer under ``/test_data`` + co-located
``dataset_dict.json`` bundles) and extends it for Inspect jobs that only set ``test_data_ref``
(e.g. Open-Telco collections) without a tokenizer parameter.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEST_DATA_DIR = "/test_data"
_JOB_SPEC_ALLOWED_ROOT = Path("/meta")
_WRITE_PROBE_NAME = ".evalhub-write-probe"


def _resolve_job_spec_path_for_read(path: str) -> Path | None:
    if not isinstance(path, str) or not path.strip():
        return None
    try:
        resolved = Path(path.strip()).resolve()
    except (OSError, ValueError):
        return None
    try:
        allowed = _JOB_SPEC_ALLOWED_ROOT.resolve()
    except OSError:
        allowed = _JOB_SPEC_ALLOWED_ROOT
    if not resolved.is_relative_to(allowed):
        return None
    return resolved


def _read_job_spec_dict_from_path(path: str) -> dict[str, Any]:
    resolved = _resolve_job_spec_path_for_read(path)
    if resolved is None:
        return {}
    try:
        with open(resolved, encoding="utf-8") as f:
            spec = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return spec if isinstance(spec, dict) else {}


def _has_test_data_ref(spec: dict[str, Any]) -> bool:
    ref = spec.get("test_data_ref")
    if not isinstance(ref, dict):
        return False
    return any(ref.get(key) for key in ("s3", "pvc", "git"))


def _extract_tokenizer_parameter(parameters: dict[str, Any]) -> str | None:
    raw = parameters.get("tokenizer")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _dataset_material_present_under_test_data(
    root_resolved: Path, tokenizer_resolved: Path
) -> bool:
    try:
        for child in root_resolved.iterdir():
            if not child.is_dir():
                continue
            try:
                cres = child.resolve()
            except OSError:
                continue
            if cres == tokenizer_resolved:
                continue
            if tokenizer_resolved.is_relative_to(cres):
                continue
            if (child / "dataset_dict.json").is_file():
                return True
    except OSError:
        return False
    return False


def _infer_auto_offline_from_local_test_data(
    parameters: dict[str, Any],
    *,
    test_data_root: str | Path | None = None,
) -> bool:
    """True when ``parameters.tokenizer`` points into ``test_data_root`` and datasets are co-located."""
    tokenizer_str = _extract_tokenizer_parameter(parameters)
    if not tokenizer_str or not tokenizer_str.startswith("/"):
        return False

    root = Path(test_data_root if test_data_root is not None else TEST_DATA_DIR)
    try:
        root_res = root.resolve()
    except OSError:
        return False
    if not root_res.is_dir():
        return False

    tokenizer_path = Path(tokenizer_str)
    try:
        tok_res = tokenizer_path.resolve()
    except OSError:
        return False

    if tok_res == root_res or not tok_res.is_relative_to(root_res):
        return False
    try:
        if not tok_res.exists():
            return False
    except OSError:
        return False

    return _dataset_material_present_under_test_data(root_res, tok_res)


def _test_data_mount_usable(test_data_root: str | Path | None = None) -> bool:
    root = Path(test_data_root if test_data_root is not None else TEST_DATA_DIR)
    try:
        if not root.is_dir():
            return False
        return any(root.iterdir())
    except OSError:
        return False


def should_use_hf_offline(
    parameters: dict[str, Any],
    *,
    job_spec_path: str | None = None,
    test_data_root: str | Path | None = None,
) -> bool:
    """Whether to pin Hugging Face caches to staged ``/test_data`` and disable Hub downloads."""
    root = test_data_root if test_data_root is not None else TEST_DATA_DIR
    if _infer_auto_offline_from_local_test_data(parameters, test_data_root=root):
        return True

    path = job_spec_path or os.environ.get("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
    spec = _read_job_spec_dict_from_path(path)
    if spec and _has_test_data_ref(spec) and _test_data_mount_usable(root):
        return True
    return False


def job_spec_requests_test_data(path: str | None = None) -> bool:
    """True when the mounted job spec JSON includes ``test_data_ref``."""
    job_path = path or os.environ.get("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
    return _has_test_data_ref(_read_job_spec_dict_from_path(job_path))


def _is_writable_dir(path: Path) -> bool:
    """True when ``path`` can be created (if absent) and written to by the current UID."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    probe = path / _WRITE_PROBE_NAME
    try:
        probe.touch()
        probe.unlink()
    except OSError:
        return False
    return True


def _hf_home_candidates(staged_root: Path) -> list[Path]:
    """Writable-cache candidates, most specific first. Never the staged data mount."""
    candidates: list[Path] = []
    explicit = os.environ.get("HF_HOME", "").strip()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    home = os.environ.get("HOME", "").strip()
    if explicit:
        candidates.append(Path(explicit))
    if xdg:
        candidates.append(Path(xdg) / "huggingface")
    if home:
        candidates.append(Path(home) / ".cache" / "huggingface")
    candidates.append(Path(tempfile.gettempdir()) / "huggingface")

    staged = staged_root.resolve() if staged_root.exists() else staged_root
    out: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved == staged or resolved.is_relative_to(staged):
            continue
        if resolved not in out:
            out.append(resolved)
    return out


def resolve_writable_hf_home(staged_root: str | Path = TEST_DATA_DIR) -> Path:
    """First writable Hugging Face cache root outside the staged test data.

    ``/test_data`` is populated by the Eval Hub sync and is routinely read-only or owned
    by another UID — on OpenShift the pod gets an arbitrary UID in group 0 — so it can
    never serve as ``HF_HOME``: huggingface_hub and datasets write lock and fingerprint
    files there even with downloads disabled.
    """
    candidates = _hf_home_candidates(Path(staged_root))
    for candidate in candidates:
        if _is_writable_dir(candidate):
            return candidate
    raise RuntimeError(
        "No writable Hugging Face cache directory found. Tried: "
        + ", ".join(str(c) for c in candidates)
        + ". Set HF_HOME to a writable path (an emptyDir volume works) on the job pod."
    )


def _staged_hub_cache(staged_root: Path) -> Path | None:
    """Locate a Hugging Face hub cache layout inside the staged test data, if there is one."""
    hub = staged_root / "hub"
    if hub.is_dir():
        return hub
    try:
        for child in staged_root.iterdir():
            if child.is_dir() and child.name.startswith(("models--", "datasets--")):
                return staged_root
    except OSError:
        return None
    return None


def configure_hf_offline_environment(
    staged_root: str,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Use local Hugging Face caches only (disconnected / no huggingface.co).

    ``HF_HOME`` and ``HF_DATASETS_CACHE`` stay on a writable path; only the hub cache is
    pointed at the staged mount, and only when that mount actually holds a hub cache
    layout. A read-only hub cache is safe to read: ``hf_hub_download`` calls
    ``os.makedirs(storage_folder, exist_ok=True)`` before its ``local_files_only``
    branch, which is a no-op when the staged repo folder already exists.

    Returns the environment values that were applied.
    """
    staged = Path(staged_root)
    hf_home = resolve_writable_hf_home(staged)
    hub_cache = _staged_hub_cache(staged) or (hf_home / "hub")
    values = {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hub_cache),
        "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_EVALUATE_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for key, value in values.items():
        os.environ[key] = value
        if env is not None:
            env[key] = value
    return values


def ensure_test_data_ready_for_offline(
    parameters: dict[str, Any],
    *,
    job_spec_path: str | None = None,
    test_data_root: str | Path | None = None,
) -> None:
    """Raise when the job expects offline data but ``/test_data`` is missing or empty."""
    root_path = test_data_root if test_data_root is not None else TEST_DATA_DIR
    if not should_use_hf_offline(
        parameters, job_spec_path=job_spec_path, test_data_root=root_path
    ):
        if job_spec_requests_test_data(job_spec_path) and not _test_data_mount_usable(root_path):
            raise RuntimeError(
                f"Job spec includes test_data_ref but {root_path} is missing or empty. "
                "Ensure the test-data init container populated /test_data before the adapter starts."
            )
        return

    root = Path(root_path)
    if not root.is_dir() or not _test_data_mount_usable(root):
        raise RuntimeError(
            f"HF offline mode was selected but {test_data_root} is missing or empty. "
            "Ensure test_data_ref is configured so the init container syncs data before the adapter runs."
        )


def seed_hf_offline_from_job_spec_file() -> None:
    """Best-effort offline env seed at import time (job file may not exist in local dev)."""
    path = os.environ.get("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
    spec = _read_job_spec_dict_from_path(path)
    parameters = spec.get("parameters") if isinstance(spec.get("parameters"), dict) else {}
    if should_use_hf_offline(parameters, job_spec_path=path):
        values = configure_hf_offline_environment(TEST_DATA_DIR)
        logger.info(
            "HF offline mode (import-time seed): HF_HOME=%s, HF_HUB_CACHE=%s, Hub downloads disabled",
            values["HF_HOME"],
            values["HF_HUB_CACHE"],
        )


def _seed_hf_offline_before_adapter_import() -> None:
    try:
        seed_hf_offline_from_job_spec_file()
    except Exception as exc:  # noqa: BLE001 — import-time seed must never block startup
        print(f"WARNING: HF offline seed skipped: {exc}", file=sys.stderr)


_seed_hf_offline_before_adapter_import()
