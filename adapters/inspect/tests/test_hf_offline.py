"""Tests for Hugging Face offline detection and env configuration."""

import json
import os
from pathlib import Path

import pytest

from _hf_offline import (
    configure_hf_offline_environment,
    job_spec_requests_test_data,
    should_use_hf_offline,
)
from _execution import build_env


_HF_ENV_KEYS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_EVALUATE_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "XDG_CACHE_HOME",
)


@pytest.fixture(autouse=True)
def restore_hf_env():
    """configure_hf_offline_environment writes os.environ process-wide; undo it per test."""
    saved = {k: os.environ.get(k) for k in _HF_ENV_KEYS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}", encoding="utf-8")


@pytest.fixture
def fake_test_data(tmp_path: Path) -> Path:
    root = tmp_path / "test_data"
    tok = root / "tokenizer"
    tok.mkdir(parents=True)
    _touch(tok / "config.json")
    bundle = root / "GSMA--ot-full--telemath"
    bundle.mkdir(parents=True)
    _touch(bundle / "dataset_dict.json")
    return root


def test_infer_offline_from_tokenizer_and_bundle(fake_test_data: Path) -> None:
    tok = fake_test_data / "tokenizer"
    params = {"tokenizer": str(tok.resolve())}
    assert should_use_hf_offline(params, test_data_root=fake_test_data)


def test_infer_offline_from_test_data_ref_in_job_spec(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "test_data"
    root.mkdir()
    _touch(root / "hub" / "datasets--GSMA--ot-full" / "refs" / "main")

    spec = {
        "test_data_ref": {
            "s3": {"bucket": "mlpipeline", "key": "offline", "secret_ref": "minio-test"},
        },
    }
    monkeypatch.setattr(
        "_hf_offline._read_job_spec_dict_from_path",
        lambda _path: spec,
    )

    assert job_spec_requests_test_data("/meta/job.json")
    assert should_use_hf_offline({}, job_spec_path="/meta/job.json", test_data_root=root)


def test_no_offline_without_test_data_ref_or_tokenizer(tmp_path: Path) -> None:
    root = tmp_path / "test_data"
    root.mkdir()
    _touch(root / "placeholder.txt")
    assert not should_use_hf_offline({}, test_data_root=root)


def test_build_env_sets_hf_offline(monkeypatch, fake_test_data: Path, job_spec_path) -> None:
    from main import InspectAdapter

    import _execution as execution_mod
    import _hf_offline as hf_offline_mod

    monkeypatch.setattr(hf_offline_mod, "TEST_DATA_DIR", str(fake_test_data))
    monkeypatch.setattr(execution_mod, "TEST_DATA_DIR", str(fake_test_data))

    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.parameters["tokenizer"] = str((fake_test_data / "tokenizer").resolve())
    env = build_env(adapter.job_spec, "standard")
    assert env.get("HF_HUB_OFFLINE") == "1"
    # HF_HOME must stay off the staged mount — huggingface_hub and datasets write there.
    assert env.get("HF_HOME") != str(fake_test_data)
    assert not Path(env["HF_HOME"]).is_relative_to(fake_test_data)


def test_configure_hf_offline_environment_updates_os_environ(monkeypatch, tmp_path: Path) -> None:
    staged = tmp_path / "test_data"
    staged.mkdir()
    writable = tmp_path / "cache" / "huggingface"
    monkeypatch.setenv("HF_HOME", str(writable))
    env: dict[str, str] = {}
    configure_hf_offline_environment(str(staged), env)
    assert env["HF_HOME"] == str(writable)
    assert env["HF_HUB_CACHE"] == str(writable / "hub")
    assert env["HF_DATASETS_CACHE"] == str(writable / "datasets")
    assert env["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HOME"] == str(writable)


def test_staged_hub_cache_is_used_for_hub_reads(monkeypatch, tmp_path: Path) -> None:
    """A staged `hub/` layout serves hub reads while HF_HOME stays writable."""
    staged = tmp_path / "test_data"
    (staged / "hub" / "datasets--GSMA--ot-full").mkdir(parents=True)
    writable = tmp_path / "cache" / "huggingface"
    monkeypatch.setenv("HF_HOME", str(writable))

    env: dict[str, str] = {}
    configure_hf_offline_environment(str(staged), env)

    assert env["HF_HUB_CACHE"] == str(staged / "hub")
    assert env["HF_HOME"] == str(writable)
    assert env["HF_DATASETS_CACHE"] == str(writable / "datasets")


def test_read_only_staged_mount_is_never_chosen_as_hf_home(monkeypatch, tmp_path: Path) -> None:
    """Regression: HF_HOME=/test_data breaks when the sync mounts it read-only."""
    staged = tmp_path / "test_data"
    (staged / "hub").mkdir(parents=True)
    staged.chmod(0o555)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "app-cache"))
    try:
        env: dict[str, str] = {}
        configure_hf_offline_environment(str(staged), env)
        assert env["HF_HOME"] == str(tmp_path / "app-cache" / "huggingface")
        assert Path(env["HF_HOME"]).is_dir()
        assert env["HF_HUB_CACHE"] == str(staged / "hub")
    finally:
        staged.chmod(0o755)


def test_hf_home_pointing_at_staged_mount_is_rejected(monkeypatch, tmp_path: Path) -> None:
    """An operator-set HF_HOME inside /test_data is ignored, not honoured."""
    staged = tmp_path / "test_data"
    staged.mkdir()
    monkeypatch.setenv("HF_HOME", str(staged))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "app-cache"))

    env: dict[str, str] = {}
    configure_hf_offline_environment(str(staged), env)

    assert env["HF_HOME"] == str(tmp_path / "app-cache" / "huggingface")
