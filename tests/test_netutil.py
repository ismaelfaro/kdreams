"""Tests for kdream.core.netutil — download resilience helpers."""
from __future__ import annotations

import pytest


class TestWithRetry:
    def test_returns_on_first_success(self):
        from kdream.core.netutil import with_retry
        assert with_retry(lambda: 42) == 42

    def test_retries_then_succeeds(self, monkeypatch):
        from kdream.core import netutil
        monkeypatch.setattr(netutil.time, "sleep", lambda s: None)
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("cdn hiccup")
            return "ok"

        retries = []
        result = netutil.with_retry(
            flaky, attempts=5,
            on_retry=lambda a, t, e: retries.append((a, type(e).__name__)),
        )
        assert result == "ok"
        assert calls["n"] == 3
        assert retries == [(1, "ConnectionError"), (2, "ConnectionError")]

    def test_raises_after_exhausting_attempts(self, monkeypatch):
        from kdream.core import netutil
        monkeypatch.setattr(netutil.time, "sleep", lambda s: None)

        def always_fails():
            raise TimeoutError("dead")

        with pytest.raises(TimeoutError):
            netutil.with_retry(always_fails, attempts=3)

    def test_attempts_from_env(self, monkeypatch):
        from kdream.core import netutil
        monkeypatch.setattr(netutil.time, "sleep", lambda s: None)
        monkeypatch.setenv("KDREAM_DOWNLOAD_RETRIES", "2")
        calls = {"n": 0}

        def fails():
            calls["n"] += 1
            raise OSError("x")

        with pytest.raises(OSError):
            netutil.with_retry(fails)
        assert calls["n"] == 2

    def test_keyboard_interrupt_not_retried(self, monkeypatch):
        from kdream.core import netutil
        monkeypatch.setattr(netutil.time, "sleep", lambda s: None)
        calls = {"n": 0}

        def interrupted():
            calls["n"] += 1
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            netutil.with_retry(interrupted, attempts=5)
        assert calls["n"] == 1


class TestNetworkEnv:
    def test_apply_network_env_noop_by_default(self, monkeypatch):
        from kdream.core.netutil import apply_network_env
        monkeypatch.delenv("KDREAM_FORCE_IPV4", raising=False)
        monkeypatch.delenv("KDREAM_DISABLE_XET", raising=False)
        assert apply_network_env() == []

    def test_disable_xet_sets_hf_env(self, monkeypatch):
        from kdream.core.netutil import apply_network_env
        monkeypatch.setenv("KDREAM_DISABLE_XET", "1")
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
        notes = apply_network_env()
        import os
        assert os.environ["HF_HUB_DISABLE_XET"] == "1"
        assert any("Xet" in n for n in notes)


class TestDoctorCLI:
    def test_doctor_runs(self):
        from click.testing import CliRunner

        from kdream.cli import cli
        result = CliRunner().invoke(cli, ["doctor", "--skip-network"])
        assert result.exit_code == 0, result.output
        assert "accelerator" in result.output
        assert "Environment knobs" in result.output
        assert "KDREAM_FORCE_IPV4" in result.output
