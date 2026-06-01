#!/usr/bin/env python3
"""
Unit tests for the HTA asset handler's raw-passthrough behavior.

These exercise the compressed-size gate and JSON passthrough without needing
the optional HolisticTraceAnalysis dependency installed.
"""

import gzip
import json

from comet_mcp.asset_handlers import hta


def _gz(obj) -> bytes:
    return gzip.compress(json.dumps(obj).encode())


class TestRawPassthrough:
    """Small assets should return the parsed trace JSON directly."""

    def test_small_asset_returns_raw_trace(self):
        trace = {"traceEvents": [{"name": "aten::add", "dur": 12}], "schema": 1}
        content = _gz(trace)

        result = hta.handle(content, "run_hta_results.json.gz")

        assert result["format"] == "raw_trace"
        assert result["asset_name"] == "run_hta_results.json.gz"
        assert result["trace"] == trace
        assert result["size_bytes"] == len(json.dumps(trace).encode())

    def test_large_compressed_asset_defers_to_summary(self, monkeypatch):
        # Force a tiny gz threshold so any real payload exceeds it.
        monkeypatch.setenv("COMET_MCP_RAW_ASSET_MAX_GZ_BYTES", "1")
        content = _gz({"traceEvents": [{"name": "aten::add"}]})

        result = hta.handle(content, "run_hta_results.json.gz")

        # Over the gz gate -> summary path; HTA is not installed in tests, so
        # we get the import error rather than a raw_trace payload.
        assert result.get("format") != "raw_trace"

    def test_threshold_is_configurable(self, monkeypatch):
        trace = {"traceEvents": [{"name": "aten::add"}]}
        content = _gz(trace)
        # Threshold comfortably above this tiny payload's compressed size.
        monkeypatch.setenv("COMET_MCP_RAW_ASSET_MAX_GZ_BYTES", str(len(content) + 10))

        result = hta.handle(content, "run_hta_results.json.gz")

        assert result["format"] == "raw_trace"


class TestTryRaw:
    """_try_raw should fall back (return None) on malformed input."""

    def test_non_gzip_returns_none(self):
        assert hta._try_raw(b"not gzip data", "x_hta_results.json.gz") is None

    def test_valid_gzip_invalid_json_returns_none(self):
        content = gzip.compress(b"this is not json")
        assert hta._try_raw(content, "x_hta_results.json.gz") is None

    def test_bad_env_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("COMET_MCP_RAW_ASSET_MAX_GZ_BYTES", "not-a-number")
        assert hta._raw_max_gz_bytes() == hta.RAW_MAX_GZ_BYTES_DEFAULT
