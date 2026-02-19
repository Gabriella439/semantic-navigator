import argparse
import json
import os
from unittest.mock import patch

import numpy
import pytest
from numpy import float32

from semantic_navigator.main import (
    ClusterTree,
    Cluster,
    Embed,
    Label,
    Labels,
    Tree,
    _show_status,
    _build_labels_schema,
    _build_model_identity,
    _count_tree_depth,
    _count_tree_leaves,
    _fmt_time,
    _parse_cli_tool,
    _parse_raw_response,
    _resolve_gpu_layers,
    _sanitize_path,
    _validate_label_count,
    case_insensitive_glob,
    cluster_hash,
    content_hash,
    count_cached_labels,
    extract_json,
    repair_json,
    to_files,
    to_pattern,
)


# -- Fixtures --

def _embed(entry: str, content: str = "") -> Embed:
    return Embed(entry=entry, content=content or entry, embedding=numpy.zeros(3, dtype=float32))


def _label(text: str = "lbl") -> Label:
    return Label(overarchingTheme="T", distinguishingFeature="F", label=text)


# -- _fmt_time --

class TestFmtTime:
    def test_under_60(self):
        assert _fmt_time(3.14) == "3.1s"

    def test_zero(self):
        assert _fmt_time(0) == "0.0s"

    def test_exactly_60(self):
        assert _fmt_time(60) == "1m 0s"

    def test_over_60(self):
        assert _fmt_time(125.7) == "2m 6s"


# -- extract_json --

class TestExtractJson:
    def test_code_fenced(self):
        text = "here\n```json\n{\"a\": 1}\n```\nafter"
        assert extract_json(text) == '{"a": 1}'

    def test_bare_json(self):
        text = 'blah {"x": 2} blah'
        assert extract_json(text) == '{"x": 2}'

    def test_no_json(self):
        assert extract_json("  hello  ") == "hello"


# -- repair_json --

class TestRepairJson:
    def test_backslash_p(self):
        assert repair_json(r'{"a": "\p"}') == r'{"a": "\\p"}'

    def test_backslash_utils(self):
        assert repair_json(r'{"a": "\utils"}') == r'{"a": "\\utils"}'

    def test_valid_escapes_untouched(self):
        original = r'{"a": "line\nbreak", "b": "tab\there"}'
        assert repair_json(original) == original


# -- case_insensitive_glob --

class TestCaseInsensitiveGlob:
    def test_letters(self):
        assert case_insensitive_glob("abc") == "[Aa][Bb][Cc]"

    def test_non_alpha(self):
        assert case_insensitive_glob("1.2") == "1.2"

    def test_mixed(self):
        assert case_insensitive_glob("a1") == "[Aa]1"


# -- content_hash / cluster_hash --

class TestHashes:
    def test_content_hash_deterministic(self):
        assert content_hash("hello") == content_hash("hello")
        assert content_hash("a") != content_hash("b")

    def test_cluster_hash_stable_order(self):
        assert cluster_hash(["b", "a"]) == cluster_hash(["a", "b"])

    def test_cluster_hash_different_content(self):
        assert cluster_hash(["a"]) != cluster_hash(["b"])


# -- _sanitize_path --

class TestSanitizePath:
    def test_replaces_slashes_and_colon(self):
        result = _sanitize_path("C:/foo/bar")
        assert "/" not in result
        assert "\\" not in result
        assert ":" not in result


# -- to_pattern --

class TestToPattern:
    def test_common_prefix_and_suffix(self):
        result = to_pattern(["src/foo.py", "src/bar.py"])
        assert result == "src/*.py: "

    def test_single_file(self):
        assert to_pattern(["only.txt"]) == "only.txt: "

    def test_no_common_parts(self):
        assert to_pattern(["abc", "xyz"]) == ""

    def test_star_insertion(self):
        result = to_pattern(["pre_a_suf", "pre_b_suf"])
        assert "*" in result


# -- to_files --

class TestToFiles:
    def test_flatten(self):
        trees = [
            Tree("a", ["f1", "f2"], []),
            Tree("b", ["f3"], []),
        ]
        assert to_files(trees) == ["f1", "f2", "f3"]


# -- _count_tree_depth / _count_tree_leaves --

class TestTreeCounts:
    def test_leaf(self):
        ct = ClusterTree(Cluster([_embed("a")]), [])
        assert _count_tree_depth(ct) == 0
        assert _count_tree_leaves(ct) == 1

    def test_nested(self):
        child1 = ClusterTree(Cluster([_embed("a")]), [])
        child2 = ClusterTree(Cluster([_embed("b")]), [])
        parent = ClusterTree(Cluster([_embed("a"), _embed("b")]), [child1, child2])
        assert _count_tree_depth(parent) == 1
        assert _count_tree_leaves(parent) == 2


# -- _build_labels_schema --

class TestBuildLabelsSchema:
    def test_with_count(self):
        schema = _build_labels_schema(3)
        arr = schema["properties"]["labels"]
        assert arr["minItems"] == 3
        assert arr["maxItems"] == 3

    def test_without_count(self):
        schema = _build_labels_schema(None)
        arr = schema["properties"]["labels"]
        assert "minItems" not in arr
        assert "maxItems" not in arr


# -- _validate_label_count --

class TestValidateLabelCount:
    def test_exact_match(self):
        parsed = Labels(labels=[_label(), _label()])
        result = _validate_label_count(parsed, 2, "test", 0)
        assert result is parsed

    def test_no_expected_count(self):
        parsed = Labels(labels=[_label()])
        result = _validate_label_count(parsed, None, "test", 0)
        assert result is parsed

    def test_truncation(self):
        parsed = Labels(labels=[_label("a"), _label("b"), _label("c")])
        result = _validate_label_count(parsed, 2, "test", 0)
        assert len(result.labels) == 2

    def test_retry_when_too_few(self):
        parsed = Labels(labels=[_label()])
        result = _validate_label_count(parsed, 3, "test", 0)
        assert result is None

    def test_padding_at_max_attempts(self):
        from semantic_navigator.main import max_count_retries
        parsed = Labels(labels=[_label()])
        result = _validate_label_count(parsed, 3, "test", max_count_retries - 1)
        assert result is not None
        assert len(result.labels) == 3


# -- _parse_raw_response --

class TestParseRawResponse:
    def test_clean_json(self):
        raw = '{"labels": [{"overarchingTheme": "T", "distinguishingFeature": "F", "label": "L"}]}'
        result = _parse_raw_response(raw, Labels, False)
        assert len(result.labels) == 1

    def test_repairable_json(self):
        raw = r'{"labels": [{"overarchingTheme": "T", "distinguishingFeature": "F", "label": "\path"}]}'
        result = _parse_raw_response(raw, Labels, False)
        assert result.labels[0].label == "\\path"

    def test_totally_invalid(self):
        with pytest.raises(Exception):
            _parse_raw_response("not json at all {{{", Labels, False)


# -- _build_model_identity --

class TestBuildModelIdentity:
    def _ns(self, **kwargs):
        defaults = {"local": None, "local_file": None, "openai": False, "completion_model": "gpt-4o-mini"}
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_local_only(self):
        result = _build_model_identity(self._ns(local="m/repo"), None)
        assert result == "local:m/repo"

    def test_cli_only(self):
        result = _build_model_identity(self._ns(), ["llm", "-m", "4o"])
        assert "cli:" in result

    def test_openai_only(self):
        result = _build_model_identity(self._ns(openai=True), None)
        assert "openai:gpt-4o-mini" in result

    def test_combo(self):
        result = _build_model_identity(self._ns(local="m", openai=True), ["llm"])
        assert "local:m" in result
        assert "cli:llm" in result
        assert "openai:" in result


# -- _resolve_gpu_layers --

class TestResolveGpuLayers:
    def test_gpu_false(self):
        assert _resolve_gpu_layers(False, None, 0) == 0

    def test_explicit_layers(self):
        assert _resolve_gpu_layers(True, 42, 0) == 42

    def test_no_model_size(self):
        assert _resolve_gpu_layers(True, None, 0) == -1


# -- count_cached_labels --

class TestCountCachedLabels:
    def test_leaf_no_cache(self):
        ct = ClusterTree(Cluster([_embed("a", "content_a")]), [])
        uncached, cached, cc = count_cached_labels(ct, "/repo", "id", cached_keys=set())
        assert uncached == 1
        assert cached == 0
        assert cc == 0

    def test_leaf_with_cache(self):
        e = _embed("a", "content_a")
        key = content_hash(e.content)
        ct = ClusterTree(Cluster([e]), [])
        uncached, cached, cc = count_cached_labels(ct, "/repo", "id", cached_keys={key})
        assert uncached == 0
        assert cached == 1

    def test_nested(self):
        e1, e2 = _embed("a", "ca"), _embed("b", "cb")
        child1 = ClusterTree(Cluster([e1]), [])
        child2 = ClusterTree(Cluster([e2]), [])
        parent = ClusterTree(Cluster([e1, e2]), [child1, child2])
        uncached, cached, cc = count_cached_labels(parent, "/repo", "id", cached_keys=set())
        assert uncached == 2
        assert cached == 0


# -- _parse_cli_tool --

class TestParseCliTool:
    def test_valid_tool(self):
        parser = argparse.ArgumentParser()
        with patch("semantic_navigator.main.shutil.which", return_value="/usr/bin/echo"):
            result = _parse_cli_tool(["--echo", "arg1"], parser)
        assert result == ["echo", "arg1"]

    def test_no_remaining(self):
        parser = argparse.ArgumentParser()
        assert _parse_cli_tool([], parser) is None

    def test_not_starting_with_dashes(self):
        parser = argparse.ArgumentParser()
        assert _parse_cli_tool(["foo"], parser) is None

    def test_tool_not_found(self):
        parser = argparse.ArgumentParser()
        with patch("semantic_navigator.main.shutil.which", return_value=None):
            with pytest.raises(SystemExit):
                _parse_cli_tool(["--nonexistent"], parser)


# -- _show_status --

class TestShowStatus:
    def test_no_files(self, tmp_path, capsys):
        """Empty directory shows 0 files."""
        _show_status(str(tmp_path), "BAAI/bge-large-en-v1.5")
        out = capsys.readouterr().out
        assert "(0 files)" in out
        assert "0/0 cached" in out

    def test_uncached_files(self, tmp_path, capsys):
        """Files without cache show as missing."""
        (tmp_path / "a.py").write_text("print('hello')", encoding="utf-8")
        (tmp_path / "b.py").write_text("print('world')", encoding="utf-8")
        _show_status(str(tmp_path), "BAAI/bge-large-en-v1.5")
        out = capsys.readouterr().out
        assert "(2 files)" in out
        assert "0/2 cached (2 missing)" in out
        assert "a.py" in out
        assert "b.py" in out

    def test_cached_embeddings(self, tmp_path, capsys):
        """Files with cached embeddings show as cached."""
        from semantic_navigator.cache import content_hash, embedding_cache_dir, save_cached_embedding
        from semantic_navigator.pipeline import _generate_paths

        (tmp_path / "a.py").write_text("hello", encoding="utf-8")
        # Compute the same hash the status function would use
        paths = _generate_paths(str(tmp_path))
        h = content_hash(f"{paths[0]}:\n\nhello")
        emb_dir = embedding_cache_dir("test-model")
        save_cached_embedding(emb_dir, h, numpy.zeros(3, dtype=float32))

        _show_status(str(tmp_path), "test-model")
        out = capsys.readouterr().out
        assert "1/1 cached (0 missing)" in out

    def test_binary_files_skipped(self, tmp_path, capsys):
        """Binary files are skipped."""
        (tmp_path / "img.bin").write_bytes(b"\x80\x81\x82\x00\xff")
        (tmp_path / "a.py").write_text("code", encoding="utf-8")
        _show_status(str(tmp_path), "BAAI/bge-large-en-v1.5")
        out = capsys.readouterr().out
        assert "(1 files)" in out

    def test_label_caches_shown(self, tmp_path, capsys):
        """Label cache directories are enumerated."""
        from semantic_navigator.cache import content_hash, label_cache_dir, save_cached_label
        from semantic_navigator.pipeline import _generate_paths

        (tmp_path / "a.py").write_text("code", encoding="utf-8")
        paths = _generate_paths(str(tmp_path))
        h = content_hash(f"{paths[0]}:\n\ncode")

        # Save a label under a known model identity
        ldir = label_cache_dir(str(tmp_path), "test-identity")
        save_cached_label(ldir, h, _label("test"))

        _show_status(str(tmp_path), "BAAI/bge-large-en-v1.5")
        out = capsys.readouterr().out
        assert "1 model identity" in out
        assert "1/1 cached" in out
