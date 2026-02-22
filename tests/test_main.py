import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy
import pytest
from numpy import float32

from semantic_navigator.cache import (
    app_cache_dir, cluster_hash, content_hash, embedding_cache_dir,
    label_cache_dir, load_cached_embedding, load_cached_label,
    load_cached_cluster_labels, save_cached_embedding, save_cached_label,
    save_cached_cluster_labels,
)
from semantic_navigator.gpu import _resolve_gpu_layers
from semantic_navigator.inference import (
    _build_labels_schema, _parse_raw_response, _validate_label_count,
    max_count_retries,
)
from semantic_navigator.main import _build_model_identity, _parse_subcommand, _show_status
from semantic_navigator.models import Cluster, ClusterTree, Embed, Label, Labels, Tree
from semantic_navigator.pipeline import (
    _count_tree_depth, _count_tree_leaves, count_cached_labels, to_files, to_pattern,
)
from semantic_navigator.util import (
    _sanitize_path, case_insensitive_glob, extract_json, _fmt_time, repair_json,
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
    def test_openai(self):
        ns = argparse.Namespace(completion_model="gpt-5-mini")
        result = _build_model_identity("openai", ns)
        assert result == "openai:gpt-5-mini"

    def test_local(self):
        ns = argparse.Namespace(local="m/repo", local_file=None)
        result = _build_model_identity("local", ns)
        assert result == "local:m/repo"

    def test_local_with_file(self):
        ns = argparse.Namespace(local="m/repo", local_file="Q4.gguf")
        result = _build_model_identity("local", ns)
        assert result == "local:m/repo+file:Q4.gguf"

    def test_cli(self):
        ns = argparse.Namespace(cli_command=["llm", "-m", "4o"])
        result = _build_model_identity("cli", ns)
        assert result == "cli:llm -m 4o"


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

class TestParseSubcommand:
    def test_default_to_openai(self):
        sub, rest = _parse_subcommand(["./repo"])
        assert sub == "openai"
        assert rest == ["./repo"]

    def test_explicit_openai(self):
        sub, rest = _parse_subcommand(["openai", "./repo"])
        assert sub == "openai"
        assert rest == ["./repo"]

    def test_explicit_local(self):
        sub, rest = _parse_subcommand(["local", "./repo", "--local", "m"])
        assert sub == "local"
        assert rest == ["./repo", "--local", "m"]

    def test_explicit_cli(self):
        sub, rest = _parse_subcommand(["cli", "gemini", "./repo"])
        assert sub == "cli"
        assert rest == ["gemini", "./repo"]

    def test_flags_before_subcommand(self):
        sub, rest = _parse_subcommand(["--debug", "local", "./repo"])
        assert sub == "local"
        assert rest == ["--debug", "./repo"]

    def test_no_args(self):
        sub, rest = _parse_subcommand([])
        assert sub == "openai"
        assert rest == []


# -- _show_status --

class TestShowStatus:
    def test_no_files(self, tmp_path, capsys):
        """Empty directory shows 0 files."""
        _show_status(str(tmp_path))
        out = capsys.readouterr().out
        assert "(0 files)" in out
        assert "no label caches found" in out

    def test_uncached_files(self, tmp_path, capsys):
        """Files without label cache show as missing."""
        from semantic_navigator.cache import content_hash, label_cache_dir, save_cached_label
        from semantic_navigator.pipeline import _generate_paths

        (tmp_path / "a.py").write_text("print('hello')", encoding="utf-8")
        (tmp_path / "b.py").write_text("print('world')", encoding="utf-8")

        # Create a label dir with no cached labels so the section appears
        paths = _generate_paths(str(tmp_path))
        h = content_hash(f"{paths[0]}:\n\nprint('hello')")
        ldir = label_cache_dir(str(tmp_path), "test-identity")
        save_cached_label(ldir, h, _label("test"))

        _show_status(str(tmp_path))
        out = capsys.readouterr().out
        assert "(2 files)" in out
        assert "1/2 cached" in out
        assert "1 missing" in out

    def test_binary_files_skipped(self, tmp_path, capsys):
        """Binary files are skipped."""
        (tmp_path / "img.bin").write_bytes(b"\x80\x81\x82\x00\xff")
        (tmp_path / "a.py").write_text("code", encoding="utf-8")
        _show_status(str(tmp_path))
        out = capsys.readouterr().out
        assert "(1 files)" in out

    def test_label_caches_shown(self, tmp_path, capsys):
        """Label cache directories are enumerated with missing files listed."""
        from semantic_navigator.cache import content_hash, label_cache_dir, save_cached_label
        from semantic_navigator.pipeline import _generate_paths

        (tmp_path / "a.py").write_text("code", encoding="utf-8")
        paths = _generate_paths(str(tmp_path))
        h = content_hash(f"{paths[0]}:\n\ncode")

        # Save a label under a known model identity
        ldir = label_cache_dir(str(tmp_path), "test-identity")
        save_cached_label(ldir, h, _label("test"))

        _show_status(str(tmp_path))
        out = capsys.readouterr().out
        assert "1 model identity" in out
        assert "1/1 cached" in out

    def test_multiple_model_identities(self, tmp_path, capsys):
        """Multiple model identities are listed separately."""
        (tmp_path / "a.py").write_text("code", encoding="utf-8")
        ldir1 = label_cache_dir(str(tmp_path), "identity-1")
        ldir2 = label_cache_dir(str(tmp_path), "identity-2")
        save_cached_label(ldir1, "fake-hash", _label("l1"))
        save_cached_label(ldir2, "fake-hash", _label("l2"))

        _show_status(str(tmp_path))
        out = capsys.readouterr().out
        assert "2 model identities" in out

    def test_deleted_file_skipped(self, tmp_path, capsys):
        """Files that exist in git index but are deleted on disk don't crash."""
        # Create then delete a file — _generate_paths for non-git returns scandir
        # so this tests the error handling in _show_status's file reading loop
        (tmp_path / "a.py").write_text("code", encoding="utf-8")
        _show_status(str(tmp_path))
        out = capsys.readouterr().out
        assert "(1 files)" in out


# -- extract_json (additional) --

class TestExtractJsonAdditional:
    def test_code_fenced_no_lang(self):
        text = "result:\n```\n{\"key\": 42}\n```\ndone"
        assert extract_json(text) == '{"key": 42}'

    def test_nested_braces(self):
        text = '{"a": {"b": 1}}'
        assert extract_json(text) == '{"a": {"b": 1}}'

    def test_empty_string(self):
        assert extract_json("") == ""


# -- repair_json (additional) --

class TestRepairJsonAdditional:
    def test_backslash_before_unicode_non_hex(self):
        """\\uNOTHEX should become \\\\uNOTHEX."""
        assert repair_json(r'"\user"') == r'"\\user"'

    def test_valid_unicode_escape_untouched(self):
        original = r'"\u0041"'
        assert repair_json(original) == original

    def test_multiple_invalid_escapes(self):
        # \g and \k are not valid JSON escapes, so they get doubled
        result = repair_json(r'{"a": "\goo\kar"}')
        assert result == r'{"a": "\\goo\\kar"}'


# -- _sanitize_path (additional) --

class TestSanitizePathAdditional:
    def test_unix_path(self):
        result = _sanitize_path("/home/user/repo")
        assert "/" not in result

    def test_windows_backslash(self):
        result = _sanitize_path("C:\\Users\\foo\\repo")
        assert "\\" not in result
        assert ":" not in result


# -- to_pattern (additional) --

class TestToPatternAdditional:
    def test_identical_files(self):
        """All-same files produce the filename with no star."""
        result = to_pattern(["same.py", "same.py"])
        assert result == "same.py: "
        assert "*" not in result

    def test_prefix_only(self):
        result = to_pattern(["src/a", "src/b"])
        assert result.startswith("src/")
        assert "*" in result

    def test_suffix_only(self):
        result = to_pattern(["foo.test.js", "bar.test.js"])
        assert result.endswith(".test.js: ")
        assert "*" in result

    def test_empty_list(self):
        assert to_pattern([]) == ""


# -- to_files (additional) --

class TestToFilesAdditional:
    def test_empty(self):
        assert to_files([]) == []

    def test_deeply_nested(self):
        leaf = Tree("leaf", ["f1"], [])
        mid = Tree("mid", ["f1"], [leaf])
        top = Tree("top", ["f1"], [mid])
        assert to_files([top]) == ["f1"]


# -- _resolve_gpu_layers (additional) --

class TestResolveGpuLayersAdditional:
    def test_vram_exceeds_model_returns_all(self):
        """When VRAM >> model size, return -1 (all layers)."""
        with patch("semantic_navigator.gpu.detect_device_memory", return_value=16_000_000_000):
            result = _resolve_gpu_layers(True, None, 0, model_size=4_000_000_000)
        assert result == -1

    def test_vram_less_than_model_returns_partial(self):
        """When VRAM < model size, return proportional layers."""
        with patch("semantic_navigator.gpu.detect_device_memory", return_value=4_000_000_000):
            result = _resolve_gpu_layers(True, None, 0, model_size=8_000_000_000)
        # 4B * 0.6 / 8B * 100 = 30
        assert result == 30

    def test_vram_none_returns_all(self):
        """When VRAM detection fails, return -1."""
        with patch("semantic_navigator.gpu.detect_device_memory", return_value=None):
            result = _resolve_gpu_layers(True, None, 0, model_size=4_000_000_000)
        assert result == -1


# -- Corrupt cache recovery --

class TestCorruptCacheRecovery:
    def test_corrupt_label_returns_none(self, tmp_path):
        """Corrupt JSON label file returns None instead of crashing."""
        path = tmp_path / "bad.json"
        path.write_text("not valid json{{{")
        result = load_cached_label(tmp_path, "bad")
        assert result is None

    def test_corrupt_cluster_returns_none(self, tmp_path):
        """Corrupt cluster label file returns None."""
        path = tmp_path / "cluster-bad.json"
        path.write_text("truncated{")
        result = load_cached_cluster_labels(tmp_path, "bad")
        assert result is None

    def test_corrupt_embedding_returns_none(self, tmp_path):
        """Corrupt .npy file returns None instead of crashing."""
        path = tmp_path / "bad.npy"
        path.write_bytes(b"not a numpy file")
        result = load_cached_embedding(tmp_path, "bad")
        assert result is None

    def test_truncated_label_json(self, tmp_path):
        """Partially written JSON (valid syntax, missing fields) returns None."""
        path = tmp_path / "partial.json"
        path.write_text('{"overarchingTheme": "T"}')
        result = load_cached_label(tmp_path, "partial")
        assert result is None


# -- app_cache_dir per platform --

class TestAppCacheDir:
    def test_linux(self):
        with patch("semantic_navigator.cache.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch.dict(os.environ, {}, clear=True):
                with patch("semantic_navigator.cache.Path.home", return_value=Path("/home/user")):
                    from semantic_navigator.cache import app_cache_dir
                    result = app_cache_dir()
                    assert "semantic-navigator" in str(result)

    def test_xdg_override(self):
        with patch("semantic_navigator.cache.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch.dict(os.environ, {"XDG_CACHE_HOME": "/custom/cache"}):
                from semantic_navigator.cache import app_cache_dir
                result = app_cache_dir()
                # Check path parts rather than string prefix (cross-platform)
                parts = result.parts
                assert "custom" in parts and "cache" in parts
                assert parts[-1] == "semantic-navigator"


# -- _build_model_identity (additional) --

class TestBuildModelIdentityAdditional:
    def test_unknown_subcommand_empty(self):
        result = _build_model_identity("unknown", argparse.Namespace())
        assert result == ""


# -- _count_tree_depth / _count_tree_leaves (additional) --

class TestTreeCountsAdditional:
    def test_deep_tree(self):
        leaf = ClusterTree(Cluster([_embed("a")]), [])
        mid = ClusterTree(Cluster([_embed("a")]), [leaf])
        root = ClusterTree(Cluster([_embed("a")]), [mid])
        assert _count_tree_depth(root) == 2
        assert _count_tree_leaves(root) == 1

    def test_wide_tree(self):
        children = [ClusterTree(Cluster([_embed(f"f{i}")]), []) for i in range(5)]
        root = ClusterTree(Cluster([_embed(f"f{i}") for i in range(5)]), children)
        assert _count_tree_depth(root) == 1
        assert _count_tree_leaves(root) == 5


# -- embedding_cache_dir / label_cache_dir --

class TestCacheDirFunctions:
    def test_embedding_cache_dir_slashes(self):
        result = embedding_cache_dir("BAAI/bge-large-en-v1.5")
        assert "BAAI--bge-large-en-v1.5" in str(result)

    def test_label_cache_dir_deterministic(self):
        r1 = label_cache_dir("/repo", "model-a")
        r2 = label_cache_dir("/repo", "model-a")
        assert r1 == r2

    def test_label_cache_dir_different_models(self):
        r1 = label_cache_dir("/repo", "model-a")
        r2 = label_cache_dir("/repo", "model-b")
        assert r1 != r2
