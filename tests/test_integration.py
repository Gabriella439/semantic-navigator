"""Integration tests — exercise multi-function workflows without real backends."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy
import pytest
from numpy import float32

from semantic_navigator.cache import (
    cluster_hash, content_hash, list_cached_keys,
    load_cached_cluster_labels, load_cached_embedding, load_cached_label,
    save_cached_cluster_labels, save_cached_embedding, save_cached_label,
)
from semantic_navigator.models import (
    Aspect, AspectPool, Cluster, ClusterTree, Embed, Facets, Label, Labels, Tree,
)
from semantic_navigator.pipeline import (
    _generate_paths, _read_file, build_cluster_tree, cluster,
    count_cached_labels, label_nodes, to_files, to_pattern,
)
from semantic_navigator.util import timed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed(name: str, vec: list[float] | None = None) -> Embed:
    if vec is None:
        vec = numpy.random.default_rng(hash(name) % 2**32).standard_normal(32).tolist()
    return Embed(
        entry=name,
        content=f"{name}:\n\ncontent of {name}",
        embedding=numpy.asarray(vec, dtype=float32),
    )


def _label(text: str = "lbl") -> Label:
    return Label(overarchingTheme="T", distinguishingFeature="F", label=text)


def _make_aspect(name: str = "test/0") -> Aspect:
    return Aspect(
        name=name, cli_command=None, local_model=None,
        local_n_ctx=None, openai_client=None, openai_model=None,
    )


# ---------------------------------------------------------------------------
# Cache round-trips (embedding, label, cluster-label)
# ---------------------------------------------------------------------------

class TestCacheRoundTrips:
    def test_embedding_save_load(self, tmp_path: Path):
        vec = numpy.array([1.0, 2.0, 3.0], dtype=float32)
        save_cached_embedding(tmp_path, "k1", vec)
        loaded = load_cached_embedding(tmp_path, "k1")
        assert loaded is not None
        numpy.testing.assert_array_equal(loaded, vec)

    def test_embedding_missing(self, tmp_path: Path):
        assert load_cached_embedding(tmp_path, "nope") is None

    def test_label_save_load(self, tmp_path: Path):
        lbl = _label("hello")
        save_cached_label(tmp_path, "k1", lbl)
        loaded = load_cached_label(tmp_path, "k1")
        assert loaded is not None
        assert loaded.label == "hello"

    def test_label_missing(self, tmp_path: Path):
        assert load_cached_label(tmp_path, "nope") is None

    def test_cluster_labels_save_load(self, tmp_path: Path):
        labels = Labels(labels=[_label("a"), _label("b")])
        save_cached_cluster_labels(tmp_path, "ckey", labels)
        loaded = load_cached_cluster_labels(tmp_path, "ckey")
        assert loaded is not None
        assert len(loaded.labels) == 2

    def test_cluster_labels_missing(self, tmp_path: Path):
        assert load_cached_cluster_labels(tmp_path, "nope") is None

    def test_list_cached_keys(self, tmp_path: Path):
        save_cached_label(tmp_path, "a", _label())
        save_cached_label(tmp_path, "b", _label())
        keys = list_cached_keys(tmp_path, ".json")
        assert keys == {"a", "b"}

    def test_list_cached_keys_empty_dir(self, tmp_path: Path):
        assert list_cached_keys(tmp_path / "nonexistent", ".json") == set()


# ---------------------------------------------------------------------------
# AspectPool async acquire / release
# ---------------------------------------------------------------------------

class TestAspectPool:
    def test_acquire_release(self):
        a1, a2 = _make_aspect("a"), _make_aspect("b")
        pool = AspectPool([a1, a2])

        async def run():
            got1 = await pool.acquire()
            got2 = await pool.acquire()
            assert {got1.name, got2.name} == {"a", "b"}
            pool.release(got1)
            got3 = await pool.acquire()
            assert got3 is got1

        asyncio.run(run())

    def test_min_local_n_ctx_none(self):
        pool = AspectPool([_make_aspect()])
        assert pool.min_local_n_ctx is None

    def test_min_local_n_ctx_value(self):
        a = Aspect("x", None, None, 4096, None, None)
        b = Aspect("y", None, None, 8192, None, None)
        pool = AspectPool([a, b])
        assert pool.min_local_n_ctx == 4096


# ---------------------------------------------------------------------------
# _generate_paths + _read_file on a temp directory
# ---------------------------------------------------------------------------

class TestFileDiscovery:
    def test_generate_paths_plain_dir(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.py").write_text("world")
        paths = _generate_paths(str(tmp_path))
        basenames = sorted(os.path.basename(p) for p in paths)
        assert basenames == ["a.txt", "b.py"]

    def test_read_file_text(self, tmp_path: Path):
        (tmp_path / "f.txt").write_text("content")

        async def run():
            return await _read_file(str(tmp_path), "f.txt")

        result = asyncio.run(run())
        assert result is not None
        path, content = result
        assert path == "f.txt"
        assert "content" in content

    def test_read_file_binary_returns_none(self, tmp_path: Path):
        (tmp_path / "img.bin").write_bytes(bytes(range(256)))

        async def run():
            return await _read_file(str(tmp_path), "img.bin")

        result = asyncio.run(run())
        assert result is None


# ---------------------------------------------------------------------------
# cluster() with synthetic embeddings
# ---------------------------------------------------------------------------

class TestClusterIntegration:
    def _make_cluster(self, n: int, dims: int = 32, seed: int = 42) -> Cluster:
        rng = numpy.random.default_rng(seed)
        embeds = []
        for i in range(n):
            vec = rng.standard_normal(dims).astype(float32)
            embeds.append(Embed(f"file{i}.py", f"file{i}.py:\n\ncontent {i}", vec))
        return embeds, Cluster(embeds)

    def test_small_cluster_returns_single(self):
        """≤ max_leaves items should not be split."""
        _, c = self._make_cluster(10)
        result = cluster(c)
        assert len(result) == 1
        assert result[0] is c

    def test_large_cluster_splits(self):
        """> max_leaves items should produce multiple clusters."""
        _, c = self._make_cluster(60)
        result = cluster(c)
        assert len(result) >= 2
        total = sum(len(sub.embeds) for sub in result)
        assert total == 60

    def test_build_cluster_tree_leaf(self):
        _, c = self._make_cluster(5)
        ct = build_cluster_tree(c)
        assert ct.children == []
        assert ct.node is c

    def test_build_cluster_tree_recursive(self):
        _, c = self._make_cluster(60)
        ct = build_cluster_tree(c)
        assert ct.children  # should have children
        from semantic_navigator.pipeline import _count_tree_leaves, _count_tree_depth
        assert _count_tree_leaves(ct) >= 2
        assert _count_tree_depth(ct) >= 1

    def test_two_distinct_groups(self):
        """Two well-separated groups should end up in different clusters."""
        rng = numpy.random.default_rng(0)
        embeds = []
        for i in range(30):
            base = numpy.array([1.0] * 16 + [0.0] * 16, dtype=float32)
            embeds.append(Embed(f"a{i}", f"a{i}", base + rng.normal(0, 0.05, 32).astype(float32)))
        for i in range(30):
            base = numpy.array([0.0] * 16 + [1.0] * 16, dtype=float32)
            embeds.append(Embed(f"b{i}", f"b{i}", base + rng.normal(0, 0.05, 32).astype(float32)))
        result = cluster(Cluster(embeds))
        assert len(result) >= 2
        # Check that the two groups are mostly separated
        for sub in result:
            names = [e.entry for e in sub.embeds]
            a_count = sum(1 for n in names if n.startswith("a"))
            b_count = sum(1 for n in names if n.startswith("b"))
            assert a_count == 0 or b_count == 0 or abs(a_count - b_count) > 10


# ---------------------------------------------------------------------------
# timed() context manager
# ---------------------------------------------------------------------------

class TestTimed:
    def test_records_timing(self):
        timings: dict[str, float] = {}
        with timed("phase", timings):
            pass
        assert "phase" in timings
        assert timings["phase"] >= 0

    def test_no_timings_dict(self, capsys):
        with timed("hello"):
            pass
        assert "hello" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# count_cached_labels with real cache directory
# ---------------------------------------------------------------------------

class TestCountCachedLabelsIntegration:
    def test_with_saved_cache(self, tmp_path: Path):
        e1 = _embed("f1")
        e2 = _embed("f2")
        # Save one label to cache
        key1 = content_hash(e1.content)
        save_cached_label(tmp_path, key1, _label("l1"))

        cached_keys = list_cached_keys(tmp_path, ".json")
        ct = ClusterTree(Cluster([e1, e2]), [])
        uncached, cached, cc = count_cached_labels(ct, "/r", "id", cached_keys=cached_keys)
        assert cached == 1
        assert uncached == 1

    def test_cluster_cache_detected(self, tmp_path: Path):
        e1 = _embed("a")
        e2 = _embed("b")
        child1 = ClusterTree(Cluster([e1]), [])
        child2 = ClusterTree(Cluster([e2]), [])
        parent = ClusterTree(Cluster([e1, e2]), [child1, child2])

        # Save a cluster-level cache entry
        all_files = [e.entry for e in parent.node.embeds]
        c_key = "cluster-" + cluster_hash(all_files)
        save_cached_label(tmp_path, c_key, _label())  # just needs the key to exist

        cached_keys = list_cached_keys(tmp_path, ".json")
        _, _, cc = count_cached_labels(parent, "/r", "id", cached_keys=cached_keys)
        assert cc == 1


# ---------------------------------------------------------------------------
# label_nodes end-to-end with mocked complete()
# ---------------------------------------------------------------------------

class TestLabelNodesIntegration:
    def test_leaf_labeling(self, tmp_path: Path):
        """Leaf node: each file gets a label via complete()."""
        e1 = _embed("src/a.py")
        e2 = _embed("src/b.py")
        ct = ClusterTree(Cluster([e1, e2]), [])

        mock_labels = Labels(labels=[_label("Label A"), _label("Label B")])

        async def run():
            from unittest.mock import MagicMock
            from tqdm import tqdm

            progress = tqdm(disable=True)

            # Patch complete + cache dir so nothing touches real FS
            with patch("semantic_navigator.pipeline.complete", new_callable=AsyncMock, return_value=mock_labels), \
                 patch("semantic_navigator.pipeline.label_cache_dir", return_value=tmp_path), \
                 patch("semantic_navigator.pipeline.load_cached_label", return_value=None), \
                 patch("semantic_navigator.pipeline.save_cached_label"):
                trees = await label_nodes(
                    MagicMock(repository="/r", model_identity="id", pool=MagicMock(min_local_n_ctx=None)),
                    ct, progress,
                )
            return trees

        trees = asyncio.run(run())
        assert len(trees) == 2
        assert all(isinstance(t, Tree) for t in trees)
        assert "Label A" in trees[0].label

    def test_cluster_labeling(self, tmp_path: Path):
        """Non-leaf node: children get labeled, then cluster gets a group label."""
        e1 = _embed("x.py")
        e2 = _embed("y.py")
        child1 = ClusterTree(Cluster([e1]), [])
        child2 = ClusterTree(Cluster([e2]), [])
        parent = ClusterTree(Cluster([e1, e2]), [child1, child2])

        leaf_labels = Labels(labels=[_label("Leaf")])
        cluster_labels = Labels(labels=[_label("Group A"), _label("Group B")])

        call_count = 0

        async def mock_complete(facets, prompt, output_type, progress=None, expected_count=None):
            nonlocal call_count
            call_count += 1
            if expected_count == 1:
                return leaf_labels
            return cluster_labels

        async def run():
            from unittest.mock import MagicMock
            from tqdm import tqdm

            progress = tqdm(disable=True)
            facets_mock = MagicMock(
                repository="/r", model_identity="id",
                pool=MagicMock(min_local_n_ctx=None),
            )

            with patch("semantic_navigator.pipeline.complete", side_effect=mock_complete), \
                 patch("semantic_navigator.pipeline.label_cache_dir", return_value=tmp_path), \
                 patch("semantic_navigator.pipeline.load_cached_label", return_value=None), \
                 patch("semantic_navigator.pipeline.save_cached_label"), \
                 patch("semantic_navigator.pipeline.load_cached_cluster_labels", return_value=None), \
                 patch("semantic_navigator.pipeline.save_cached_cluster_labels"):
                trees = await label_nodes(facets_mock, parent, progress)
            return trees

        trees = asyncio.run(run())
        assert len(trees) == 2
        # Each tree should have children (the leaf trees)
        assert all(isinstance(t, Tree) for t in trees)
        assert call_count >= 2  # at least leaf batches + cluster


# ---------------------------------------------------------------------------
# End-to-end: cluster → build_cluster_tree → tree structure
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_cluster_to_tree_structure(self):
        """Synthetic data through cluster + tree building produces valid structure."""
        rng = numpy.random.default_rng(99)
        embeds = [
            Embed(f"f{i}.py", f"f{i}.py:\n\ncontent", rng.standard_normal(32).astype(float32))
            for i in range(40)
        ]
        c = Cluster(embeds)

        ct = build_cluster_tree(c)

        # Verify tree invariants
        from semantic_navigator.pipeline import _count_tree_leaves, _count_tree_depth
        leaves = _count_tree_leaves(ct)
        assert leaves >= 2
        depth = _count_tree_depth(ct)
        assert depth >= 1

        # All original embeds reachable from leaves
        def collect_leaf_embeds(node: ClusterTree) -> list[Embed]:
            if not node.children:
                return list(node.node.embeds)
            return [e for child in node.children for e in collect_leaf_embeds(child)]

        leaf_embeds = collect_leaf_embeds(ct)
        leaf_entries = sorted(e.entry for e in leaf_embeds)
        original_entries = sorted(e.entry for e in embeds)
        assert leaf_entries == original_entries

    def test_to_pattern_on_clustered_files(self):
        """to_pattern works on realistic file lists from clusters."""
        files = ["src/utils/auth.py", "src/utils/crypto.py", "src/utils/hash.py"]
        pattern = to_pattern(files)
        assert "src/utils/" in pattern
        assert ".py" in pattern
        assert "*" in pattern

    def test_to_files_on_nested_trees(self):
        """to_files flattens a realistic multi-level tree."""
        inner = [
            Tree("a.py: Auth helper", ["a.py"], []),
            Tree("b.py: Crypto util", ["b.py"], []),
        ]
        outer = [
            Tree("src/utils/*: Utilities", ["a.py", "b.py"], inner),
            Tree("main.py: Entry point", ["main.py"], []),
        ]
        assert to_files(outer) == ["a.py", "b.py", "main.py"]


# ---------------------------------------------------------------------------
# AspectPool.remove() — drains queue and removes dead aspect
# ---------------------------------------------------------------------------

class TestAspectPoolRemove:
    def test_remove_drains_dead_aspect(self):
        a1, a2, a3 = _make_aspect("a"), _make_aspect("b"), _make_aspect("c")
        pool = AspectPool([a1, a2, a3])

        async def run():
            # Acquire one to populate the queue
            got = await pool.acquire()
            pool.release(got)
            # Remove a2
            remaining = pool.remove(a2)
            assert remaining == 2
            assert a2 not in pool.aspects
            # Verify we can only acquire a1 and a3
            acquired = set()
            acquired.add((await pool.acquire()).name)
            acquired.add((await pool.acquire()).name)
            assert acquired == {"a", "c"}

        asyncio.run(run())

    def test_remove_last_aspect(self):
        a = _make_aspect("only")
        pool = AspectPool([a])

        async def run():
            await pool.acquire()
            remaining = pool.remove(a)
            assert remaining == 0
            assert pool.aspects == []

        asyncio.run(run())

    def test_remove_aspect_not_in_queue(self):
        """Remove an aspect that was acquired (not in queue) still removes from list."""
        a1, a2 = _make_aspect("a"), _make_aspect("b")
        pool = AspectPool([a1, a2])

        async def run():
            got = await pool.acquire()  # removes from queue
            remaining = pool.remove(got)
            assert remaining == 1

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Corrupt cache recovery in integration context
# ---------------------------------------------------------------------------

class TestCorruptCacheIntegration:
    def test_corrupt_label_triggers_relabel(self, tmp_path: Path):
        """Corrupt label cache returns None, treated as uncached."""
        (tmp_path / "bad-hash.json").write_text("{{invalid")
        from semantic_navigator.cache import load_cached_label
        result = load_cached_label(tmp_path, "bad-hash")
        assert result is None

    def test_corrupt_embedding_triggers_recompute(self, tmp_path: Path):
        """Corrupt embedding returns None, treated as uncached."""
        (tmp_path / "bad.npy").write_bytes(b"\x00\x01\x02")
        from semantic_navigator.cache import load_cached_embedding
        result = load_cached_embedding(tmp_path, "bad")
        assert result is None

    def test_overwrite_corrupt_with_valid(self, tmp_path: Path):
        """Saving over a corrupt file produces a valid cached entry."""
        (tmp_path / "key.json").write_text("corrupt")
        assert load_cached_label(tmp_path, "key") is None
        save_cached_label(tmp_path, "key", _label("fixed"))
        loaded = load_cached_label(tmp_path, "key")
        assert loaded is not None
        assert loaded.label == "fixed"


# ---------------------------------------------------------------------------
# _generate_paths forward-slash consistency
# ---------------------------------------------------------------------------

class TestGeneratePathsConsistency:
    def test_paths_use_forward_slashes(self, tmp_path: Path):
        """All returned paths use forward slashes regardless of OS."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "file.py").write_text("code")
        # scandir fallback only lists immediate children, so test top-level
        (tmp_path / "top.py").write_text("top")
        paths = _generate_paths(str(tmp_path))
        for p in paths:
            assert "\\" not in p, f"Path contains backslash: {p}"

    def test_empty_directory(self, tmp_path: Path):
        paths = _generate_paths(str(tmp_path))
        assert paths == []

    def test_only_binary_files(self, tmp_path: Path):
        """Directory with only binary files still returns paths (filtering happens in _read_file)."""
        (tmp_path / "img.bin").write_bytes(b"\x80\x81")
        paths = _generate_paths(str(tmp_path))
        assert len(paths) == 1


# ---------------------------------------------------------------------------
# _read_file edge cases
# ---------------------------------------------------------------------------

class TestReadFileEdgeCases:
    def test_empty_file(self, tmp_path: Path):
        (tmp_path / "empty.py").write_text("")

        async def run():
            return await _read_file(str(tmp_path), "empty.py")

        result = asyncio.run(run())
        assert result is not None
        path, content = result
        assert path == "empty.py"
        assert "empty.py:\n\n" in content

    def test_unicode_content(self, tmp_path: Path):
        (tmp_path / "uni.py").write_text("# 日本語コメント\nprint('hello')", encoding="utf-8")

        async def run():
            return await _read_file(str(tmp_path), "uni.py")

        result = asyncio.run(run())
        assert result is not None
        assert "日本語" in result[1]

    def test_nonexistent_file(self, tmp_path: Path):
        async def run():
            return await _read_file(str(tmp_path), "does_not_exist.py")

        # Should not crash — returns None or raises handled exception
        try:
            result = asyncio.run(run())
        except FileNotFoundError:
            pass  # acceptable


# ---------------------------------------------------------------------------
# Empty cluster through pipeline
# ---------------------------------------------------------------------------

class TestEmptyCluster:
    def test_empty_cluster_returns_single(self):
        """Cluster with no embeds returns itself as single cluster."""
        c = Cluster([])
        result = cluster(c)
        assert len(result) == 1
        assert result[0] is c

    def test_empty_cluster_tree(self):
        c = Cluster([])
        ct = build_cluster_tree(c)
        assert ct.children == []
        assert ct.node is c

    def test_single_file_cluster(self):
        """Cluster with one file is a leaf."""
        e = _embed("only.py")
        c = Cluster([e])
        ct = build_cluster_tree(c)
        assert ct.children == []
        from semantic_navigator.pipeline import _count_tree_leaves
        assert _count_tree_leaves(ct) == 1


# ---------------------------------------------------------------------------
# Atomic write behavior
# ---------------------------------------------------------------------------

class TestAtomicWrites:
    def test_label_write_no_temp_files_left(self, tmp_path: Path):
        """After save, no .tmp files remain in the directory."""
        save_cached_label(tmp_path, "k1", _label("test"))
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_embedding_write_no_temp_files_left(self, tmp_path: Path):
        import numpy
        vec = numpy.array([1.0, 2.0], dtype=float32)
        save_cached_embedding(tmp_path, "k1", vec)
        tmp_files = list(tmp_path.glob("*.tmp"))
        npy_tmps = list(tmp_path.glob("*.npy.npy"))
        assert tmp_files == []
        assert npy_tmps == [], "Double .npy extension detected"

    def test_cluster_label_write_no_temp_files(self, tmp_path: Path):
        labels = Labels(labels=[_label("a")])
        save_cached_cluster_labels(tmp_path, "ck", labels)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_overwrite_existing_label(self, tmp_path: Path):
        """Overwriting a label produces valid data (atomic replace)."""
        save_cached_label(tmp_path, "k", _label("old"))
        save_cached_label(tmp_path, "k", _label("new"))
        loaded = load_cached_label(tmp_path, "k")
        assert loaded.label == "new"


# ---------------------------------------------------------------------------
# Cached label reuse in label_nodes
# ---------------------------------------------------------------------------

class TestLabelNodesCachedReuse:
    def test_cached_labels_skip_inference(self, tmp_path: Path):
        """When all labels are cached, complete() is never called."""
        e1 = _embed("a.py")
        ct = ClusterTree(Cluster([e1]), [])
        cached_label = _label("Cached Label")

        call_count = 0

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return Labels(labels=[_label("Should Not Appear")])

        async def run():
            from unittest.mock import MagicMock
            from tqdm import tqdm

            progress = tqdm(disable=True)
            facets_mock = MagicMock(
                repository="/r", model_identity="id",
                pool=MagicMock(min_local_n_ctx=None),
            )

            with patch("semantic_navigator.pipeline.complete", side_effect=mock_complete), \
                 patch("semantic_navigator.pipeline.label_cache_dir", return_value=tmp_path), \
                 patch("semantic_navigator.pipeline.load_cached_label", return_value=cached_label), \
                 patch("semantic_navigator.pipeline.save_cached_label"):
                trees = await label_nodes(facets_mock, ct, progress)
            return trees

        trees = asyncio.run(run())
        assert len(trees) == 1
        assert "Cached Label" in trees[0].label
        assert call_count == 0, "complete() should not be called when all labels are cached"
