import aiofiles
import asyncio
import collections
import itertools
import json
import math
import numpy
import os
import scipy
import sklearn
import sklearn.cluster
import sklearn.preprocessing
import sklearn.utils
import sklearn.utils.extmath

from dulwich.errors import NotGitRepository
from dulwich.repo import Repo
from fastembed import TextEmbedding
from numpy import float32
from numpy.typing import NDArray
from pathlib import PurePath, PurePosixPath
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from semantic_navigator.models import (
    Cluster, ClusterTree, Embed, Facets, Tree,
)
from semantic_navigator.cache import (
    cluster_hash, content_hash, embedding_cache_dir, label_cache_dir,
    list_cached_keys, load_cached_cluster_labels, load_cached_embedding,
    load_cached_label, save_cached_cluster_labels, save_cached_embedding,
    save_cached_label,
)
from semantic_navigator.models import Label, Labels
from semantic_navigator.util import timed
from semantic_navigator.inference import complete

max_clusters = 20
max_leaves = 20


def _generate_paths(directory: str) -> list[str]:
    """Generate file paths from git index or directory scan.
    Returns forward-slash paths for consistent cache keys across platforms."""
    try:
        repo = Repo.discover(directory)
        paths = []
        # Normalize both to PurePosixPath for cross-platform consistency:
        # - repo.path may use OS-native separators
        # - git index paths always use forward slashes
        repo_root = PurePosixPath(PurePath(repo.path).as_posix())
        subdir = PurePosixPath(PurePath(directory).as_posix())
        subdirectory = subdir.relative_to(repo_root)
        for bytestring in repo.open_index().paths():
            path = bytestring.decode("utf-8")  # already forward-slash
            try:
                relative_path = PurePosixPath(path).relative_to(subdirectory)
                paths.append(str(relative_path))
            except ValueError:
                pass
        return paths
    except NotGitRepository:
        return [
            entry.name for entry in os.scandir(directory)
            if entry.is_file(follow_symlinks = False)
        ]


async def _read_file(directory: str, path: str) -> tuple[str, str] | None:
    """Read a single file, returning (path, content) or None for non-text files."""
    try:
        absolute_path = os.path.join(directory, path)
        async with aiofiles.open(absolute_path, "rb") as handle:
            bytestring = await handle.read()
            text = bytestring.decode("utf-8")
            return (path, f"{path}:\n\n{text}")
    except UnicodeDecodeError:
        return None
    except IsADirectoryError:
        # Submodules or symlinks to directories
        return None


async def _embed_with_openai(facets: Facets, contents: list[str]) -> list[NDArray[float32]]:
    """Embed contents using OpenAI API with token-aware chunking."""
    import tiktoken
    try:
        encoding = tiktoken.encoding_for_model(facets.openai_embedding_model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    max_tokens_per_embed = 8192
    max_embeds_per_batch = 2048

    truncated_contents = []
    for content in contents:
        tokens = encoding.encode(content)
        if len(tokens) > max_tokens_per_embed:
            tokens = tokens[:max_tokens_per_embed]
            truncated_contents.append(encoding.decode(tokens))
        else:
            truncated_contents.append(content)

    async def openai_embed_batch(batch: list[str]) -> list[NDArray[float32]]:
        response = await facets.openai_client.embeddings.create(
            model = facets.openai_embedding_model,
            input = batch,
        )
        return [numpy.asarray(datum.embedding, float32) for datum in response.data]

    from itertools import batched
    batches = list(batched(truncated_contents, max_embeds_per_batch))
    new_embeddings_nested = await tqdm_asyncio.gather(
        *(openai_embed_batch(list(batch)) for batch in batches),
        desc = f"Embedding contents ({len(contents)} uncached)",
        unit = "batch",
        leave = False,
    )
    return [e for batch_result in new_embeddings_nested for e in batch_result]


def _embed_with_fastembed(facets: Facets, contents: list[str]) -> list[NDArray[float32]]:
    """Embed contents using fastembed with GPU fallback."""
    def do_embed(model: TextEmbedding, desc: str) -> list[NDArray[float32]]:
        return [
            numpy.asarray(e, float32)
            for e in tqdm(
                model.embed(contents, batch_size = facets.batch_size),
                desc = desc,
                unit = "file",
                total = len(contents),
                leave = False
            )
        ]

    try:
        return do_embed(facets.embedding_model, f"Embedding contents ({len(contents)} uncached)")
    except Exception as e:
        if not facets.gpu:
            raise
        print(f"\nGPU embedding failed ({e}), falling back to CPU...")
        cpu_model = TextEmbedding(model_name = facets.embedding_model_name, providers = ["CPUExecutionProvider"])
        return do_embed(cpu_model, "Embedding contents (CPU)")


async def embed(facets: Facets, directory: str) -> Cluster:
    file_paths = _generate_paths(directory)

    tasks = tqdm_asyncio.gather(
        *(_read_file(directory, path) for path in file_paths),
        desc = "Reading files",
        unit = "file",
        leave = False
    )

    results = [ result for result in await tasks if result is not None ]

    if not results:
        return Cluster([])

    paths, contents = zip(*results)

    cdir = embedding_cache_dir(facets.embedding_model_name)
    cached_emb_keys = list_cached_keys(cdir, ".npy")
    embeddings: list[NDArray[float32]] = [None] * len(contents)  # type: ignore[list-item]
    uncached_indices = []

    for i, content in enumerate(contents):
        key = content_hash(content)
        if key in cached_emb_keys:
            embeddings[i] = load_cached_embedding(cdir, key)
        else:
            uncached_indices.append(i)

    n_cached = len(contents) - len(uncached_indices)
    if n_cached > 0:
        print(f"Embeddings: {n_cached}/{len(contents)} cached, {len(uncached_indices)} to compute")

    if uncached_indices:
        uncached_contents = [contents[i] for i in uncached_indices]

        if facets.openai_client is not None:
            new_embeddings = await _embed_with_openai(facets, uncached_contents)
        else:
            new_embeddings = _embed_with_fastembed(facets, uncached_contents)

        for i, embedding in zip(uncached_indices, new_embeddings):
            embeddings[i] = embedding
            save_cached_embedding(cdir, content_hash(contents[i]), embedding)

    embeds = [
        Embed(path, content, embedding)
        for path, content, embedding in zip(paths, contents, embeddings)
    ]

    return Cluster(embeds)


def cluster(input: Cluster) -> list[Cluster]:
    N = len(input.embeds)

    if N <= max_leaves:
        return [input]

    entries, contents, embeddings = zip(*(
        (embed.entry, embed.content, embed.embedding)
        for embed in input.embeds
    ))

    normalized = sklearn.preprocessing.normalize(embeddings)

    def get_nearest_neighbors(n_neighbors: int) -> tuple[int, int, NearestNeighbors]:
        nearest_neighbors = NearestNeighbors(
            n_neighbors = n_neighbors,
            metric = "cosine",
            n_jobs = -1
        ).fit(normalized)

        graph = nearest_neighbors.kneighbors_graph(
            mode = "connectivity"
        )

        n_components, _ = scipy.sparse.csgraph.connected_components(
            graph,
            directed = False
        )

        return n_components, n_neighbors, nearest_neighbors

    candidate_neighbor_counts = list(itertools.takewhile(
        lambda x: x < N,
        (round(math.exp(n)) for n in itertools.count())
    )) + [ math.floor(N / 2) ]

    results = [
        get_nearest_neighbors(n_neighbors)
        for n_neighbors in candidate_neighbor_counts
    ]

    connected = [
        (n_neighbors, nearest_neighbors)
        for n_components, n_neighbors, nearest_neighbors in results
        if n_components == 1
    ]
    if not connected:
        # No neighbor count produced a connected graph; fall back to treating as a single cluster
        return [input]
    n_neighbors, nearest_neighbors = connected[0]

    distances, indices = nearest_neighbors.kneighbors()

    sigmas = distances[:, -1]

    rows    = numpy.repeat(numpy.arange(N), n_neighbors)
    columns = indices.reshape(-1)

    d = distances.reshape(-1)

    sigma_i = numpy.repeat(sigmas, n_neighbors)
    sigma_j = sigmas[columns]

    denominator = numpy.maximum(sigma_i * sigma_j, 1e-12)
    data = numpy.exp(-(d * d) / denominator).astype(numpy.float32)

    affinity = scipy.sparse.coo_matrix((data, (rows, columns)), shape = (N, N)).tocsr()

    affinity = (affinity + affinity.T) * 0.5
    affinity.setdiag(1.0)
    affinity.eliminate_zeros()

    random_state = sklearn.utils.check_random_state(0)

    laplacian, dd = scipy.sparse.csgraph.laplacian(
        affinity,
        normed = True,
        return_diag = True
    )

    laplacian = laplacian.tocoo()
    laplacian.data[laplacian.row == laplacian.col] = 1
    laplacian = laplacian.tocsr()

    laplacian *= -1
    v0 = random_state.uniform(-1, 1, N)

    if max_clusters + 1 < N:
        k = max_clusters + 1

        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            laplacian,
            k = k,
            sigma = 1.0,
            which = 'LM',
            tol = 0.0,
            v0 = v0
        )
    else:
        k = N

        eigenvalues, eigenvectors = scipy.linalg.eigh(
            laplacian.toarray(),
            check_finite = False
        )

    indices = numpy.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[indices]

    eigenvectors = eigenvectors[:, indices]

    wide_spectral_embeddings = eigenvectors.T / dd
    wide_spectral_embeddings = sklearn.utils.extmath._deterministic_vector_sign_flip(wide_spectral_embeddings)
    wide_spectral_embeddings = wide_spectral_embeddings[1:k].T
    eigenvalues = eigenvalues * -1

    n_clusters = numpy.argmax(numpy.diff(eigenvalues[1:])) + 2

    spectral_embeddings = wide_spectral_embeddings[:, :n_clusters]

    spectral_embeddings = sklearn.preprocessing.normalize(spectral_embeddings)

    labels = sklearn.cluster.KMeans(
        n_clusters = n_clusters,
        random_state = 0,
        n_init = "auto"
    ).fit_predict(spectral_embeddings)

    groups = collections.OrderedDict()

    for (label, entry, content, embedding) in zip(labels, entries, contents, embeddings):
        groups.setdefault(label, []).append(Embed(entry, content, embedding))

    return [ Cluster(embeds) for embeds in groups.values() ]


def _count_tree_depth(ct: ClusterTree) -> int:
    if not ct.children:
        return 0
    return 1 + max(_count_tree_depth(c) for c in ct.children)


def _count_tree_leaves(ct: ClusterTree) -> int:
    if not ct.children:
        return 1
    return sum(_count_tree_leaves(c) for c in ct.children)


def build_cluster_tree(c: Cluster) -> ClusterTree:
    children = cluster(c)
    if len(children) == 1:
        return ClusterTree(c, [])
    return ClusterTree(c, [build_cluster_tree(child) for child in children])


def to_pattern(files: list[str]) -> str:
    prefix = os.path.commonprefix(files)
    suffix = os.path.commonprefix([ file[len(prefix):][::-1] for file in files ])[::-1]

    middle_parts = [file[len(prefix):-len(suffix)] if suffix else file[len(prefix):] for file in files]
    star = "*" if any(middle_parts) else ""

    if not prefix and not suffix:
        return ""
    if not suffix:
        return f"{prefix}{star}: "
    if not prefix:
        return f"{star}{suffix}: "
    return f"{prefix}{star}{suffix}: "


def to_files(trees: list[Tree]) -> list[str]:
    return [ file for tree in trees for file in tree.files ]


def count_cached_labels(ct: ClusterTree, repository: str, model_identity: str, cached_keys: set[str] | None = None) -> tuple[int, int, int]:
    """Count cached vs uncached file labels in the tree.
    Returns (uncached_files, cached_files, cached_clusters)."""
    if cached_keys is None:
        cached_keys = list_cached_keys(label_cache_dir(repository, model_identity), ".json")
    if not ct.children:
        uncached = sum(
            1 for embed in ct.node.embeds
            if content_hash(embed.content) not in cached_keys
        )
        cached = len(ct.node.embeds) - uncached
        return (uncached, cached, 0)

    total_uncached = 0
    total_cached = 0
    total_cached_clusters = 0
    for child in ct.children:
        uncached, cached, cc = count_cached_labels(child, repository, model_identity, cached_keys)
        total_uncached += uncached
        total_cached += cached
        total_cached_clusters += cc
    # Check if this cluster node itself is cached
    all_files = [e.entry for e in ct.node.embeds]
    c_key = "cluster-" + cluster_hash(all_files)
    if c_key in cached_keys:
        total_cached_clusters += 1
    return (total_uncached, total_cached, total_cached_clusters)


def _count_expected_calls(ct: ClusterTree, cached_keys: set[str]) -> int:
    """Estimate the number of LLM calls needed to label the tree.
    Each cluster node = 1 call, each leaf with uncached files = 1+ calls."""
    if not ct.children:
        has_uncached = any(
            content_hash(embed.content) not in cached_keys
            for embed in ct.node.embeds
        )
        return 1 if has_uncached else 0

    total = sum(_count_expected_calls(child, cached_keys) for child in ct.children)
    # This cluster node itself will make 1 call (cached or not, it updates progress)
    total += 1
    return total


async def _label_leaf_node(facets: Facets, ct: ClusterTree, progress: tqdm) -> list[Tree]:
    """Handle leaf clusters: batch files, prompt for labels, cache results."""
    ldir = label_cache_dir(facets.repository, facets.model_identity)
    cached_labels: dict[int, Label] = {}
    uncached_embeds: list[tuple[int, Embed]] = []

    for i, embed in enumerate(ct.node.embeds):
        cached = load_cached_label(ldir, content_hash(embed.content))
        if cached is not None:
            cached_labels[i] = cached
        else:
            uncached_embeds.append((i, embed))

    if uncached_embeds:
        # Split into batches for local models to avoid exceeding context window
        if facets.pool.min_local_n_ctx is not None:
            # ~1.5 chars per token for code, reserve 4096 tokens for response,
            # 1000 chars for prompt instructions/schema overhead
            max_chars = max(int((facets.pool.min_local_n_ctx - 4096) * 1.5) - 1000, 1000)

            def render_embed(embed: Embed) -> str:
                content = embed.content
                if len(content) > max_chars:
                    content = content[:max_chars] + "\n... (truncated)"
                return f"# File: {embed.entry}\n\n{content}"

            batches: list[list[tuple[int, Embed]]] = []
            batch: list[tuple[int, Embed]] = []
            batch_size = 0
            for item in uncached_embeds:
                size = min(len(item[1].content), max_chars) + len(item[1].entry) + 20
                if batch and batch_size + size > max_chars:
                    batches.append(batch)
                    batch = [item]
                    batch_size = size
                else:
                    batch.append(item)
                    batch_size += size
            if batch:
                batches.append(batch)
        else:
            def render_embed(embed: Embed) -> str:
                return f"# File: {embed.entry}\n\n{embed.content}"

            batches = [uncached_embeds]

        batch_sizes = [len(b) for b in batches]
        tqdm.write(f"  Leaf: {len(uncached_embeds)} files in {len(batches)} batch{'es' if len(batches) != 1 else ''} (avg {sum(batch_sizes)/len(batch_sizes):.1f} files/batch)")

        schema = json.dumps(Labels.model_json_schema(), indent=2)

        async def process_batch(batch: list[tuple[int, Embed]]) -> None:
            rendered_embeds = "\n\n".join([ render_embed(embed) for _, embed in batch ])

            prompt = (
                f"Label each file in 3 to 7 words. Don't include file path/names in descriptions.\n"
                f"Return exactly {len(batch)} label{'s' if len(batch) != 1 else ''}, one per file.\n\n"
                f"{rendered_embeds}\n\n"
                f"Respond with ONLY valid JSON matching this schema (no markdown, no code fences, no other text):\n{schema}"
            )

            result = await complete(facets, prompt, Labels, progress, expected_count=len(batch))

            for (i, embed), label in zip(batch, result.labels):
                cached_labels[i] = label
                # Save each file label immediately so progress survives crashes
                save_cached_label(ldir, content_hash(embed.content), label)

        await asyncio.gather(*(process_batch(batch) for batch in batches))

    return [
        Tree(
            f"{embed.entry}: {cached_labels[i].label}" if i in cached_labels else embed.entry,
            [ embed.entry ],
            [],
        )
        for i, embed in enumerate(ct.node.embeds)
    ]


async def _label_cluster_node(facets: Facets, ct: ClusterTree, progress: tqdm) -> list[Tree]:
    """Handle non-leaf clusters: recurse children, cache cluster labels."""
    treess = await asyncio.gather(
        *(label_nodes(facets, child, progress) for child in ct.children),
    )

    # Check cluster label cache
    ldir = label_cache_dir(facets.repository, facets.model_identity)
    all_files = [f for trees in treess for tree in trees for f in tree.files]
    c_key = cluster_hash(all_files)
    cached_cluster = load_cached_cluster_labels(ldir, c_key)

    if cached_cluster is not None and len(cached_cluster.labels) == len(treess):
        tqdm.write(f"  Cluster ({len(treess)} children): cached")
        if progress is not None:
            progress.update(1)
        return [
            Tree(f"{to_pattern(to_files(trees))}{label.label}", to_files(trees), trees)
            for label, trees in zip(cached_cluster.labels, treess)
        ]

    def render_cluster(trees: list[Tree]) -> str:
        rendered_trees = "\n".join([ tree.label for tree in trees ])
        return f"# Cluster\n\n{rendered_trees}"

    rendered_clusters = "\n\n".join([ render_cluster(trees) for trees in treess ])

    schema = json.dumps(Labels.model_json_schema(), indent=2)
    prompt = (
        f"Label each cluster in 2 words. Don't include file path/names in labels.\n"
        f"Return exactly {len(treess)} label{'s' if len(treess) != 1 else ''}, one per cluster.\n\n"
        f"{rendered_clusters}\n\n"
        f"Respond with ONLY valid JSON matching this schema (no markdown, no code fences, no other text):\n{schema}"
    )

    labels = await complete(facets, prompt, Labels, progress, expected_count=len(treess))

    # Cache cluster labels for crash recovery
    save_cached_cluster_labels(ldir, c_key, labels)

    return [
        Tree(f"{to_pattern(to_files(trees))}{label.label}", to_files(trees), trees)
        for label, trees in zip(labels.labels, treess)
    ]


async def label_nodes(facets: Facets, ct: ClusterTree, progress: tqdm) -> list[Tree]:
    if not ct.children:
        return await _label_leaf_node(facets, ct, progress)
    return await _label_cluster_node(facets, ct, progress)


async def tree(facets: Facets, label: str, c: Cluster, timings: dict[str, float] | None = None) -> Tree:
    with timed("Clustering", timings):
        ct = build_cluster_tree(c)
    depth = _count_tree_depth(ct)
    leaves = _count_tree_leaves(ct)
    print(f"  {leaves} leaf clusters, depth {depth}")
    uncached_files, cached_files, cached_clusters = count_cached_labels(ct, facets.repository, facets.model_identity)
    total_files = cached_files + uncached_files
    cache_parts = []
    if cached_files > 0:
        cache_parts.append(f"{cached_files}/{total_files} file labels")
    if cached_clusters > 0:
        cache_parts.append(f"{cached_clusters} cluster labels")
    if cache_parts:
        print(f"Cache: {', '.join(cache_parts)}", flush=True)
    if uncached_files > 0:
        print(f"Labeling {uncached_files} uncached files...", flush=True)
    cached_keys = list_cached_keys(label_cache_dir(facets.repository, facets.model_identity), ".json")
    total_calls = _count_expected_calls(ct, cached_keys)
    with timed("Labeling", timings):
        with tqdm(total = total_calls, desc = "Labeling", unit = "call", leave = False) as progress:
            children = await label_nodes(facets, ct, progress)

    return Tree(label, to_files(children), children)
