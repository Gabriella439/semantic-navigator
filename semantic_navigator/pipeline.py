import aiofiles
import asyncio
import collections
import itertools
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
from itertools import batched, chain
from numpy import float32
from numpy.typing import NDArray
from pathlib import PurePath
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from typing import Iterable

from semantic_navigator.models import (
    Cluster, Embed, Facets, Labels, Tree,
)
from semantic_navigator.util import to_files, to_pattern

max_clusters = 20

max_leaves = 20


def initialize(
    completion_model: str,
    embedding_model_name: str,
    local_embedding_model: str | None = None,
) -> Facets:
    from openai import AsyncOpenAI

    openai_client = AsyncOpenAI()

    if local_embedding_model is not None:
        # fastembed for local embeddings
        from fastembed import TextEmbedding
        embedding = TextEmbedding(model_name=local_embedding_model)
        openai_embedding_model = None
    else:
        # OpenAI for embeddings (default)
        embedding = None
        openai_embedding_model = embedding_model_name

    return Facets(
        openai_client = openai_client,
        embedding_model = embedding,
        embedding_model_name = embedding_model_name,
        completion_model = completion_model,
        openai_embedding_model = openai_embedding_model,
    )


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

    batches = list(batched(truncated_contents, max_embeds_per_batch))
    new_embeddings_nested = await tqdm_asyncio.gather(
        *(openai_embed_batch(list(batch)) for batch in batches),
        desc = f"Embedding contents ({len(contents)} files)",
        unit = "batch",
        leave = False,
    )
    return [e for batch_result in new_embeddings_nested for e in batch_result]


def _embed_with_fastembed(facets: Facets, contents: list[str]) -> list[NDArray[float32]]:
    """Embed contents using fastembed."""
    return [
        numpy.asarray(e, float32)
        for e in tqdm(
            facets.embedding_model.embed(contents),
            desc = f"Embedding contents ({len(contents)} files)",
            unit = "file",
            total = len(contents),
            leave = False
        )
    ]


async def embed(facets: Facets, directory: str) -> Cluster:
    try:
        repo = Repo.discover(directory)

        def generate_paths() -> Iterable[str]:
            for bytestring in repo.open_index().paths():
                path = bytestring.decode("utf-8")

                subdirectory = PurePath(directory).relative_to(repo.path)

                try:
                    relative_path = PurePath(path).relative_to(subdirectory)

                    yield str(relative_path)
                except ValueError:
                    pass

    except NotGitRepository:
        def generate_paths() -> Iterable[str]:
            for entry in os.scandir(directory):
                if entry.is_file(follow_symlinks = False):
                    yield entry.path

    tasks = tqdm_asyncio.gather(
        *(_read_file(directory, path) for path in generate_paths()),
        desc = "Reading files",
        unit = "file",
        leave = False
    )

    results = [result for result in await tasks if result is not None]

    if not results:
        return Cluster([])

    paths, contents = zip(*results)

    if facets.openai_embedding_model is not None:
        embeddings = await _embed_with_openai(facets, list(contents))
    else:
        embeddings = _embed_with_fastembed(facets, list(contents))

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

    # The following code computes an affinity matrix using a radial basis
    # function with an adaptive σ.  See:
    #
    #     L. Zelnik-Manor, P. Perona (2004), "Self-Tuning Spectral Clustering"

    normalized = sklearn.preprocessing.normalize(embeddings)

    # The original paper suggests setting K (`n_neighbors`) to 7.  Here we do
    # something a little fancier and try to find a low value of `n_neighbors`
    # that produces one connected component.  This usually ends up being around
    # 7 anyway.
    #
    # The reason we want to avoid multiple connected components is because if
    # we have more than one connected component then those connected components
    # will dominate the clusters suggested by spectral clustering.  We don't
    # want that because we don't want spectral clustering to degenerate to the
    # same result as K nearest neighbors.  We want the K nearest neighbors
    # algorithm to weakly inform the spectral clustering algorithm without
    # dominating the result.
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

    # We don't attempt to find the absolute lowest value of K (`n_neighbors`).
    # Instead we just sample a few values and pick a "small enough" one.
    candidate_neighbor_counts = list(itertools.takewhile(
        lambda x: x < N,
        (round(math.exp(n)) for n in itertools.count())
    )) + [ math.floor(N / 2) ]

    results = [
        get_nearest_neighbors(n_neighbors)
        for n_neighbors in candidate_neighbor_counts
    ]

    # Find the first sample value of K (`n_neighbors`) that produces one
    # connected component.  There's guaranteed to be at least one since the
    # very last value we sample (⌊N/2⌋) always produces one connected
    # component.
    n_neighbors, nearest_neighbors = [
        (n_neighbors, nearest_neighbors)
        for n_components, n_neighbors, nearest_neighbors in results
        if n_components == 1
    ][0]

    distances, indices = nearest_neighbors.kneighbors()

    # sigmas[i] = the distance of semantic embedding #i to its Kth nearest
    # neighbor
    sigmas = distances[:, -1]

    rows    = numpy.repeat(numpy.arange(N), n_neighbors)
    columns = indices.reshape(-1)

    d = distances.reshape(-1)

    sigma_i = numpy.repeat(sigmas, n_neighbors)
    sigma_j = sigmas[columns]

    denominator = numpy.maximum(sigma_i * sigma_j, 1e-12)
    data = numpy.exp(-(d * d) / denominator).astype(numpy.float32)

    # Affinity: A_ij = exp(-d(x_i, x_j)^2 / (σ_i σ_j))
    affinity = scipy.sparse.coo_matrix((data, (rows, columns)), shape = (N, N)).tocsr()

    affinity = (affinity + affinity.T) * 0.5
    affinity.setdiag(1.0)
    affinity.eliminate_zeros()

    # The following code is basically `sklearn.manifold.spectral_embeddings`,
    # but exploded out so that we can get access to the eigenvalues, which are
    # normally not exposed by the function.  We'll need those eigenvalues
    # later.
    random_state = sklearn.utils.check_random_state(0)

    laplacian, dd = scipy.sparse.csgraph.laplacian(
        affinity,
        normed = True,
        return_diag = True
    )

    # laplacian = set_diag(laplacian, 1, True)
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

    # Find the optimal cluster count by looking for the largest eigengap
    #
    # The reason the suggested cluster count is not just:
    #
    #     numpy.argmax(numpy.diff(eigenvalues)) + 1
    #
    # … is because we want at least two clusters
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


async def label_nodes(facets: Facets, c: Cluster, depth: int) -> list[Tree]:
    children = cluster(c)

    if len(children) == 1:
        def render_embed(embed: Embed) -> str:
            return f"# File: {embed.entry}\n\n{embed.content}"

        rendered_embeds = "\n\n".join([ render_embed(embed) for embed in c.embeds ])

        input = f"Label each file in 3 to 7 words.  Don't include file path/names in descriptions.\n\n{rendered_embeds}"

        response = await facets.openai_client.responses.parse(
            model = facets.completion_model,
            input = input,
            text_format = Labels
        )

        assert response.output_parsed is not None

        # assert len(response.output_parsed.labels) == len(c.embeds)

        return [
            Tree(f"{embed.entry}: {label.label}", [ embed.entry ], [])
            for label, embed in zip(response.output_parsed.labels, c.embeds)
        ]

    else:
        if depth == 0:
            treess = await tqdm_asyncio.gather(
                *(label_nodes(facets, child, depth + 1) for child in children),
                desc = "Labeling clusters",
                unit = "cluster",
                leave = False
            )
        else:
            treess = await asyncio.gather(
                *(label_nodes(facets, child, depth + 1) for child in children),
            )

        def render_cluster(trees: list[Tree]) -> str:
            rendered_trees = "\n".join([ tree.label for tree in trees ])

            return f"# Cluster\n\n{rendered_trees}"

        rendered_clusters = "\n\n".join([ render_cluster(trees) for trees in treess ])

        input = f"Label each cluster in 2 words.  Don't include file path/names in labels.\n\n{rendered_clusters}"

        response = await facets.openai_client.responses.parse(
            model = facets.completion_model,
            input = input,
            text_format = Labels
        )

        assert response.output_parsed is not None

        # assert len(response.output_parsed.labels) == len(children)

        return [
            Tree(f"{to_pattern(to_files(trees))}{label.label}", to_files(trees), trees)
            for label, trees in zip(response.output_parsed.labels, treess)
        ]


async def tree(facets: Facets, label: str, c: Cluster) -> Tree:
    children = await label_nodes(facets, c, 0)

    return Tree(label, to_files(children), children)
