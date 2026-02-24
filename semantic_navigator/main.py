import aiofiles
import argparse
import asyncio
import collections
import itertools
import math
import numpy
import os
import psutil
import scipy
import sklearn
import textual
import textual.app
import textual.widgets
import tiktoken

from dataclasses import dataclass
from dulwich.errors import NotGitRepository
from dulwich.repo import Repo
from itertools import batched, chain
from numpy import float32
from numpy.typing import NDArray
from pathlib import PurePath
from pydantic import BaseModel
from openai import AsyncOpenAI
from sklearn.neighbors import NearestNeighbors
from tiktoken import Encoding
from typing import Iterable
from tqdm.asyncio import tqdm_asyncio

max_clusters = 20

@dataclass(frozen = True)
class Facets:
    openai_client: AsyncOpenAI
    embedding_model: str
    completion_model: str
    embedding_encoding: Encoding
    completion_encoding: Encoding

def initialize(completion_model: str, embedding_model: str) -> Facets:
    openai_client = AsyncOpenAI()

    embedding_model = embedding_model

    completion_model = completion_model

    embedding_encoding = tiktoken.encoding_for_model(embedding_model)

    try:
        completion_encoding = tiktoken.encoding_for_model(completion_model)
    except KeyError:
        completion_encoding = tiktoken.get_encoding("o200k_base")

    return Facets(
        openai_client = openai_client,
        embedding_model = embedding_model,
        completion_model = completion_model,
        embedding_encoding = embedding_encoding,
        completion_encoding = completion_encoding
    )

@dataclass(frozen = True)
class Embed:
    entry: str
    content: str
    embedding: NDArray[float32]

@dataclass(frozen = True)
class Cluster:
    embeds: list[Embed]

max_tokens_per_embed = 8192

max_tokens_per_batch_embed = 300000

# Increase the soft limit on the number of open file descriptors and return the
# available number of descriptors our code should use
def tune_open_descriptors() -> int:
    # Use the current number of file descriptors in use to estimate how many
    # file descriptors to reserve
    num_fds = psutil.Process(os.getpid()).num_fds()

    reserve = 3 * num_fds

    if os.name == "posix":
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

        desired = max(soft, hard - reserve)

        resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))

        return hard
    else:
        # A quick search shows that the default limit (imposed by stdio) is 512
        # on Windows, which is probably a safe value to use for all non-POSIX
        # platforms
        return 512 - reserve

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

    # Half the smallest `ulimit` value commonly found in the wild (1024 on
    # Linux)
    available_descriptors = tune_open_descriptors()

    semaphore = asyncio.Semaphore(available_descriptors)

    async def read(path) -> list[tuple[str, str]]:
        try:
            absolute_path = os.path.join(directory, path)

            async with semaphore, aiofiles.open(absolute_path, "rb") as handle:
                prefix = f"{path}:\n\n"

                bytestring = await handle.read()

                text = bytestring.decode("utf-8")

                prefix_tokens = facets.embedding_encoding.encode(prefix)
                text_tokens   = facets.embedding_encoding.encode(text)

                max_tokens_per_chunk = max_tokens_per_embed - len(prefix_tokens)

                return [
                    (path, facets.embedding_encoding.decode(prefix_tokens + list(chunk)))

                    # TODO: This currently only takes the first chunk because
                    # GPT has trouble labeling chunks in order when multiple
                    # chunks have the same file name.  Remove the `[:1]` when
                    # this is fixed.
                    for chunk in list(batched(text_tokens, max_tokens_per_chunk))[:1]
                ]

        except UnicodeDecodeError:
            # Ignore files that aren't UTF-8
            return [ ]

        except IsADirectoryError:
            # This can happen when a "file" listed by the repository is:
            #
            # - a submodule
            # - a symlink to a directory
            #
            # TODO: The submodule case can and should be fixed and properly
            # handled
            return [ ]

    tasks = tqdm_asyncio.gather(
        *(read(path) for path in generate_paths()),
        desc = "Reading files",
        unit = "file",
        leave = False
    )

    results = list(chain.from_iterable(await tasks))

    if not results:
        return Cluster([])

    paths, contents = zip(*results)

    max_embeds = math.floor(max_tokens_per_batch_embed / max_tokens_per_embed)

    async def embed_batch(input) -> list[NDArray[float32]]:
        response = await facets.openai_client.embeddings.create(
            model = facets.embedding_model,
            input = input
        )

        return [
            numpy.asarray(datum.embedding, float32) for datum in response.data
        ]

    tasks = tqdm_asyncio.gather(
        *(embed_batch(input) for input in batched(contents, max_embeds)),
        desc = "Embedding contents",
        unit = "batch",
        leave = False
    )

    embeddings = list(chain.from_iterable(await tasks))

    embeds = [
        Embed(path, content, embedding)
        for path, content, embedding in zip(paths, contents, embeddings)
    ]

    return Cluster(embeds)

# The clustering algorithm can go as low as 1 here, but we set it higher for
# two reasons:
#
# - it's easier for users to navigate when there is more branching at the
#   leaves
# - this also avoids straining the tree visualizer, which doesn't like a really
#   deeply nested tree structure.
max_leaves = 20

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

@dataclass(frozen = True)
class Tree:
    label: str
    files: list[str]
    children: list["Tree"]

def to_pattern(files: list[str]) -> str:
    prefix = os.path.commonprefix(files)
    suffix = os.path.commonprefix([ file[len(prefix):][::-1] for file in files ])[::-1]

    if suffix:
        if any([ file[len(prefix):-len(suffix)] for file in files ]):
            star = "*"
        else:
            star = ""
    else:
        if any([ file[len(prefix):] for file in files ]):
            star = "*"
        else:
            star = ""

    if prefix:
        if suffix:
            return f"{prefix}{star}{suffix}: "
        else:
            return f"{prefix}{star}: "
    else:
        if suffix:
            return f"{star}{suffix}: "
        else:
            return ""

class Label(BaseModel):
    overarchingTheme: str
    distinguishingFeature: str
    label: str
    
class Labels(BaseModel):
    labels: list[Label]

def to_files(trees: list[Tree]) -> list[str]:
    return [ file for tree in trees for file in tree.files ]

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

class UI(textual.app.App):
    def __init__(self, tree_):
        super().__init__()
        self.tree_ = tree_

    async def on_mount(self):
        self.treeview = textual.widgets.Tree(f"{self.tree_.label} ({len(self.tree_.files)})")

        def loop(node, children):
            for child in children:
                if len(child.files) <= 1:
                    n = node.add(child.label)
                    n.allow_expand = False
                else:
                    n = node.add(f"{child.label} ({len(child.files)})")
                    n.allow_expand = True

                    loop(n, child.children)

        loop(self.treeview.root, self.tree_.children)

        self.mount(self.treeview)

def main():
    parser = argparse.ArgumentParser(
        prog = "semantic-navigator",
        description = "Cluster documents by semantic facets",
    )

    parser.add_argument("repository")
    parser.add_argument("--completion-model", default = "gpt-5-mini")
    parser.add_argument("--embedding-model", default = "text-embedding-3-large")
    arguments = parser.parse_args()

    facets = initialize(arguments.completion_model, arguments.embedding_model)

    async def async_tasks():
        initial_cluster = await embed(facets, arguments.repository)

        tree_ = await tree(facets, arguments.repository, initial_cluster)

        return tree_

    tree_ = asyncio.run(async_tasks())

    UI(tree_).run()

if __name__ == "__main__":
    main()
