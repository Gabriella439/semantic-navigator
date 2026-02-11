import aiofiles
import argparse
import asyncio
import collections
import itertools
import math
import numpy
import os
import scipy
import sklearn
import textual
import textual.app
import textual.widgets
import tiktoken

from dataclasses import dataclass
from git import Repo
from numpy import float32
from numpy.typing import NDArray
from pydantic import BaseModel
from openai import AsyncOpenAI
from tiktoken import Encoding
from tqdm.asyncio import tqdm_asyncio

max_tokens_per_embed = 8192

max_tokens_per_batch_embed = 300000

max_leaves = 7

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

async def embed(facets: Facets, repository: str) -> Cluster:
    repo = Repo(repository)

    async def read(path):
        absolute_path = os.path.join(repository, path)

        try:
            async with aiofiles.open(absolute_path, "rb") as handle:
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
                    for chunk in list(itertools.batched(text_tokens, max_tokens_per_chunk))[:1]
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
        *(read(path) for path, _ in repo.index.entries),
        desc = "Reading files",
        unit = "file",
        leave = False
    )

    results = list(itertools.chain.from_iterable(await tasks))

    if not results:
        return Cluster([])

    paths, contents = zip(*results)

    max_embeds = math.floor(max_tokens_per_batch_embed / max_tokens_per_embed)

    async def embed_batch(input):
        response = await facets.openai_client.embeddings.create(
            model = facets.embedding_model,
            input = input
        )

        return [
            numpy.asarray(datum.embedding, float32) for datum in response.data
        ]

    tasks = tqdm_asyncio.gather(
        *(embed_batch(input) for input in itertools.batched(contents, max_embeds)),
        desc = "Embedding contents",
        unit = "batch",
        leave = False
    )

    embeddings = list(itertools.chain.from_iterable(await tasks))

    embeds = [
        Embed(path, content, embedding)
        for path, content, embedding in zip(paths, contents, embeddings)
    ]

    return Cluster(embeds)

def cluster(input: Cluster) -> list[Cluster]:
    if len(input.embeds) <= max_leaves:
        return []

    entries, contents, embeddings = zip(*(
        (embed.entry, embed.content, embed.embedding)
        for embed in input.embeds
    ))

    N = len(embeddings)

    # The following code computes an affinity matrix using a radial basis
    # function with an adaptive adaptive σ.  See:
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
    # will dominate the clusters suggested by spectral cluster.  We don't want
    # that because we don't want spectral clustering to degenerate to the same
    # result as K nearest neighbors.  We want the K nearest neighbors algorithm
    # to weakly inform the spectral clustering algorithm without dominating the
    # result.
    def get_nearest_neighbors(n_neighbors):
        nearest_neighbors = sklearn.neighbors.NearestNeighbors(
            n_neighbors = n_neighbors,
            metric = "cosine",
            n_jobs = -1
        ).fit(normalized)

        directed_graph = nearest_neighbors.kneighbors_graph(
            mode = "connectivity"
        )

        undirected_graph = directed_graph.maximum(directed_graph.T)

        components, _ = scipy.sparse.csgraph.connected_components(undirected_graph)

        return components, n_neighbors, nearest_neighbors

    # We don't attempt to find the absolute lowest value of K (`n_neighbors`).
    # Instead we just sample a few values and pick a "small enough" one.
    candidate_neighbor_counts = list(itertools.takewhile(
        lambda x: x < N,
        (round(math.exp(n)) for n in itertools.count())
    )) + [ math.ceil(N / 2) ]

    results = [
        get_nearest_neighbors(n_neighbors)
        for n_neighbors in candidate_neighbor_counts
    ]

    # Find the first sample value of K (`n_neighbors`) that produces one
    # connected component.  There's guaranteed to be at least one since the
    # very last value we sample (⌈N/2⌉) always produces one connected
    # component.
    n_neighbors, nearest_neighbors = [
        (n_neighbors, nearest_neighbors)
        for components, n_neighbors, nearest_neighbors in results
        if components == 1
    ][0]

    distances, indices = nearest_neighbors.kneighbors(normalized)

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

    # This is actually the *maximum* number of clusters that the algorithm can
    # return.
    #
    # The algorithm is actually fast enough to return a much larger number of
    # clusters and sometimes you find much more optimal clusterings at much
    # higher cluster counts.  For example, I've seen repositories where the
    # optimal cluster count was 600+ (and the results were indeed good based on
    # visual inspection).  However, we cap the maximum cluster count to avoid
    # presenting too many clusters to the user at any level of the decision
    # tree.
    #
    # As a bonus, capping the cluster count improves performance, too.
    n_clusters = min(N - 1, max_clusters)

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
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        laplacian,
        k = n_clusters,
        sigma = 1.0,
        which = 'LM',
        tol = 0.0,
        v0 = v0
    )
    wide_spectral_embeddings = eigenvectors.T[n_clusters::-1] * dd
    wide_spectral_embeddings = sklearn.utils.extmath._deterministic_vector_sign_flip(wide_spectral_embeddings)
    wide_spectral_embeddings = wide_spectral_embeddings[1:n_clusters].T
    eigenvalues = eigenvalues[n_clusters::-1]
    eigenvalues *= -1

    # Find the optimal cluster count by looking for the largest eigengap
    #
    # The reason the suggested cluster count is not just:
    #
    #     numpy.argmax(numpy.diff(eigenvalues))
    #
    # … is because we want at least two clusters (otherwise what's the point?).
    n_clusters = numpy.argmax(numpy.diff(eigenvalues)[2:]) + 2

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

def to_pattern(files):
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

class Labels(BaseModel):
    labels: list[str]

async def label_nodes(facets: Facets, c: Cluster) -> list[Tree]:
    if len(c.embeds) <= max_leaves:
        def render_embed(embed: Embed) -> str:
            return f"# File: {embed.entry}\n\n{embed.content}"

        rendered_embeds = "\n\n".join([ render_embed(embed) for embed in c.embeds ])

        input = f"Describe in ≈3 words what distinguishes each one of these files from the other files.  Don't include the file path/name in the description.\n\n{rendered_embeds}"

        response = await facets.openai_client.responses.parse(
            model = facets.completion_model,
            input = input,
            text_format = Labels
        )

        assert response.output_parsed is not None

        # assert len(response.output_parsed.labels) == len(c.embeds)

        return [
            Tree(f"{embed.entry}: {label}", [ embed.entry ], [])
            for label, embed in zip(response.output_parsed.labels, c.embeds)
        ]

    else:
        children = cluster(c)

        treess = await tqdm_asyncio.gather(
            *(label_nodes(facets, child) for child in children),
            desc = "Labeling clusters",
            unit = "cluster",
            leave = False
        )

        def render_cluster(trees: list[Tree]) -> str:
            rendered_trees = "\n".join([ tree.label for tree in trees ])

            return f"# Cluster\n\n{rendered_trees}"

        rendered_clusters = "\n\n".join([ render_cluster(trees) for trees in treess ])

        input = f"Describe in ≈3 words what distinguishes each one of these clusters from the other clusters:\n\n{rendered_clusters}"

        response = await facets.openai_client.responses.parse(
            model = facets.completion_model,
            input = input,
            text_format = Labels
        )

        assert response.output_parsed is not None

        # assert len(response.output_parsed.labels) == len(children)

        def to_files(trees: list[Tree]) -> list[str]:
            return [
                file
                for tree in trees
                for file in tree.files
            ]

        return [
            Tree(f"{to_pattern(to_files(trees))}{label}", to_files(trees), trees)
            for label, trees in zip(response.output_parsed.labels, treess)
        ]

async def tree(facets: Facets, label: str, c: Cluster) -> Tree:
    children = await label_nodes(facets, c)

    files = [ file for child in children for file in child.files ]

    return Tree(label, files, children)

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
        prog = "facets",
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
