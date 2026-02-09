import aiofiles
import argparse
import asyncio
import collections
import dataclasses
import git
import itertools
import math
import numpy
import openai
import os
import pathlib
import scipy
import sklearn
import textual
import textual.app
import textual.widgets
import tiktoken

from git import Repo
from numpy import float32
from numpy.typing import NDArray
from openai import AsyncOpenAI
from tiktoken import Encoding
from typing import TypeVar

max_tokens_per_embed = 8192

max_tokens_per_batch_embed = 300000

@dataclasses.dataclass(frozen=True)
class Facets:
    openai_client: AsyncOpenAI
    embedding_model: str
    completion_model: str
    embedding_encoding: Encoding
    completion_encoding: Encoding

def initialize() -> Facets:
    openai_client = AsyncOpenAI()

    embedding_model = "text-embedding-3-large"

    completion_model = "gpt-5.2"

    embedding_encoding = tiktoken.encoding_for_model(embedding_model)

    completion_encoding = tiktoken.get_encoding("o200k_base")

    return Facets(
        openai_client = openai_client,
        embedding_model = embedding_model,
        completion_model = completion_model,
        embedding_encoding = embedding_encoding,
        completion_encoding = completion_encoding
    )

@dataclasses.dataclass(frozen=True)
class Embed:
    entry: str
    embedding: NDArray[float32]

@dataclasses.dataclass(frozen=True)
class Cluster:
    embeds: list[Embed]

async def embed(facets: Facets, repository: str) -> Cluster:
    print("[+] Embedding files")

    repo = Repo(repository)

    async def read(entry):
        absolute_path = os.path.join(repository, entry)

        try:
            async with aiofiles.open(absolute_path, "rb") as handle:
                bytestring = await handle.read()

                text = bytestring.decode("utf-8")

                annotated = f"{entry}:\n\n{text}"

                tokens = facets.embedding_encoding.encode(annotated)

                # TODO: chunk instead of truncate
                truncated = tokens[:max_tokens_per_embed]

                return [ (entry, facets.embedding_encoding.decode(truncated)) ]

        except UnicodeDecodeError:
            # Ignore documents that aren't UTF-8
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

    results = list(itertools.chain.from_iterable(await asyncio.gather(*(read(entry) for entry, _ in repo.index.entries))))

    entries, contents = zip(*results)

    max_embeds = math.floor(max_tokens_per_batch_embed / max_tokens_per_embed)

    async def embed_batch(input):
        response = await facets.openai_client.embeddings.create(
          model=facets.embedding_model,
          input=input
        )

        return [ numpy.asarray(datum.embedding, float32) for datum in response.data ]

    embeddings = list(itertools.chain.from_iterable(await asyncio.gather(*(embed_batch(input) for input in itertools.batched(contents, max_embeds)))))

    return Cluster([ Embed(entry, embedding) for entry, embedding in zip(entries, embeddings) ])

def cluster(input: Cluster) -> list[Cluster]:
    print("[+] Clustering embeddings")

    if len(input.embeds) < 7:
        return []

    entries, embeddings = zip(*((embed.entry, embed.embedding) for embed in input.embeds))

    N = len(embeddings)

    normalized = sklearn.preprocessing.normalize(embeddings)

    # Find the smallest value for `n_neighbors` that produces one connected
    # component under nearest neighbors
    #
    # If we pick a value of `n_neighbors` that is too small and build an
    # affinity matrix from the corresponding nearest_neighbors matrix then
    # spectral clustering is only going to identify clusters found by the
    # nearest neighbors algorithm, which is not what we want. We only want the
    # nearest neighbors algorithm to weakly inform the choice of radius for the
    # radial-basis function.
    def get_nearest_neighbors(n_neighbors):
        nearest_neighbors = sklearn.neighbors.NearestNeighbors(
          n_neighbors=n_neighbors,
          metric="cosine",
          n_jobs=-1
        ).fit(normalized)

        directed_graph = nearest_neighbors.kneighbors_graph(mode="connectivity")

        undirected_graph = directed_graph.maximum(directed_graph.T)

        components, _ = scipy.sparse.csgraph.connected_components(undirected_graph)

        return components, n_neighbors, nearest_neighbors

    candidate_neighbor_counts = itertools.takewhile(
        lambda x: x < N,
        (round(math.exp(n)) for n in itertools.count())
    )

    results = [
        get_nearest_neighbors(n_neighbors)
        for n_neighbors in candidate_neighbor_counts
    ]

    n_neighbors, nearest_neighbors = [
        (n_neighbors, nearest_neighbors)
        for components, n_neighbors, nearest_neighbors in results
        if components == 1
    ][0]

    # Compute an adaptive sigma for our radial basis function based on
    # neighborhood size.  See:
    #
    #     Fischer, I., & Poland, J. (2004). New methods for spectral clustering.
    #     Technical Report No. IDSIA-12-04, Dalle Molle Institute for
    #     Artificial Intelligence, Manno-Lugano, Switzerland.
    distances, indices = nearest_neighbors.kneighbors(normalized)

    sigmas = distances[:, -1]

    rows = numpy.repeat(numpy.arange(N), n_neighbors)
    columns = indices.reshape(-1)
    d = distances.reshape(-1)

    sigma_i = numpy.repeat(sigmas, n_neighbors)
    sigma_j = sigmas[columns]

    denominator = numpy.maximum(sigma_i * sigma_j, 1e-12)

    data = numpy.exp(-(d * d) / denominator).astype(numpy.float32)

    similarities = scipy.sparse.coo_matrix((data, (rows, columns)), shape=(N, N)).tocsr()

    affinity = (similarities + similarities.T) * 0.5
    affinity.setdiag(1.0)
    affinity.eliminate_zeros()

    # This is basically `sklearn.manifold.spectral_embedding`, but exploded
    # out so that we can get access to the eigenvalues, which are normally not
    # exposed by the function.  We'll need those eigenvalues later

    # This is actually the *maximum* number of clusters that the algorithm can
    # return.
    #
    # The algorithm is actually fast enough to return a much larger number of
    # clusters and sometimes you find much more optimal clusterings at much
    # higher cluster counts.  For example, I've seen repositories where the
    # optimal cluster count was 600+.  However, we cap the maximum cluster
    # count at 20 because we don't want to present more than that many choices
    # to the user at any level of the decision tree.  Ideally we present around
    # ≈7 choices but capping at 20 is just being conservative.
    #
    # As a bonus, capping at 20 improves performance, too.
    n_clusters = min(N - 1, 20)

    random_state = sklearn.utils.check_random_state(0)

    laplacian, dd = scipy.sparse.csgraph.laplacian(
      affinity,
      normed=True,
      return_diag=True
    )

    # laplacian = set_diag(laplacian, 1, True)
    laplacian = laplacian.tocoo()
    laplacian.data[laplacian.row == laplacian.col] = 1
    laplacian = laplacian.tocsr()

    laplacian *= -1
    v0 = random_state.uniform(-1, 1, N)
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
      laplacian,
      k=n_clusters,
      sigma=1.0,
      which='LM',
      tol=0.0,
      v0=v0
    )
    full_embedding = eigenvectors.T[n_clusters::-1] * dd
    full_embedding = sklearn.utils.extmath._deterministic_vector_sign_flip(full_embedding)
    full_embedding = full_embedding[1:n_clusters].T
    eigenvalues = eigenvalues[n_clusters::-1]
    eigenvalues *= -1

    # The reason the suggested cluster count is not just:
    #
    #     numpy.argmax(numpy.diff(eigenvalues))
    #
    # … is because we want at least two clusters (otherwise what's the point?).
    n_clusters = numpy.argmax(numpy.diff(eigenvalues)[2:]) + 2

    embedding = full_embedding[:, :n_clusters]

    normalized_embedding = sklearn.preprocessing.normalize(embedding)

    labels = sklearn.cluster.KMeans(
      n_clusters=n_clusters,
      random_state=0,
      n_init="auto"
    ).fit_predict(normalized_embedding)

    groups = collections.OrderedDict()

    for (label, entry, vector) in zip(labels, entries, full_embedding):
        groups.setdefault(label, []).append(Embed(entry, vector))

    return [ Cluster(embeds) for embeds in groups.values() ]

@dataclasses.dataclass(frozen=True)
class Tree:
    label: str
    children: list["Tree"]

A = TypeVar('A')

def zipper(elements: list[A]) -> list[tuple[list[A], A, list[A]]]:
    return [
        (elements[:index], element, elements[index + 1:])
        for index, element in enumerate(elements)
    ]

async def label_leaves(facets: Facets, c: Cluster) -> list[Tree]:
    async def label(element: tuple[list[Embed], Embed, list[Embed]]) -> Tree:
        prefix, embed, suffix = element

        here = embed.entry

        other = "\n".join([ embed.entry for embed in prefix + suffix ])

        input = f"Describe in a few words what distinguishes this file:\n\n{here}\n\n… from these other files:\n\n{other}\n\nYour response in its entirety should be a succinct description (≈3 words) without any explanation/context/rationale because the full text of what you say will be used as the file label without any trimming."

        response = await facets.openai_client.responses.create(
            model = facets.completion_model,
            input = input
        )

        children = [ Tree(here, []) ]
        return Tree(response.output_text, children)

    trees = await asyncio.gather(*(label(tuple) for tuple in zipper(c.embeds)))

    files = Tree(f"[{len(c.embeds)} files]", [ Tree(embed.entry, []) for embed in c.embeds ])

    return [ files ] + trees

async def label_nodes(facets: Facets, c: Cluster) -> list[Tree]:
    if len(c.embeds) <= 7:
        return await label_leaves(facets, c)
    else:
        async def label(element: tuple[list[list[Tree]], list[Tree], list[list[Tree]]]) -> Tree:
            def render_trees(trees: list[Tree]) -> str:
                return "\n".join([ tree.label for tree in trees ])

            prefix, trees, suffix = element

            here = render_trees(trees)

            other = "\n\n".join([ render_trees(trees) for trees in prefix + suffix ])

            input = f"Describe in a few words what distinguish this cluster of labels:\n\n{here}\n\n… from these other clusters of labels:\n\n{other}\n\nYour response in its entirety should be a succinct description (≈3 words) without any explanation/context/rationale because the full text of what you say will be used as the file label without any trimming."

            response = await facets.openai_client.responses.create(
                model = facets.completion_model,
                input = input
            )

            return Tree(response.output_text, trees)

        children = cluster(c)

        treess = await asyncio.gather(*(label_nodes(facets, child) for child in children))

        trees = await asyncio.gather(*(label(tuple) for tuple in zipper(treess)))
        files = Tree(f"[{len(c.embeds)} files]", [ Tree(embed.entry, []) for embed in c.embeds ])

        return [ files ] + trees
    
async def tree(facets: Facets, label: str, c: Cluster) -> Tree:
    trees = await label_nodes(facets, c)

    return Tree(label, trees)

class UI(textual.app.App):
    def __init__(self, tree_):
        super().__init__()
        self.tree_ = tree_

    async def on_mount(self):
        self.treeview = textual.widgets.Tree(self.tree_.label)
        def loop(node, children):
            for child in children:
                n = node.add(child.label)
                n.allow_expand = bool(child.children)
                loop(n, child.children)

        loop(self.treeview.root, self.tree_.children)

        self.mount(self.treeview)

def main():
    parser = argparse.ArgumentParser(
        prog='facets',
        description='Cluster documents by semantic facets',
    )

    parser.add_argument('repository')
    arguments = parser.parse_args()

    facets = initialize()

    async def async_tasks():
        initial_cluster = await embed(facets, arguments.repository)

        tree_ = await tree(facets, arguments.repository, initial_cluster)

        return tree_

    tree_ = asyncio.run(async_tasks())

    UI(tree_).run()

main()
