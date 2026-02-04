import aiofiles
import argparse
import asyncio
import collections
import git
import itertools
import math
import numpy
import openai
import os
import pathlib
import scipy
import sklearn
import tiktoken

max_tokens_per_embed = 8192

max_tokens_per_batch_embed = 300000

async def main():
    parser = argparse.ArgumentParser(
      prog='facets',
      description='Cluster documents by semantic facets',
    )

    parser.add_argument('repository')

    arguments = parser.parse_args()

    repository = git.Repo(arguments.repository)

    embedding_model = "text-embedding-3-large"

    completion_model = "gpt-5.2"

    openai_client = openai.AsyncOpenAI()

    embedding_encoding = tiktoken.encoding_for_model(embedding_model)

    completion_encoding = tiktoken.get_encoding("o200k_base")

    # This value was chosen based on hand-testing of what gave the best balance
    # of performance and reliability
    semaphore = asyncio.Semaphore(148)

    print("[+] Reading files")

    async def read(entry):
        absolute_path = os.path.join(arguments.repository, entry)

        try:
            async with aiofiles.open(absolute_path, "rb") as handle:
                bytestring = await handle.read()

                text = bytestring.decode("utf-8")

                annotated = f"{entry}:\n\n{text}"

                tokens = embedding_encoding.encode(annotated)

                # TODO: chunk instead of truncate
                truncated = tokens[:max_tokens_per_embed]

                return [ (entry, embedding_encoding.decode(truncated)) ]

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

    results = list(itertools.chain.from_iterable(await asyncio.gather(*(read(entry) for entry, _ in repository.index.entries))))

    entries, contents = zip(*results)

    print("[+] Embedding file contents")

    max_embeds = math.floor(max_tokens_per_batch_embed / max_tokens_per_embed)

    async def embed(input):
        response = await openai_client.embeddings.create(
          model=embedding_model,
          input=input
        )

        return [ datum.embedding for datum in response.data ]

    embeddings = list(itertools.chain.from_iterable(await asyncio.gather(*(embed(input) for input in itertools.batched(contents, max_embeds)))))

    print("[+] Clustering embeddings")

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
        groups.setdefault(label, []).append((entry, vector))

    print("[+] Labeling clusters")
    def render_group(values):
        def key(value):
            _, vector = value

            return scipy.linalg.norm(vector)

        entries = [ entry for entry, vector in sorted(values, reverse=True, key=key) ]

        desired_entries = 403

        step = math.ceil(len(entries) / desired_entries)

        entries = entries[::step]

        extra_files = len(values) - len(entries)
        if extra_files > 0:
            entries.append(f"… [{extra_files} more files]")

        return "\n".join(sorted(entries))

    async def label(item):
        my_label, my_values = item

        other_values = [
            value
            for other_label, other_values in groups.items()
            if my_label != other_label
            for value in other_values
        ]

        prompt = f"Vector embedding plus clustering produced the following cluster of files:\n\n{render_group(my_values)}\n\nDescribe in a few words what distinguishes that cluster of files from these other files in the project that don't belong to that cluster:\n\n{render_group(other_values)}\n\nYour response in its entirety should be a succinct description (≈3 words) without any explanation/context/rationale because the full text of what you say will be used as the user-facing cluster label without any trimming"

        tokens = completion_encoding.encode(prompt)

        truncated = tokens[:128000]

        input = completion_encoding.decode(truncated)

        summary = await openai_client.responses.create(model = completion_model, input = input)

        return summary.output_text, render_group(my_values)

    results = await asyncio.gather(*(label(item) for item in groups.items()))

    print("")

    for summary, entries in results:
        print(f"# {summary}")

        print("")

        print(entries)

        print("")

asyncio.run(main())
