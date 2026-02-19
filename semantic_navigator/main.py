import aiofiles
import argparse
import asyncio
import collections
import hashlib
import itertools
import json
import math
import numpy
import os
import re
import scipy
import shlex
import shutil
import sklearn
import subprocess
import sys
import textual
import textual.app
import textual.widgets

from dataclasses import dataclass
from dulwich.errors import NotGitRepository
from dulwich.repo import Repo
from fastembed import TextEmbedding
from itertools import chain
from numpy import float32
from numpy.typing import NDArray
from pathlib import Path, PurePath
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from typing import Iterable, TypeVar

T = TypeVar("T", bound=BaseModel)

max_clusters = 20

@dataclass(frozen = True)
class Aspect:
    """A single inference backend (CLI tool or local model)."""
    name: str
    cli_command: list[str] | None
    local_model: object | None
    local_n_ctx: int | None

class AspectPool:
    """Async pool that distributes work across multiple inference backends."""
    def __init__(self, aspects: list[Aspect]):
        self.aspects = aspects
        self._queue: asyncio.Queue[Aspect] | None = None

    def _ensure_queue(self) -> asyncio.Queue[Aspect]:
        if self._queue is None:
            self._queue = asyncio.Queue()
            for a in self.aspects:
                self._queue.put_nowait(a)
        return self._queue

    async def acquire(self) -> Aspect:
        return await self._ensure_queue().get()

    def release(self, aspect: Aspect) -> None:
        self._ensure_queue().put_nowait(aspect)

    @property
    def min_local_n_ctx(self) -> int | None:
        ctxs = [a.local_n_ctx for a in self.aspects if a.local_n_ctx is not None]
        return min(ctxs) if ctxs else None

@dataclass(frozen = True)
class Facets:
    pool: AspectPool
    embedding_model: TextEmbedding
    embedding_model_name: str
    gpu: bool
    batch_size: int
    timeout: int
    debug: bool

def list_devices():
    result = subprocess.run(
        ["powershell", "-Command",
         "Get-CimInstance Win32_VideoController | Select-Object -Property Name"],
        capture_output = True, text = True
    )
    names = [
        line.strip() for line in result.stdout.strip().splitlines()
        if line.strip() and line.strip() != "Name" and not line.strip().startswith("----")
    ]
    print("GPU devices:")
    for i, name in enumerate(names):
        vram = detect_device_memory(True, i)
        if vram is not None:
            print(f"  {i}: {name} ({vram / 1e9:.1f} GB)")
        else:
            print(f"  {i}: {name} (unknown VRAM)")

def case_insensitive_glob(pattern: str) -> str:
    return ''.join(
        f'[{c.upper()}{c.lower()}]' if c.isalpha() else c
        for c in pattern
    )

gguf_quant_preference = [
    "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q6_k", "q3_k_l",
    "q3_k_m", "q8_0", "q4_0", "q5_0", "q3_k_s", "q2_k",
    "fp16", "f16",
]

def detect_device_memory(gpu: bool, device: int) -> int | None:
    """Detect memory in bytes for the target device. Returns None if unknown."""
    try:
        if sys.platform == "win32":
            if gpu:
                reg_path = (
                    "HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\"
                    f"{{4d36e968-e325-11ce-bfc1-08002be10318}}\\{device:04d}"
                )
                result = subprocess.run(
                    ["powershell", "-Command",
                     f"(Get-ItemProperty '{reg_path}'"
                     " -Name 'HardwareInformation.qwMemorySize'"
                     " -ErrorAction SilentlyContinue)."
                     "'HardwareInformation.qwMemorySize'"],
                    capture_output=True, text=True, timeout=10,
                )
                value = result.stdout.strip()
                if value:
                    return int(value)
            else:
                result = subprocess.run(
                    ["powershell", "-Command",
                     "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                    capture_output=True, text=True, timeout=10,
                )
                return int(result.stdout.strip())
        else:
            if not gpu:
                return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, subprocess.TimeoutExpired, AttributeError):
        pass
    return None

def select_best_gguf(repo_id: str, memory_budget: int | None) -> tuple[list[str], int]:
    """Auto-select the best GGUF for the device. Returns (filenames, total_bytes)."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    gguf_files = [
        (s.rfilename, s.size or 0)
        for s in info.siblings
        if s.rfilename.endswith(".gguf")
    ]

    if not gguf_files:
        raise ValueError(f"No .gguf files found in {repo_id}")

    # Group split files by base name
    groups: dict[str, list[tuple[str, int]]] = {}
    for filename, size in gguf_files:
        base = re.sub(r'-\d{5}-of-\d{5}\.gguf$', '.gguf', filename)
        groups.setdefault(base, []).append((filename, size))

    candidates = [
        (base, sum(s for _, s in files), len(files) == 1, files)
        for base, files in groups.items()
    ]

    if memory_budget is not None:
        budget = int(memory_budget * 0.8)
        fitting = [c for c in candidates if c[1] <= budget]
        if fitting:
            # Largest model that fits = best quality; prefer single files as tiebreaker
            fitting.sort(key=lambda c: (-c[1], not c[2]))
            chosen = fitting[0]
        else:
            # Nothing fits — pick smallest
            candidates.sort(key=lambda c: (c[1], not c[2]))
            chosen = candidates[0]
    else:
        # Memory unknown — fall back to quant preference
        def quant_rank(base: str) -> int:
            lower = base.lower()
            for i, quant in enumerate(gguf_quant_preference):
                if quant in lower:
                    return i
            return len(gguf_quant_preference)

        single = [c for c in candidates if c[2]]
        pool = single if single else candidates
        chosen = min(pool, key=lambda c: quant_rank(c[0]))

    base, total_size, is_single, files = chosen

    sorted_files = sorted(f[0] for f in files)
    return sorted_files, total_size

def _resolve_gpu_layers(gpu: bool, gpu_layers: int | None, device: int, model_size: int | None = None) -> int:
    """Compute n_gpu_layers for llama.cpp."""
    if not gpu:
        return 0
    if gpu_layers is not None:
        return gpu_layers
    if model_size:
        vram = detect_device_memory(True, device)
        if vram:
            return int(100 * vram * 0.6 / model_size)
    return -1

def _estimate_model_size(local: str, local_file: str | None) -> int | None:
    """Estimate model file size in bytes without loading. Returns None if unknown."""
    is_local_file = os.path.exists(local) or "\\" in local or local.count("/") > 1
    if is_local_file:
        return os.path.getsize(local)
    if local_file is None:
        try:
            _, model_size = select_best_gguf(local, None)
            return model_size
        except Exception:
            return None
    return None

def _create_local_model(local: str, local_file: str | None, gpu: bool, device: int, gpu_layers: int | None, n_ctx: int, debug: bool) -> object:
    from llama_cpp import Llama

    is_local_file = os.path.exists(local) or "\\" in local or local.count("/") > 1

    if is_local_file:
        model_size = os.path.getsize(local)
        n_gpu_layers = _resolve_gpu_layers(gpu, gpu_layers, device, model_size)
        print(f"Local model: {local} ({'GPU ' + str(device) if gpu else 'CPU'})")
        return Llama(
            model_path = local,
            n_gpu_layers = n_gpu_layers,
            n_ctx = n_ctx,
            verbose = debug,
        )
    else:
        if local_file is None:
            memory_budget = detect_device_memory(gpu, device)
            local_files, model_size = select_best_gguf(local, memory_budget)
            n_gpu_layers = _resolve_gpu_layers(gpu, gpu_layers, device, model_size)
            print(f"Local model: {local} (auto-selected: {local_files[0]}, {model_size / 1e9:.1f} GB, {'GPU ' + str(device) if gpu else 'CPU'})")
        else:
            local_files = [case_insensitive_glob(local_file)]
            n_gpu_layers = _resolve_gpu_layers(gpu, gpu_layers, device)
            print(f"Local model: {local} (file: {local_files[0]}, {'GPU ' + str(device) if gpu else 'CPU'})")
        return Llama.from_pretrained(
            repo_id = local,
            filename = local_files[0],
            additional_files = local_files[1:] or None,
            n_gpu_layers = n_gpu_layers,
            n_ctx = n_ctx,
            verbose = debug,
        )

def initialize(cli_command: list[str] | None, local: str | None, local_file: str | None, embedding_model: str, gpu: bool, cpu: bool, cpu_offload: bool, devices: list[int], gpu_layers: int | None, batch_size: int | None, concurrency: int, n_ctx: int, timeout: int, debug: bool) -> Facets:
    aspects: list[Aspect] = []

    if local is not None:
        if gpu:
            model_size = _estimate_model_size(local, local_file)
            for device in devices:
                vram = detect_device_memory(True, device)
                if model_size is not None and vram is not None and vram < model_size:
                    print(f"Skipping GPU {device}: insufficient VRAM ({vram / 1e9:.1f} GB) for model ({model_size / 1e9:.1f} GB)")
                    continue
                model = _create_local_model(local, local_file, True, device, gpu_layers, n_ctx, debug)
                aspects.append(Aspect(
                    name = f"local/gpu:{device}",
                    cli_command = None,
                    local_model = model,
                    local_n_ctx = model.n_ctx(),
                ))
            if cpu:
                model = _create_local_model(local, local_file, False, 0, 0, n_ctx, debug)
                aspects.append(Aspect(
                    name = "local/cpu",
                    cli_command = None,
                    local_model = model,
                    local_n_ctx = model.n_ctx(),
                ))
        else:
            model = _create_local_model(local, local_file, False, 0, 0, n_ctx, debug)
            aspects.append(Aspect(
                name = "local/cpu",
                cli_command = None,
                local_model = model,
                local_n_ctx = model.n_ctx(),
            ))

    if cli_command is not None:
        print(f"CLI tool: {shlex.join(cli_command)} (concurrency {concurrency})")
        for i in range(concurrency):
            aspects.append(Aspect(
                name = f"cli/{i}",
                cli_command = cli_command,
                local_model = None,
                local_n_ctx = None,
            ))

    if gpu:
        device = devices[0]
        providers = [("DmlExecutionProvider", {"device_id": device})]
        if cpu_offload:
            providers.append("CPUExecutionProvider")
    else:
        providers = ["CPUExecutionProvider"]
    if batch_size is None:
        batch_size = 32 if gpu else 256
    print(f"Loading embedding model ({embedding_model}, batch size {batch_size})...")
    embedding = TextEmbedding(model_name = embedding_model, providers = providers)
    print(f"Initialized {len(aspects)} aspect{'s' if len(aspects) != 1 else ''}: {', '.join(a.name for a in aspects)}")
    return Facets(
        pool = AspectPool(aspects),
        embedding_model = embedding,
        embedding_model_name = embedding_model,
        gpu = gpu,
        batch_size = batch_size,
        timeout = timeout,
        debug = debug,
    )

def extract_json(text: str) -> str:
    """Extract JSON from text that may contain code fences or surrounding text."""
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text.strip()

def repair_json(text: str) -> str:
    """Fix invalid escape sequences in JSON strings."""
    # Fix \X where X is not a valid JSON escape character
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    # Fix \uXXXX where XXXX aren't all hex digits (e.g. \utils)
    text = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', text)
    return text

def _build_labels_schema(count: int | None) -> dict:
    """Build a JSON schema for Labels with optional exact item count."""
    label_schema = {
        "type": "object",
        "properties": {
            "overarchingTheme": {"type": "string"},
            "distinguishingFeature": {"type": "string"},
            "label": {"type": "string"},
        },
        "required": ["overarchingTheme", "distinguishingFeature", "label"],
    }
    array_schema: dict = {"type": "array", "items": label_schema}
    if count is not None:
        array_schema["minItems"] = count
        array_schema["maxItems"] = count
    return {
        "type": "object",
        "properties": {"labels": array_schema},
        "required": ["labels"],
    }

def _local_complete(model: object, prompt: str, response_format: dict | None = None) -> str:
    kwargs: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    result = model.create_chat_completion(**kwargs)
    return result["choices"][0]["message"]["content"]

max_retries = 60
max_count_retries = 5

async def complete(facets: Facets, prompt: str, output_type: type[T], progress: tqdm | None = None, expected_count: int | None = None) -> T:
    """Run CLI tool or local model with prompt, parse JSON response into Pydantic model."""
    for attempt in range(max_retries):
        if facets.debug:
            print(f"\n[debug] prompt ({len(prompt)} chars):\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}")
        aspect = await facets.pool.acquire()
        try:
            if aspect.local_model is not None:
                response_format = None
                if expected_count is not None:
                    response_format = {
                        "type": "json_object",
                        "schema": _build_labels_schema(expected_count),
                    }
                loop = asyncio.get_event_loop()
                raw = await loop.run_in_executor(None, _local_complete, aspect.local_model, prompt, response_format)
            else:
                if facets.debug:
                    cmd = shlex.join(aspect.cli_command)
                    print(f"[debug] command: {cmd}")
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: subprocess.run(
                    aspect.cli_command,
                    input=prompt.encode(),
                    capture_output=True,
                    timeout=facets.timeout,
                ))
                raw = result.stdout.decode()
                if facets.debug:
                    print(f"[debug] exit code: {result.returncode}")
                    print(f"[debug] stdout ({len(raw)} chars):\n{raw[:500]}{'...' if len(raw) > 500 else ''}")
                    if result.stderr:
                        print(f"[debug] stderr: {result.stderr.decode()[:500]}")
                if result.returncode != 0:
                    raise RuntimeError(
                        f"CLI command failed (exit {result.returncode}): {result.stderr.decode()}"
                    )
        finally:
            facets.pool.release(aspect)
        if facets.debug:
            print(f"[debug] raw output ({len(raw)} chars):\n{raw[:500]}{'...' if len(raw) > 500 else ''}")
        extracted = extract_json(raw)
        if facets.debug:
            print(f"[debug] extracted JSON: {extracted[:500]}{'...' if len(extracted) > 500 else ''}")
        try:
            result = output_type.model_validate_json(extracted)
        except Exception:
            repaired = repair_json(extracted)
            if facets.debug:
                print(f"[debug] repaired JSON: {repaired[:500]}{'...' if len(repaired) > 500 else ''}")
            try:
                result = output_type.model_validate_json(repaired)
            except Exception as e:
                if attempt < max_retries - 1:
                    tqdm.write(f"Retrying ({attempt + 1}/{max_retries}): {e}")
                    continue
                raise
        if expected_count is not None and hasattr(result, 'labels') and len(result.labels) != expected_count:
            if len(result.labels) > expected_count:
                tqdm.write(f"Truncating {len(result.labels)} labels to {expected_count}")
                result.labels = result.labels[:expected_count]
            elif attempt < max_count_retries - 1:
                tqdm.write(f"Retrying ({attempt + 1}/{max_count_retries}): expected {expected_count} labels, got {len(result.labels)}")
                continue
            else:
                # Pad with generic labels rather than retrying forever
                tqdm.write(f"Padding {len(result.labels)} labels to {expected_count} (gave up after {max_count_retries} attempts)")
                while len(result.labels) < expected_count:
                    result.labels.append(Label(
                        overarchingTheme = "Miscellaneous",
                        distinguishingFeature = "Ungrouped",
                        label = "Miscellaneous",
                    ))
        if progress is not None:
            progress.update(1)
        return result

class Label(BaseModel):
    overarchingTheme: str
    distinguishingFeature: str
    label: str

class Labels(BaseModel):
    labels: list[Label]

def app_cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "semantic-navigator"

def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

def embedding_cache_dir(model_name: str) -> Path:
    return app_cache_dir() / "embeddings" / model_name.replace("/", "--")

def load_cached_embedding(directory: Path, key: str) -> NDArray[float32] | None:
    path = directory / f"{key}.npy"
    if path.exists():
        return numpy.load(path).astype(float32)
    return None

def save_cached_embedding(directory: Path, key: str, embedding: NDArray[float32]) -> None:
    directory.mkdir(parents = True, exist_ok = True)
    numpy.save(directory / f"{key}.npy", embedding)

def label_cache_dir() -> Path:
    return app_cache_dir() / "labels"

def load_cached_label(directory: Path, key: str) -> Label | None:
    path = directory / f"{key}.json"
    if path.exists():
        return Label.model_validate_json(path.read_text())
    return None

def save_cached_label(directory: Path, key: str, label: Label) -> None:
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / f"{key}.json"
    path.write_text(label.model_dump_json())

@dataclass(frozen = True)
class Embed:
    entry: str
    content: str
    embedding: NDArray[float32]

@dataclass(frozen = True)
class Cluster:
    embeds: list[Embed]

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

    async def read(path) -> tuple[str, str] | None:
        try:
            absolute_path = os.path.join(directory, path)

            async with aiofiles.open(absolute_path, "rb") as handle:
                bytestring = await handle.read()

                text = bytestring.decode("utf-8")

                return (path, f"{path}:\n\n{text}")

        except UnicodeDecodeError:
            # Ignore files that aren't UTF-8
            return None

        except IsADirectoryError:
            # This can happen when a "file" listed by the repository is:
            #
            # - a submodule
            # - a symlink to a directory
            #
            # TODO: The submodule case can and should be fixed and properly
            # handled
            return None

    tasks = tqdm_asyncio.gather(
        *(read(path) for path in generate_paths()),
        desc = "Reading files",
        unit = "file",
        leave = False
    )

    results = [ result for result in await tasks if result is not None ]

    if not results:
        return Cluster([])

    paths, contents = zip(*results)

    cdir = embedding_cache_dir(facets.embedding_model_name)
    embeddings: list[NDArray[float32]] = [None] * len(contents)  # type: ignore[list-item]
    uncached_indices = []

    for i, content in enumerate(contents):
        cached = load_cached_embedding(cdir, content_hash(content))
        if cached is not None:
            embeddings[i] = cached
        else:
            uncached_indices.append(i)

    n_cached = len(contents) - len(uncached_indices)
    if n_cached > 0:
        print(f"Loaded {n_cached}/{len(contents)} embeddings from cache")

    if uncached_indices:
        uncached_contents = [contents[i] for i in uncached_indices]

        def do_embed(model: TextEmbedding, desc: str) -> list[NDArray[float32]]:
            return [
                numpy.asarray(e, float32)
                for e in tqdm(
                    model.embed(uncached_contents, batch_size = facets.batch_size),
                    desc = desc,
                    unit = "file",
                    total = len(uncached_contents),
                    leave = False
                )
            ]

        try:
            new_embeddings = do_embed(facets.embedding_model, f"Embedding contents ({len(uncached_indices)} uncached)")
        except Exception as e:
            if not facets.gpu:
                raise
            print(f"\nGPU embedding failed ({e}), falling back to CPU...")
            cpu_model = TextEmbedding(model_name = facets.embedding_model_name, providers = ["CPUExecutionProvider"])
            new_embeddings = do_embed(cpu_model, "Embedding contents (CPU)")

        for i, embedding in zip(uncached_indices, new_embeddings):
            embeddings[i] = embedding
            save_cached_embedding(cdir, content_hash(contents[i]), embedding)

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
class ClusterTree:
    node: Cluster
    children: list["ClusterTree"]

def build_cluster_tree(c: Cluster) -> ClusterTree:
    children = cluster(c)
    if len(children) == 1:
        return ClusterTree(c, [])
    else:
        return ClusterTree(c, [build_cluster_tree(child) for child in children])

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

def to_files(trees: list[Tree]) -> list[str]:
    return [ file for tree in trees for file in tree.files ]

def count_llm_calls(ct: ClusterTree) -> tuple[int, int]:
    """Count how many LLM calls will be needed.
    Returns (total_calls, cached_file_labels)."""
    if not ct.children:
        ldir = label_cache_dir()
        uncached = sum(
            1 for embed in ct.node.embeds
            if load_cached_label(ldir, content_hash(embed.content)) is None
        )
        cached = len(ct.node.embeds) - uncached
        return (1 if uncached > 0 else 0, cached)
    else:
        total = 0
        total_cached = 0
        for child in ct.children:
            calls, cached = count_llm_calls(child)
            total += calls
            total_cached += cached
        return (total + 1, total_cached)  # +1 for cluster summarization

async def label_nodes(facets: Facets, ct: ClusterTree, progress: tqdm) -> list[Tree]:
    if not ct.children:
        ldir = label_cache_dir()
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
                max_chars = int((facets.pool.min_local_n_ctx - 4096) * 1.5) - 1000

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

            schema = json.dumps(Labels.model_json_schema(), indent=2)

            for batch in batches:
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
                    save_cached_label(ldir, content_hash(embed.content), label)

        return [
            Tree(
                f"{embed.entry}: {cached_labels[i].label}" if i in cached_labels else embed.entry,
                [ embed.entry ],
                [],
            )
            for i, embed in enumerate(ct.node.embeds)
        ]

    else:
        treess = await asyncio.gather(
            *(label_nodes(facets, child, progress) for child in ct.children),
        )

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

        return [
            Tree(f"{to_pattern(to_files(trees))}{label.label}", to_files(trees), trees)
            for label, trees in zip(labels.labels, treess)
        ]

async def tree(facets: Facets, label: str, c: Cluster) -> Tree:
    ct = build_cluster_tree(c)
    total_calls, cached_labels = count_llm_calls(ct)
    if cached_labels > 0:
        print(f"Loaded {cached_labels} file labels from cache", flush=True)
    if total_calls > 0:
        print(f"Labeling with {total_calls} API call{'s' if total_calls != 1 else ''}...", flush=True)
    with tqdm(total = total_calls, desc = "Labeling", unit = "call", leave = False) as progress:
        children = await label_nodes(facets, ct, progress)

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
        prog = "facets",
        description = "Cluster documents by semantic facets",
    )

    parser.add_argument("repository", nargs = "?")
    parser.add_argument("--embedding-model", default = "BAAI/bge-large-en-v1.5")
    parser.add_argument("--gpu", action = "store_true")
    parser.add_argument("--cpu", action = "store_true", help = "Add a CPU local model worker alongside GPU workers")
    parser.add_argument("--cpu-offload", action = "store_true")
    parser.add_argument("--device", type = str, default = "0", help = "GPU device IDs, comma-separated (e.g. 0,1)")
    parser.add_argument("--gpu-layers", type = int, default = None)
    parser.add_argument("--batch-size", type = int, default = None)
    parser.add_argument("--concurrency", type = int, default = None)
    parser.add_argument("--timeout", type = int, default = 60)
    parser.add_argument("--list-devices", action = "store_true")
    parser.add_argument("--local", default = None)
    parser.add_argument("--local-file", default = None)
    parser.add_argument("--n-ctx", type = int, default = None)
    parser.add_argument("--debug", action = "store_true")
    arguments, remaining = parser.parse_known_args()

    if arguments.list_devices:
        list_devices()
        return

    if arguments.repository is None:
        parser.error("the following arguments are required: repository")

    # Extract CLI tool from unknown flags: --gemini → ["gemini"], --llm -m gpt-4o → ["llm", "-m", "gpt-4o"]
    has_cli_tool = remaining and remaining[0].startswith("--")

    if arguments.local is None and not has_cli_tool:
        parser.error("no CLI tool specified (e.g. --gemini, --llm) and no --local model provided")

    if arguments.cpu_offload and not arguments.gpu:
        parser.error("--cpu-offload requires --gpu")

    if arguments.cpu and not arguments.gpu:
        parser.error("--cpu only makes sense with --gpu (without --gpu, CPU is already the default)")

    if arguments.gpu_layers is not None and arguments.local is None:
        parser.error("--gpu-layers requires --local")

    if arguments.local_file is not None and arguments.local is None:
        parser.error("--local-file requires --local")

    if arguments.n_ctx is not None and arguments.local is None:
        parser.error("--n-ctx requires --local")

    if arguments.concurrency is not None and arguments.local is not None and not has_cli_tool:
        parser.error("--concurrency has no effect with --local only (local model concurrency is 1 per device)")

    cli_command = None
    if has_cli_tool:
        tool_name = remaining[0][2:]
        if shutil.which(tool_name) is None:
            parser.error(f"CLI tool '{tool_name}' not found on PATH")
        cli_command = [tool_name] + remaining[1:]

    try:
        devices = [int(d.strip()) for d in arguments.device.split(",")]
    except ValueError:
        parser.error(f"--device must be comma-separated integers, got: {arguments.device}")

    if arguments.n_ctx is None:
        arguments.n_ctx = 32768
    if arguments.concurrency is None:
        arguments.concurrency = 4

    facets = initialize(cli_command, arguments.local, arguments.local_file, arguments.embedding_model, arguments.gpu, arguments.cpu, arguments.cpu_offload, devices, arguments.gpu_layers, arguments.batch_size, arguments.concurrency, arguments.n_ctx, arguments.timeout, arguments.debug)

    async def async_tasks():
        initial_cluster = await embed(facets, arguments.repository)

        print(f"Clustering {len(initial_cluster.embeds)} files...")
        tree_ = await tree(facets, arguments.repository, initial_cluster)

        print("Done!")
        return tree_

    tree_ = asyncio.run(async_tasks())

    UI(tree_).run()

if __name__ == "__main__":
    main()
