import asyncio

from dataclasses import dataclass
from fastembed import TextEmbedding
from numpy import float32
from numpy.typing import NDArray
from pydantic import BaseModel


class Label(BaseModel):
    overarchingTheme: str
    distinguishingFeature: str
    label: str


class Labels(BaseModel):
    labels: list[Label]


@dataclass(frozen = True)
class Embed:
    entry: str
    content: str
    embedding: NDArray[float32]


@dataclass(frozen = True)
class Cluster:
    embeds: list[Embed]


@dataclass(frozen = True)
class ClusterTree:
    node: Cluster
    children: list["ClusterTree"]


@dataclass(frozen = True)
class Tree:
    label: str
    files: list[str]
    children: list["Tree"]


@dataclass(frozen = True)
class Aspect:
    """A single inference backend (CLI tool, local model, or OpenAI)."""
    name: str
    cli_command: list[str] | None
    local_model: object | None
    local_n_ctx: int | None
    openai_client: object | None
    openai_model: str | None


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
    repository: str
    model_identity: str
    pool: AspectPool
    embedding_model: TextEmbedding | None
    embedding_model_name: str
    gpu: bool
    batch_size: int
    timeout: int
    debug: bool
    openai_client: object | None
    openai_embedding_model: str | None
