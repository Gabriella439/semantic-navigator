from dataclasses import dataclass
from numpy import float32
from numpy.typing import NDArray
from pydantic import BaseModel
from openai import AsyncOpenAI
from tiktoken import Encoding


@dataclass(frozen = True)
class Facets:
    openai_client: AsyncOpenAI
    embedding_model: str
    completion_model: str
    embedding_encoding: Encoding
    completion_encoding: Encoding


@dataclass(frozen = True)
class Embed:
    entry: str
    content: str
    embedding: NDArray[float32]


@dataclass(frozen = True)
class Cluster:
    embeds: list[Embed]


@dataclass(frozen = True)
class Tree:
    label: str
    files: list[str]
    children: list["Tree"]


class Label(BaseModel):
    overarchingTheme: str
    distinguishingFeature: str
    label: str


class Labels(BaseModel):
    labels: list[Label]
