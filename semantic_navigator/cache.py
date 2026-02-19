import hashlib
import numpy
import os
import sys

from numpy import float32
from numpy.typing import NDArray
from pathlib import Path

from semantic_navigator.models import Label, Labels


def app_cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "semantic-navigator"


def repo_cache_dir(repository: str) -> Path:
    from semantic_navigator.util import _sanitize_path
    return app_cache_dir() / "repos" / _sanitize_path(repository)


def embedding_cache_dir(model_name: str) -> Path:
    return app_cache_dir() / "embeddings" / model_name.replace("/", "--")


def label_cache_dir(repository: str, model_identity: str) -> Path:
    safe_id = hashlib.sha256(model_identity.encode()).hexdigest()[:16]
    return repo_cache_dir(repository) / "labels" / safe_id


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def cluster_hash(files: list[str]) -> str:
    """Stable hash for a cluster based on its sorted file list."""
    return hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest()


def list_cached_keys(directory: Path, suffix: str) -> set[str]:
    """List all cached keys in a directory by stripping the suffix from filenames."""
    if not directory.exists():
        return set()
    return {p.stem for p in directory.iterdir() if p.suffix == suffix}


def load_cached_embedding(directory: Path, key: str) -> NDArray[float32] | None:
    path = directory / f"{key}.npy"
    if path.exists():
        return numpy.load(path).astype(float32)
    return None


def save_cached_embedding(directory: Path, key: str, embedding: NDArray[float32]) -> None:
    directory.mkdir(parents = True, exist_ok = True)
    numpy.save(directory / f"{key}.npy", embedding)


def load_cached_label(directory: Path, key: str) -> Label | None:
    path = directory / f"{key}.json"
    if path.exists():
        return Label.model_validate_json(path.read_text())
    return None


def save_cached_label(directory: Path, key: str, label: Label) -> None:
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / f"{key}.json"
    path.write_text(label.model_dump_json())


def load_cached_cluster_labels(directory: Path, key: str) -> Labels | None:
    path = directory / f"cluster-{key}.json"
    if path.exists():
        try:
            return Labels.model_validate_json(path.read_text())
        except Exception as e:
            print(f"Warning: corrupt cluster cache {path}: {e}", file=sys.stderr)
            return None
    return None


def save_cached_cluster_labels(directory: Path, key: str, labels: Labels) -> None:
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / f"cluster-{key}.json"
    path.write_text(labels.model_dump_json())
