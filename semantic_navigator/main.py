import argparse
import asyncio
import shlex
import shutil
import time

import textual
import textual.app
import textual.widgets

# Re-export all public names for backwards compatibility
from semantic_navigator.models import (  # noqa: F401
    Aspect, AspectPool, Cluster, ClusterTree, Embed, Facets, Label, Labels, Tree,
)
from semantic_navigator.util import (  # noqa: F401
    _fmt_time, _sanitize_path, case_insensitive_glob, extract_json, repair_json, timed,
)
from semantic_navigator.cache import (  # noqa: F401
    app_cache_dir, cluster_hash, content_hash, embedding_cache_dir, label_cache_dir,
    list_cached_keys, load_cached_cluster_labels, load_cached_embedding, load_cached_label,
    repo_cache_dir, save_cached_cluster_labels, save_cached_embedding, save_cached_label,
)
from semantic_navigator.gpu import (  # noqa: F401
    _detect_gpu_memory_linux, _detect_gpu_memory_windows, _list_devices_linux,
    _resolve_gpu_layers, detect_device_memory, gguf_quant_preference, list_devices,
    select_best_gguf,
)
from semantic_navigator.inference import (  # noqa: F401
    _build_labels_schema, _create_cli_aspects, _create_embedding_model,
    _create_local_aspects, _create_local_model, _create_openai_aspects,
    _invoke_cli, _invoke_local, _invoke_openai, _local_complete,
    _parse_raw_response, _validate_label_count, complete, initialize,
    max_count_retries, max_retries,
)
from semantic_navigator.pipeline import (  # noqa: F401
    _count_tree_depth, _count_tree_leaves, _embed_with_fastembed, _embed_with_openai,
    _generate_paths, _label_cluster_node, _label_leaf_node, _read_file,
    build_cluster_tree, cluster, count_cached_labels, embed, label_nodes,
    max_clusters, max_leaves, to_files, to_pattern, tree,
)


class UI(textual.app.App):
    BINDINGS = [("slash", "focus_search", "Search"), ("escape", "clear_search", "Clear")]

    def __init__(self, tree_):
        super().__init__()
        self.tree_ = tree_

    async def on_mount(self):
        self.search_input = textual.widgets.Input(placeholder="Search (press / to focus)...")
        self.search_input.display = False
        self.treeview = textual.widgets.Tree(f"{self.tree_.label} ({len(self.tree_.files)})")
        self._build_tree()
        await self.mount(self.search_input)
        await self.mount(self.treeview)

    def _build_tree(self, filter_text: str = ""):
        self.treeview.clear()
        self.treeview.root.set_label(f"{self.tree_.label} ({len(self.tree_.files)})")

        def matches(child: Tree, text: str) -> bool:
            if text in child.label.lower():
                return True
            return any(text in f.lower() for f in child.files)

        def loop(node, children):
            for child in children:
                if filter_text and not matches(child, filter_text):
                    continue
                if len(child.files) <= 1:
                    n = node.add(child.label)
                    n.allow_expand = False
                else:
                    n = node.add(f"{child.label} ({len(child.files)})")
                    n.allow_expand = True
                    loop(n, child.children)

        loop(self.treeview.root, self.tree_.children)
        if filter_text:
            self.treeview.root.expand_all()

    def action_focus_search(self):
        self.search_input.display = True
        self.search_input.focus()

    def action_clear_search(self):
        self.search_input.value = ""
        self.search_input.display = False
        self._build_tree()
        self.treeview.focus()

    def on_input_changed(self, event: textual.widgets.Input.Changed):
        self._build_tree(event.value.strip().lower())


def _handle_erase_models():
    """Handle the --erase-models command."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        if not cache_info.repos:
            print("No downloaded models found.")
            return
        total_size = sum(r.size_on_disk for r in cache_info.repos)
        print(f"Downloaded models ({total_size / 1e9:.1f} GB):")
        for repo in sorted(cache_info.repos, key=lambda r: r.size_on_disk, reverse=True):
            print(f"  {repo.repo_id} ({repo.size_on_disk / 1e9:.1f} GB)")
        confirm = input("\nDelete all downloaded models? [y/N] ").strip().lower()
        if confirm == "y":
            delete_strategy = cache_info.delete_revisions(
                [r.commit_hash for repo in cache_info.repos for r in repo.revisions]
            )
            delete_strategy.execute()
            print("Done.")
        else:
            print("Aborted.")
    except ImportError:
        print("huggingface_hub is not installed.")


def _flush_cache(repository: str):
    """Delete cached labels for the given repository (preserves embeddings)."""
    repo_dir = repo_cache_dir(repository)
    if repo_dir.exists():
        size = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file())
        shutil.rmtree(repo_dir)
        print(f"Flushed repo cache for {repository} ({size / 1e6:.1f} MB)")
    else:
        print(f"No cache found.")


def _show_status(repository: str):
    """Show cache status for a repository."""
    import os
    from pathlib import Path

    file_paths = _generate_paths(repository)

    # Read files and compute content hashes
    file_hashes: list[tuple[str, str]] = []  # (path, hash)
    for path in file_paths:
        try:
            absolute_path = os.path.join(repository, path)
            with open(absolute_path, "rb") as f:
                text = f.read().decode("utf-8")
            file_hashes.append((path, content_hash(f"{path}:\n\n{text}")))
        except (UnicodeDecodeError, IsADirectoryError):
            pass

    total = len(file_hashes)

    print(f"Repository: {repository} ({total} files)")
    print()

    # Label caches
    repo_dir = repo_cache_dir(repository)
    labels_dir = repo_dir / "labels"
    if not labels_dir.exists():
        print("Labels: no label caches found")
        return

    label_dirs = [d for d in labels_dir.iterdir() if d.is_dir()]
    if not label_dirs:
        print("Labels: no label caches found")
        return

    print(f"Labels ({len(label_dirs)} model {'identity' if len(label_dirs) == 1 else 'identities'}):")
    for ldir in sorted(label_dirs):
        cached_keys = list_cached_keys(ldir, ".json")
        lbl_cached = sum(1 for _, h in file_hashes if h in cached_keys)
        lbl_missing = total - lbl_cached
        status = f"  [{ldir.name}]: {lbl_cached}/{total} cached"
        if lbl_missing > 0:
            status += f" ({lbl_missing} missing)"
        print(status)
        if lbl_missing > 0:
            print("  Missing:")
            for path, h in file_hashes:
                if h not in cached_keys:
                    print(f"    {path}")


def _flush_labels(repository: str):
    """Delete cached labels only (keeps embeddings)."""
    repo_dir = repo_cache_dir(repository)
    labels_dir = repo_dir / "labels"
    if labels_dir.exists():
        size = sum(f.stat().st_size for f in labels_dir.rglob("*") if f.is_file())
        shutil.rmtree(labels_dir)
        print(f"Flushed label cache for {repository} ({size / 1e6:.1f} MB)")
    else:
        print(f"No label cache found for {repository}")


def _parse_cli_tool(remaining: list[str], parser: argparse.ArgumentParser) -> list[str] | None:
    """Parse CLI tool from remaining args. Returns command list or None."""
    if not remaining or not remaining[0].startswith("--"):
        return None
    tool_name = remaining[0][2:]
    if shutil.which(tool_name) is None:
        parser.error(f"CLI tool '{tool_name}' not found on PATH")
    return [tool_name] + remaining[1:]


def _build_model_identity(arguments: argparse.Namespace, cli_command: list[str] | None) -> str:
    """Build model identity string from arguments."""
    identity_parts = []
    if arguments.local:
        identity_parts.append(f"local:{arguments.local}")
        if arguments.local_file:
            identity_parts.append(f"file:{arguments.local_file}")
    if cli_command:
        identity_parts.append(f"cli:{shlex.join(cli_command)}")
    if arguments.openai:
        identity_parts.append(f"openai:{arguments.completion_model}")
    return "+".join(identity_parts)


def main():
    parser = argparse.ArgumentParser(
        prog = "semantic-navigator",
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
    parser.add_argument("--flush-cache", action = "store_true", help = "Delete cached labels for the given repository (preserves embeddings)")
    parser.add_argument("--flush-labels", action = "store_true", help = "Delete cached labels only (keeps embeddings)")
    parser.add_argument("--status", action = "store_true", help = "Show cache status for the repository")
    parser.add_argument("--erase-models", action = "store_true", help = "Delete downloaded HuggingFace models")
    parser.add_argument("--openai", action = "store_true", help = "Use OpenAI API for labeling (requires OPENAI_API_KEY)")
    parser.add_argument("--completion-model", default = "gpt-4o-mini", help = "OpenAI model for labeling (default: gpt-4o-mini)")
    parser.add_argument("--openai-embedding-model", default = None, help = "Use OpenAI API for embeddings (e.g. text-embedding-3-large)")
    parser.add_argument("--local", default = None)
    parser.add_argument("--local-file", default = None)
    parser.add_argument("--n-ctx", type = int, default = None)
    parser.add_argument("--debug", action = "store_true")
    arguments, remaining = parser.parse_known_args()

    if arguments.list_devices:
        list_devices()
        return

    if arguments.erase_models:
        _handle_erase_models()
        return

    if arguments.repository is None:
        parser.error("the following arguments are required: repository")

    if arguments.flush_cache:
        _flush_cache(arguments.repository)
        return

    if arguments.flush_labels:
        _flush_labels(arguments.repository)
        return

    if arguments.status:
        _show_status(arguments.repository)
        return

    cli_command = _parse_cli_tool(remaining, parser)
    has_cli_tool = cli_command is not None

    if arguments.local is None and not has_cli_tool and not arguments.openai:
        parser.error("no backend specified (e.g. --openai, --gemini, --llm, or --local)")

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

    if arguments.concurrency is not None and arguments.local is not None and not has_cli_tool and not arguments.openai:
        parser.error("--concurrency has no effect with --local only (local model concurrency is 1 per device)")

    try:
        devices = [int(d.strip()) for d in arguments.device.split(",")]
    except ValueError:
        parser.error(f"--device must be comma-separated integers, got: {arguments.device}")

    if arguments.n_ctx is None:
        arguments.n_ctx = 8192
    if arguments.concurrency is None:
        arguments.concurrency = 4

    model_identity = _build_model_identity(arguments, cli_command)
    openai_model = arguments.completion_model if arguments.openai else None

    # When using --openai with --openai-embedding-model, override the embedding model name for cache identity
    embedding_model_name = arguments.embedding_model
    if arguments.openai_embedding_model:
        embedding_model_name = arguments.openai_embedding_model

    facets = initialize(arguments.repository, model_identity, cli_command, arguments.local, arguments.local_file, embedding_model_name, arguments.gpu, arguments.cpu, arguments.cpu_offload, devices, arguments.gpu_layers, arguments.batch_size, arguments.concurrency, arguments.n_ctx, arguments.timeout, arguments.debug, openai_model=openai_model, openai_embedding_model=arguments.openai_embedding_model)

    async def async_tasks():
        timings: dict[str, float] = {}
        total_start = time.monotonic()

        with timed("Reading & embedding", timings):
            initial_cluster = await embed(facets, arguments.repository)

        print(f"Processing {len(initial_cluster.embeds)} files...")
        tree_ = await tree(facets, arguments.repository, initial_cluster, timings)

        total = time.monotonic() - total_start
        parts = " | ".join(f"{k}: {_fmt_time(v)}" for k, v in timings.items())
        print(f"Done! Total: {_fmt_time(total)} ({parts})")
        return tree_

    tree_ = asyncio.run(async_tasks())

    UI(tree_).run()

if __name__ == "__main__":
    main()
