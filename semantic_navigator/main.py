import argparse
import asyncio
import shlex
import shutil
import sys
import time

import textual
import textual.app
import textual.widgets

from semantic_navigator.cache import content_hash, list_cached_keys, repo_cache_dir
from semantic_navigator.gpu import list_devices
from semantic_navigator.inference import initialize
from semantic_navigator.models import Facets, Tree
from semantic_navigator.pipeline import _generate_paths, embed, tree
from semantic_navigator.util import _fmt_time, timed


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
    repo_dir = repo_cache_dir(repository)
    if repo_dir.exists():
        size = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file())
        shutil.rmtree(repo_dir)
        print(f"Flushed repo cache for {repository} ({size / 1e6:.1f} MB)")
    else:
        print(f"No cache found.")


def _flush_labels(repository: str):
    repo_dir = repo_cache_dir(repository)
    labels_dir = repo_dir / "labels"
    if labels_dir.exists():
        size = sum(f.stat().st_size for f in labels_dir.rglob("*") if f.is_file())
        shutil.rmtree(labels_dir)
        print(f"Flushed label cache for {repository} ({size / 1e6:.1f} MB)")
    else:
        print(f"No label cache found for {repository}")


def _show_status(repository: str):
    import os

    file_paths = _generate_paths(repository)

    file_hashes: list[tuple[str, str]] = []
    for path in file_paths:
        try:
            absolute_path = os.path.join(repository, path)
            with open(absolute_path, "rb") as f:
                text = f.read().decode("utf-8")
            file_hashes.append((path, content_hash(f"{path}:\n\n{text}")))
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError, PermissionError):
            pass

    total = len(file_hashes)
    print(f"Repository: {repository} ({total} files)")
    print()

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


_SUBCOMMANDS = {"openai", "local", "cli"}


def _parse_subcommand(argv: list[str]) -> tuple[str, list[str]]:
    """Detect subcommand from argv, defaulting to 'openai'."""
    for i, arg in enumerate(argv):
        if arg.startswith("-"):
            continue
        if arg in _SUBCOMMANDS:
            return arg, argv[:i] + argv[i + 1:]
        break
    return "openai", argv


def _build_model_identity(subcommand: str, arguments: argparse.Namespace) -> str:
    if subcommand == "openai":
        return f"openai:{arguments.completion_model}"
    elif subcommand == "local":
        parts = [f"local:{arguments.local}"]
        if arguments.local_file:
            parts.append(f"file:{arguments.local_file}")
        return "+".join(parts)
    elif subcommand == "cli":
        return f"cli:{shlex.join(arguments.cli_command)}"
    return ""


def main():
    # Pre-parse to detect subcommand before argparse runs
    argv = sys.argv[1:]
    subcommand, remaining = _parse_subcommand(argv)

    parser = argparse.ArgumentParser(
        prog = "semantic-navigator",
        description = "Cluster documents by semantic facets",
    )

    # Utility flags (available on all subcommands)
    parser.add_argument("--list-devices", action = "store_true")
    parser.add_argument("--erase-models", action = "store_true")
    parser.add_argument("--flush-cache", action = "store_true")
    parser.add_argument("--flush-labels", action = "store_true")
    parser.add_argument("--status", action = "store_true")
    parser.add_argument("--debug", action = "store_true")

    # Subcommand-specific flags
    if subcommand == "openai":
        parser.add_argument("repository", nargs = "?")
        parser.add_argument("--completion-model", default = "gpt-5-mini",
                            help = "OpenAI model for labeling (default: gpt-5-mini)")
        parser.add_argument("--embedding-model", default = "text-embedding-3-large",
                            help = "OpenAI embedding model (default: text-embedding-3-large)")
        parser.add_argument("--concurrency", type = int, default = 4)
        parser.add_argument("--timeout", type = int, default = 60)

    elif subcommand == "local":
        parser.add_argument("repository", nargs = "?")
        parser.add_argument("--embedding-model", default = "BAAI/bge-large-en-v1.5",
                            help = "fastembed model for local embeddings (default: BAAI/bge-large-en-v1.5)")
        parser.add_argument("--local", required = True,
                            help = "HuggingFace model repo ID for GGUF completions")
        parser.add_argument("--local-file", default = None,
                            help = "Specific GGUF file name within the model repo")
        parser.add_argument("--gpu", action = "store_true")
        parser.add_argument("--cpu", action = "store_true",
                            help = "Add a CPU local model worker alongside GPU workers")
        parser.add_argument("--cpu-offload", action = "store_true")
        parser.add_argument("--device", type = str, default = "0",
                            help = "GPU device IDs, comma-separated (e.g. 0,1)")
        parser.add_argument("--gpu-layers", type = int, default = None)
        parser.add_argument("--batch-size", type = int, default = None)
        parser.add_argument("--n-ctx", type = int, default = None)
        parser.add_argument("--timeout", type = int, default = 60)

    elif subcommand == "cli":
        parser.add_argument("tool", help = "CLI tool name (e.g. gemini, llm)")
        parser.add_argument("repository", nargs = "?")
        parser.add_argument("--embedding-model", default = "BAAI/bge-large-en-v1.5",
                            help = "fastembed model for local embeddings (default: BAAI/bge-large-en-v1.5)")
        parser.add_argument("--concurrency", type = int, default = 4)
        parser.add_argument("--timeout", type = int, default = 60)

    if subcommand == "cli":
        arguments, tool_remaining = parser.parse_known_args(remaining)
        arguments.cli_command = [arguments.tool] + tool_remaining
    else:
        arguments = parser.parse_args(remaining)

    # Handle utility commands
    if arguments.list_devices:
        list_devices()
        return

    if arguments.erase_models:
        _handle_erase_models()
        return

    if not hasattr(arguments, "repository") or arguments.repository is None:
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

    # Subcommand-specific validation and initialization
    if subcommand == "openai":
        model_identity = _build_model_identity(subcommand, arguments)
        facets = initialize(
            arguments.repository, model_identity, None, None, None,
            arguments.embedding_model, False, False, False, [0], None,
            None, arguments.concurrency, 8192, arguments.timeout,
            arguments.debug, openai_model=arguments.completion_model,
            openai_embedding_model=arguments.embedding_model,
        )

    elif subcommand == "local":
        if arguments.cpu_offload and not arguments.gpu:
            parser.error("--cpu-offload requires --gpu")
        if arguments.cpu and not arguments.gpu:
            parser.error("--cpu only makes sense with --gpu (without --gpu, CPU is already the default)")

        try:
            devices = [int(d.strip()) for d in arguments.device.split(",")]
        except ValueError:
            parser.error(f"--device must be comma-separated integers, got: {arguments.device}")

        if arguments.n_ctx is None:
            arguments.n_ctx = 8192

        model_identity = _build_model_identity(subcommand, arguments)
        facets = initialize(
            arguments.repository, model_identity, None, arguments.local,
            arguments.local_file, arguments.embedding_model, arguments.gpu,
            arguments.cpu, arguments.cpu_offload, devices, arguments.gpu_layers,
            arguments.batch_size, 1, arguments.n_ctx, arguments.timeout,
            arguments.debug,
        )

    elif subcommand == "cli":
        if shutil.which(arguments.tool) is None:
            parser.error(f"CLI tool '{arguments.tool}' not found on PATH")

        model_identity = _build_model_identity(subcommand, arguments)
        facets = initialize(
            arguments.repository, model_identity, arguments.cli_command,
            None, None, arguments.embedding_model, False, False, False,
            [0], None, None, arguments.concurrency, 8192, arguments.timeout,
            arguments.debug,
        )

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

    try:
        tree_ = asyncio.run(async_tasks())
    except KeyboardInterrupt:
        print("\nInterrupted. Progress has been cached and will resume on next run.")
        return

    UI(tree_).run()

if __name__ == "__main__":
    main()
