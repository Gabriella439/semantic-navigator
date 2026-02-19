import argparse
import asyncio
import shutil

import textual
import textual.app
import textual.widgets

from semantic_navigator.cache import repo_cache_dir
from semantic_navigator.models import Tree
from semantic_navigator.pipeline import initialize, embed, tree


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


def _flush_cache(repository: str):
    """Delete cached labels for the given repository (preserves embeddings)."""
    repo_dir = repo_cache_dir(repository)
    if repo_dir.exists():
        size = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file())
        shutil.rmtree(repo_dir)
        print(f"Flushed repo cache for {repository} ({size / 1e6:.1f} MB)")
    else:
        print(f"No cache found.")


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


def _show_status(repository: str):
    """Show cache status for a repository."""
    import os
    from semantic_navigator.cache import content_hash, label_cache_dir, list_cached_keys
    from semantic_navigator.pipeline import _generate_paths

    file_paths = _generate_paths(directory=repository)

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


def main():
    parser = argparse.ArgumentParser(
        prog = "semantic-navigator",
        description = "Cluster documents by semantic facets",
    )

    parser.add_argument("repository", nargs = "?")
    parser.add_argument("--completion-model", default = "gpt-5-mini")
    parser.add_argument("--embedding-model", default = "text-embedding-3-large",
                        help = "OpenAI embedding model (default: text-embedding-3-large)")
    parser.add_argument("--local-embedding-model", default = None,
                        help = "Use local fastembed model instead of OpenAI (e.g. BAAI/bge-large-en-v1.5)")
    parser.add_argument("--flush-cache", action = "store_true",
                        help = "Delete cached labels for the given repository (preserves embeddings)")
    parser.add_argument("--flush-labels", action = "store_true",
                        help = "Delete cached labels only (keeps embeddings)")
    parser.add_argument("--status", action = "store_true",
                        help = "Show cache status for the repository")
    arguments = parser.parse_args()

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

    facets = initialize(
        arguments.repository,
        arguments.completion_model,
        arguments.embedding_model,
        local_embedding_model = arguments.local_embedding_model,
    )

    async def async_tasks():
        initial_cluster = await embed(facets, arguments.repository)

        tree_ = await tree(facets, arguments.repository, initial_cluster)

        return tree_

    tree_ = asyncio.run(async_tasks())

    UI(tree_).run()

if __name__ == "__main__":
    main()
