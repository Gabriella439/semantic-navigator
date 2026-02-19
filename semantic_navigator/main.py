import argparse
import asyncio

import textual
import textual.app
import textual.widgets

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

def main():
    parser = argparse.ArgumentParser(
        prog = "semantic-navigator",
        description = "Cluster documents by semantic facets",
    )

    parser.add_argument("repository")
    parser.add_argument("--completion-model", default = "gpt-4o-mini")
    parser.add_argument("--embedding-model", default = "BAAI/bge-large-en-v1.5",
                        help = "fastembed model for local embeddings (default: BAAI/bge-large-en-v1.5)")
    parser.add_argument("--openai-embedding-model", default = None,
                        help = "Use OpenAI embedding model instead of local fastembed (e.g. text-embedding-3-small)")
    arguments = parser.parse_args()

    facets = initialize(
        arguments.completion_model,
        arguments.embedding_model,
        openai_embedding_model = arguments.openai_embedding_model,
    )

    async def async_tasks():
        initial_cluster = await embed(facets, arguments.repository)

        tree_ = await tree(facets, arguments.repository, initial_cluster)

        return tree_

    tree_ = asyncio.run(async_tasks())

    UI(tree_).run()

if __name__ == "__main__":
    main()
