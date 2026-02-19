# Semantic project navigator

Introductory blog post about this tool: [Browse code by meaning](https://haskellforall.com/2026/02/browse-code-by-meaning).

This project provides a tool that lets you browse a repository's files
by their meaning:

https://github.com/user-attachments/assets/95f04087-453d-413e-8661-5bcd7dce4062


## Setup

Currently I've only built and run this script using uv and Nix.  However, you
can feel free to submit pull requests for other installation instructions if
you've vetted them.

You will need a labeling backend — one of:
- A CLI AI tool that reads from stdin and writes to stdout (e.g. `gemini`, `llm`, `aichat`)
- OpenAI API via `--openai` (requires `OPENAI_API_KEY`)
- A local GGUF model via `--local`

You can combine multiple backends (e.g. `--local model --openai`) and work will
be distributed across them.

### uv

You can run the script in a single command, like this:

```ShellSession
$ uvx git+https://github.com/Gabriella439/semantic-navigator ./path/to/repository
```

… or you can install the script:

```ShellSession
$ uv tool install git+https://github.com/Gabriella439/semantic-navigator

$ semantic-navigator ./path/to/repository
```

### Nix

You can run the script in a single command, like this:

```ShellSession
$ nix run github:Gabriella439/semantic-navigator -- ./path/to/repository
```

… or you can build the script and run it separately:

```ShellSession
$ nix build github:Gabriella439/semantic-navigator

$ ./result/bin/semantic-navigator ./path/to/repository
```

… or you can install the script:

```ShellSession
$ nix profile install github:Gabriella439/semantic-navigator

$ semantic-navigator ./path/to/repository
```

## Usage

Depending on the size of the project it will probably take between a few
seconds to a minute to produce a tree viewer.  You must specify which backend
to use:

```ShellSession
# Gemini CLI
$ semantic-navigator --gemini ./path/to/repository

# Simon Willison's llm with GPT-4o
$ semantic-navigator ./path/to/repository --llm -m gpt-4o

# aichat
$ semantic-navigator --aichat ./path/to/repository
```

Any CLI tool that reads a prompt from stdin and writes its response to stdout
will work — just pass `--<tool-name>` and any additional arguments.

### OpenAI

You can use OpenAI directly via `--openai`. Install the optional dependency
first:

```ShellSession
$ uv sync --extra openai
```

Then run with `--openai`:

```ShellSession
# Default (gpt-4o-mini for labeling, text-embedding-3-small for embeddings)
$ semantic-navigator --openai ./path/to/repository

# Custom completion model
$ semantic-navigator --openai --completion-model gpt-4o ./path/to/repository

# Higher-quality embeddings
$ semantic-navigator --openai --openai-embedding-model text-embedding-3-large ./path/to/repository

# OpenAI for labeling, local fastembed for embeddings
$ semantic-navigator --openai --openai-embedding-model local ./path/to/repository
```

This requires the `OPENAI_API_KEY` environment variable to be set.

### Local LLM

Instead of an external CLI tool, you can use a local GGUF model via
`llama-cpp-python`. Install the optional dependency first:

```ShellSession
$ uv sync --extra local
```

Then use `--local` with either a local file path or a Hugging Face repo ID:

```ShellSession
# Local GGUF file
$ semantic-navigator --local ./models/qwen2.5-7b-q4_k_m.gguf ./repo

# Hugging Face repo (auto-selects best quantization for your hardware)
$ semantic-navigator --local Qwen/Qwen2.5-7B-Instruct-GGUF ./repo

# Specific quantization
$ semantic-navigator --local bartowski/Qwen2.5-7B-Instruct-GGUF --local-file "*Q8_0.gguf" ./repo
```

### GPU acceleration

Use `--gpu` to enable GPU offloading for both the embedding model (DirectML)
and local LLM inference (Vulkan, CUDA, etc.):

```ShellSession
# GPU-accelerated local model
$ semantic-navigator --local Qwen/Qwen2.5-7B-Instruct-GGUF --gpu ./repo

# Multiple GPUs (work distributed across devices)
$ semantic-navigator --local Qwen/Qwen2.5-7B-Instruct-GGUF --gpu --device 0,1 ./repo

# GPU for local model + CPU fallback worker
$ semantic-navigator --local Qwen/Qwen2.5-7B-Instruct-GGUF --gpu --cpu ./repo

# GPU with CPU offload for embeddings
$ semantic-navigator --local Qwen/Qwen2.5-7B-Instruct-GGUF --gpu --cpu-offload ./repo
```

Install the GPU extra for DirectML-accelerated embeddings:

```ShellSession
$ uv sync --extra gpu
```

To see available GPU devices and their VRAM:

```ShellSession
$ semantic-navigator --list-devices
```

When auto-selecting a GGUF quantization from Hugging Face, the tool picks the
largest model that fits in 60% of available VRAM (reserving space for KV cache
and runtime overhead).

### Combining backends

You can use multiple backends simultaneously — work is distributed across them:

```ShellSession
# Local model + OpenAI running in parallel
$ semantic-navigator --local Qwen/Qwen2.5-7B-Instruct-GGUF --openai ./repo

# Local model + CLI tool
$ semantic-navigator --local Qwen/Qwen2.5-7B-Instruct-GGUF --gemini ./repo
```

### Advanced options

```
--concurrency N       Number of concurrent workers for CLI/OpenAI backends (default: 4)
--n-ctx N             Context window size for local models (default: 8192)
--gpu-layers N        Override automatic GPU layer count for local models
--batch-size N        Embedding batch size (default: 256 CPU, 32 GPU)
--timeout N           Timeout in seconds for CLI tool calls (default: 60)
--embedding-model M   Override fastembed model (default: BAAI/bge-large-en-v1.5)
--debug               Print prompts, raw outputs, and debug info
```

## Caching

Results are cached to avoid redundant work on subsequent runs:

- **Embeddings** are cached globally (shared across repositories) since they
  depend only on file content, not the labeling model
- **Labels** are cached per-repository and per-model-identity, so switching
  backends produces fresh labels

Cache management commands:

```ShellSession
# Show label cache coverage for a repository
$ semantic-navigator --status ./repo

# Delete label cache for a repository (keeps embeddings)
$ semantic-navigator --flush-labels ./repo

# Delete all cached data for a repository (labels + repo metadata)
$ semantic-navigator --flush-cache ./repo

# Delete downloaded HuggingFace models
$ semantic-navigator --erase-models
```

If the tool crashes during labeling, completed labels are preserved — the next
run picks up where it left off.

## How it works

For small repositories (up to 20 files) you won't see any clusters and the tool
will just summarize the individual files:

![](./images/small.png)

This is a tradeoff the tool makes for ergonomic reasons: the tool avoids
subdividing clusters with 20 files or fewer.

For a medium-sized repository you'll begin to see top-level clusters:

![](./images/medium.png)

The label for each cluster describes the files within that cluster and will
also display a file pattern if all files within the cluster begin with the same
prefix or suffix.  In the above example the "Project Prelude" doesn't display a
file pattern because there is no common prefix or suffix within the cluster,
whereas the "Condition Rendering" cluster displays a file pattern of
`*/Condition.dhall` because both files within the cluster share the same
suffix.

For an even larger repository you'll begin to see nested clusters:

![](./images/large.png)

On a somewhat modern MacBook this tool can handle up to ≈10,000 files within a
few minutes.

You can use this tool on any text documents; not just code!  For example,
here's the result when running the tool on the repository for my self-hosted
blog:

![](./images/haskellforall.png)

In other words, this tool isn't just a code indexer or project indexer; it's a
general file indexer.

## Development

If you use Nix and `direnv` this project provides a `.envrc` which
automatically provides a virtual environment with all the necessary
dependencies (both Python and non-Python dependencies).

Otherwise if you don't use `direnv` you can enter the virtual environment using:

```ShellSession
$ nix develop
```

… and you can test any of the setup commands with a local checkout by replacing
`github:Gabriella439/semantic-navigator` with `.`, like this:

```ShellSession
$ nix run . -- ./path/to/repository
```

To run the test suite:

```ShellSession
$ uv run pytest
```

Embeddings are generated locally by default using [fastembed](https://github.com/qdrant/fastembed) (ONNX-based), so no API key is needed for the embedding step (unless you opt into OpenAI embeddings via `--openai`).
